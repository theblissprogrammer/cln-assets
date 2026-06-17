# -*- coding: utf-8 -*-
"""PHASE 1b — FULL EXTRACTION per verified bilingual speaker: pull substantial single-source audio per
language (consistent mic = clean features), FILTER to the speaker (drop interviewer/other voices via
resemblyzer), segment, Whisper-transcribe both, compute clean features (VQ grain + delivery + idiolect).
Output = reliable per-speaker L1/L2 feature table + transcripts + (tar of audio for training)."""
import os, glob, json, warnings, subprocess, numpy as np, librosa, soundfile as sf, parselmouth, torch
warnings.filterwarnings("ignore")
from parselmouth.praat import call
from resemblyzer import VoiceEncoder, preprocess_wav
import whisper
SR=24000; st=lambda hz:12*np.log2(np.maximum(hz,1e-6)/55.0)
os.makedirs("corpus", exist_ok=True)
enc=VoiceEncoder(verbose=False)
def loud(w): r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
def emb16(w): e=enc.embed_utterance(preprocess_wav(loud(w),source_sr=16000)); return e/np.linalg.norm(e)
asr=whisper.load_model("small")
EN_FILL=set("uh um ah er hmm".split()); EN_MARK=["you know","i mean","kind of","sort of"]; EN_MARK1=set("like so right actually well basically".split())
AR_MARK=["شوف","يعني","طيب","والله","خلاص","بقى","يا اخي"]
def idiolect(words, lang):
    if not words: return dict(marker_min=0,pause_min=0,meangap=0)
    toks=[w['w'].strip().lower() for w in words]; dur=(words[-1]['e']-words[0]['s'])/60+1e-9
    low=' '.join(toks)
    if lang=='en': nm=sum(1 for t in toks if t in EN_FILL or t in EN_MARK1)+sum(low.count(m) for m in EN_MARK)
    else: nm=sum(low.count(m) for m in AR_MARK)
    gaps=np.array([words[i]['s']-words[i-1]['e'] for i in range(1,len(words)) if words[i]['s']>words[i-1]['e']])
    return dict(marker_min=nm/dur, pause_min=int((gaps>0.4).sum())/dur, meangap=float(gaps.mean()) if len(gaps) else 0)
def vq_deliv(w):
    snd=parselmouth.Sound((w/(np.max(np.abs(w))+1e-9)).astype(np.float64),sampling_frequency=SR)
    o={}
    try: o['HNR']=float(call(snd.to_harmonicity_cc(0.01,75,0.1,1.0),'Get mean',0,0))
    except: o['HNR']=np.nan
    try:
        pp=call(snd,'To PointProcess (periodic, cc)',75,500)
        o['jit']=float(call(pp,'Get jitter (local)',0,0,0.0001,0.02,1.3)*100); o['shim']=float(call([snd,pp],'Get shimmer (local)',0,0,0.0001,0.02,1.3,1.6)*100)
    except: o['jit']=o['shim']=np.nan
    S=np.abs(librosa.stft(w.astype('float32'),n_fft=2048))**2; f=librosa.fft_frequencies(sr=SR,n_fft=2048); b=(f>200)&(f<8000)
    o['tilt']=float(np.polyfit(np.log10(f[b]),10*np.log10(S[b].mean(1)+1e-9),1)[0])
    f0=snd.to_pitch(0.01,65,500).selected_array['frequency']; v=f0[f0>0]
    o['f0med']=float(np.median(v)); o['range']=float(np.percentile(st(v),95)-np.percentile(st(v),5)); o['dyn']=float(np.mean(np.abs(np.diff(st(f0[f0>0]))))/0.01)
    return o
QUERIES=json.load(open("verified_queries.json"))   # {name:[en_query,ar_query]}
TABLE={}
for name,(enq,arq) in QUERIES.items():
    rec={}
    for lang,q in [('en',enq),('ar',arq)]:
        d=f"corpus/{name}_{lang}.wav"
        if not os.path.exists(d):
            subprocess.run(["yt-dlp",f"ytsearch1:{q}","--download-sections","*30-540","-x","--audio-format","wav","-q","-o",f"corpus/{name}_{lang}.%(ext)s"],capture_output=True,timeout=300)
        if not os.path.exists(d): continue
        w16,_=librosa.load(d,sr=16000,mono=True)
        iv=librosa.effects.split(w16,top_db=30); segs=[(s,e) for s,e in iv if (e-s)>1.5*16000]
        if len(segs)<5: continue
        E=[emb16(w16[s:e]) for s,e in segs]; c=np.mean(E,0); c/=np.linalg.norm(c)
        keep=[segs[i] for i in range(len(segs)) if float(E[i]@c)>0.80]   # filter to dominant speaker
        if len(keep)<5: continue
        w,_=librosa.load(d,sr=SR,mono=True)
        spk=np.concatenate([w[int(s*SR/16000):int(e*SR/16000)] for s,e in keep])
        tr=asr.transcribe(d, language=lang, word_timestamps=True)
        words=[{'w':x['word'].strip(),'s':x['start'],'e':x['end']} for sg in tr['segments'] for x in sg.get('words',[])]
        rec[lang]={**vq_deliv(spk), **idiolect(words,lang), 'sec':len(spk)/SR}
    if 'en' in rec and 'ar' in rec:
        TABLE[name]=rec; print(f"{name:14s} EN {rec['en']['sec']:.0f}s AR {rec['ar']['sec']:.0f}s | f0med {rec['en']['f0med']:.0f}/{rec['ar']['f0med']:.0f} range {rec['en']['range']:.1f}/{rec['ar']['range']:.1f} HNR {rec['en']['HNR']:.1f}/{rec['ar']['HNR']:.1f} mark/min {rec['en']['marker_min']:.1f}/{rec['ar']['marker_min']:.1f}",flush=True)
json.dump(TABLE, open("corpus/feature_table.json","w"))
# clean L1<->L2 correlations
print(f"\nCLEAN feature table: {len(TABLE)} speakers")
for f in ['HNR','jit','shim','tilt','f0med','range','dyn','marker_min']:
    l1=np.array([TABLE[s]['en'][f] for s in TABLE]); l2=np.array([TABLE[s]['ar'][f] for s in TABLE]); ok=~(np.isnan(l1)|np.isnan(l2))
    if ok.sum()>3:
        print(f"  {f:10s} L1<->L2 corr {np.corrcoef(l1[ok],l2[ok])[0,1]:+.2f}  shift {np.mean(l2[ok]-l1[ok]):+.1f}",flush=True)
print("MEASURE_DONE",flush=True)
