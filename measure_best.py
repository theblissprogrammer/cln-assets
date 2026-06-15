# -*- coding: utf-8 -*-
"""Measure final best-recipe: per setting (msa/egy x e0.5/e1.0) averaged resem/UTMOS/WER + delivery
scalars vs HER. Best-of-takes by resem per passage. Upload the best clip per setting for the ear."""
import os, json, glob, warnings, subprocess, numpy as np, librosa, soundfile as sf, torch
warnings.filterwarnings("ignore")
import parselmouth
from resemblyzer import VoiceEncoder, preprocess_wav
SR=24000
st=lambda hz:12*np.log2(np.maximum(hz,1e-6)/55.0)
enc=VoiceEncoder(verbose=False)
def loud(w): r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
def remb(w16): e=enc.embed_utterance(preprocess_wav(loud(w16),source_sr=16000)); return e/np.linalg.norm(e)
y,_=librosa.load("her_audio.wav",sr=SR,mono=True); iv=librosa.effects.split(y,top_db=30)
herv=np.mean([remb(librosa.resample(y[s:e],orig_sr=SR,target_sr=16000)) for s,e in iv if (e-s)>1.5*SR],0); herv/=np.linalg.norm(herv)
UT=torch.hub.load("tarepan/SpeechMOS:v1.2.0","utmos22_strong",trust_repo=True); UT.eval()
def utmos(w):
    w16=librosa.resample(w.astype(np.float32),orig_sr=SR,target_sr=16000)
    with torch.no_grad(): return float(UT(torch.from_numpy(w16)[None],16000))
def resem(w): return float(remb(librosa.resample(w.astype(np.float32),orig_sr=SR,target_sr=16000))@herv)
def deliv(w):
    f0=parselmouth.Sound(w.astype(np.float64),sampling_frequency=SR).to_pitch(0.01,100,500).selected_array['frequency']; v=f0[f0>0]
    if len(v)<20: return (np.nan,np.nan,np.nan)
    rng=np.percentile(st(v),95)-np.percentile(st(v),5); dyn=np.mean(np.abs(np.diff(st(f0[f0>0]))))/0.01
    rt=librosa.feature.rms(y=w.astype(np.float32),frame_length=int(0.025*SR),hop_length=int(0.01*SR))[0]
    n=min(len(f0),len(rt)); m=f0[:n]>0
    fec=np.corrcoef(st(f0[:n][m]),20*np.log10(rt[:n][m]+1e-6))[0,1] if m.sum()>10 else np.nan
    return (rng,dyn,fec)
hr=deliv(y); print(f"HER delivery: f0_range={hr[0]:.1f} f0_dyn={hr[1]:.1f} f0_ecorr={hr[2]:+.2f}",flush=True)
import whisper, jiwer
asr=whisper.load_model("small")
TX=json.load(open("best_texts.json"))
def norm(s): return ''.join(c for c in s if c.isalnum() or c==' ').strip()
def litter(p):
    try:
        r=subprocess.run(["curl","-s","-F","reqtype=fileupload","-F","time=72h","-F",f"fileToUpload=@{p}","https://litterbox.catbox.moe/resources/internals/api.php"],capture_output=True,text=True,timeout=60)
        return r.stdout.strip()
    except Exception: return "ERR"
print("\nsetting    resem  UTMOS  WER   f0range f0dyn f0ecorr",flush=True)
for tag,texts in [("msa",TX["MSA"]),("egy",TX["EGY"])]:
    for exag in [0.5,1.0]:
        d=f"best_{tag}_e{int(exag*10)}"
        if not os.path.isdir(d): continue
        rs=[];us=[];ws=[];dr=[];dd=[];df=[];bestclip=None;bestr=-9
        for i in range(len(texts)):
            takes=sorted(glob.glob(f"{d}/p{i}_t*.wav"))
            if not takes: continue
            sc=[(resem(librosa.load(t,sr=SR)[0]),t) for t in takes]; br,bt=max(sc)
            w,_=librosa.load(bt,sr=SR); rs.append(br); us.append(utmos(w))
            try: ws.append(jiwer.wer(norm(texts[i]),norm(asr.transcribe(bt,language="ar")["text"])))
            except Exception: ws.append(np.nan)
            a,b,c=deliv(w); dr.append(a);dd.append(b);df.append(c)
            if br>bestr: bestr,bestclip=br,bt
        print(f"BEST {tag}_e{exag} {np.nanmean(rs):.3f}  {np.nanmean(us):.2f}  {np.nanmean(ws):.2f}  {np.nanmean(dr):5.1f}  {np.nanmean(dd):5.1f}  {np.nanmean(df):+.2f}  (her {hr[0]:.1f}/{hr[1]:.1f}/{hr[2]:+.2f})",flush=True)
        if bestclip: print(f"URL {tag}_e{exag} {litter(bestclip)}",flush=True)
print("MEASURE_DONE",flush=True)
