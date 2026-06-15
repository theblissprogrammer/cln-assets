# -*- coding: utf-8 -*-
"""Measure the generation-time sweep: per (ref,exag) -> resem->her, UTMOS, Arabic WER, + delivery
scalars (f0_range_st, f0_dyn, f0_energy_corr) vs HER. Best-of-3 by resem per passage. Upload best clips."""
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
    f0=parselmouth.Sound(w.astype(np.float64),sampling_frequency=SR).to_pitch(0.01,100,500).selected_array['frequency']
    v=f0[f0>0]
    if len(v)<20: return (np.nan,np.nan,np.nan)
    sv=st(v); rng=np.percentile(sv,95)-np.percentile(sv,5)
    dyn=np.mean(np.abs(np.diff(st(f0[f0>0]))))/0.01
    rt=librosa.feature.rms(y=w.astype(np.float32),frame_length=int(0.025*SR),hop_length=int(0.01*SR))[0]
    n=min(len(f0),len(rt)); m=f0[:n]>0
    fec=np.corrcoef(st(f0[:n][m]),20*np.log10(rt[:n][m]+1e-6))[0,1] if m.sum()>10 else np.nan
    return (rng,dyn,fec)
# her reference delivery (full)
hr=deliv(y); print(f"HER delivery: f0_range={hr[0]:.1f} f0_dyn={hr[1]:.1f} f0_energy_corr={hr[2]:.2f}",flush=True)

import whisper, jiwer
asr=whisper.load_model("small")
PASS=json.load(open("passages.json"))
def norm(s): return ''.join(c for c in s if c.isalnum() or c==' ').strip()
def wer_of(path,ref):
    try:
        t=asr.transcribe(path,language="ar")["text"]
        return jiwer.wer(norm(ref),norm(t))
    except Exception: return np.nan

def litter(p):
    try:
        r=subprocess.run(["curl","-s","-F","reqtype=fileupload","-F","time=72h","-F",f"fileToUpload=@{p}","https://litterbox.catbox.moe/resources/internals/api.php"],capture_output=True,text=True,timeout=60)
        return r.stdout.strip()
    except Exception as e: return "ERR"

print("\nsetting      resem  UTMOS   WER   f0range f0dyn f0ecorr",flush=True)
results={}
for ref in ["std","expr"]:
    for exag in [0.5,1.0,1.5,2.0]:
        d=f"sw_{ref}_e{int(exag*10)}"
        if not os.path.isdir(d): continue
        rs=[];us=[];ws=[];dr=[];dd=[];df=[]
        for i in range(len(PASS)):
            takes=sorted(glob.glob(f"{d}/p{i}_t*.wav"))
            if not takes: continue
            scored=[(resem(librosa.load(t,sr=SR)[0]),t) for t in takes]
            best_r,best=max(scored)
            w,_=librosa.load(best,sr=SR)
            rs.append(best_r); us.append(utmos(w)); ws.append(wer_of(best,PASS[i]))
            a,b,c=deliv(w); dr.append(a);dd.append(b);df.append(c)
        tag=f"{ref}_e{exag}"
        results[tag]=dict(resem=np.nanmean(rs),utmos=np.nanmean(us),wer=np.nanmean(ws),
                          f0range=np.nanmean(dr),f0dyn=np.nanmean(dd),fec=np.nanmean(df))
        r=results[tag]
        print(f"SWEEP {tag:9s} {r['resem']:.3f}  {r['utmos']:.2f}  {r['wer']:.2f}  {r['f0range']:5.1f}  {r['f0dyn']:5.1f}  {r['fec']:+.2f}  (her {hr[0]:.1f}/{hr[1]:.1f}/{hr[2]:+.2f})",flush=True)

# upload best take of a few key settings for the ear
for tag in ["std_e0.5","expr_e1.5","expr_e2.0"]:
    ref,exag=tag.split("_e"); d=f"sw_{ref}_e{int(float(exag)*10)}"
    t=sorted(glob.glob(f"{d}/p0_t*.wav"))
    if t: print(f"URL {tag} {litter(t[0])}",flush=True)
json.dump({k:{kk:float(vv) for kk,vv in v.items()} for k,v in results.items()},open("sweep_results.json","w"))
print("MEASURE_DONE",flush=True)
