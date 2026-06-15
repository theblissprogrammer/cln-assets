# -*- coding: utf-8 -*-
import glob, os, json, re, subprocess, warnings, numpy as np, librosa, torch
warnings.filterwarnings("ignore")
from resemblyzer import VoiceEncoder, preprocess_wav
def loud(w):
    r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
enc=VoiceEncoder(verbose=False)
def emb(w16): e=enc.embed_utterance(preprocess_wav(loud(w16),source_sr=16000)); return e/np.linalg.norm(e)
y16,_=librosa.load("her_audio.wav",sr=16000,mono=True); iv=librosa.effects.split(y16,top_db=30)
hs=[y16[s:e] for s,e in iv if (e-s)>1.5*16000]
herv=np.mean([emb(s) for s in hs],0); herv/=np.linalg.norm(herv)
ut=torch.hub.load("tarepan/SpeechMOS:v1.2.0","utmos22_strong",trust_repo=True); ut.eval()
def utmos(p):
    w,_=librosa.load(p,sr=16000,mono=True)
    with torch.no_grad(): return float(ut(torch.tensor(w).float().unsqueeze(0),16000))
def rsim(p):
    w,_=librosa.load(p,sr=24000,mono=True); return float(emb(librosa.resample(w,orig_sr=24000,target_sr=16000))@herv)
def f0range(p):
    w,_=librosa.load(p,sr=16000,mono=True); f0,_,_=librosa.pyin(w,fmin=80,fmax=400,sr=16000); v=f0[~np.isnan(f0)]
    return float(12*np.log2((np.percentile(v,95)+1e-9)/(np.percentile(v,5)+1e-9))) if len(v)>5 else 0.0
def upload(p):
    try:
        r=subprocess.run(["curl","-s","--max-time","60","-F","reqtype=fileupload","-F","time=72h","-F",f"fileToUpload=@{p}","https://litterbox.catbox.moe/resources/internals/api.php"],capture_output=True,text=True,timeout=70)
        return r.stdout.strip() if r.stdout.strip().startswith("http") else None
    except Exception: return None
print("===== SUMMARY (Chatterbox best-of-N + emotion; select max-resem take per sentence) =====", flush=True)
for d in sorted(glob.glob("out_ex*")):
    sents={}
    for p in glob.glob(f"{d}/*.wav"):
        i=os.path.basename(p).split("_")[0]; sents.setdefault(i,[]).append(p)
    best=[]
    for i,takes in sorted(sents.items()):
        bp=max(takes,key=rsim); best.append(bp)
    R=[rsim(p) for p in best]; U=[utmos(p) for p in best]; F=[f0range(p) for p in best]
    print(f"SUMMARY {d:8s} bestResemHER={np.mean(R):.3f} (per={[round(r,2) for r in R]}) UTMOS={np.mean(U):.2f} f0range={np.mean(F):.1f}", flush=True)
    # upload 2 best clips
    for k,bp in enumerate(best[:2]):
        subprocess.run(["ffmpeg","-y","-t","7","-i",bp,"-ar","24000","-ac","1","-b:a","64k",f"best_{d}_{k}.mp3"],capture_output=True)
        print(f"URL best_{d}_{k} {upload(f'best_{d}_{k}.mp3')}", flush=True)
print("MEASURE_DONE", flush=True)
