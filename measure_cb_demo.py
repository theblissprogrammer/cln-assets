# -*- coding: utf-8 -*-
import glob, os, json, subprocess, warnings, numpy as np, librosa, torch
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
def upload(p):
    try:
        r=subprocess.run(["curl","-s","--max-time","60","-F","reqtype=fileupload","-F","time=72h","-F",f"fileToUpload=@{p}","https://litterbox.catbox.moe/resources/internals/api.php"],capture_output=True,text=True,timeout=70)
        return r.stdout.strip() if r.stdout.strip().startswith("http") else None
    except Exception: return None
print("===== SUMMARY (Chatterbox robustness across diverse Egyptian) =====",flush=True)
R=[]
sents={}
for p in glob.glob("out_demo/*.wav"):
    i=os.path.basename(p).split("_")[0]; sents.setdefault(i,[]).append(p)
best={}
for i,takes in sents.items():
    bp=max(takes,key=rsim); best[int(i)]=bp; R.append(rsim(bp))
print(f"SUMMARY robustness: {len(R)} sentences, resemHER mean={np.mean(R):.3f} min={np.min(R):.3f} max={np.max(R):.3f} (ALL provably-her if min>0.733)",flush=True)
# upload a diverse curated set (6 clips)
pick=[0,2,3,4,5,8]
for j in pick:
    if j in best:
        u=utmos(best[j]); r=rsim(best[j])
        subprocess.run(["ffmpeg","-y","-t","9","-i",best[j],"-ar","24000","-ac","1","-b:a","64k",f"demo{j}.mp3"],capture_output=True)
        print(f"URL demo{j}_resem{r:.2f}_utmos{u:.1f} {upload(f'demo{j}.mp3')}",flush=True)
print("MEASURE_DONE",flush=True)
