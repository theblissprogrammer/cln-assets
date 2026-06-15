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
import whisper, jiwer
asr=whisper.load_model("medium")
def norm_ar(t):
    t=re.sub(r'[ً-ْٰـ]','',t).replace('أ','ا').replace('إ','ا').replace('آ','ا').replace('ى','ي').replace('ة','ه').replace('ؤ','و').replace('ئ','ي')
    return re.sub(r'\s+',' ',re.sub(r'[^ء-ي\s]',' ',t)).strip()
def wer_of(p,ref):
    try: hyp=asr.transcribe(p,language='ar',fp16=False)['text']
    except Exception: return 1.0
    r,h=norm_ar(ref),norm_ar(hyp); return float(jiwer.wer(r,h)) if r else 1.0
TEXTS=json.load(open("texts.json"))
def grp(d):
    fs=sorted(glob.glob(f"{d}/*.wav"),key=lambda x:int(os.path.splitext(os.path.basename(x))[0]))
    if not fs: return None
    return (float(np.mean([rsim(p) for p in fs])), float(np.mean([utmos(p) for p in fs])),
            float(np.mean([wer_of(p,TEXTS[int(os.path.splitext(os.path.basename(p))[0])]) for p in fs])))
print("===== SUMMARY (Chatterbox + EGYPTIAN dialect) =====", flush=True)
for d in sorted(glob.glob("out_cb_*")):
    g=grp(d)
    if g: print(f"SUMMARY {d:12s} resemHER={g[0]:.3f} UTMOS={g[1]:.2f} WER={g[2]:.3f}", flush=True)
def upload(p):
    try:
        r=subprocess.run(["curl","-s","--max-time","60","-F","reqtype=fileupload","-F","time=72h","-F",f"fileToUpload=@{p}","https://litterbox.catbox.moe/resources/internals/api.php"],capture_output=True,text=True,timeout=70)
        return r.stdout.strip() if r.stdout.strip().startswith("http") else None
    except Exception: return None
for cfg in ["0p3","0p5"]:
    for i in range(2):
        src=f"out_cb_{cfg}/{i}.wav"
        if os.path.exists(src):
            subprocess.run(["ffmpeg","-y","-t","7","-i",src,"-ar","24000","-ac","1","-b:a","64k",f"cbegy_{cfg}_{i}.mp3"],capture_output=True)
            print(f"URL cbegy_{cfg}_{i} {upload(f'cbegy_{cfg}_{i}.mp3')}", flush=True)
print("MEASURE_DONE", flush=True)
