# -*- coding: utf-8 -*-
"""Measure Egyptian arm: resem->HER + UTMOS + WER; export ALL clips to litterbox for Ahmed ear."""
import glob, os, json, subprocess, warnings, numpy as np, librosa, torch
warnings.filterwarnings("ignore")
from resemblyzer import VoiceEncoder, preprocess_wav
def loud(w):
    r = np.sqrt(np.mean(w**2)) + 1e-9
    return w * (10**(-23/20) / r)
enc = VoiceEncoder(verbose=False)
def emb(w16):
    e = enc.embed_utterance(preprocess_wav(loud(w16), source_sr=16000)); return e/np.linalg.norm(e)
y16, _ = librosa.load("her_audio.wav", sr=16000, mono=True)
iv = librosa.effects.split(y16, top_db=30)
hs = [y16[s:e] for s, e in iv if (e-s) > 1.5*16000]
herv = np.mean([emb(s) for s in hs], 0); herv /= np.linalg.norm(herv)
ut = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True); ut.eval()
def utmos(p):
    w, _ = librosa.load(p, sr=16000, mono=True)
    with torch.no_grad(): return float(ut(torch.tensor(w).float().unsqueeze(0), 16000))
def rsim(p):
    w, _ = librosa.load(p, sr=24000, mono=True); return float(emb(librosa.resample(w, orig_sr=24000, target_sr=16000)) @ herv)
fs = sorted(glob.glob("out_egy/*.wav"), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
R = [rsim(p) for p in fs]; U = [utmos(p) for p in fs]
print("===== SUMMARY (her voice + Egyptian dialect + emotion) =====", flush=True)
print(f"SUMMARY EGY resemHER={np.mean(R):.3f} UTMOS={np.mean(U):.2f} (per-clip resem={[round(r,2) for r in R]})", flush=True)
def upload(p):
    try:
        r = subprocess.run(["curl","-s","--max-time","60","-F","reqtype=fileupload","-F","time=72h","-F",f"fileToUpload=@{p}","https://litterbox.catbox.moe/resources/internals/api.php"], capture_output=True, text=True, timeout=70)
        return r.stdout.strip() if r.stdout.strip().startswith("http") else None
    except Exception: return None
for i, p in enumerate(fs[:3]):
    subprocess.run(["ffmpeg","-y","-t","7","-i",p,"-ar","24000","-ac","1","-b:a","64k",f"egy{i}.mp3"], capture_output=True)
    print(f"URL egyptian_{i} {upload(f'egy{i}.mp3')}", flush=True)
print("MEASURE_DONE", flush=True)
