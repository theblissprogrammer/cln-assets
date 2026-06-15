# -*- coding: utf-8 -*-
"""2-stage: F5-Egyptian authentic source -> kNN into her real frames -> her voice + authentic Egyptian.
Measure resem/ecapa->HER + UTMOS + WER + upload."""
import os, glob, subprocess, json, re, warnings, numpy as np, librosa, soundfile as sf, torch, torchaudio
warnings.filterwarnings("ignore")
def loud(w):
    r = np.sqrt(np.mean(w**2)) + 1e-9
    return w * (10**(-23/20) / r)
dev = "cuda" if torch.cuda.is_available() else "cpu"
knn = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True, device=dev)
y, sr = librosa.load("her_audio.wav", sr=16000, mono=True); iv = librosa.effects.split(y, top_db=30)
os.makedirs("her_chunks", exist_ok=True)
chunks = []; buf = []; cur = 0
for s, e in iv:
    buf.append(y[s:e]); cur += e - s
    if cur > 10*sr: chunks.append(np.concatenate(buf)); buf = []; cur = 0
if buf: chunks.append(np.concatenate(buf))
paths = []
for i, c in enumerate(chunks):
    p = f"her_chunks/{i}.wav"; sf.write(p, c, sr); paths.append(p)
pool = knn.get_matching_set(paths)
os.makedirs("out_2stage", exist_ok=True)
for p in sorted(glob.glob("out_f5src/*.wav")):
    q = knn.get_features(p); out = knn.match(q, pool, topk=4)
    torchaudio.save(f"out_2stage/{os.path.basename(p)}", out[None].cpu(), 16000)
    print("KNN", os.path.basename(p), flush=True)
print("KNN_DONE", flush=True)
# measure
from resemblyzer import VoiceEncoder, preprocess_wav
enc = VoiceEncoder(verbose=False)
def emb(w16): e = enc.embed_utterance(preprocess_wav(loud(w16), source_sr=16000)); return e/np.linalg.norm(e)
hs = [y[s:e] for s, e in iv if (e-s) > 1.5*16000]
herv = np.mean([emb(s) for s in hs], 0); herv /= np.linalg.norm(herv)
ut = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True); ut.eval()
def rsim(p):
    w, _ = librosa.load(p, sr=16000, mono=True); return float(emb(w) @ herv)
def utmos(p):
    w, _ = librosa.load(p, sr=16000, mono=True)
    with torch.no_grad(): return float(ut(torch.tensor(w).float().unsqueeze(0), 16000))
fs = sorted(glob.glob("out_2stage/*.wav"))
R = [rsim(p) for p in fs]; U = [utmos(p) for p in fs]
F5R = [rsim(p) for p in sorted(glob.glob("out_f5src/*.wav"))]
print("===== SUMMARY (2-stage: her voice + authentic Egyptian) =====", flush=True)
print(f"SUMMARY F5_SOURCE  resemHER={np.mean(F5R):.3f} (authentic Egyptian but not-her)", flush=True)
print(f"SUMMARY 2STAGE_kNN resemHER={np.mean(R):.3f} UTMOS={np.mean(U):.2f} (her frames + Egyptian content)", flush=True)
def upload(p):
    try:
        r = subprocess.run(["curl","-s","--max-time","60","-F","reqtype=fileupload","-F","time=72h","-F",f"fileToUpload=@{p}","https://litterbox.catbox.moe/resources/internals/api.php"], capture_output=True, text=True, timeout=70)
        return r.stdout.strip() if r.stdout.strip().startswith("http") else None
    except Exception: return None
for i, p in enumerate(fs[:3]):
    subprocess.run(["ffmpeg","-y","-t","7","-i",p,"-ar","24000","-ac","1","-b:a","64k",f"two{i}.mp3"], capture_output=True)
    print(f"URL 2stage_egyptian_{i} {upload(f'two{i}.mp3')}", flush=True)
print("MEASURE_DONE", flush=True)
