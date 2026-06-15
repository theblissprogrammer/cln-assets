# -*- coding: utf-8 -*-
"""Measure each (topk,sigma) render: resem/ecapa->HER + UTMOS(static killed?) + WER. Export best+synth to litterbox."""
import glob, os, json, re, subprocess, warnings, numpy as np, librosa, torch
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
from speechbrain.inference.speaker import EncoderClassifier
dev = "cuda" if torch.cuda.is_available() else "cpu"
ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="/tmp/ecapa", run_opts={"device": dev})
def eemb(w16):
    with torch.no_grad():
        v = ec.encode_batch(torch.tensor(w16).float().unsqueeze(0).to(dev)).squeeze().cpu().numpy()
    return v/np.linalg.norm(v)
hereca = np.mean([eemb(loud(s)) for s in hs], 0); hereca /= np.linalg.norm(hereca)
ut = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True); ut.eval()
def utmos(p):
    w, _ = librosa.load(p, sr=16000, mono=True)
    with torch.no_grad(): return float(ut(torch.tensor(w).float().unsqueeze(0), 16000))
def rsim(p):
    w, _ = librosa.load(p, sr=24000, mono=True); return float(emb(librosa.resample(w, orig_sr=24000, target_sr=16000)) @ herv)
def esim(p):
    w, _ = librosa.load(p, sr=16000, mono=True); return float(eemb(loud(w)) @ hereca)
def grp(d):
    fs = sorted(glob.glob(f"{d}/*.wav"))
    return (float(np.mean([rsim(p) for p in fs])), float(np.mean([esim(p) for p in fs])), float(np.mean([utmos(p) for p in fs])))
print("===== SUMMARY (Vocos render debug: kill static[UTMOS up] while holding identity?) =====", flush=True)
print("SUMMARY BASELINES: SYNTH UTMOS~2.92 resem0.86 ecapa0.52 | 16kHz-kNN UTMOS~2.50 resem0.85 ecapa0.55 | prev-Vocos UTMOS? resem0.85 ecapa0.57", flush=True)
results = {}
for d in sorted(glob.glob("out_v_*")):
    r, e, u = grp(d); results[d] = (r, e, u)
    print(f"SUMMARY {d:16s} resemHER={r:.3f} ecapaHER={e:.3f} UTMOS={u:.2f}", flush=True)
# pick best: highest UTMOS among resem>=0.82
ok = {d: v for d, v in results.items() if v[0] >= 0.82}
best = max(ok or results, key=lambda d: results[d][2])
print(f"SUMMARY BEST={best} {results[best]}", flush=True)
def upload(p):
    try:
        r = subprocess.run(["curl","-s","--max-time","60","-F","reqtype=fileupload","-F","time=72h","-F",f"fileToUpload=@{p}","https://litterbox.catbox.moe/resources/internals/api.php"], capture_output=True, text=True, timeout=70)
        return r.stdout.strip() if r.stdout.strip().startswith("http") else None
    except Exception: return None
src = sorted(glob.glob(f"{best}/*.wav"))[0]
subprocess.run(["ffmpeg","-y","-t","6","-i",src,"-ar","24000","-ac","1","-b:a","64k","best.mp3"], capture_output=True)
print(f"URL vocos_debug_best {upload('best.mp3')}", flush=True)
print("MEASURE_DONE", flush=True)
