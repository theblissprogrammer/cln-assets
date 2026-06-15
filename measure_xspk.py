# -*- coding: utf-8 -*-
"""Cross-speaker dub verdict. DUB wants resemHER >> resemDONOR (her identity wins)
AND arousal pulled toward DONOR (donor emotion transferred)."""
import glob, warnings, numpy as np, librosa
warnings.filterwarnings("ignore")
from ser_lib import load_ser
from resemblyzer import VoiceEncoder, preprocess_wav

def loud(w):
    r = np.sqrt(np.mean(w**2)) + 1e-9
    return w * (10**(-23/20) / r)

ser = load_ser(); print("SER loaded", flush=True)
enc = VoiceEncoder(verbose=False)
def emb(w16):
    e = enc.embed_utterance(preprocess_wav(loud(w16), source_sr=16000)); return e / np.linalg.norm(e)

y, _ = librosa.load("her_audio.wav", sr=16000, mono=True)
iv = librosa.effects.split(y, top_db=30)
hsegs = [y[s:e] for s, e in iv if (e - s) > 1.5 * 16000]
herv = np.mean([emb(s) for s in hsegs], 0); herv /= np.linalg.norm(herv)
dfiles = sorted(glob.glob("donor_raw/*.wav"))
donorv = np.mean([emb(librosa.load(f, sr=16000, mono=True)[0]) for f in dfiles], 0)
donorv /= np.linalg.norm(donorv)
print(f"[centroids] her-vs-donor sim={float(herv@donorv):.3f} (low=clearly different speakers)", flush=True)

def scores(p):
    w24, _ = librosa.load(p, sr=24000, mono=True)
    e = emb(librosa.resample(w24, orig_sr=24000, target_sr=16000))
    w16, _ = librosa.load(p, sr=16000, mono=True); a, v = ser(loud(w16))
    return float(e @ herv), float(e @ donorv), a
def grp(outdir):
    fs = sorted(glob.glob(f"{outdir}/*.wav")); H = []; D = []; A = []
    for p in fs:
        h, d, a = scores(p); H.append(h); D.append(d); A.append(a)
    return float(np.mean(H)), float(np.mean(D)), float(np.mean(A))

print("===== SUMMARY (DUB wants resemHER>>resemDONOR AND arousal toward DONOR) =====", flush=True)
for lang in ["en", "ar"]:
    for tag in ["HER", "DONOR", "DUB"]:
        h, d, a = grp(f"out_{lang}_{tag}")
        print(f"SUMMARY {lang.upper()} {tag} resemHER={h:.3f} resemDONOR={d:.3f} arousal={a:.3f}", flush=True)
print("MEASURE_DONE", flush=True)
