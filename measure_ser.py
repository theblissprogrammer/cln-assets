# -*- coding: utf-8 -*-
"""Does output emotion track source emotion? Pearson corr(source arousal, output arousal)
per language, plus resemblyzer identity. EN = same-lang SER control; AR = cross-lingual dub test."""
import glob, os, json, warnings, numpy as np, librosa
warnings.filterwarnings("ignore")
from ser_lib import load_ser
from resemblyzer import VoiceEncoder, preprocess_wav

def loud(w):
    r = np.sqrt(np.mean(w**2)) + 1e-9
    return w * (10**(-23/20) / r)

def pearson(a, b):
    a = np.array(a, float); b = np.array(b, float)
    if a.std() < 1e-9 or b.std() < 1e-9: return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

ser = load_ser(); print("SER loaded", flush=True)

# --- her resemblyzer centroid ---
y, sr = librosa.load("her_audio.wav", sr=16000, mono=True)
iv = librosa.effects.split(y, top_db=30)
hsegs = [y[s:e] for s, e in iv if (e - s) > 1.5 * 16000]
enc = VoiceEncoder(verbose=False)
E = np.array([(lambda e: e / np.linalg.norm(e))(
    enc.embed_utterance(preprocess_wav(loud(s), source_sr=16000))) for s in hsegs])
herv = np.mean(E, 0); herv /= np.linalg.norm(herv)
def rsim(p):
    w, _ = librosa.load(p, sr=24000, mono=True)
    e = enc.embed_utterance(preprocess_wav(loud(w), source_sr=24000)); e /= np.linalg.norm(e)
    return float(e @ herv)

src = json.load(open("source_ser.json")); N = len(src)
src_ar = [d["arousal"] for d in src]; src_va = [d["valence"] for d in src]

print("===== SUMMARY (want output arousal to TRACK source arousal; resem identity held) =====", flush=True)
for lang, outdir in [("EN", "out_en"), ("AR", "out_ar")]:
    o_ar = []; o_va = []; o_re = []
    for k in range(N):
        fs = sorted(glob.glob(f"{outdir}/r{k}_*.wav"))
        aa = []; vv = []; rr = []
        for p in fs:
            w16, _ = librosa.load(p, sr=16000, mono=True)
            a, v = ser(loud(w16)); aa.append(a); vv.append(v); rr.append(rsim(p))
        o_ar.append(float(np.mean(aa))); o_va.append(float(np.mean(vv))); o_re.append(float(np.mean(rr)))
    print(f"SUMMARY {lang}_OUT arousal={[round(x,3) for x in o_ar]} resem={[round(x,3) for x in o_re]}", flush=True)
    print(f"SUMMARY {lang}_CORR src->out arousal_r={pearson(src_ar,o_ar):.3f} "
          f"valence_r={pearson(src_va,o_va):.3f} resem_mean={np.mean(o_re):.3f}", flush=True)
print(f"SUMMARY SRC arousal={src_ar} valence={src_va}", flush=True)
print("MEASURE_DONE", flush=True)
