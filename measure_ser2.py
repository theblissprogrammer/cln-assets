# -*- coding: utf-8 -*-
"""Max-contrast measure: SER arousal EXPR-output vs CALM-output per language + resem identity."""
import glob, json, warnings, numpy as np, librosa
warnings.filterwarnings("ignore")
from ser_lib import load_ser
from resemblyzer import VoiceEncoder, preprocess_wav

def loud(w):
    r = np.sqrt(np.mean(w**2)) + 1e-9
    return w * (10**(-23/20) / r)

ser = load_ser(); print("SER loaded", flush=True)

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

def grp(outdir):
    fs = sorted(glob.glob(f"{outdir}/*.wav")); aa = []; vv = []; rr = []
    for p in fs:
        w16, _ = librosa.load(p, sr=16000, mono=True)
        a, v = ser(loud(w16)); aa.append(a); vv.append(v); rr.append(rsim(p))
    return float(np.mean(aa)), float(np.mean(vv)), float(np.mean(rr))

src = json.load(open("src2.json"))
print("===== SUMMARY (max-contrast: does SER arousal move EXPR>CALM? identity held?) =====", flush=True)
print(f"SUMMARY SRC calm_arousal={src['calm_arousal']} expr_arousal={src['expr_arousal']} "
      f"delta={round(src['expr_arousal']-src['calm_arousal'],3)}", flush=True)
for lang in ["en", "ar"]:
    ea, ev, er = grp(f"out_{lang}_expr"); ca, cv, cr = grp(f"out_{lang}_calm")
    print(f"SUMMARY {lang.upper()} EXPR arousal={ea:.3f} resem={er:.3f} | "
          f"CALM arousal={ca:.3f} resem={cr:.3f} | AROUSAL_DELTA={ea-ca:+.3f}", flush=True)
print("MEASURE_DONE", flush=True)
