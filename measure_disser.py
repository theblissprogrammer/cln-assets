# -*- coding: utf-8 -*-
"""Does output arousal follow the GPT-latent source? gpt_fraction = (EgptCspk - CgptEspk)/(EE - CC).
~1 => emotion lives in gpt-latent (disentangled dub knob). ~0 => not separable that way."""
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
    fs = sorted(glob.glob(f"{outdir}/*.wav")); aa = []; rr = []
    for p in fs:
        w16, _ = librosa.load(p, sr=16000, mono=True)
        a, v = ser(loud(w16)); aa.append(a); rr.append(rsim(p))
    return float(np.mean(aa)), float(np.mean(rr))

src = json.load(open("srcd.json"))
print("===== SUMMARY (does arousal follow GPT-latent? identity stays her?) =====", flush=True)
print(f"SUMMARY SRC calm={src['calm_arousal']} expr={src['expr_arousal']}", flush=True)
for lang in ["en", "ar"]:
    A = {}; R = {}
    for tag in ["EE", "CC", "EgptCspk", "CgptEspk"]:
        A[tag], R[tag] = grp(f"out_{lang}_{tag}")
    anchor = A["EE"] - A["CC"]
    gpt_frac = (A["EgptCspk"] - A["CgptEspk"]) / anchor if abs(anchor) > 1e-6 else float("nan")
    print(f"SUMMARY {lang.upper()} arousal EE={A['EE']:.3f} CC={A['CC']:.3f} "
          f"EgptCspk={A['EgptCspk']:.3f} CgptEspk={A['CgptEspk']:.3f} | "
          f"resem EE={R['EE']:.3f} CC={R['CC']:.3f} EgptCspk={R['EgptCspk']:.3f} CgptEspk={R['CgptEspk']:.3f}", flush=True)
    print(f"SUMMARY {lang.upper()} GPT_FRACTION={gpt_frac:.2f}  (1=emotion in gpt-latent, 0=not)", flush=True)
print("MEASURE_DONE", flush=True)
