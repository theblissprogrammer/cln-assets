# -*- coding: utf-8 -*-
"""STEP 0 measure: does her SOURCE drift cross-lingually, where ECAPA drops?
Glottal/VQ stats for her_EN_real vs her_EN_clone vs her_AR_clone vs generic_AR.
+ resem/ECAPA -> HER for EN vs AR clones. Prints DECISION-0."""
import glob, warnings, numpy as np, librosa
warnings.filterwarnings("ignore")
from glottal import glottal_feats
from resemblyzer import VoiceEncoder, preprocess_wav

def loud(w):
    r = np.sqrt(np.mean(w**2)) + 1e-9
    return w * (10**(-23/20) / r)

FEATS = ["h1h2", "hnr", "cpps", "tilt", "f0_mean", "f0_std"]
def group_feats(items):
    rows = []
    for it in items:
        try:
            rows.append(glottal_feats(it))
        except Exception as e:
            print("  feat err", str(e)[:50], flush=True)
    out = {}
    for k in FEATS:
        vals = [r[k] for r in rows if k in r and not (isinstance(r[k], float) and np.isnan(r[k]))]
        out[k] = round(float(np.mean(vals)), 2) if vals else float('nan')
    return out

# --- her real EN segments (sample) ---
y, sr = librosa.load("her_audio.wav", sr=24000, mono=True)
iv = librosa.effects.split(y, top_db=30)
hsegs = [y[s:e] for s, e in iv if (e - s) > 2.0 * 24000]
hsegs = hsegs[::max(1, len(hsegs)//15)][:15]
print(f"her real segments analyzed: {len(hsegs)}", flush=True)

G = {}
G["HER_EN_real"] = group_feats([loud(s) for s in hsegs])
G["HER_EN_clone"] = group_feats(sorted(glob.glob("out_en/*.wav")))
G["HER_AR_clone"] = group_feats(sorted(glob.glob("out_ar/*.wav")))
G["GENERIC_AR"]  = group_feats(sorted(glob.glob("out_ar_gen/*.wav")))

# --- resem + ECAPA -> HER (confirm the cross-lingual fine-texture gap) ---
enc = VoiceEncoder(verbose=False)
def emb(w16):
    e = enc.embed_utterance(preprocess_wav(loud(w16), source_sr=16000)); return e / np.linalg.norm(e)
y16, _ = librosa.load("her_audio.wav", sr=16000, mono=True)
iv16 = librosa.effects.split(y16, top_db=30)
herv = np.mean([emb(y16[s:e]) for s, e in iv16 if (e - s) > 1.5*16000], 0); herv /= np.linalg.norm(herv)
def rsim(p):
    w, _ = librosa.load(p, sr=24000, mono=True)
    return float(emb(librosa.resample(w, orig_sr=24000, target_sr=16000)) @ herv)
ecapa = None
try:
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="/tmp/ecapa", run_opts={"device": dev})
    def eemb(w16):
        import torch as T
        with T.no_grad():
            v = ec.encode_batch(T.tensor(w16).float().unsqueeze(0).to(dev)).squeeze().cpu().numpy()
        return v / np.linalg.norm(v)
    hereca = np.mean([eemb(loud(y16[s:e])) for s, e in iv16 if (e - s) > 1.5*16000], 0); hereca /= np.linalg.norm(hereca)
    def esim(p):
        w, _ = librosa.load(p, sr=16000, mono=True); return float(eemb(loud(w)) @ hereca)
    ecapa = True
except Exception as e:
    print("ecapa unavail:", str(e)[:60], flush=True)

def idscore(outdir):
    fs = sorted(glob.glob(f"{outdir}/*.wav"))
    r = float(np.mean([rsim(p) for p in fs]))
    e = float(np.mean([esim(p) for p in fs])) if ecapa else float('nan')
    return r, e
r_en, e_en = idscore("out_en")
r_ar, e_ar = idscore("out_ar")

print("===== SUMMARY (STEP 0: does her SOURCE drift cross-lingually where ECAPA drops?) =====", flush=True)
for g in ["HER_EN_real", "HER_EN_clone", "HER_AR_clone", "GENERIC_AR"]:
    print(f"SUMMARY GLOTTAL {g:13s} " + " ".join(f"{k}={G[g][k]}" for k in FEATS), flush=True)
print(f"SUMMARY IDENTITY EN_clone resem={r_en:.3f} ecapa={e_en:.3f} | AR_clone resem={r_ar:.3f} ecapa={e_ar:.3f}", flush=True)

# DECISION-0: does AR-clone source diverge from her EN-real MORE than the same-lang EN-clone does?
def dist(a, b, keys):
    return {k: (abs(a[k]-b[k]) if not (np.isnan(a[k]) or np.isnan(b[k])) else float('nan')) for k in keys}
SRC = ["h1h2", "hnr", "cpps", "tilt"]   # source/voice-quality (pitch-independent)
d_ar = dist(G["HER_AR_clone"], G["HER_EN_real"], SRC)
d_en = dist(G["HER_EN_clone"], G["HER_EN_real"], SRC)
d_gen = dist(G["HER_AR_clone"], G["GENERIC_AR"], SRC)
print("SUMMARY SRCDRIFT AR-vs-HERreal=" + str(d_ar), flush=True)
print("SUMMARY SRCDRIFT ENclone-vs-HERreal(noise floor)=" + str(d_en), flush=True)
print("SUMMARY SRCDRIFT AR-vs-GENERIC(toward generic? small=yes)=" + str(d_gen), flush=True)
drift = sum(1 for k in SRC if not np.isnan(d_ar[k]) and not np.isnan(d_en[k]) and d_ar[k] > d_en[k] + 0.5)
ecapa_drop = (not np.isnan(e_ar) and not np.isnan(e_en) and e_ar < e_en - 0.03)
print(f"SUMMARY DECISION0 src_features_drifting={drift}/4 ecapa_drops_xling={ecapa_drop} -> "
      f"{'PREMISE ALIVE (source drifts where ECAPA drops) -> STEP 1' if drift>=2 and ecapa_drop else 'WEAK (source preserved or ECAPA ok) -> reconsider/rank-2'}", flush=True)
print("MEASURE_DONE", flush=True)
