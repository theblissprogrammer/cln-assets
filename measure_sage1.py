# -*- coding: utf-8 -*-
"""SAGE STEP 1 measure: per-arm resem/ECAPA -> HER + glottal feats, and the GATE:
does B (F0+frozen source) beat A (F0 only) on ECAPA->HER, per-utterance, beyond noise?
Also reports her real ECAPA leave-one-out std as the significance ruler."""
import glob, os, warnings, numpy as np, librosa
warnings.filterwarnings("ignore")
from glottal import glottal_feats
from resemblyzer import VoiceEncoder, preprocess_wav

def loud(w):
    r = np.sqrt(np.mean(w**2)) + 1e-9
    return w * (10**(-23/20) / r)

enc = VoiceEncoder(verbose=False)
def emb(w16):
    e = enc.embed_utterance(preprocess_wav(loud(w16), source_sr=16000)); return e / np.linalg.norm(e)
y16, _ = librosa.load("her_audio.wav", sr=16000, mono=True)
iv = librosa.effects.split(y16, top_db=30)
hs = [y16[s:e] for s, e in iv if (e - s) > 1.5*16000]
HE = np.array([emb(s) for s in hs]); herv = HE.mean(0); herv /= np.linalg.norm(herv)
# resem LOO
rloo = np.array([float(HE[i] @ (lambda c: c/np.linalg.norm(c))(np.delete(HE, i, 0).mean(0))) for i in range(len(HE))])
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
    EE = np.array([eemb(loud(s)) for s in hs]); hereca = EE.mean(0); hereca /= np.linalg.norm(hereca)
    eloo = np.array([float(EE[i] @ (lambda c: c/np.linalg.norm(c))(np.delete(EE, i, 0).mean(0))) for i in range(len(EE))])
    def esim(p):
        w, _ = librosa.load(p, sr=16000, mono=True); return float(eemb(loud(w)) @ hereca)
    ecapa = True
    print(f"[ruler] her real ECAPA LOO mean={eloo.mean():.3f} std={eloo.std():.3f} | resem LOO mean={rloo.mean():.3f} std={rloo.std():.3f}", flush=True)
except Exception as e:
    print("ecapa unavail:", str(e)[:60], flush=True)

ARMS = ["out_RAW", "out_A", "out_B", "out_C"]
names = sorted(os.path.basename(p) for p in glob.glob("out_A/*.wav"))
per = {a: {"r": [], "e": []} for a in ARMS}
for a in ARMS:
    for n in names:
        p = f"{a}/{n}"
        if not os.path.exists(p):
            continue
        per[a]["r"].append(rsim(p))
        per[a]["e"].append(esim(p) if ecapa else float('nan'))

print("===== SUMMARY (SAGE STEP 1: does frozen source [B] beat F0-only [A] on ECAPA->HER?) =====", flush=True)
FEATS = ["h1h2", "hnr", "tilt"]
for a in ARMS:
    r = np.array(per[a]["r"]); e = np.array(per[a]["e"])
    gf = [glottal_feats(p) for p in sorted(glob.glob(f"{a}/*.wav"))]
    gm = {k: round(float(np.nanmean([d[k] for d in gf])), 2) for k in FEATS}
    tag = a.replace("out_", "")
    print(f"SUMMARY ARM {tag:4s} resem={r.mean():.3f} ecapa={np.nanmean(e):.3f} | " +
          " ".join(f"{k}={gm[k]}" for k in FEATS), flush=True)

# GATE: paired B vs A on ECAPA
rA, eA = np.array(per["out_A"]["r"]), np.array(per["out_A"]["e"])
rB, eB = np.array(per["out_B"]["r"]), np.array(per["out_B"]["e"])
if ecapa:
    dE = eB - eA; dR = rB - rA
    print(f"SUMMARY PAIRED B-vs-A ecapa: per-utt={[round(x,3) for x in dE]}", flush=True)
    print(f"SUMMARY GATE mean_dEcapa={dE.mean():+.4f} std={dE.std():.4f} (n={len(dE)}) | "
          f"her_real_ecapa_LOO_std={eloo.std():.3f} | mean_dResem={dR.mean():+.4f} | resemB={rB.mean():.3f}", flush=True)
    win = dE.mean() > 0 and dE.mean() > dE.std()/np.sqrt(len(dE)) and rB.mean() >= 0.86
    strong = dE.mean() > eloo.std()
    print(f"SUMMARY VERDICT B>A_significant={win} B>A_beyond_realLOOstd={strong} -> "
          f"{'SOURCE CHANNEL ADDS IDENTITY (novelty alive)' if win else 'WASH -> source channel does not beat F0 transplant (cheap kill)'}", flush=True)
print("MEASURE_DONE", flush=True)
