# -*- coding: utf-8 -*-
"""rank-2 gate: does kNN-VC (her real frames) keep her identity cross-lingually,
beating synthesis on ECAPA and not leaking the donor? Compare: kNN output vs the generic-Arabic source."""
import glob, os, warnings, numpy as np, librosa
warnings.filterwarnings("ignore")
from resemblyzer import VoiceEncoder, preprocess_wav

def loud(w):
    r = np.sqrt(np.mean(w**2)) + 1e-9
    return w * (10**(-23/20) / r)

enc = VoiceEncoder(verbose=False)
def emb(w16):
    e = enc.embed_utterance(preprocess_wav(loud(w16), source_sr=16000)); return e / np.linalg.norm(e)

# her centroid + LOO
y16, _ = librosa.load("her_audio.wav", sr=16000, mono=True)
iv = librosa.effects.split(y16, top_db=30)
hs = [y16[s:e] for s, e in iv if (e - s) > 1.5*16000]
HE = np.array([emb(s) for s in hs]); herv = HE.mean(0); herv /= np.linalg.norm(herv)
rloo = np.array([float(HE[i] @ (lambda c: c/np.linalg.norm(c))(np.delete(HE, i, 0).mean(0))) for i in range(len(HE))])

# source (generic Arabic donor) centroid
src_files = sorted(glob.glob("out_ar_src/*.wav"))
SE = np.array([emb(librosa.load(p, sr=16000, mono=True)[0]) for p in src_files]); srcv = SE.mean(0); srcv /= np.linalg.norm(srcv)
print(f"[centroids] her-vs-source sim={float(herv@srcv):.3f}  her resem LOO mean={rloo.mean():.3f} std={rloo.std():.3f}", flush=True)

# ECAPA
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
    print(f"[ruler] her real ECAPA LOO mean={eloo.mean():.3f} std={eloo.std():.3f}", flush=True)
except Exception as e:
    print("ecapa unavail:", str(e)[:60], flush=True)

def scores(p):
    w24, _ = librosa.load(p, sr=24000, mono=True)
    e = emb(librosa.resample(w24, orig_sr=24000, target_sr=16000))
    rh, rs = float(e @ herv), float(e @ srcv)
    ee = (esim(p) if ecapa else float('nan'))
    return rh, rs, ee
def grp(d):
    fs = sorted(glob.glob(f"{d}/*.wav")); R = []; S = []; E = []
    for p in fs:
        rh, rs, ee = scores(p); R.append(rh); S.append(rs); E.append(ee)
    return float(np.mean(R)), float(np.mean(S)), float(np.nanmean(E))

print("===== SUMMARY (rank-2 kNN: her real frames -> Arabic; identity vs donor; beat synthesis ecapa 0.529?) =====", flush=True)
sr_h, sr_s, sr_e = grp("out_ar_src")     # the generic Arabic source (should be -> SOURCE, not HER)
kr_h, kr_s, kr_e = grp("out_knn_ar")     # kNN converted to her
print(f"SUMMARY SRC_generic   resemHER={sr_h:.3f} resemSRC={sr_s:.3f} ecapaHER={sr_e:.3f}", flush=True)
print(f"SUMMARY KNN_to_her     resemHER={kr_h:.3f} resemSRC={kr_s:.3f} ecapaHER={kr_e:.3f}", flush=True)
print(f"SUMMARY BASELINE_synth direct-XTTS-Arabic resemHER~0.863 ecapaHER~0.529 (the bar)", flush=True)
id_wins = kr_h > kr_s + 0.1
beats_synth = (not np.isnan(kr_e)) and kr_e > 0.529
print(f"SUMMARY VERDICT identity_wins_vs_donor={id_wins} (resemHER {kr_h:.3f} vs resemSRC {kr_s:.3f}) | "
      f"beats_synth_ecapa={beats_synth} ({kr_e:.3f} vs 0.529) | resemHER_toward_0.85={kr_h>=0.83}", flush=True)
print("MEASURE_DONE", flush=True)
