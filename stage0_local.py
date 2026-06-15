# -*- coding: utf-8 -*-
"""Stage-0 double-dissociation probe (LOCAL, no GPU): edit ONLY her filter, keep her real residual.
A=recon floor | B=VTL/McAdams warp (predict identity COLLAPSES) | C=formant/constriction shift
(predict phoneme moves, identity SURVIVES). resem->HER per arm. Saves wavs for the ear."""
import os, warnings, numpy as np, librosa, soundfile as sf
warnings.filterwarnings("ignore")
from resemblyzer import VoiceEncoder, preprocess_wav
from source_lpc import analyze, synth, mcadams, shift_formant

SR = 16000
def loud(w):
    r = np.sqrt(np.mean(w**2)) + 1e-9
    return w * (10**(-23/20) / r)
enc = VoiceEncoder(verbose=False)
def emb(w):
    e = enc.embed_utterance(preprocess_wav(loud(w), source_sr=SR)); return e/np.linalg.norm(e)

y, _ = librosa.load("her_audio.wav", sr=SR, mono=True)
iv = librosa.effects.split(y, top_db=30)
segs = [y[s:e] for s, e in iv if (e-s) > 3*SR][:6]   # a few clean ~3s+ her segments
# her centroid from ALL her segments
allsegs = [y[s:e] for s, e in iv if (e-s) > 1.5*SR]
herv = np.mean([emb(s) for s in allsegs], 0); herv /= np.linalg.norm(herv)

ARMS = {
    "A_recon":      None,
    "B_vtl_0.80":   (lambda a: mcadams(a, 0.80)),
    "B_vtl_1.20":   (lambda a: mcadams(a, 1.20)),
    "C_F2_up1.20":  (lambda a: shift_formant(a, 2, 1.20)),
    "C_F2_dn0.83":  (lambda a: shift_formant(a, 2, 0.83)),
    "C_F1_up1.20":  (lambda a: shift_formant(a, 1, 1.20)),
}
os.makedirs("/tmp/ear_clips/stage0", exist_ok=True)
print("her real centroid built from", len(allsegs), "segments", flush=True)
results = {}
for name, mod in ARMS.items():
    sims = []
    for j, sg in enumerate(segs):
        P = analyze(sg, SR, order=18)
        ysyn = synth(P, mod)
        ysyn = np.nan_to_num(ysyn)
        sims.append(float(emb(ysyn) @ herv))
        if j == 0:
            sf.write(f"/tmp/ear_clips/stage0/{name}.wav", ysyn/(np.max(np.abs(ysyn))+1e-9)*0.95, SR)
    results[name] = float(np.mean(sims))
    print(f"ARM {name:14s} resem->HER = {np.mean(sims):.3f}  (per={[round(s,2) for s in sims]})", flush=True)

print("\n=== READ ===", flush=True)
A = results["A_recon"]; B = min(results["B_vtl_0.80"], results["B_vtl_1.20"]); C = max(results["C_F2_up1.20"], results["C_F2_dn0.83"], results["C_F1_up1.20"])
print(f"recon floor A={A:.3f} | VTL-warp(min) B={B:.3f} (predict << A) | constriction(best) C={C:.3f} (predict ~ A)", flush=True)
print(f"VTL drop A-B = {A-B:.3f} | constriction drop A-C = {A-C:.3f}", flush=True)
print(f"VERDICT: {'PREMISE SUPPORTED (constriction preserves identity, VTL collapses it)' if (A-C) < (A-B) - 0.03 else 'INCONCLUSIVE/CONTRA (constriction leaks identity too)'}", flush=True)
