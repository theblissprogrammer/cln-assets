# -*- coding: utf-8 -*-
"""SAGE STEP 1 arms: build RAW / A(F0) / B(F0+source) / C(source-only) for each Arabic base clone."""
import os, glob, warnings, numpy as np, librosa, soundfile as sf
warnings.filterwarnings("ignore")
from glottal import glottal_feats
from source_world import decompose, synth, transplant_f0, tilt_correct, ap_match

SR = 24000
def loud(w):
    r = np.sqrt(np.mean(w**2)) + 1e-9
    return w * (10**(-23/20) / r)

# --- her source targets from her real EN ---
y, _ = librosa.load("her_audio.wav", sr=SR, mono=True)
iv = librosa.effects.split(y, top_db=30)
hsegs = [y[s:e] for s, e in iv if (e - s) > 2.0 * SR]
hsegs = hsegs[::max(1, len(hsegs)//15)][:15]
import pyworld as pw
lf0 = []
for s in hsegs:
    f0, t = pw.harvest(np.ascontiguousarray(s.astype(np.float64)), SR)
    f0 = pw.stonemask(np.ascontiguousarray(s.astype(np.float64)), f0, t, SR)
    lf0.append(np.log(f0[f0 > 0]))
lf0 = np.concatenate(lf0)
her_logmean, her_logstd = float(lf0.mean()), float(lf0.std())
hf = [glottal_feats(loud(s)) for s in hsegs]
her_tilt = float(np.nanmean([d["tilt"] for d in hf]))
her_hnr = float(np.nanmean([d["hnr"] for d in hf]))
print(f"HER TARGETS logf0_mean={her_logmean:.3f} logf0_std={her_logstd:.3f} tilt={her_tilt:.2f} hnr={her_hnr:.2f}", flush=True)

for d in ["out_RAW", "out_A", "out_B", "out_C"]:
    os.makedirs(d, exist_ok=True)

for p in sorted(glob.glob("out_ar_base/*.wav")):
    name = os.path.basename(p)
    x, _ = librosa.load(p, sr=SR, mono=True)
    f0, sp, ap = decompose(x, SR)
    uf = glottal_feats(loud(x))
    d_tilt = her_tilt - uf["tilt"]   # apply to make tilt match her
    d_hnr = her_hnr - uf["hnr"]
    f0_her = transplant_f0(f0, her_logmean, her_logstd)
    sp_src = tilt_correct(sp, SR, d_tilt)
    ap_src = ap_match(ap, d_hnr)
    sf.write(f"out_RAW/{name}", synth(f0,     sp,     ap,     SR), SR)       # round-trip floor
    sf.write(f"out_A/{name}",   synth(f0_her, sp,     ap,     SR), SR)       # F0 transplant only (bar)
    sf.write(f"out_B/{name}",   synth(f0_her, sp_src, ap_src, SR), SR)       # F0 + frozen source
    sf.write(f"out_C/{name}",   synth(f0,     sp_src, ap_src, SR), SR)       # source only, native F0
    print(f"ARMS {name} dtilt={d_tilt:.1f} dhnr={d_hnr:.1f}", flush=True)
print("ARMS_DONE", flush=True)
