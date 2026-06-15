# -*- coding: utf-8 -*-
"""Sweep blend-alpha (F0 contour) x energy-reimpose to find the delivery-gain vs naturalness/identity
sweet spot on CB-Arabic, with impostor specificity at the best setting."""
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, librosa, soundfile as sf
import op_core as op, deliv_core as dc
SR=24000
hy,_=librosa.load("her_audio.wav",sr=SR,mono=True)
op.her_resem_centroid(hy)
HER_BANK=op.contour_bank(hy); HER_RANGE=float(np.median([d['rng'] for d in HER_BANK]))
IMP_BANK=[]
for f in sorted(os.listdir("imp2"))[:10]:
    w,_=librosa.load(f"imp2/{f}",sr=SR,mono=True); vd=op.f0_of(w); vd=vd[vd>0]
    if len(vd)==0 or np.median(vd)<165: continue
    IMP_BANK+=op.contour_bank(w)
S=dc.her_space()
clips=["ex5_0","ex5_1","ex12_0","ex12_1","egy3_0","egy5_0"]
W={c:librosa.load(f"cb_ar/{c}.mp3",sr=SR,mono=True)[0] for c in clips}
print(f"her_self {S['her_self']:.3f} generic {S['imp_sim']:.3f} (thr~0.22) | her_range {HER_RANGE:.1f}st",flush=True)

def measure(fn,tag,save=False):
    ds=[];um=[];rs=[]
    for c in clips:
        out=W[c] if fn is None else np.nan_to_num(fn(W[c]))
        ds.append(dc.delivery_sim(out)); um.append(op.utmos(out)); rs.append(op.resem(out))
        if save and c=="ex12_0": sf.write(f"/tmp/blend_{tag}.wav",out/(np.max(np.abs(out))+1e-9)*0.95,SR)
    print(f"  {tag:22s} dlv {np.nanmean(ds):.3f}  UTMOS {np.mean(um):.3f}  resem {np.mean(rs):.3f}",flush=True)
    return np.nanmean(ds),np.mean(um),np.mean(rs)

print("\n=== sweep ===",flush=True)
measure(None,"orig")
for a in [0.4,0.7,1.0]:
    measure(lambda x,a=a: op.blend_transplant(x,HER_BANK,HER_RANGE,alpha=a),f"F0blend_a{a}")
for sgn in [0.5,0.9]:
    measure(lambda x,s=sgn: op.energy_reimpose(x,strength=s),f"energy_s{sgn}")
# combo: gentle F0 blend + energy
measure(lambda x: op.energy_reimpose(op.blend_transplant(x,HER_BANK,HER_RANGE,alpha=0.5),0.7),"combo_a0.5_e0.7",save=True)
print("\n=== specificity (best combo, her bank vs impostor bank) ===",flush=True)
measure(lambda x: op.energy_reimpose(op.blend_transplant(x,HER_BANK,HER_RANGE,alpha=0.5),0.7),"HER_combo")
measure(lambda x: op.energy_reimpose(op.blend_transplant(x,IMP_BANK,HER_RANGE,alpha=0.5),0.7),"IMPOSTOR_combo")
print("MEASURE_DONE",flush=True)
