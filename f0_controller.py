# -*- coding: utf-8 -*-
"""STAGE-B delivery controller (the genuinely-ours, explicit, auditable piece):
per-phrase register-relative F0 RANGE/DYNAMISM injection via faithful PSOLA, calibrated to HER
measured delivery prior (f0range 16.1 st, f0dyn 47.3). The ONE proven-individual channel (AUC 0.85).
Fixes the earlier global-register bug: scale each phrase's F0 around ITS OWN median.
Control input = her measured scalars (NOT a black-box embedding) -> leakage-proof by construction."""
import warnings; warnings.filterwarnings("ignore")
import json, numpy as np, librosa, parselmouth
from parselmouth.praat import call
SR=24000
st=lambda hz:12*np.log2(np.maximum(hz,1e-6)/55.0)

def f0_of(w): return parselmouth.Sound(w.astype(np.float64),sampling_frequency=SR).to_pitch(0.01,100,500).selected_array['frequency']
def runs_of(w): return [(s,e) for s,e in librosa.effects.split(w.astype(np.float32),top_db=30) if (e-s)>0.10*SR]

def measure_range_dyn(w):
    f0=f0_of(w); v=f0[f0>0]
    if len(v)<10: return (np.nan,np.nan)
    return (float(np.percentile(st(v),95)-np.percentile(st(v),5)),
            float(np.mean(np.abs(np.diff(st(f0[f0>0]))))/0.01))

def inject_f0(w, target_range, max_scale=2.2):
    """Per-phrase: scale F0 excursions around each phrase's OWN median so the clip's overall F0-range
    moves toward target_range (her). Faithful PSOLA (r=0.997). Returns rendered audio."""
    cur_range,_=measure_range_dyn(w)
    if not np.isfinite(cur_range) or cur_range<1: return w
    scale=float(np.clip(target_range/cur_range, 1.0, max_scale))
    snd=parselmouth.Sound(w.astype(np.float64),sampling_frequency=SR)
    manip=call(snd,"To Manipulation",0.01,75,500); pt=call(manip,"Extract pitch tier")
    n=call(pt,"Get number of points")
    if n<3: return w
    pts=[(call(pt,"Get time from index",i),call(pt,"Get value at index",i)) for i in range(1,n+1)]
    iv=runs_of(w); f0=f0_of(w)
    regs={}
    for ri,(s,e) in enumerate(iv):
        a=int(s/(0.01*SR)); b=int(e/(0.01*SR)); seg=f0[a:b]; vv=seg>0
        regs[ri]=float(np.median(st(seg[vv]))) if vv.sum()>=2 else None
    def reg_of(t):
        for ri,(s,e) in enumerate(iv):
            if s/SR-0.03<=t<=e/SR+0.03: return regs.get(ri)
        return None
    call(pt,"Remove points between",0,1e9)
    for t,v in pts:
        r=reg_of(t)
        if r is None: call(pt,"Add point",t,float(v)); continue
        new_st=r+(st(v)-r)*scale                 # scale excursion around THIS phrase's register
        call(pt,"Add point",t,float(np.clip(55*2**(new_st/12),60,600)))
    call([manip,pt],"Replace pitch tier")
    return np.array(call(manip,"Get resynthesis (overlap-add)").values[0])

if __name__=="__main__":
    import glob, deliv_core as dc, op_core as op
    hy,_=librosa.load("her_audio.wav",sr=SR,mono=True); op.her_resem_centroid(hy); dc.her_space()
    prior=json.load(open("train/her_delivery_prior.json")); TR=prior["f0range"]
    print(f"HER target f0range={TR:.1f} st, f0dyn={prior['f0dyn']:.1f}",flush=True)
    print(f"{'clip':16s} {'range b->a':>12s} {'dyn b->a':>11s} {'dlv b->a':>13s} {'resem b->a':>13s} {'UTMOS b->a':>13s}",flush=True)
    clips=sorted(glob.glob('cb_ar/ex*.mp3'))[:2]+sorted(glob.glob('best_clips/*.wav'))
    for f in clips:
        w,_=librosa.load(f,sr=SR,mono=True)
        o=np.nan_to_num(inject_f0(w,TR))
        rb,db=measure_range_dyn(w); ra,da=measure_range_dyn(o)
        name=f.split('/')[-1][:15]
        print(f"{name:16s} {rb:5.1f}->{ra:5.1f}  {db:4.0f}->{da:4.0f}  {dc.delivery_sim(w):+.2f}->{dc.delivery_sim(o):+.2f}  {op.resem(w):.3f}->{op.resem(o):.3f}  {op.utmos(w):.2f}->{op.utmos(o):.2f}",flush=True)
    print("MEASURE_DONE",flush=True)
