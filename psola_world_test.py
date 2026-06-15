# -*- coding: utf-8 -*-
"""
Re-imposition PATH de-risk: which method re-imposes a pitch contour with the SMALLEST naturalness tax?
Project lesson: WORLD FULL re-synth craters UTMOS (job_prosody 2.07). But that was full re-synth.
PSOLA (Praat, pitch-synchronous OLA) edits F0/timing while preserving formants -> should keep quality.
Measure UTMOS of: original clean CB-Arabic clip | WORLD round-trip (no mod) | PSOLA round-trip (no mod)
| PSOLA with pitch-excursion x1.6 (a real delivery edit). Smallest tax = the re-imposition path.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, librosa, soundfile as sf, torch
import parselmouth
from parselmouth.praat import call
import source_world as sw

SR=24000
UT=torch.hub.load("tarepan/SpeechMOS:v1.2.0","utmos22_strong",trust_repo=True); UT.eval()
def utmos(w):
    w16=librosa.resample(w.astype(np.float32),orig_sr=SR,target_sr=16000)
    with torch.no_grad(): return round(float(UT(torch.from_numpy(w16)[None],16000)),3)

def world_roundtrip(w, dyn_scale=1.0):
    f0,sp,ap=sw.decompose(w,SR)
    if dyn_scale!=1.0:
        v=f0>0; lf=np.log(f0[v]); m=lf.mean(); f0[v]=np.exp(m+(lf-m)*dyn_scale)
    return sw.synth(f0,sp,ap,SR)

def psola_roundtrip(w, exc_scale=1.0):
    snd=parselmouth.Sound(w.astype(np.float64),sampling_frequency=SR)
    manip=call(snd,"To Manipulation",0.01,75,500)
    pt=call(manip,"Extract pitch tier")
    if exc_scale!=1.0:
        n=call(pt,"Get number of points")
        mean=call(pt,"Get mean (curve)",0,0)
        # rebuild excursion-scaled tier
        times=[call(pt,"Get time from index",i) for i in range(1,n+1)]
        vals =[call(pt,"Get value at index",i) for i in range(1,n+1)]
        call(pt,"Remove points between",0,1e9)
        for t,vv in zip(times,vals):
            call(pt,"Add point",t, mean*(vv/mean)**exc_scale)
    call([manip,pt],"Replace pitch tier")
    out=call(manip,"Get resynthesis (overlap-add)")
    return np.array(out.values[0])

clip="cb_ar/ex5_0.mp3"
w,_=librosa.load(clip,sr=SR,mono=True)
rows=[("original_CB", w)]
rows.append(("WORLD_roundtrip_nomod", world_roundtrip(w,1.0)))
rows.append(("WORLD_dyn_x1.6",        world_roundtrip(w,1.6)))
rows.append(("PSOLA_roundtrip_nomod", psola_roundtrip(w,1.0)))
rows.append(("PSOLA_exc_x1.6",        psola_roundtrip(w,1.6)))

print(f"{'method':24s} {'UTMOS':>7s}  {'tax_vs_orig':>11s}",flush=True)
base=None
for name,sig in rows:
    sig=np.nan_to_num(sig); u=utmos(sig)
    if base is None: base=u
    print(f"{name:24s} {u:7.3f}  {u-base:+11.3f}",flush=True)
    sf.write(f"/tmp/psola_{name}.wav", sig/(np.max(np.abs(sig))+1e-9)*0.95, SR)
print("MEASURE_DONE",flush=True)
