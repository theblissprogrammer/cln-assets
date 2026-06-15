# -*- coding: utf-8 -*-
"""
DELIVERY-FINGERPRINT DIAGNOSTIC (Stage-0 of the delivery program).
Q: (a) is her delivery fingerprint a STABLE within-her signature?
   (b) does Chatterbox-Arabic land INSIDE or OUTSIDE her natural delivery range
       (= which delivery components are LOST cross-lingually = what an operator must transfer)?
Honest confound flags: G1 voice-quality is CHANNEL-sensitive (her=noisy YouTube vs CB=clean studio);
G3 timing is LANGUAGE-bound + needs long audio. The CHANNEL/LEVEL-ROBUST, language-invariant core = F0-DYNAMICS (G2) + emphasis-coupling.
"""
import os, numpy as np, librosa, warnings, json
warnings.filterwarnings("ignore")
from delivery_fingerprint import fingerprint, ALLFEATS

SR=24000
G1=["hnr","cpps","h1h2","tilt","jitter","shimmer"]            # channel-sensitive
G2=["f0_med","f0_range_st","f0_iqr_st","f0_dyn_st_s","declination_st_s","voiced_frac"]  # CHANNEL-ROBUST core
G3=["artic_rate","pause_rate","pause_mean","pause_frac","rhythm_irreg"]  # language-bound + needs long audio
G4=["rms_range_db","rms_std_db","emphasis_peak","f0_energy_corr"]        # energy: level-norm helps; f0_energy_corr robust
ROBUST=["f0_range_st","f0_iqr_st","f0_dyn_st_s","declination_st_s","f0_energy_corr","emphasis_peak"]

def her_segments(path, seglen=5.0, n=40):
    y,_=librosa.load(path,sr=SR,mono=True)
    iv=librosa.effects.split(y,top_db=30)
    # stitch consecutive voiced regions into ~seglen windows that preserve her real pausing
    segs=[]; cur_s=iv[0][0];
    i=0
    while i<len(iv) and len(segs)<n:
        s=iv[i][0]
        e=s+int(seglen*SR)
        chunk=y[s:e]
        if len(chunk)>=int(seglen*SR*0.8):
            segs.append(chunk)
        # advance past this window
        while i<len(iv) and iv[i][0]<e: i+=1
    return segs

print("extracting her within-distribution...", flush=True)
hs=her_segments("her_audio.wav")
her=[fingerprint(s) for s in hs]
print(f"her segments: {len(her)}", flush=True)
mu={k:np.nanmean([h[k] for h in her]) for k in ALLFEATS}
sd={k:np.nanstd ([h[k] for h in her]) for k in ALLFEATS}
cv={k:abs(sd[k]/mu[k]) if abs(mu[k])>1e-6 else np.nan for k in ALLFEATS}

print("extracting Chatterbox-Arabic clips...", flush=True)
cb_files=sorted([f for f in os.listdir("cb_ar") if f.endswith(".mp3")])
cb={f:fingerprint(os.path.join("cb_ar",f)) for f in cb_files}
cbmu={k:np.nanmean([cb[f][k] for f in cb_files]) for k in ALLFEATS}

def z(k, val):
    return (val-mu[k])/sd[k] if sd[k]>1e-9 else float('nan')

print("\n================ WITHIN-HER STABILITY (is the fingerprint a real signature?) ================", flush=True)
print(f"{'FEAT':18s} {'her_mean':>9s} {'her_std':>8s} {'her_CV':>7s}  group", flush=True)
for grp,name in [(G1,'G1-voiceQ(channel!)'),(G2,'G2-F0dyn(ROBUST)'),(G3,'G3-timing(lang/long)'),(G4,'G4-energy')]:
    for k in grp:
        print(f"{k:18s} {mu[k]:9.3f} {sd[k]:8.3f} {cv[k]*100 if np.isfinite(cv[k]) else float('nan'):6.0f}%  {name}", flush=True)

print("\n================ THE GAP: Chatterbox-Arabic vs her natural delivery range ================", flush=True)
print("(z = SDs outside her range. |z|<1 = within her; |z|>2 = LOST/outside her signature)", flush=True)
print(f"{'FEAT':18s} {'her_mean':>9s} {'cb_AR_mean':>10s} {'z':>6s}   verdict", flush=True)
for grp,name in [(G2,'G2-F0dyn(ROBUST)'),(G4,'G4-energy'),(G1,'G1-voiceQ(channel-confounded)'),(G3,'G3-timing(short-clip-degenerate)')]:
    print(f"  --- {name} ---", flush=True)
    for k in grp:
        zz=z(k,cbmu[k])
        v = "within-her" if abs(zz)<1 else ("drift" if abs(zz)<2 else "OUTSIDE/LOST")
        print(f"{k:18s} {mu[k]:9.3f} {cbmu[k]:10.3f} {zz:6.2f}   {v}", flush=True)

# headline: mean |z| over the channel/level-ROBUST language-invariant core
robz=[abs(z(k,cbmu[k])) for k in ROBUST if np.isfinite(z(k,cbmu[k]))]
allz=[abs(z(k,cbmu[k])) for k in (G2+G4) if np.isfinite(z(k,cbmu[k]))]
print("\n================ SUMMARY ================", flush=True)
print(f"SUMMARY robust_core_mean_|z| = {np.mean(robz):.2f}  (features: {ROBUST})", flush=True)
print(f"SUMMARY G2+G4_mean_|z|       = {np.mean(allz):.2f}", flush=True)
nlost=sum(1 for k in ROBUST if np.isfinite(z(k,cbmu[k])) and abs(z(k,cbmu[k]))>=2)
ndrift=sum(1 for k in ROBUST if np.isfinite(z(k,cbmu[k])) and 1<=abs(z(k,cbmu[k]))<2)
print(f"SUMMARY robust_core: {nlost}/{len(ROBUST)} OUTSIDE(|z|>=2), {ndrift}/{len(ROBUST)} drift(1<=|z|<2)", flush=True)
print(f"VERDICT: {'GAP EXISTS -> delivery program is ALIVE (Chatterbox does NOT carry her delivery fingerprint cross-lingually)' if np.mean(robz)>=1.0 else 'NO clear gap on robust core -> Chatterbox already carries delivery (program may be moot; need finer test)'}", flush=True)
json.dump({"mu":mu,"sd":sd,"cbmu":cbmu}, open("/tmp/delivery_diag.json","w"))
print("MEASURE_DONE", flush=True)
