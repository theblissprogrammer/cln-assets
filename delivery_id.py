# -*- coding: utf-8 -*-
"""
FOUNDATION TEST: is DELIVERY (timbre-free "way of talking") speaker-discriminative?
If her held-out delivery vectors separate her from impostors (low EER) using NO timbre/spectral
info, then delivery is a genuine identity channel SEPARATE from the timbre channel that
resemblyzer/Chatterbox already capture -> Ahmed's two-layer thesis validated, rigorously.
Then: where does the Chatterbox-Arabic clone land in delivery space -- her region or generic?

Delivery vector (NO timbre): F0-CONTOUR SHAPE (DCT, register+range removed) + F0 dynamics +
energy dynamics + rhythm/timing. Standardized across all speakers' segments.
CONFOUND (flagged): impostors=LibriSpeech read audiobook, her=conversational YouTube -> part of
the separation is read-vs-conversational STYLE, not pure her-identity. We restrict impostors to
FEMALE (f0-matched) to remove the easy register cue and stress the delivery-shape signal.
"""
import os, numpy as np, librosa, warnings
warnings.filterwarnings("ignore")
import parselmouth
from scipy.fftpack import dct
from sklearn.metrics import roc_curve

SR=24000
def f0_track(w):
    snd=parselmouth.Sound(w.astype(np.float64), sampling_frequency=SR)
    p=snd.to_pitch(0.01,100,500); return p.selected_array['frequency']

def voiced_runs(f0v, minlen=20):
    runs=[]; cur=[]
    for i,v in enumerate(f0v):
        if v>0: cur.append(i)
        elif cur: runs.append(cur); cur=[]
    if cur: runs.append(cur)
    return [r for r in runs if len(r)>=minlen]

def delivery_vector(w):
    """timbre-FREE delivery descriptor."""
    f0v=f0_track(w)
    voiced=f0v[f0v>0]
    if len(voiced)<30: return None
    st=lambda hz:12*np.log2(np.maximum(hz,1e-6)/55.0)
    sv=st(voiced)
    # --- F0 dynamics (scalars) ---
    f0_range=np.percentile(sv,95)-np.percentile(sv,5)
    f0_iqr=np.percentile(sv,75)-np.percentile(sv,25)
    dd=np.abs(np.diff(st(f0v[f0v>0]))); f0_dyn=np.mean(dd)/0.01
    # --- F0 CONTOUR SHAPE (register+range removed -> pure shape) via mean DCT over runs ---
    runs=voiced_runs(f0v)
    shp=[]
    for r in runs:
        c=st(f0v[r]); c=c-np.mean(c)                     # remove register
        s=np.std(c); c=c/s if s>1e-6 else c              # remove range -> SHAPE only
        c=np.interp(np.linspace(0,1,64), np.linspace(0,1,len(c)), c)
        shp.append(dct(c,norm='ortho')[1:6])             # coeffs 1..5 (skip 0=mean)
    shp=np.mean(shp,0) if shp else np.zeros(5)
    # run-slope (declination tendency) distribution
    slopes=[np.polyfit(np.arange(len(r))*0.01, st(f0v[r]),1)[0] for r in runs if len(r)>=15]
    declin=np.mean(slopes) if slopes else 0.0
    declin_sd=np.std(slopes) if len(slopes)>1 else 0.0
    # --- energy dynamics + emphasis coupling ---
    rms=librosa.feature.rms(y=w.astype(np.float32),frame_length=1024,hop_length=256)[0]
    rdb=20*np.log10(rms+1e-6); rdb=rdb[rdb>rdb.max()-40]
    rms_std=np.std(rdb) if len(rdb)>2 else 0.0
    hop=0.01
    rt=librosa.feature.rms(y=w.astype(np.float32),frame_length=int(0.025*SR),hop_length=int(hop*SR))[0]
    n=min(len(f0v),len(rt)); m=f0v[:n]>0
    fecorr=np.corrcoef(st(f0v[:n][m]),20*np.log10(rt[:n][m]+1e-6))[0,1] if m.sum()>10 else 0.0
    # --- rhythm/timing ---
    iv=librosa.effects.split(w.astype(np.float32),top_db=30); dur=len(w)/SR
    segd=np.array([(e-s)/SR for s,e in iv]) if len(iv) else np.array([0.0])
    rhythm=np.std(segd)/(np.mean(segd)+1e-9) if len(segd)>2 else 0.0
    artic=len(iv)/dur if dur>0 else 0.0
    voiced_frac=float(np.mean(f0v>0))
    vec=np.array([f0_range,f0_iqr,f0_dyn,declin,declin_sd,*shp,rms_std,fecorr,rhythm,artic,voiced_frac],dtype=float)
    return np.nan_to_num(vec)

def segs_of(w, seglen=4.0, hop=3.0, maxn=30):
    out=[]; L=int(seglen*SR); H=int(hop*SR)
    iv=librosa.effects.split(w,top_db=30)
    if len(iv)==0: return out
    s0=iv[0][0]; i=s0
    while i+L<=len(w) and len(out)<maxn:
        out.append(w[i:i+L]); i+=H
    return out

print("== building speaker segment sets ==",flush=True)
hy,_=librosa.load("her_audio.wav",sr=SR,mono=True)
HER=segs_of(hy, seglen=4.0, hop=4.0, maxn=40)
imp={}
for f in sorted(os.listdir("imp2")):
    w,_=librosa.load(f"imp2/{f}",sr=SR,mono=True)
    f0=f0_track(w); vd=f0[f0>0]
    fmed=np.median(vd) if len(vd) else 0
    if fmed<165: continue                       # FEMALE only (f0-match her ~242) -> fair test
    s=segs_of(w,seglen=4.0,hop=3.0,maxn=6)
    if len(s)>=2: imp[f]=s
print(f"her segs {len(HER)} | female impostors {len(imp)} ({list(imp)[:6]}...)",flush=True)

# vectors
def vstack(segs):
    vs=[delivery_vector(s) for s in segs]; return [v for v in vs if v is not None]
HV=vstack(HER)
IV={k:vstack(v) for k,v in imp.items()}; IV={k:v for k,v in IV.items() if v}
allv=np.array(HV+[v for vs in IV.values() for v in vs])
mu=allv.mean(0); sd=allv.std(0)+1e-9
zn=lambda V:[(v-mu)/sd for v in V]
HVz=zn(HV); IVz={k:zn(v) for k,v in IV.items()}
her_cent=np.mean(HVz,0)
imp_cents={k:np.mean(v,0) for k,v in IVz.items()}

def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
# LOO positives: her seg vs her centroid (leave-one-out)
pos=[]
for i,v in enumerate(HVz):
    c=np.mean([HVz[j] for j in range(len(HVz)) if j!=i],0)
    pos.append(cos(v,c))
# negatives: each impostor seg vs HER centroid
neg=[cos(v,her_cent) for vs in IVz.values() for v in vs]
labels=[1]*len(pos)+[0]*len(neg); scores=pos+neg
fpr,tpr,thr=roc_curve(labels,scores); fnr=1-tpr
k=int(np.nanargmin(np.abs(fnr-fpr))); eer=(fpr[k]+fnr[k])/2

print("\n================ DELIVERY-ONLY SPEAKER DISCRIMINATION (timbre-free) ================",flush=True)
print(f"her LOO self-sim:   mean {np.mean(pos):.3f}  min {np.min(pos):.3f}  p10 {np.percentile(pos,10):.3f}",flush=True)
print(f"impostor->her sim:  mean {np.mean(neg):.3f}  max {np.max(neg):.3f}  p90 {np.percentile(neg,90):.3f}  (n={len(neg)})",flush=True)
print(f"SUMMARY DELIVERY_EER = {eer*100:.1f}%   threshold {thr[k]:.3f}",flush=True)
print(f"VERDICT: {'DELIVERY IS A REAL IDENTITY CHANNEL (timbre-free delivery separates her from female impostors)' if eer<0.25 else 'delivery weakly discriminative (EER high) -> shape signal noisy at this scale'}",flush=True)

# where does the Chatterbox-Arabic clone land in DELIVERY space?
print("\n================ does the Chatterbox-Arabic CLONE capture her DELIVERY? ================",flush=True)
for tag,files in [("CB-AR ex0.5",["ex5_0","ex5_1"]),("CB-AR ex1.2",["ex12_0","ex12_1"]),("CB-AR egy",["egy3_0","egy5_0"])]:
    sims=[]
    for fn in files:
        w,_=librosa.load(f"cb_ar/{fn}.mp3",sr=SR,mono=True)
        v=delivery_vector(w)
        if v is None: continue
        vz=(v-mu)/sd; sims.append(cos(vz,her_cent))
    impref=np.mean([cos(c,her_cent) for c in imp_cents.values()])
    print(f"{tag:14s} delivery-sim->her = {np.mean(sims):.3f}   (her-LOO {np.mean(pos):.3f} | avg impostor {impref:.3f})",flush=True)
print("MEASURE_DONE",flush=True)
