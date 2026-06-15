# -*- coding: utf-8 -*-
"""Reusable delivery-space scorer (timbre-free): build her centroid + score any clip's delivery-sim->her."""
import os, numpy as np, librosa, warnings
warnings.filterwarnings("ignore")
import parselmouth
from scipy.fftpack import dct
SR=24000
def f0_track(w):
    return parselmouth.Sound(w.astype(np.float64),sampling_frequency=SR).to_pitch(0.01,100,500).selected_array['frequency']
def _runs(f0v,minlen=20):
    r=[];c=[]
    for i,v in enumerate(f0v):
        if v>0:c.append(i)
        elif c:r.append(c);c=[]
    if c:r.append(c)
    return [x for x in r if len(x)>=minlen]
def delivery_vector(w):
    f0v=f0_track(w); voiced=f0v[f0v>0]
    if len(voiced)<30: return None
    st=lambda hz:12*np.log2(np.maximum(hz,1e-6)/55.0); sv=st(voiced)
    f0_range=np.percentile(sv,95)-np.percentile(sv,5); f0_iqr=np.percentile(sv,75)-np.percentile(sv,25)
    f0_dyn=np.mean(np.abs(np.diff(st(f0v[f0v>0]))))/0.01
    shp=[]
    for r in _runs(f0v):
        c=st(f0v[r]);c=c-np.mean(c);s=np.std(c);c=c/s if s>1e-6 else c
        c=np.interp(np.linspace(0,1,64),np.linspace(0,1,len(c)),c); shp.append(dct(c,norm='ortho')[1:6])
    shp=np.mean(shp,0) if shp else np.zeros(5)
    sl=[np.polyfit(np.arange(len(r))*0.01,st(f0v[r]),1)[0] for r in _runs(f0v) if len(r)>=15]
    declin=np.mean(sl) if sl else 0.0; declin_sd=np.std(sl) if len(sl)>1 else 0.0
    rms=librosa.feature.rms(y=w.astype(np.float32),frame_length=1024,hop_length=256)[0]
    rdb=20*np.log10(rms+1e-6); rdb=rdb[rdb>rdb.max()-40]; rms_std=np.std(rdb) if len(rdb)>2 else 0.0
    rt=librosa.feature.rms(y=w.astype(np.float32),frame_length=int(0.025*SR),hop_length=int(0.01*SR))[0]
    n=min(len(f0v),len(rt)); m=f0v[:n]>0
    fecorr=np.corrcoef(st(f0v[:n][m]),20*np.log10(rt[:n][m]+1e-6))[0,1] if m.sum()>10 else 0.0
    iv=librosa.effects.split(w.astype(np.float32),top_db=30); dur=len(w)/SR
    segd=np.array([(e-s)/SR for s,e in iv]) if len(iv) else np.array([0.0])
    rhythm=np.std(segd)/(np.mean(segd)+1e-9) if len(segd)>2 else 0.0; artic=len(iv)/dur if dur>0 else 0.0
    return np.nan_to_num(np.array([f0_range,f0_iqr,f0_dyn,declin,declin_sd,*shp,rms_std,fecorr,rhythm,artic,float(np.mean(f0v>0))]))
def segs_of(w,seglen=4.0,hop=4.0,maxn=40):
    out=[];L=int(seglen*SR);H=int(hop*SR);iv=librosa.effects.split(w,top_db=30)
    if len(iv)==0:return out
    i=iv[0][0]
    while i+L<=len(w) and len(out)<maxn: out.append(w[i:i+L]); i+=H
    return out
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))

_CACHE={}
def her_space(her_path="her_audio.wav", imp_dir="imp2"):
    if _CACHE: return _CACHE
    hy,_=librosa.load(her_path,sr=SR,mono=True)
    HV=[v for v in (delivery_vector(s) for s in segs_of(hy,4.0,4.0,40)) if v is not None]
    IV=[]
    for f in sorted(os.listdir(imp_dir)):
        w,_=librosa.load(f"{imp_dir}/{f}",sr=SR,mono=True); f0=f0_track(w); vd=f0[f0>0]
        if len(vd)==0 or np.median(vd)<165: continue
        for s in segs_of(w,4.0,3.0,6):
            v=delivery_vector(s)
            if v is not None: IV.append(v)
    allv=np.array(HV+IV); mu=allv.mean(0); sd=allv.std(0)+1e-9
    HVz=[(v-mu)/sd for v in HV]; her_cent=np.mean(HVz,0)
    her_self=np.mean([cos(v,her_cent) for v in HVz])
    imp_sim=np.mean([cos((v-mu)/sd,her_cent) for v in IV])
    _CACHE.update(dict(mu=mu,sd=sd,her_cent=her_cent,her_self=her_self,imp_sim=imp_sim))
    return _CACHE
def delivery_sim(w):
    S=her_space(); v=delivery_vector(w)
    if v is None: return float('nan')
    return cos((v-S['mu'])/S['sd'], S['her_cent'])
