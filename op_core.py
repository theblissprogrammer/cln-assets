# -*- coding: utf-8 -*-
"""Reusable delivery-operator core (no top-level execution). PSOLA contour edit (faithful, r=0.997),
energy-envelope edit, contour bank, blended multi-channel transplant + metric loaders."""
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, librosa, torch
import parselmouth
from parselmouth.praat import call
from resemblyzer import VoiceEncoder, preprocess_wav
SR=24000
st=lambda hz:12*np.log2(np.maximum(hz,1e-6)/55.0)
hz=lambda s:55.0*2**(np.asarray(s)/12)

# ---- metrics ----
_enc=VoiceEncoder(verbose=False); _UT=[None]; _herv=[None]
def loud(w): r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
def remb(w):
    w16=librosa.resample(w.astype(np.float32),orig_sr=SR,target_sr=16000)
    e=_enc.embed_utterance(preprocess_wav(loud(w16),source_sr=16000)); return e/np.linalg.norm(e)
def her_resem_centroid(hy):
    iv=librosa.effects.split(hy,top_db=30)
    v=np.mean([remb(hy[s:e]) for s,e in iv if (e-s)>1.5*SR],0); _herv[0]=v/np.linalg.norm(v)
def resem(w): return float(remb(w)@_herv[0])
def utmos(w):
    if _UT[0] is None:
        _UT[0]=torch.hub.load("tarepan/SpeechMOS:v1.2.0","utmos22_strong",trust_repo=True); _UT[0].eval()
    w16=librosa.resample(w.astype(np.float32),orig_sr=SR,target_sr=16000)
    with torch.no_grad(): return round(float(_UT[0](torch.from_numpy(w16)[None],16000)),3)

# ---- audio helpers ----
def runs_of(w): return [(s,e) for s,e in librosa.effects.split(w.astype(np.float32),top_db=30)]
def f0_of(w): return parselmouth.Sound(w.astype(np.float64),sampling_frequency=SR).to_pitch(0.01,100,500).selected_array['frequency']

def run_shape(f0,s,e):
    a=int(s/(0.01*SR)); b=int(e/(0.01*SR)); seg=f0[a:b]; v=seg>0
    if v.sum()<12: return None
    c=st(seg[v]); rng=np.percentile(c,90)-np.percentile(c,10); reg=np.median(c)
    cc=c-np.mean(c); sd=np.std(cc); cc=cc/sd if sd>1e-6 else cc
    cc=np.interp(np.linspace(0,1,64),np.linspace(0,1,len(cc)),cc)
    return dict(shape=cc,nfr=int(v.sum()),rng=float(rng),reg=float(reg))

def contour_bank(w):
    f0=f0_of(w); iv=runs_of(w); bank=[]
    for j,(s,e) in enumerate(iv):
        r=run_shape(f0,s,e)
        if r is None: continue
        r['final']=((iv[j+1][0]-e)/SR if j+1<len(iv) else 9.9)>0.4
        bank.append(r)
    return bank

def _edit_pitch(w, newst_fn):
    """newst_fn(run_idx, pos, orig_st_at_point, run_reg, clip_shape_val)->new semitone. Faithful PSOLA."""
    snd=parselmouth.Sound(w.astype(np.float64),sampling_frequency=SR)
    manip=call(snd,"To Manipulation",0.01,75,500); pt=call(manip,"Extract pitch tier")
    n=call(pt,"Get number of points")
    if n<3: return w
    pts=[(call(pt,"Get time from index",i),call(pt,"Get value at index",i)) for i in range(1,n+1)]
    iv=runs_of(w); f0=f0_of(w)
    info={}
    for ri,(s,e) in enumerate(iv):
        r=run_shape(f0,s,e); info[ri]=(s/SR,e/SR,r)
    def find(t):
        for ri,(ts,te,r) in info.items():
            if ts-0.02<=t<=te+0.02: return ri,ts,te,r
        return None,None,None,None
    call(pt,"Remove points between",0,1e9)
    for t,v in pts:
        ri,ts,te,r=find(t)
        if ri is None or r is None: call(pt,"Add point",t,float(v)); continue
        pos=np.clip((t-ts)/(te-ts+1e-9),0,1)
        clip_sh=float(np.interp(pos,np.linspace(0,1,64),r['shape']))
        ns=newst_fn(ri,pos,st(v),r['reg'],r['rng'],clip_sh)
        call(pt,"Add point",t,float(np.clip(hz(ns),60,600)))
    call([manip,pt],"Replace pitch tier")
    return np.array(call(manip,"Get resynthesis (overlap-add)").values[0])

def flatten(w): return _edit_pitch(w, lambda ri,pos,os,reg,rng,csh: reg)

def blend_transplant(w, bank, target_range, alpha=0.6, seed=1):
    """new_shape = (1-a)*clip_shape + a*donor_shape ; range -> target_range ; keep run register."""
    rs=np.random.RandomState(seed); iv=runs_of(w); f0=f0_of(w); donors={}
    for ri,(s,e) in enumerate(iv):
        r=run_shape(f0,s,e)
        if r is None: continue
        gap=(iv[ri+1][0]-e)/SR if ri+1<len(iv) else 9.9
        cand=[d for d in bank if d['final']==(gap>0.4)] or bank
        cand=sorted(cand,key=lambda d:abs(d['nfr']-r['nfr']))[:max(8,len(cand)//4)]
        donors[ri]=cand[rs.randint(len(cand))]['shape']
    def fn(ri,pos,os,reg,rng,csh):
        dsh=float(np.interp(pos,np.linspace(0,1,64),donors[ri])) if ri in donors else csh
        sh=(1-alpha)*csh + alpha*dsh
        rng_use=(1-alpha)*rng + alpha*target_range
        return reg + rng_use*sh
    return _edit_pitch(w, fn)

def energy_reimpose(w, strength=0.6):
    """amplitude edit: sharpen emphasis (expand energy dynamics) gently -- cleaner than pitch surgery.
    raise loud frames, lower quiet frames around the local mean (her dynamics are wider)."""
    hop=int(0.005*SR); fl=int(0.025*SR)
    rms=librosa.feature.rms(y=w.astype(np.float32),frame_length=fl,hop_length=hop)[0]
    edb=20*np.log10(rms+1e-6); m=edb[edb>edb.max()-40].mean()
    gain_db=strength*(edb-m)                          # expand around mean
    gain=10**(gain_db/20)
    g=np.interp(np.arange(len(w)), np.arange(len(gain))*hop, gain)
    out=w*g
    return out/(np.max(np.abs(out))+1e-9)*0.95
