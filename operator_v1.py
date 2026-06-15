# -*- coding: utf-8 -*-
"""
DELIVERY OPERATOR v1 (the decisive test): re-impose her MEASURED delivery channels onto a clean
Chatterbox-Arabic clip via PSOLA, language-agnostically (energy peaks = emphasis anchors, no Arabic NLP).
Targets the measured gaps: pitch RANGE/dynamism (clone compressed), pitch-ENERGY COUPLING
(f0_energy_corr her +0.15 vs clone -0.03, the z=-2.03 most-lost channel), phrase-final FALL.
Q: does it move delivery-sim->her (0.17 -> toward 0.42) while UTMOS stays clean + resemblyzer held?
"""
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, librosa, soundfile as sf, torch
import parselmouth
from parselmouth.praat import call
from resemblyzer import VoiceEncoder, preprocess_wav
import deliv_core as dc
SR=24000

# --- her resemblyzer centroid + UTMOS ---
enc=VoiceEncoder(verbose=False)
def loud(w): r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
def remb(w):
    w16=librosa.resample(w.astype(np.float32),orig_sr=SR,target_sr=16000)
    e=enc.embed_utterance(preprocess_wav(loud(w16),source_sr=16000)); return e/np.linalg.norm(e)
hy,_=librosa.load("her_audio.wav",sr=SR,mono=True)
iv=librosa.effects.split(hy,top_db=30); herv=np.mean([remb(hy[s:e]) for s,e in iv if (e-s)>1.5*SR],0); herv/=np.linalg.norm(herv)
def resem(w): return float(remb(w)@herv)
UT=torch.hub.load("tarepan/SpeechMOS:v1.2.0","utmos22_strong",trust_repo=True); UT.eval()
def utmos(w):
    w16=librosa.resample(w.astype(np.float32),orig_sr=SR,target_sr=16000)
    with torch.no_grad(): return round(float(UT(torch.from_numpy(w16)[None],16000)),3)

def energy_env(w):
    rt=librosa.feature.rms(y=w.astype(np.float32),frame_length=int(0.025*SR),hop_length=int(0.005*SR))[0]
    t=np.arange(len(rt))*0.005
    edb=20*np.log10(rt+1e-6)
    z=(edb-edb.mean())/(edb.std()+1e-9)
    return t,z

def phrase_final_ramp(w, times):
    """ramp 0->1 over the last 40% of each voiced phrase (for boundary fall)."""
    iv=librosa.effects.split(w.astype(np.float32),top_db=30)
    ramp=np.zeros(len(times))
    for s,e in iv:
        ts,te=s/SR,e/SR; L=te-ts
        for i,t in enumerate(times):
            if ts<=t<=te:
                frac=(t-ts)/(L+1e-9)
                ramp[i]=max(ramp[i], max(0.0,(frac-0.6)/0.4))
    return ramp

def operator(w, exc=1.5, couple_st=2.0, finalfall_st=2.5):
    snd=parselmouth.Sound(w.astype(np.float64),sampling_frequency=SR)
    manip=call(snd,"To Manipulation",0.01,75,500)
    pt=call(manip,"Extract pitch tier")
    n=call(pt,"Get number of points")
    if n<3: return w
    reg=call(pt,"Get mean (curve)",0,0)
    times=[call(pt,"Get time from index",i) for i in range(1,n+1)]
    vals =[call(pt,"Get value at index",i) for i in range(1,n+1)]
    te,ez=energy_env(w);
    ramp=phrase_final_ramp(w, times)
    ez_at=np.interp(times, te, ez)
    newvals=[]
    for v,ezi,ri in zip(vals,ez_at,ramp):
        st_off = 12*np.log2(max(v,1e-6)/reg)*exc        # expand excursion (range/dynamism)
        st_off += couple_st*ezi                          # couple pitch to energy (emphasis)
        st_off -= finalfall_st*ri                        # phrase-final fall
        newvals.append(reg*2**(st_off/12))
    call(pt,"Remove points between",0,1e9)
    for t,nv in zip(times,newvals): call(pt,"Add point",t,float(np.clip(nv,60,600)))
    call([manip,pt],"Replace pitch tier")
    out=call(manip,"Get resynthesis (overlap-add)")
    return np.array(out.values[0])

S=dc.her_space()
print(f"her delivery self-sim {S['her_self']:.3f} | generic impostor {S['imp_sim']:.3f}  (her/not-her ~0.22)",flush=True)
clips=["ex5_0","ex5_1","ex12_0","ex12_1","egy3_0","egy5_0"]
ARMS=[("orig",None),
      ("op_mild",dict(exc=1.3,couple_st=1.5,finalfall_st=1.5)),
      ("op_med", dict(exc=1.6,couple_st=2.5,finalfall_st=2.5)),
      ("op_strong",dict(exc=2.0,couple_st=3.5,finalfall_st=3.0))]
print(f"\n{'arm':10s} {'dlv_sim->her':>12s} {'UTMOS':>7s} {'resem->her':>11s}",flush=True)
agg={}
for an,kw in ARMS:
    ds=[];um=[];rs=[]
    for c in clips:
        w,_=librosa.load(f"cb_ar/{c}.mp3",sr=SR,mono=True)
        out=w if kw is None else np.nan_to_num(operator(w,**kw))
        ds.append(dc.delivery_sim(out)); um.append(utmos(out)); rs.append(resem(out))
        if c=="ex12_0": sf.write(f"/tmp/op_{an}.wav", out/(np.max(np.abs(out))+1e-9)*0.95, SR)
    agg[an]=(np.nanmean(ds),np.mean(um),np.mean(rs))
    print(f"{an:10s} {np.nanmean(ds):12.3f} {np.mean(um):7.3f} {np.mean(rs):11.3f}",flush=True)
o=agg['orig']
print(f"\nSUMMARY orig dlv {o[0]:.3f} -> best-op dlv {max(a[0] for a in agg.values()):.3f}  (target her 0.42); UTMOS/resem deltas above",flush=True)
print("MEASURE_DONE",flush=True)
