# -*- coding: utf-8 -*-
"""
GATE-4 CORE (de-risk the deepest novelty, local, no GPU):
Is her delivery a LEARNABLE, CONTENT-RELATIVE, INDIVIDUAL predictor  f_her: slot -> delivery value?

If YES, the cross-lingual transfer can move the PREDICTOR (fit on her EN slots, applied to AR slots)
instead of frame-aligned VALUES -> sidesteps the no-1:1-alignment wall AND the
learned-embedding-leakage wall (Sigurgeirsson-King ICASSP2023) by construction.

slot           = an inter-pausal phrase (language-agnostic unit)
slot FEATURES  = position_in_utt, is_final, is_initial, n_syll(energy-peak proxy), dur, rel_energy   (NO phonemes/timbre)
slot VALUES    = declination(st/s), f0_range_st, contour DCT1-3 (register-removed log-F0), energy_prom   (delivery)

TESTS:
 (A) PREDICTABILITY: her-trained ridge, 5-fold CV on her phrases -> R^2 > 0 means her delivery is
     content-STRUCTURED (beats predicting her global mean = the 'neutral' baseline).
 (B) INDIVIDUALITY: train ridge on GENERIC (pooled female impostors), apply to her held-out.
     If her-trained R^2 >> generic-trained R^2 on HER data -> the predictor is HERS, not a universal
     English-prosody rule -> f_her is a personal operator worth transferring.
"""
import os, numpy as np, librosa, warnings
warnings.filterwarnings("ignore")
import parselmouth
from scipy.fftpack import dct
from scipy.signal import find_peaks
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

SR=24000
def f0_of(w):
    snd=parselmouth.Sound(w.astype(np.float64),sampling_frequency=SR)
    return snd.to_pitch(0.01,100,500).selected_array['frequency']
st=lambda hz:12*np.log2(np.maximum(hz,1e-6)/55.0)

def phrases(w):
    """inter-pausal phrases + utterance grouping by >0.35s gaps. returns list of dicts."""
    iv=librosa.effects.split(w,top_db=30)
    iv=[(s,e) for s,e in iv if (e-s)>0.30*SR]
    if len(iv)<4: return []
    # utterance groups by big gaps
    groups=[]; cur=[0]
    for i in range(1,len(iv)):
        gap=(iv[i][0]-iv[i-1][1])/SR
        if gap>0.35: groups.append(cur); cur=[i]
        else: cur.append(i)
    groups.append(cur)
    f0=f0_of(w)
    rms=librosa.feature.rms(y=w.astype(np.float32),frame_length=1024,hop_length=256)[0]
    utt_eng=np.mean(20*np.log10(rms+1e-6))
    out=[]
    for g in groups:
        n=len(g)
        for k,idx in enumerate(g):
            s,e=iv[idx]
            fs=int(s/256); fe=max(int(e/256),fs+1)
            # F0 over phrase (voiced)
            a=int(s*0.01*SR/SR/0.01)  # frame idx in f0 (10ms hop): s/ (0.01*SR)
            fa=int(s/(0.01*SR)); fb=int(e/(0.01*SR))
            seg_f0=f0[fa:fb]; v=seg_f0>0
            if v.sum()<8: continue
            lf=st(seg_f0[v])
            # value channels
            x=np.arange(v.sum())*0.01
            declin=np.polyfit(x,lf,1)[0] if v.sum()>=8 else 0.0
            f0range=np.percentile(lf,95)-np.percentile(lf,5)
            c=lf-lf.mean(); s_=np.std(c); c=c/s_ if s_>1e-6 else c
            c=np.interp(np.linspace(0,1,32),np.linspace(0,1,len(c)),c)
            d=dct(c,norm='ortho')[1:4]
            # phrase energy + syllable proxy
            seg_rms=rms[fs:fe]; eng=np.mean(20*np.log10(seg_rms+1e-6))
            env=seg_rms/(seg_rms.max()+1e-9)
            pk,_=find_peaks(env,height=0.45,distance=12)  # ~120ms apart
            nsyll=max(len(pk),1)
            dur=(e-s)/SR
            feat=[k/max(n-1,1), float(k==n-1), float(k==0), nsyll, dur, eng-utt_eng]
            val=[declin, f0range, d[0], d[1], d[2], np.mean(env)]
            out.append((feat,val))
    return out

def collect(path):
    w,_=librosa.load(path,sr=SR,mono=True)
    return phrases(w)

print("== segmenting her into phrase slots ==",flush=True)
her=collect("her_audio.wav")
print(f"her phrases: {len(her)}",flush=True)
gen=[]
for f in sorted(os.listdir("imp2")):
    w,_=librosa.load(f"imp2/{f}",sr=SR,mono=True)
    f0=f0_of(w); vd=f0[f0>0]
    if len(vd)==0 or np.median(vd)<165: continue   # female only
    gen+=phrases(w)
print(f"generic(female impostor) phrases: {len(gen)}",flush=True)

CH=["declin","f0range","dct1","dct2","dct3","energy_prom"]
Xh=np.array([f for f,v in her]); Yh=np.array([v for f,v in her])
Xg=np.array([f for f,v in gen]); Yg=np.array([v for f,v in gen])

def r2(y,yhat):
    ss=np.sum((y-y.mean())**2); return 1-np.sum((y-yhat)**2)/(ss+1e-12)

print("\n================ (A) PREDICTABILITY + (B) INDIVIDUALITY (per delivery channel, R^2 on HER held-out) ================",flush=True)
print(f"{'channel':12s} {'her-trained':>12s} {'generic-trained':>16s} {'gain(her-gen)':>14s}",flush=True)
kf=KFold(5,shuffle=True,random_state=0)
her_r2={}; gen_r2={}
for j,ch in enumerate(CH):
    # her-trained 5-fold CV
    preds=np.zeros(len(Xh))
    for tr,te in kf.split(Xh):
        sc=StandardScaler().fit(Xh[tr])
        m=Ridge(alpha=1.0).fit(sc.transform(Xh[tr]),Yh[tr,j])
        preds[te]=m.predict(sc.transform(Xh[te]))
    her_r2[ch]=r2(Yh[:,j],preds)
    # generic-trained -> her (all her as held-out)
    scg=StandardScaler().fit(Xg)
    mg=Ridge(alpha=1.0).fit(scg.transform(Xg),Yg[:,j])
    gp=mg.predict(scg.transform(Xh))
    gen_r2[ch]=r2(Yh[:,j],gp)
    print(f"{ch:12s} {her_r2[ch]:12.3f} {gen_r2[ch]:16.3f} {her_r2[ch]-gen_r2[ch]:14.3f}",flush=True)

mh=np.mean(list(her_r2.values())); mg2=np.mean(list(gen_r2.values()))
print(f"\nSUMMARY her_mean_R2={mh:.3f}  generic_mean_R2={mg2:.3f}  individuality_gain={mh-mg2:.3f}",flush=True)
print(f"VERDICT_A_predictable: {'YES (delivery is content-structured, beats neutral/global-mean)' if mh>0.05 else 'WEAK (delivery ~ noise at phrase level)'}",flush=True)
print(f"VERDICT_B_individual:  {'YES (her predictor beats generic on her data -> personal operator)' if (mh-mg2)>0.02 else 'NO (her predictor ~ generic English-prosody rule -> not individual)'}",flush=True)
print("MEASURE_DONE",flush=True)
