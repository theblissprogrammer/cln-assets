# -*- coding: utf-8 -*-
"""PHASE 2a — from the bilingual corpus, build the per-speaker L1(EN) vs L2(AR) feature table and learn
the CROSS-LINGUAL TRANSFORMATION: which identity features are INVARIANT (transfer L1->L2 = the real
identity) vs which SHIFT systematically by language (the thing zero-shot clones get wrong). With 26
speakers we test if L2 is PREDICTABLE from L1 -> basis for calibrating a new speaker's clone to their
(predicted) L2 identity. VQ grain (HNR/jitter/shimmer/tilt) + delivery (f0med/range/dyn/coupling)."""
import os, glob, warnings, numpy as np, librosa, parselmouth
warnings.filterwarnings("ignore")
from parselmouth.praat import call
SR=24000; st=lambda hz:12*np.log2(np.maximum(hz,1e-6)/55.0)
def feats(paths):
    H=[];J=[];SH=[];CP=[];TI=[];FM=[];FR=[];FD=[];CO=[]
    for p in paths:
        try:
            w,_=librosa.load(p,sr=SR,mono=True); w=w/(np.max(np.abs(w))+1e-9)
            snd=parselmouth.Sound(w.astype(np.float64),sampling_frequency=SR)
            H.append(call(snd.to_harmonicity_cc(0.01,75,0.1,1.0),'Get mean',0,0))
            pp=call(snd,'To PointProcess (periodic, cc)',75,500)
            J.append(call(pp,'Get jitter (local)',0,0,0.0001,0.02,1.3)*100)
            SH.append(call([snd,pp],'Get shimmer (local)',0,0,0.0001,0.02,1.3,1.6)*100)
            try: CP.append(call(call(snd,'To PowerCepstrogram',60,0.002,5000,50),'Get CPPS',False,0.01,0.001,60,330,0.05,'Parabolic',0.001,0,'Straight','Robust'))
            except: pass
            S=np.abs(librosa.stft(w.astype('float32'),n_fft=2048))**2; f=librosa.fft_frequencies(sr=SR,n_fft=2048); b=(f>200)&(f<8000)
            TI.append(float(np.polyfit(np.log10(f[b]),10*np.log10(S[b].mean(1)+1e-9),1)[0]))
            f0=snd.to_pitch(0.01,65,500).selected_array['frequency']; v=f0[f0>0]
            if len(v)>20:
                FM.append(float(np.median(v))); FR.append(float(np.percentile(st(v),95)-np.percentile(st(v),5)))
                FD.append(float(np.mean(np.abs(np.diff(st(f0[f0>0]))))/0.01))
                rms=librosa.feature.rms(y=w.astype('float32'),frame_length=int(0.025*SR),hop_length=int(0.01*SR))[0]
                n=min(len(f0),len(rms)); m=f0[:n]>0
                CO.append(float(np.corrcoef(st(f0[:n][m]),20*np.log10(rms[:n][m]+1e-6))[0,1]) if m.sum()>10 else 0)
        except Exception: pass
    g=lambda x: float(np.median(x)) if x else float('nan')
    return dict(HNR=g(H),jit=g(J),shim=g(SH),CPP=g(CP),tilt=g(TI),f0med=g(FM),range=g(FR),dyn=g(FD),coup=g(CO))

speakers=sorted(set(os.path.basename(p).split('_en_')[0].split('_ar_')[0] for p in glob.glob('bicorpus/*.wav')))
rows={}
for sp in speakers:
    en=sorted(glob.glob(f'bicorpus/{sp}_en_*.wav')); ar=sorted(glob.glob(f'bicorpus/{sp}_ar_*.wav'))
    if not en or not ar: continue
    rows[sp]=(feats(en),feats(ar))
print(f"feature table: {len(rows)} bilingual speakers\n")
FEATS=['HNR','jit','shim','tilt','f0med','range','dyn','coup']
import numpy as np
print("CROSS-LINGUAL TRANSFORMATION (across speakers): is each feature INVARIANT (L1=L2=identity) or SHIFTED (language)?")
print(f"{'feat':6s} {'L1_mean':>8s} {'L2_mean':>8s} {'meanShift':>9s} {'L1<->L2 corr':>12s}  interpretation")
for f in FEATS:
    l1=np.array([rows[s][0][f] for s in rows]); l2=np.array([rows[s][1][f] for s in rows])
    ok=~(np.isnan(l1)|np.isnan(l2)); l1,l2=l1[ok],l2[ok]
    corr=float(np.corrcoef(l1,l2)[0,1]) if len(l1)>3 else float('nan')
    shift=float(np.mean(l2-l1))
    interp = "INVARIANT identity (predictable L1->L2)" if corr>0.5 and abs(shift)<0.3*abs(np.mean(l1)+1e-9) else ("SHIFTS by language" if abs(shift)>0.2*abs(np.mean(l1)+1e-9) else "noisy")
    print(f"{f:6s} {np.mean(l1):8.1f} {np.mean(l2):8.1f} {shift:+9.1f} {corr:12.2f}  {interp}")
print("\n=> features with high L1<->L2 corr = a speaker's identity we can PREDICT/transfer; systematic shifts = the L1->L2 language effect to apply.")
print("DONE")
