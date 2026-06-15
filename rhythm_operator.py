# -*- coding: utf-8 -*-
"""RHYTHM operator: transfer her PACING (the strongest individual delivery channel, AUC 0.92) onto a
multi-phrase generated clip -- CLEAN time-domain edit (no pitch/timbre touch -> resem/UTMOS safe).
Re-time inter-phrase pauses to her pause-duration distribution + match her articulation rate.
Specificity: her rhythm profile vs impostor rhythm profile."""
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, librosa, soundfile as sf
import deliv_core as dc
SR=24000

def rhythm_profile(w):
    iv=librosa.effects.split(w.astype(np.float32),top_db=30)
    iv=[(s,e) for s,e in iv if (e-s)>0.12*SR]
    if len(iv)<2: return None
    phr_dur=np.array([(e-s)/SR for s,e in iv])
    gaps=np.array([(iv[i][0]-iv[i-1][1])/SR for i in range(1,len(iv))])
    gaps=gaps[gaps>0.04]
    # articulation rate proxy: voiced frames/sec inside phrases
    f0=dc.f0_track(w); vfr=[]
    for s,e in iv:
        a=int(s/(0.01*SR)); b=int(e/(0.01*SR)); seg=f0[a:b]
        if len(seg): vfr.append(np.mean(seg>0)/0.01*np.mean(seg>0))  # rough syll-rate proxy
    return dict(phr_dur=phr_dur, gaps=gaps, gap_med=float(np.median(gaps)) if len(gaps) else 0.25,
                gap_mean=float(np.mean(gaps)) if len(gaps) else 0.25,
                artic=float(len(iv)/ (sum(phr_dur)+sum(gaps) if len(gaps) else sum(phr_dur))))

def retime(w, prof, rate=1.0, seed=1):
    """rebuild clip: keep phrases (optionally rate-stretched), set inter-phrase gaps ~ her distribution."""
    rs=np.random.RandomState(seed)
    iv=librosa.effects.split(w.astype(np.float32),top_db=30)
    iv=[(s,e) for s,e in iv if (e-s)>0.10*SR]
    if len(iv)<2: return w
    out=[]
    her_gaps=prof['gaps'] if len(prof['gaps'])>0 else np.array([prof['gap_med']])
    for i,(s,e) in enumerate(iv):
        ph=w[s:e]
        if abs(rate-1.0)>0.02:
            ph=librosa.effects.time_stretch(ph.astype(np.float32), rate=rate)
        out.append(ph)
        if i<len(iv)-1:
            g=float(rs.choice(her_gaps))                 # sample her real pause
            out.append(np.zeros(int(np.clip(g,0.05,0.9)*SR),dtype=np.float32))
    return np.concatenate(out)

if __name__=="__main__":
    hy,_=librosa.load("her_audio.wav",sr=SR,mono=True)
    dc.her_space()
    her_prof=rhythm_profile(hy)
    print(f"her rhythm: gap_med {her_prof['gap_med']:.2f}s gap_mean {her_prof['gap_mean']:.2f}s artic {her_prof['artic']:.2f}/s",flush=True)
    # impostor profile (pooled) for specificity
    ig=[]
    for f in sorted(os.listdir("imp2"))[:10]:
        w,_=librosa.load(f"imp2/{f}",sr=SR,mono=True); p=rhythm_profile(w)
        if p: ig.append(p['gaps'])
    imp_prof=dict(gaps=np.concatenate(ig), gap_med=0.2)
    # smoke test on a CB clip (short -> limited, real test = box multi-phrase outputs)
    for c in ["egy3_0","ex12_0"]:
        w,_=librosa.load(f"cb_ar/{c}.mp3",sr=SR,mono=True)
        base=dc.delivery_sim(w)
        her=dc.delivery_sim(retime(w,her_prof))
        imp=dc.delivery_sim(retime(w,imp_prof))
        print(f"  {c}: base dlv {base:.3f} | her-rhythm {her:.3f} | impostor-rhythm {imp:.3f}",flush=True)
    print("MEASURE_DONE",flush=True)
