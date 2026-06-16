# -*- coding: utf-8 -*-
"""Build a MULTI-SPEAKER corpus (CREMA-D: 91 actors x 6 emotions) with wide DELIVERY variation, to
train the delivery adapter ONCE -> apply ZERO-SHOT to any voice (replicable+fast). PARALLEL downloads
(robust per-file). Hold out actors 1079-1091 for the zero-shot gate. Self-supervised delivery_vec."""
import os, csv, warnings, urllib.request, numpy as np, librosa, soundfile as sf
warnings.filterwarnings("ignore")
from concurrent.futures import ThreadPoolExecutor, as_completed
import parselmouth
SR=24000; st=lambda hz:12*np.log2(np.maximum(hz,1e-6)/55.0)
os.makedirs("train/wavs",exist_ok=True); os.makedirs("cd_dl",exist_ok=True)
BASE="https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV"
TEXT={"IEO":"It's eleven o'clock.","TIE":"That is exactly what happened.","IOM":"I'm on my way to the meeting.",
      "IWW":"I wonder what this is about.","TAI":"The airplane is almost full.","MTI":"Maybe tomorrow it will be cold.",
      "IWL":"I would like a new alarm clock.","ITH":"I think I have a doctor's appointment.","DFA":"Don't forget a jacket.",
      "ITS":"I think I've seen this before.","TSI":"The surface is slick.","WSI":"We'll stop in a couple of minutes."}
SENT=["IEO","TIE","DFA","WSI","TAI"]; EMO=["ANG","HAP","SAD","NEU","FEA","DIS"]
TRAIN_ACT=list(range(1001,1079))   # 78 actors

cands=[(act,s,e,f"{act}_{s}_{e}_XX.wav") for act in TRAIN_ACT for s in SENT for e in EMO]
def fetch(c):
    act,s,e,fn=c; p=f"cd_dl/{fn}"
    if os.path.exists(p) and os.path.getsize(p)>2000: return c
    try:
        urllib.request.urlretrieve(f"{BASE}/{fn}", p)
        return c if (os.path.exists(p) and os.path.getsize(p)>2000) else None
    except Exception:
        return None
print(f"downloading {len(cands)} candidates (parallel)...",flush=True)
ok=[]
with ThreadPoolExecutor(max_workers=24) as ex:
    futs={ex.submit(fetch,c):c for c in cands}
    for n,f in enumerate(as_completed(futs)):
        r=f.result()
        if r: ok.append(r)
        if (n+1)%300==0: print(f"  fetched {len(ok)}/{n+1}",flush=True)
print(f"downloaded {len(ok)} clips; extracting features...",flush=True)

def feats(w):
    f=parselmouth.Sound(w.astype(np.float64),sampling_frequency=SR).to_pitch(0.01,65,500).selected_array['frequency']; v=f[f>0]
    if len(v)<8: return None
    rng=float(np.percentile(st(v),95)-np.percentile(st(v),5)); dyn=float(np.mean(np.abs(np.diff(st(f[f>0]))))/0.01)
    rms=librosa.feature.rms(y=w.astype(np.float32),frame_length=int(0.025*SR),hop_length=int(0.01*SR))[0]
    n=min(len(f),len(rms)); m=f[:n]>0
    cpl=float(np.corrcoef(st(f[:n][m]),20*np.log10(rms[:n][m]+1e-6))[0,1]) if m.sum()>10 else 0.0
    return rng,dyn,cpl
rows=[]
for act,s,e,fn in ok:
    try:
        w,_=librosa.load(f"cd_dl/{fn}",sr=SR,mono=True)
        if len(w)<0.6*SR: continue
        fe=feats(w)
        if fe is None: continue
        out=f"train/wavs/{fn}"; sf.write(out,w,SR)
        rows.append(dict(wav=out,dur=round(len(w)/SR,2),text=TEXT[s],spk=str(act),
                         f0range=round(fe[0],3),f0dyn=round(fe[1],3),coupling=round(fe[2],3)))
    except Exception: pass
with open("train/manifest.csv","w",newline="") as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
arr=lambda k:np.array([r[k] for r in rows])
print(f"\nMULTISPK CORPUS: {len(set(r['spk'] for r in rows))} speakers, {len(rows)} clips",flush=True)
print(f"  f0range spread p10-p90 [{np.percentile(arr('f0range'),10):.1f},{np.percentile(arr('f0range'),90):.1f}] median {np.median(arr('f0range')):.1f} WIDE",flush=True)
print("CORPUS_DONE",flush=True)
