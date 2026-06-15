# -*- coding: utf-8 -*-
"""Definitive identity proof: her real LOO (positives) vs ~35 LibriSpeech impostors (negatives) -> EER + threshold.
Then position the clone scores (XTTS 0.86, Chatterbox 0.92) vs the impostor distribution."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, librosa, torchaudio, glob
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.metrics import roc_curve
def loud(w):
    r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
enc=VoiceEncoder(verbose=False)
def emb(w16): e=enc.embed_utterance(preprocess_wav(loud(w16),source_sr=16000)); return e/np.linalg.norm(e)
y,_=librosa.load("her_audio.wav",sr=16000,mono=True); iv=librosa.effects.split(y,top_db=30)
hs=[y[s:e] for s,e in iv if (e-s)>1.5*16000]
E=np.array([emb(s) for s in hs]); herv=E.mean(0); herv/=np.linalg.norm(herv)
pos=[float(E[i]@(lambda c:c/np.linalg.norm(c))(np.delete(E,i,0).mean(0))) for i in range(len(E))]
print("her segments:",len(hs),flush=True)
ds=torchaudio.datasets.LIBRISPEECH(".",url="dev-clean",download=True)
print("librispeech ready",flush=True)
seen=set(); neg=[]
for i in range(len(ds)):
    item=ds[i]; wav,sr,spk=item[0],item[1],item[3]
    if spk in seen: continue
    seen.add(spk)
    w16=librosa.resample(wav.squeeze().numpy(),orig_sr=sr,target_sr=16000)
    if len(w16)<1.0*16000: continue
    neg.append(float(emb(w16)@herv))
    if len(neg)>=35: break
neg=np.array(neg)
labels=[1]*len(pos)+[0]*len(neg); scores=list(pos)+list(neg)
fpr,tpr,thr=roc_curve(labels,scores); fnr=1-tpr
k=int(np.nanargmin(np.abs(fnr-fpr))); eer=(fpr[k]+fnr[k])/2
print(f"IMPOSTOR_CALIB her_LOO_pos mean={np.mean(pos):.3f} min={np.min(pos):.3f} p10={np.percentile(pos,10):.3f}",flush=True)
print(f"IMPOSTOR_CALIB impostor_neg mean={np.mean(neg):.3f} max={np.max(neg):.3f} p90={np.percentile(neg,90):.3f} (n={len(neg)})",flush=True)
print(f"IMPOSTOR_CALIB EER={eer*100:.1f}% decision_threshold={thr[k]:.3f}",flush=True)
for name,sc in [("XTTS-Arabic",0.86),("Chatterbox-Arabic",0.92)]:
    print(f"IMPOSTOR_CALIB CLONE {name} score={sc} above_EER_thr={bool(sc>thr[k])} above_ALL_{len(neg)}_impostors={bool(sc>np.max(neg))}",flush=True)
print("MEASURE_DONE",flush=True)
