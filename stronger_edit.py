# -*- coding: utf-8 -*-
"""Stronger AUDIBLE constriction edits: clearly change the VOWEL (F1/F2 combined) while keeping
her anatomy-scale + real residual. Does an audibly-different sound keep her identity?"""
import os, warnings, numpy as np, librosa, soundfile as sf
warnings.filterwarnings("ignore")
from resemblyzer import VoiceEncoder, preprocess_wav
from source_lpc import analyze, synth, shift_formant
SR=16000
def loud(w): r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
enc=VoiceEncoder(verbose=False)
def emb(w): e=enc.embed_utterance(preprocess_wav(loud(w),source_sr=SR)); return e/np.linalg.norm(e)
y,_=librosa.load("her_audio.wav",sr=SR,mono=True); iv=librosa.effects.split(y,top_db=30)
segs=[y[s:e] for s,e in iv if (e-s)>3*SR][:6]
alls=[y[s:e] for s,e in iv if (e-s)>1.5*SR]
herv=np.mean([emb(s) for s in alls],0); herv/=np.linalg.norm(herv)
def combo(a, f1, f2):
    a2=shift_formant(a,1,f1); a2=shift_formant(a2,2,f2); return a2
EDITS={
  "toEE_F1x0.5_F2x1.7": (lambda a: combo(a,0.5,1.7)),   # high front -> "ee"
  "toOO_F1x0.6_F2x0.55":(lambda a: combo(a,0.6,0.55)),  # high back -> "oo"
  "toAA_F1x1.5_F2x0.85":(lambda a: combo(a,1.5,0.85)),  # open -> "aa"
  "toER_F1x1.0_F2x1.5": (lambda a: combo(a,1.0,1.5)),   # front shift
}
os.makedirs("/tmp/ear_clips/strong",exist_ok=True)
print(f"her centroid from {len(alls)} segs",flush=True)
for name,mod in EDITS.items():
    sims=[]
    for j,sg in enumerate(segs):
        ys=np.nan_to_num(synth(analyze(sg,SR,18),mod))
        sims.append(float(emb(ys)@herv))
        if j==1: sf.write(f"/tmp/ear_clips/strong/{name}.wav", ys/(np.max(np.abs(ys))+1e-9)*0.95, SR)
    print(f"EDIT {name:22s} resem->HER={np.mean(sims):.3f}",flush=True)
