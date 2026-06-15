# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings("ignore")
import numpy as np, librosa, soundfile as sf, whisper
from source_lpc import analyze, synth, shift_formant
SR=16000
def combo(a,f1,f2): return shift_formant(shift_formant(a,1,f1),2,f2)
y,_=librosa.load("her_audio.wav",sr=SR,mono=True); iv=librosa.effects.split(y,top_db=30)
seg=[y[s:e] for s,e in iv if (e-s)>4*SR][0]
sf.write("/tmp/o_orig.wav", seg, SR)
sf.write("/tmp/o_subtle.wav", np.nan_to_num(synth(analyze(seg,SR,18), lambda a:shift_formant(a,2,1.2))), SR)
sf.write("/tmp/o_strong_ee.wav", np.nan_to_num(synth(analyze(seg,SR,18), lambda a:combo(a,0.5,1.7))), SR)
sf.write("/tmp/o_strong_oo.wav", np.nan_to_num(synth(analyze(seg,SR,18), lambda a:combo(a,0.6,0.55))), SR)
m=whisper.load_model("base")
from resemblyzer import VoiceEncoder, preprocess_wav
def loud(w): r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
enc=VoiceEncoder(verbose=False)
def emb(w): e=enc.embed_utterance(preprocess_wav(loud(w),source_sr=SR)); return e/np.linalg.norm(e)
alls=[y[s:e] for s,e in iv if (e-s)>1.5*SR]; herv=np.mean([emb(s) for s in alls],0); herv/=np.linalg.norm(herv)
print("="*60)
for f in ["o_orig","o_subtle","o_strong_ee","o_strong_oo"]:
    txt=m.transcribe(f"/tmp/{f}.wav",language="en")["text"].strip()
    w,_=librosa.load(f"/tmp/{f}.wav",sr=SR); r=float(emb(w)@herv)
    print(f"{f:14s} resem->HER={r:.3f}  STT: {txt[:80]}")
print("="*60)
