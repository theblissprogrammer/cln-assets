# -*- coding: utf-8 -*-
"""Polish Chatterbox: best-of-N (max identity) + exaggeration sweep (her emotion), Egyptian text."""
import os, json, warnings, numpy as np, librosa, soundfile as sf, torch
warnings.filterwarnings("ignore")
try:
    import perth
    if getattr(perth, "PerthImplicitWatermarker", None) is None:
        class _NoWM:
            def __init__(self,*a,**k): pass
            def apply_watermark(self,wav,*a,**k): return wav
            def get_watermark(self,*a,**k): return None
        perth.PerthImplicitWatermarker = _NoWM
except Exception: pass
def loud(w):
    r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
y,_=librosa.load("her_audio.wav",sr=24000,mono=True); iv=librosa.effects.split(y,top_db=30)
buf=[];cur=0
for s,e in iv:
    buf.append(y[s:e]);cur+=e-s
    if cur>30*24000: break
sf.write("her_ref30.wav", loud(np.concatenate(buf)[:30*24000]),24000)
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
model=ChatterboxMultilingualTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
SR=getattr(model,"sr",24000)
EGY=["هنتقابل تاني يوم الخميس عشان نخلّص المراجعة.",
     "بصّت من الشباك مدة طويلة من غير ما تقول أي حاجة.",
     "مشينا على طول البحر شوية وإحنا بنتكلم في حاجات عادية.",
     "بعد كل اللي عدّى علينا، رجعنا تاني لنفس المكان."]
json.dump(EGY,open("texts.json","w"))
N=4
for exag in [0.5, 1.2]:
    d=f"out_ex{int(exag*10)}"; os.makedirs(d,exist_ok=True)
    for i,t in enumerate(EGY):
        for n in range(N):
            try:
                wav=model.generate(t, language_id="ar", audio_prompt_path="her_ref30.wav", cfg_weight=0.5, exaggeration=exag)
                sf.write(f"{d}/{i}_t{n}.wav", wav.squeeze().detach().cpu().numpy(), SR)
            except Exception as ex:
                print("GEN_ERR",exag,i,n,str(ex)[:80],flush=True)
    print("EXAG",exag,flush=True)
print("GEN_DONE",flush=True)
