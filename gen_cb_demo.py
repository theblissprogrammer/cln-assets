# -*- coding: utf-8 -*-
"""Robustness + comprehensive demo: diverse Egyptian sentences (Q/long/emotional/numbers), best-of-3."""
import os, json, warnings, numpy as np, librosa, soundfile as sf, torch
warnings.filterwarnings("ignore")
try:
    import perth
    if getattr(perth,"PerthImplicitWatermarker",None) is None:
        class _N:
            def __init__(s,*a,**k): pass
            def apply_watermark(s,w,*a,**k): return w
            def get_watermark(s,*a,**k): return None
        perth.PerthImplicitWatermarker=_N
except Exception: pass
def loud(w):
    r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
y,_=librosa.load("her_audio.wav",sr=24000,mono=True); iv=librosa.effects.split(y,top_db=30)
buf=[];cur=0
for s,e in iv:
    buf.append(y[s:e]);cur+=e-s
    if cur>30*24000: break
sf.write("her_ref30.wav",loud(np.concatenate(buf)[:30*24000]),24000)
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
model=ChatterboxMultilingualTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
SR=getattr(model,"sr",24000)
# diverse Egyptian colloquial: (text, exaggeration)
ITEMS=[("إنت جاي إمتى بالظبط؟",0.5),
       ("الأكل كان جامد أوي النهارده.",0.5),
       ("ما تتخيلش أنا فرحان قد إيه دلوقتي!",1.3),
       ("كنا قاعدين على القهوة بنتكلم في كل حاجة لحد ما الدنيا ضلمت وقررنا نمشي.",0.5),
       ("حسيت بوحدة غريبة وأنا ماشي لوحدي في الشارع.",1.0),
       ("الحساب طلع ميتين وخمسة وسبعين جنيه.",0.5),
       ("استنى لحظة، أنا مش فاهم إنت بتقول إيه.",0.8),
       ("يلا بينا نتمشى شوية، الجو حلو بره.",0.6),
       ("والله العظيم ده أحسن خبر سمعته من زمان!",1.3),
       ("لو سمحت ممكن تعيد اللي قلته تاني؟",0.5)]
json.dump([t for t,_ in ITEMS],open("texts.json","w"))
os.makedirs("out_demo",exist_ok=True)
for i,(t,ex) in enumerate(ITEMS):
    bestw=None; bestn=-9
    for n in range(3):
        try:
            w=model.generate(t,language_id="ar",audio_prompt_path="her_ref30.wav",cfg_weight=0.4,exaggeration=ex)
            a=w.squeeze().detach().cpu().numpy()
            sf.write(f"out_demo/{i}_t{n}.wav",a,SR)
        except Exception as e: print("GEN_ERR",i,n,str(e)[:60],flush=True)
    print("DEMO",i,flush=True)
print("GEN_DONE",flush=True)
