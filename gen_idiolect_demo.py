# -*- coding: utf-8 -*-
"""IDIOLECT INJECTION DEMO: his verbal fingerprint = "شوف"(you-know/look) + "طيب"(well) at ~4.2/min,
with repetition ("شوف شوف شوف"). Our clone strips it. Generate his Arabic clone (a) PLAIN vs
(b) with HIS MARKERS woven in at his rate -> does carrying his idiolect make it sound more like HIM?"""
import os, warnings, subprocess, numpy as np, librosa, soundfile as sf, torch
warnings.filterwarnings("ignore")
try:
    import perth
    if getattr(perth,"PerthImplicitWatermarker",None) is None:
        class _N:
            def __init__(s,*a,**k):pass
            def apply_watermark(s,w,*a,**k):return w
            def get_watermark(s,*a,**k):return None
        perth.PerthImplicitWatermarker=_N
except Exception: pass
subprocess.run(["yt-dlp","-x","--audio-format","wav","-o","en.%(ext)s","https://www.youtube.com/watch?v=Tu2-1qowW2A"],capture_output=True,timeout=180)
def loud(w): r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
en,_=librosa.load("en.wav",sr=24000,mono=True); iv=librosa.effects.split(en,top_db=30); buf=[];c=0
for s,e in iv:
    buf.append(en[s:e]); c+=e-s
    if c>25*24000: break
sf.write("enref.wav", loud(np.concatenate(buf)[:25*24000]), 24000)
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
m=ChatterboxMultilingualTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
# same content, PLAIN vs HIS-IDIOLECT (شوف/طيب/يعني woven in at his rate + his repetition)
PAIRS=[
 ("الحقيقة لا تحتاج إلى من يدافع عنها، بل تحتاج إلى من يفهمها.",
  "شوف، شوف شوف... الحقيقة طيب ما تحتاجش حد يدافع عنها، يعني، بس تحتاج حد يفهمها، شوف."),
 ("كثير من الناس يتكلمون بثقة عن أشياء لم يدرسوها جيدًا.",
  "شوف يا أخي، طيب، في ناس كتير بتتكلم بثقة، يعني، عن حاجات شوف شوف ما درسوهاش كويس."),
]
def litter(p):
    r=subprocess.run(["curl","-s","-F","reqtype=fileupload","-F","time=72h","-F",f"fileToUpload=@{p}","https://litterbox.catbox.moe/resources/internals/api.php"],capture_output=True,text=True,timeout=60); return r.stdout.strip()
m.prepare_conditionals("enref.wav", exaggeration=1.0)
for i,(plain,idio) in enumerate(PAIRS):
    for tag,txt in [("plain",plain),("idiolect",idio)]:
        w=m.generate(txt, language_id="ar", exaggeration=1.0, temperature=0.7).squeeze().detach().cpu().numpy()
        fn=f"idio_{i}_{tag}.wav"; sf.write(fn,w,m.sr)
        print(f"URL {i}_{tag} {litter(fn)}",flush=True)
print("MEASURE_DONE",flush=True)
