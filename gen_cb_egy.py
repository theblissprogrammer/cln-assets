# -*- coding: utf-8 -*-
"""Chatterbox-Multilingual + EGYPTIAN-dialect text (combine the engine identity-win with dialect)."""
import os, json, glob, warnings, numpy as np, librosa, soundfile as sf, torch
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
    r = np.sqrt(np.mean(w**2)) + 1e-9
    return w * (10**(-23/20) / r)
y, _ = librosa.load("her_audio.wav", sr=24000, mono=True); iv = librosa.effects.split(y, top_db=30)
buf = []; cur = 0
for s, e in iv:
    buf.append(y[s:e]); cur += e - s
    if cur > 30*24000: break
sf.write("her_ref30.wav", loud(np.concatenate(buf)[:30*24000]), 24000)
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
SR = getattr(model, "sr", 24000)
EGY = ["هنتقابل تاني يوم الخميس عشان نخلّص المراجعة.",
       "معظم الكراتين كانت متجهّزة خلاص ومستنية جنب الباب.",
       "بصّت من الشباك مدة طويلة من غير ما تقول أي حاجة.",
       "مشينا على طول البحر شوية وإحنا بنتكلم في حاجات عادية.",
       "بعد كل اللي عدّى علينا، رجعنا تاني لنفس المكان.",
       "كانت الجنينة هادية خالص الصبح بدري."]
json.dump(EGY, open("texts.json", "w"))
for cfg in [0.0, 0.3, 0.5]:
    d = f"out_cb_{str(cfg).replace('.','p')}"; os.makedirs(d, exist_ok=True)
    for i, t in enumerate(EGY):
        try:
            wav = model.generate(t, language_id="ar", audio_prompt_path="her_ref30.wav", cfg_weight=cfg)
            sf.write(f"{d}/{i}.wav", wav.squeeze().detach().cpu().numpy(), SR)
        except Exception as ex:
            print("GEN_ERR", cfg, i, str(ex)[:100], flush=True)
    print("CBEGY", cfg, flush=True)
print("GEN_DONE", flush=True)
