# -*- coding: utf-8 -*-
"""Chatterbox-Multilingual Arabic clone of her (EN ref -> Arabic), cfg_weight sweep."""
import os, json, warnings, numpy as np, librosa, soundfile as sf, torch
warnings.filterwarnings("ignore")
def loud(w):
    r = np.sqrt(np.mean(w**2)) + 1e-9
    return w * (10**(-23/20) / r)
y, _ = librosa.load("her_audio.wav", sr=24000, mono=True)
iv = librosa.effects.split(y, top_db=30)
buf = []; cur = 0
for s, e in iv:
    buf.append(y[s:e]); cur += e - s
    if cur > 30*24000: break
ref = np.concatenate(buf)[:30*24000]
sf.write("her_ref30.wav", loud(ref), 24000)
print("ref written", flush=True)
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
print("chatterbox loaded, sr=", getattr(model, "sr", "?"), flush=True)
AR = ["سوف نلتقي مرة أخرى يوم الخميس لإنهاء المراجعة.",
      "معظم الصناديق كانت معبأة بالفعل وتنتظر بجانب الباب.",
      "نظرت من النافذة لفترة طويلة دون أن تقول شيئاً.",
      "مشينا على طول الشاطئ لبعض الوقت نتحدث عن أشياء عادية.",
      "بعد كل ما مررنا به انتهى بنا الأمر في نفس المكان.",
      "كانت الحديقة هادئة تماماً في ذلك الصباح الباكر."]
json.dump(AR, open("texts.json", "w"))
SR = getattr(model, "sr", 24000)
for cfg in [0.5, 0.3, 0.0]:
    d = f"out_cb_{str(cfg).replace('.','p')}"; os.makedirs(d, exist_ok=True)
    for i, t in enumerate(AR):
        try:
            wav = model.generate(t, language_id="ar", audio_prompt_path="her_ref30.wav", cfg_weight=cfg)
            arr = wav.squeeze().detach().cpu().numpy()
            sf.write(f"{d}/{i}.wav", arr, SR)
        except Exception as ex:
            print("GEN_ERR", cfg, i, str(ex)[:120], flush=True)
    print("CB", cfg, flush=True)
print("GEN_DONE", flush=True)
