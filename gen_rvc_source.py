# -*- coding: utf-8 -*-
import os, numpy as np, soundfile as sf, torch, warnings
warnings.filterwarnings("ignore")
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda" if torch.cuda.is_available() else "cpu")
m = tts.synthesizer.tts_model
g = list(m.speaker_manager.speakers.keys())[0]; gi = m.speaker_manager.speakers[g]
gl, sp = gi["gpt_cond_latent"], gi["speaker_embedding"]
AR = ["سوف نلتقي مرة أخرى يوم الخميس لإنهاء المراجعة.",
      "معظم الصناديق كانت معبأة بالفعل وتنتظر بجانب الباب.",
      "نظرت من النافذة لفترة طويلة دون أن تقول شيئاً.",
      "مشينا على طول الشاطئ لبعض الوقت نتحدث عن أشياء عادية.",
      "بعد كل ما مررنا به انتهى بنا الأمر في نفس المكان.",
      "كانت الحديقة هادئة تماماً في ذلك الصباح الباكر."]
os.makedirs("/workspace/src", exist_ok=True)
for i, t in enumerate(AR):
    o = m.inference(t, "ar", gl, sp, temperature=0.7, enable_text_splitting=True)
    sf.write(f"/workspace/src/{i}.wav", np.asarray(o["wav"]), 24000)
print("SRC_GEN", len(AR), flush=True)
