# -*- coding: utf-8 -*-
"""rank-2 kNN gate: generate Arabic content in a NON-her (generic) voice = the query.
kNN-VC will convert these into HER real frames; the test is whether identity wins cross-lingually."""
import os, numpy as np, soundfile as sf, torch, warnings
warnings.filterwarnings("ignore")
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda" if torch.cuda.is_available() else "cpu")
m = tts.synthesizer.tts_model
gen_name = list(m.speaker_manager.speakers.keys())[0]
ginfo = m.speaker_manager.speakers[gen_name]
gl, sp = ginfo["gpt_cond_latent"], ginfo["speaker_embedding"]
print(f"generic Arabic source speaker: {gen_name}", flush=True)
AR = ["سوف نلتقي مرة أخرى يوم الخميس لإنهاء المراجعة.",
      "معظم الصناديق كانت معبأة بالفعل وتنتظر بجانب الباب.",
      "نظرت من النافذة لفترة طويلة دون أن تقول شيئاً.",
      "مشينا على طول الشاطئ لبعض الوقت نتحدث عن أشياء عادية.",
      "بعد كل ما مررنا به، انتهى بنا الأمر في نفس المكان.",
      "كانت الحديقة هادئة تماماً في ذلك الصباح الباكر."]
os.makedirs("out_ar_src", exist_ok=True)
for i, t in enumerate(AR):
    o = m.inference(t, "ar", gl, sp, temperature=0.7, enable_text_splitting=True)
    sf.write(f"out_ar_src/{i}.wav", np.asarray(o["wav"]), 24000)
print("GEN_DONE", flush=True)
