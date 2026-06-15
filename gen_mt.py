# -*- coding: utf-8 -*-
"""metric-test gen: her XTTS Arabic (synthesis) + generic Arabic (kNN source)."""
import os, glob, warnings, numpy as np, librosa, soundfile as sf, torch
warnings.filterwarnings("ignore")
def loud(w):
    r = np.sqrt(np.mean(w**2)) + 1e-9
    return w * (10**(-23/20) / r)
y, sr = librosa.load("her_audio.wav", sr=24000, mono=True)
iv = librosa.effects.split(y, top_db=30)
segs = []; buf = []; cur = 0
for s, e in iv:
    buf.append(y[s:e]); cur += e - s
    if cur > 10*sr:
        segs.append(np.concatenate(buf)); buf = []; cur = 0
os.makedirs("ref_her", exist_ok=True)
mid = len(segs)//2
for k, s in enumerate(segs[max(0, mid-2):mid+2][:4]):
    sf.write(f"ref_her/h{k}.wav", loud(s), 24000)
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda" if torch.cuda.is_available() else "cpu")
m = tts.synthesizer.tts_model
gl, sp = m.get_conditioning_latents(audio_path=sorted(glob.glob("ref_her/*.wav")), max_ref_length=30, gpt_cond_len=30, gpt_cond_chunk_len=6)
gname = list(m.speaker_manager.speakers.keys())[0]; gi = m.speaker_manager.speakers[gname]
gl_g, sp_g = gi["gpt_cond_latent"], gi["speaker_embedding"]
AR = ["سوف نلتقي مرة أخرى يوم الخميس لإنهاء المراجعة.",
      "معظم الصناديق كانت معبأة بالفعل وتنتظر بجانب الباب.",
      "نظرت من النافذة لفترة طويلة دون أن تقول شيئاً.",
      "مشينا على طول الشاطئ لبعض الوقت نتحدث عن أشياء عادية.",
      "بعد كل ما مررنا به انتهى بنا الأمر في نفس المكان.",
      "كانت الحديقة هادئة تماماً في ذلك الصباح الباكر."]
import json; json.dump(AR, open("texts.json", "w"))
os.makedirs("out_synth", exist_ok=True); os.makedirs("out_src", exist_ok=True)
for i, t in enumerate(AR):
    o = m.inference(t, "ar", gl, sp, temperature=0.7, enable_text_splitting=True); sf.write(f"out_synth/{i}.wav", np.asarray(o["wav"]), 24000)
    o = m.inference(t, "ar", gl_g, sp_g, temperature=0.7, enable_text_splitting=True); sf.write(f"out_src/{i}.wav", np.asarray(o["wav"]), 24000)
print("GEN_DONE", flush=True)
