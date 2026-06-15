# -*- coding: utf-8 -*-
"""SAGE STEP 1 gen: her XTTS Arabic clones = the base content (proven core: her voice + Arabic).
The arms then re-impose her frozen source on these."""
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
    if cur > 10 * sr:
        segs.append(np.concatenate(buf)); buf = []; cur = 0
if buf and cur > 6 * sr:
    segs.append(np.concatenate(buf))
os.makedirs("ref_her", exist_ok=True)
mid = len(segs) // 2
for k, s in enumerate(segs[max(0, mid-2):mid+2][:4]):
    sf.write(f"ref_her/h{k}.wav", loud(s), 24000)

from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda" if torch.cuda.is_available() else "cpu")
m = tts.synthesizer.tts_model
refs = sorted(glob.glob("ref_her/*.wav"))
gl, sp = m.get_conditioning_latents(audio_path=refs, max_ref_length=30, gpt_cond_len=30, gpt_cond_chunk_len=6)

AR = ["سوف نلتقي مرة أخرى يوم الخميس لإنهاء المراجعة.",
      "معظم الصناديق كانت معبأة بالفعل وتنتظر بجانب الباب.",
      "نظرت من النافذة لفترة طويلة دون أن تقول شيئاً.",
      "مشينا على طول الشاطئ لبعض الوقت نتحدث عن أشياء عادية.",
      "بعد كل ما مررنا به، انتهى بنا الأمر في نفس المكان.",
      "لم أكن أتوقع أن تتغير الأمور بهذه السرعة هذا العام.",
      "كانت الحديقة هادئة تماماً في ذلك الصباح الباكر.",
      "أخبرني أنه سيعود قبل حلول الظلام بكثير."]
os.makedirs("out_ar_base", exist_ok=True)
for i, txt in enumerate(AR):
    o = m.inference(txt, "ar", gl, sp, temperature=0.7, enable_text_splitting=True)
    sf.write(f"out_ar_base/{i}.wav", np.asarray(o["wav"]), 24000)
print(f"GEN_DONE {len(AR)} Arabic base clones", flush=True)
