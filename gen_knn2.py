# -*- coding: utf-8 -*-
"""rank-2 step2 gen: her XTTS-Arabic for POOL (phoneme coverage, disjoint content) and TEST (queries),
plus generic XTTS-Arabic TEST queries. AR_POOL != AR_TEST to avoid same-words leakage."""
import os, glob, json, warnings, numpy as np, librosa, soundfile as sf, torch
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
gl, sp = m.get_conditioning_latents(audio_path=sorted(glob.glob("ref_her/*.wav")), max_ref_length=30, gpt_cond_len=30, gpt_cond_chunk_len=6)
gen_name = list(m.speaker_manager.speakers.keys())[0]
ginfo = m.speaker_manager.speakers[gen_name]
gl_g, sp_g = ginfo["gpt_cond_latent"], ginfo["speaker_embedding"]

AR_TEST = ["سوف نلتقي مرة أخرى يوم الخميس لإنهاء المراجعة.",
           "معظم الصناديق كانت معبأة بالفعل وتنتظر بجانب الباب.",
           "نظرت من النافذة لفترة طويلة دون أن تقول شيئاً.",
           "مشينا على طول الشاطئ لبعض الوقت نتحدث عن أشياء عادية.",
           "بعد كل ما مررنا به انتهى بنا الأمر في نفس المكان.",
           "كانت الحديقة هادئة تماماً في ذلك الصباح الباكر."]
AR_POOL = ["الطقس صار حاراً جداً في الصيف هذا العام.",
           "ضحك الأطفال بصوت عالٍ في الحديقة الكبيرة.",
           "قرأ الأستاذ القصيدة بصوت واضح أمام الطلاب.",
           "حضرت الاجتماع المهم في الطابق الرابع صباحاً.",
           "عبرنا الطريق السريع بحذر شديد في الظلام.",
           "شربت فنجان القهوة الساخن قبل أن أخرج.",
           "وصف الطبيب الدواء المناسب للحالة الصعبة.",
           "غسلت الملابس ونشرتها تحت أشعة الشمس."]
json.dump(AR_TEST, open("texts.json", "w"))

def gen(texts, lang, g, s, outdir):
    os.makedirs(outdir, exist_ok=True)
    for i, t in enumerate(texts):
        o = m.inference(t, lang, g, s, temperature=0.7, enable_text_splitting=True)
        sf.write(f"{outdir}/{i}.wav", np.asarray(o["wav"]), 24000)

gen(AR_TEST, "ar", gl,   sp,   "out_ar_herQ")   # her XTTS Arabic (synthesis bar + her-ish query)
gen(AR_TEST, "ar", gl_g, sp_g, "out_ar_genQ")   # generic Arabic query
gen(AR_POOL, "ar", gl,   sp,   "out_ar_pool")   # her XTTS Arabic, DIFFERENT content (pool augmentation)
print("GEN_DONE", flush=True)
