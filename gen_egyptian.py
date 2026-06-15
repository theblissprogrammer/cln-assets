# -*- coding: utf-8 -*-
"""Product path: her voice (expressive conditioning) + EGYPTIAN-dialect text via XTTS."""
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
    if cur > 8*sr:
        segs.append(np.concatenate(buf)); buf = []; cur = 0
if buf and cur > 5*sr: segs.append(np.concatenate(buf))
rng = []
for s in segs:
    w16 = librosa.resample(s, orig_sr=24000, target_sr=16000)
    f0, _, _ = librosa.pyin(w16, fmin=80, fmax=400, sr=16000); f0v = f0[~np.isnan(f0)]
    rng.append(12*np.log2((np.percentile(f0v,95)+1e-9)/(np.percentile(f0v,5)+1e-9)) if len(f0v) > 5 else 0)
order = np.argsort(rng)
os.makedirs("ref_expr", exist_ok=True)
for k, i in enumerate(order[::-1][:4]):
    sf.write(f"ref_expr/e{k}.wav", loud(segs[i]), 24000)
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda" if torch.cuda.is_available() else "cpu")
m = tts.synthesizer.tts_model
gl, sp = m.get_conditioning_latents(audio_path=sorted(glob.glob("ref_expr/*.wav")), max_ref_length=30, gpt_cond_len=30, gpt_cond_chunk_len=6)
# Egyptian colloquial Arabic
EGY = ["هنتقابل تاني يوم الخميس عشان نخلّص المراجعة.",
       "معظم الكراتين كانت متجهّزة خلاص ومستنية جنب الباب.",
       "بصّت من الشباك مدة طويلة من غير ما تقول أي حاجة.",
       "مشينا على طول البحر شوية وإحنا بنتكلم في حاجات عادية.",
       "بعد كل اللي عدّى علينا، رجعنا تاني لنفس المكان.",
       "كانت الجنينة هادية خالص الصبح بدري."]
json.dump(EGY, open("texts.json", "w"))
os.makedirs("out_egy", exist_ok=True)
for i, t in enumerate(EGY):
    o = m.inference(t, "ar", gl, sp, temperature=0.75, enable_text_splitting=True)
    sf.write(f"out_egy/{i}.wav", np.asarray(o["wav"]), 24000)
print("GEN_DONE", flush=True)
