# -*- coding: utf-8 -*-
"""HYBRID gen: her-EXPRESSIVE-conditioned XTTS Arabic (carries her emotion) + generic XTTS Arabic.
These become kNN queries; her real frames are the pool."""
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
    if cur > 8 * sr:
        segs.append(np.concatenate(buf)); buf = []; cur = 0
if buf and cur > 5 * sr:
    segs.append(np.concatenate(buf))
# pick most EXPRESSIVE segments by pitch range (her emotion)
rng = []
for s in segs:
    w16 = librosa.resample(s, orig_sr=24000, target_sr=16000)
    f0, _, _ = librosa.pyin(w16, fmin=80, fmax=400, sr=16000); f0v = f0[~np.isnan(f0)]
    rng.append(12*np.log2((np.percentile(f0v, 95)+1e-9)/(np.percentile(f0v, 5)+1e-9)) if len(f0v) > 5 else 0)
order = np.argsort(rng)
os.makedirs("ref_expr", exist_ok=True)
for k, i in enumerate(order[::-1][:4]):
    sf.write(f"ref_expr/e{k}.wav", loud(segs[i]), 24000)
print(f"EXPR ref f0-ranges: {[round(rng[i],1) for i in order[::-1][:4]]}", flush=True)

from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda" if torch.cuda.is_available() else "cpu")
m = tts.synthesizer.tts_model
gl_e, sp_e = m.get_conditioning_latents(audio_path=sorted(glob.glob("ref_expr/*.wav")), max_ref_length=30, gpt_cond_len=30, gpt_cond_chunk_len=6)
gname = list(m.speaker_manager.speakers.keys())[0]; gi = m.speaker_manager.speakers[gname]
gl_g, sp_g = gi["gpt_cond_latent"], gi["speaker_embedding"]

AR = ["سوف نلتقي مرة أخرى يوم الخميس لإنهاء المراجعة.",
      "معظم الصناديق كانت معبأة بالفعل وتنتظر بجانب الباب.",
      "نظرت من النافذة لفترة طويلة دون أن تقول شيئاً.",
      "مشينا على طول الشاطئ لبعض الوقت نتحدث عن أشياء عادية.",
      "بعد كل ما مررنا به انتهى بنا الأمر في نفس المكان.",
      "كانت الحديقة هادئة تماماً في ذلك الصباح الباكر."]
json.dump(AR, open("texts.json", "w"))
def gen(g, s, outdir):
    os.makedirs(outdir, exist_ok=True)
    for i, t in enumerate(AR):
        o = m.inference(t, "ar", g, s, temperature=0.75, enable_text_splitting=True)
        sf.write(f"{outdir}/{i}.wav", np.asarray(o["wav"]), 24000)
gen(gl_e, sp_e, "out_exprAr")   # her emotion (expressive conditioning)
gen(gl_g, sp_g, "out_genAr")    # generic neutral
print("GEN_DONE", flush=True)
