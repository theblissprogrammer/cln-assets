# -*- coding: utf-8 -*-
"""STEP 0 gen: her XTTS clones in EN + AR, plus a generic (non-her) XTTS Arabic voice.
For the source-drift diagnostic — does her glottal source survive into Arabic?"""
import os, glob, warnings, numpy as np, librosa, soundfile as sf, torch
warnings.filterwarnings("ignore")

def loud(w):
    r = np.sqrt(np.mean(w**2)) + 1e-9
    return w * (10**(-23/20) / r)

# her conditioning refs (4 central-ish clips)
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
print(f"her refs: {len(glob.glob('ref_her/*.wav'))}", flush=True)

from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda" if torch.cuda.is_available() else "cpu")
m = tts.synthesizer.tts_model
refs = sorted(glob.glob("ref_her/*.wav"))
gl_her, sp_her = m.get_conditioning_latents(audio_path=refs, max_ref_length=30, gpt_cond_len=30, gpt_cond_chunk_len=6)

# generic non-her XTTS speaker (built-in) for an Arabic VQ reference point
gen_name = list(m.speaker_manager.speakers.keys())[0]
ginfo = m.speaker_manager.speakers[gen_name]
gl_gen, sp_gen = ginfo["gpt_cond_latent"], ginfo["speaker_embedding"]
print(f"generic speaker: {gen_name}", flush=True)

EN = ["We will meet again on Thursday to finish going over everything.",
      "Most of the boxes were already packed and waiting by the door.",
      "She looked out the window for a long time without saying anything.",
      "They walked along the beach for a while talking about ordinary things.",
      "After everything we went through, we ended up in the same place."]
AR = ["سوف نلتقي مرة أخرى يوم الخميس لإنهاء المراجعة.",
      "معظم الصناديق كانت معبأة بالفعل وتنتظر بجانب الباب.",
      "نظرت من النافذة لفترة طويلة دون أن تقول شيئاً.",
      "مشينا على طول الشاطئ لبعض الوقت نتحدث عن أشياء عادية.",
      "بعد كل ما مررنا به، انتهى بنا الأمر في نفس المكان."]

def gen(texts, lang, gl, sp, outdir):
    os.makedirs(outdir, exist_ok=True)
    for i, txt in enumerate(texts):
        o = m.inference(txt, lang, gl, sp, temperature=0.7, enable_text_splitting=True)
        sf.write(f"{outdir}/{i}.wav", np.asarray(o["wav"]), 24000)
    print("GEN", outdir, flush=True)

gen(EN, "en", gl_her, sp_her, "out_en")        # her, English  (same-lang control)
gen(AR, "ar", gl_her, sp_her, "out_ar")        # her, Arabic    (the question)
gen(AR, "ar", gl_gen, sp_gen, "out_ar_gen")    # generic, Arabic (non-her VQ reference)
print("GEN_DONE", flush=True)
