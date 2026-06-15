# -*- coding: utf-8 -*-
"""Max-contrast bucket test: her top-K vs bottom-K arousal clips → rich EXPR/CALM refs,
generate EN+AR from each. Tests the CEILING of XTTS emotion transfer (SER arousal)."""
import os, glob, json, warnings, numpy as np, librosa, soundfile as sf, torch
warnings.filterwarnings("ignore")
from ser_lib import load_ser

def loud(w):
    r = np.sqrt(np.mean(w**2)) + 1e-9
    return w * (10**(-23/20) / r)

ser = load_ser(); print("SER loaded", flush=True)

y, sr = librosa.load("her_audio.wav", sr=24000, mono=True)
iv = librosa.effects.split(y, top_db=30)
segs = []; buf = []; cur = 0
for s, e in iv:
    buf.append(y[s:e]); cur += e - s
    if cur > 8 * sr:
        segs.append(np.concatenate(buf)); buf = []; cur = 0
if buf and cur > 5 * sr:
    segs.append(np.concatenate(buf))

ar = []
for s in segs:
    w16 = librosa.resample(s, orig_sr=24000, target_sr=16000)
    a, v = ser(loud(w16)); ar.append(a)
ar = np.array(ar); order = np.argsort(ar)
K = 4
calm_idx = order[:K]; expr_idx = order[-K:]
os.makedirs("ref_calm", exist_ok=True); os.makedirs("ref_expr", exist_ok=True)
for k, i in enumerate(calm_idx): sf.write(f"ref_calm/c{k}.wav", loud(segs[i]), 24000)
for k, i in enumerate(expr_idx): sf.write(f"ref_expr/e{k}.wav", loud(segs[i]), 24000)
src = {"calm_arousal": round(float(ar[calm_idx].mean()), 3),
       "expr_arousal": round(float(ar[expr_idx].mean()), 3)}
json.dump(src, open("src2.json", "w"))
print("BUCKETS calm_arousal", src["calm_arousal"], "expr_arousal", src["expr_arousal"],
      "delta", round(src["expr_arousal"] - src["calm_arousal"], 3), flush=True)

from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda" if torch.cuda.is_available() else "cpu")
m = tts.synthesizer.tts_model
EN = [("e1", "We will meet again on Thursday to finish going over everything."),
      ("e2", "Most of the boxes were already packed and waiting by the door.")]
AR = [("a1", "سوف نلتقي مرة أخرى يوم الخميس لإنهاء المراجعة."),
      ("a2", "معظم الصناديق كانت معبأة بالفعل وتنتظر بجانب الباب.")]
for tag, refdir in [("EXPR", "ref_expr"), ("CALM", "ref_calm")]:
    refs = sorted(glob.glob(f"{refdir}/*.wav"))
    gl, sp = m.get_conditioning_latents(audio_path=refs, max_ref_length=30, gpt_cond_len=30, gpt_cond_chunk_len=6)
    for lang, sents in [("en", EN), ("ar", AR)]:
        outdir = f"out_{lang}_{tag.lower()}"; os.makedirs(outdir, exist_ok=True)
        for sid, txt in sents:
            for t in range(3):
                o = m.inference(txt, lang, gl, sp, temperature=0.75, enable_text_splitting=True)
                sf.write(f"{outdir}/{sid}_t{t}.wav", np.asarray(o["wav"]), 24000)
        print("GEN", tag, lang, flush=True)
print("GEN_DONE", flush=True)
