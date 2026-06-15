# -*- coding: utf-8 -*-
"""Pick her source clips spanning an arousal range, single-clip-condition XTTS,
generate fixed EN + AR sentences from each. Records source SER for correlation."""
import os, json, warnings, numpy as np, librosa, soundfile as sf, torch
warnings.filterwarnings("ignore")
from ser_lib import load_ser

def loud(w):
    r = np.sqrt(np.mean(w**2)) + 1e-9
    return w * (10**(-23/20) / r)

ser = load_ser(); print("SER loaded", flush=True)

# --- split her into ~>=10s segments (need length for stable conditioning + SER) ---
y, sr = librosa.load("her_audio.wav", sr=24000, mono=True)
iv = librosa.effects.split(y, top_db=30)
segs = []; buf = []; cur = 0
for s, e in iv:
    buf.append(y[s:e]); cur += e - s
    if cur > 10 * sr:
        segs.append(np.concatenate(buf)); buf = []; cur = 0
if buf and cur > 6 * sr:
    segs.append(np.concatenate(buf))
print(f"SEG count={len(segs)}", flush=True)

# --- SER each segment, pick N spanning arousal range ---
ar = []
for s in segs:
    w16 = librosa.resample(s, orig_sr=24000, target_sr=16000)
    a, v = ser(loud(w16)); ar.append(a)
ar = np.array(ar); order = np.argsort(ar)
N = 6
idx = [int(order[int(round(p * (len(order) - 1)))]) for p in np.linspace(0, 1, N)]

os.makedirs("refs", exist_ok=True); src = []
for k, i in enumerate(idx):
    sf.write(f"refs/r{k}.wav", loud(segs[i]), 24000)
    w16 = librosa.resample(segs[i], orig_sr=24000, target_sr=16000)
    a, v = ser(loud(w16))
    src.append({"ref": k, "arousal": round(a, 3), "valence": round(v, 3),
                "dur": round(len(segs[i]) / 24000, 1)})
json.dump(src, open("source_ser.json", "w"))
print("SRC_AROUSAL", [d["arousal"] for d in src], flush=True)
print("SRC_VALENCE", [d["valence"] for d in src], flush=True)

# --- generate EN + AR from each ref (single-clip conditioning) ---
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda" if torch.cuda.is_available() else "cpu")
m = tts.synthesizer.tts_model
EN = [("e1", "We will meet again on Thursday to finish going over everything."),
      ("e2", "Most of the boxes were already packed and waiting by the door.")]
AR = [("a1", "سوف نلتقي مرة أخرى يوم الخميس لإنهاء المراجعة."),
      ("a2", "معظم الصناديق كانت معبأة بالفعل وتنتظر بجانب الباب.")]
for lang, sents, outdir in [("en", EN, "out_en"), ("ar", AR, "out_ar")]:
    os.makedirs(outdir, exist_ok=True)
    for k in range(N):
        gl, sp = m.get_conditioning_latents(audio_path=[f"refs/r{k}.wav"],
                                            max_ref_length=30, gpt_cond_len=30, gpt_cond_chunk_len=6)
        for sid, txt in sents:
            for t in range(2):
                o = m.inference(txt, lang, gl, sp, temperature=0.75, enable_text_splitting=True)
                sf.write(f"{outdir}/r{k}_{sid}_t{t}.wav", np.asarray(o["wav"]), 24000)
        print("GEN", lang, "ref", k, flush=True)
print("GEN_DONE", flush=True)
