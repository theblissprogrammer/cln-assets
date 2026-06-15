# -*- coding: utf-8 -*-
"""Cross-speaker dub: donor (different speaker, high-arousal) drives GPT-latent (emotion),
her clips drive speaker-emb (identity). DUB = gpt_donor + spk_her. Controls HER, DONOR."""
import os, glob, warnings, numpy as np, librosa, soundfile as sf, torch
warnings.filterwarnings("ignore")
from ser_lib import load_ser

def loud(w):
    r = np.sqrt(np.mean(w**2)) + 1e-9
    return w * (10**(-23/20) / r)

ser = load_ser(); print("SER loaded", flush=True)

# --- donor: concat CREMA-D angry clips (same actor = coherent donor identity) ---
dfiles = sorted(glob.glob("donor_raw/*.wav"))
assert len(dfiles) >= 4, f"need >=4 donor clips, got {len(dfiles)}"
dws = []
for f in dfiles:
    w, _ = librosa.load(f, sr=24000, mono=True); dws.append(w)
donor = np.concatenate([np.concatenate([w, np.zeros(int(0.15 * 24000))]) for w in dws])
os.makedirs("ref_donor", exist_ok=True)
sf.write("ref_donor/d0.wav", loud(donor), 24000)
da, dv = ser(loud(librosa.resample(donor, orig_sr=24000, target_sr=16000)))
print(f"DONOR clips={len(dfiles)} dur={len(donor)/24000:.1f}s arousal={da:.3f}", flush=True)

# --- her identity refs (calm bucket, so her own emotion doesn't compete with donor's) ---
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
os.makedirs("ref_her", exist_ok=True)
for k, i in enumerate(order[:4]): sf.write(f"ref_her/h{k}.wav", loud(segs[i]), 24000)
print(f"HER calm refs=4 arousal~{float(ar[order[:4]].mean()):.3f}", flush=True)

from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda" if torch.cuda.is_available() else "cpu")
m = tts.synthesizer.tts_model
def latents(refdir):
    refs = sorted(glob.glob(f"{refdir}/*.wav"))
    return m.get_conditioning_latents(audio_path=refs, max_ref_length=30, gpt_cond_len=30, gpt_cond_chunk_len=6)
gpt_d, spk_d = latents("ref_donor")
gpt_h, spk_h = latents("ref_her")
EN = [("e1", "We will meet again on Thursday to finish going over everything."),
      ("e2", "Most of the boxes were already packed and waiting by the door.")]
AR = [("a1", "سوف نلتقي مرة أخرى يوم الخميس لإنهاء المراجعة."),
      ("a2", "معظم الصناديق كانت معبأة بالفعل وتنتظر بجانب الباب.")]
configs = [("HER", gpt_h, spk_h), ("DONOR", gpt_d, spk_d), ("DUB", gpt_d, spk_h)]
for tag, gl, sp in configs:
    for lang, sents in [("en", EN), ("ar", AR)]:
        outdir = f"out_{lang}_{tag}"; os.makedirs(outdir, exist_ok=True)
        for sid, txt in sents:
            for t in range(3):
                o = m.inference(txt, lang, gl, sp, temperature=0.75, enable_text_splitting=True)
                sf.write(f"{outdir}/{sid}_t{t}.wav", np.asarray(o["wav"]), 24000)
    print("GEN", tag, flush=True)
print("GEN_DONE", flush=True)
