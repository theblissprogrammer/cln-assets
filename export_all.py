# -*- coding: utf-8 -*-
"""Export audio for Ahmed's ear: her real voice, XTTS-synthesis Arabic, kNN-exemplar Arabic.
mp3-compress + base64 to stdout (SSH dead, tmpfiles blocks datacenter IPs)."""
import os, glob, base64, subprocess, warnings, numpy as np, librosa, soundfile as sf, torch
warnings.filterwarnings("ignore")
def loud(w):
    r = np.sqrt(np.mean(w**2)) + 1e-9
    return w * (10**(-23/20) / r)

y, sr = librosa.load("her_audio.wav", sr=24000, mono=True)
iv = librosa.effects.split(y, top_db=30)
seg = max([y[s:e] for s, e in iv], key=len)[:5*24000]
sf.write("clip_her_real.wav", loud(seg), 24000)
segs = []; buf = []; cur = 0
for s, e in iv:
    buf.append(y[s:e]); cur += e - s
    if cur > 10*24000:
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
AR = ["سوف نلتقي مرة أخرى يوم الخميس لإنهاء المراجعة."]
for i, t in enumerate(AR):
    o = m.inference(t, "ar", gl, sp, temperature=0.7, enable_text_splitting=True); sf.write(f"synth_{i}.wav", np.asarray(o["wav"]), 24000)
    o = m.inference(t, "ar", gl_g, sp_g, temperature=0.7, enable_text_splitting=True); sf.write(f"genq_{i}.wav", np.asarray(o["wav"]), 24000)
del tts, m; torch.cuda.empty_cache()

import torchaudio
knn = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True, device="cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("her_chunks", exist_ok=True)
y16, _ = librosa.load("her_audio.wav", sr=16000, mono=True); iv16 = librosa.effects.split(y16, top_db=30)
chs = []; buf = []; cur = 0
for s, e in iv16:
    buf.append(y16[s:e]); cur += e - s
    if cur > 10*16000:
        chs.append(np.concatenate(buf)); buf = []; cur = 0
if buf:
    chs.append(np.concatenate(buf))
paths = []
for i, c in enumerate(chs):
    p = f"her_chunks/{i}.wav"; sf.write(p, c, 16000); paths.append(p)
pool = knn.get_matching_set(paths)
for i in range(len(AR)):
    q = knn.get_features(f"genq_{i}.wav"); out = knn.match(q, pool, topk=4)
    torchaudio.save(f"exemplar_{i}.wav", out[None].cpu(), 16000)

def upload_litterbox(path):
    try:
        r = subprocess.run(["curl", "-s", "--max-time", "60", "-F", "reqtype=fileupload", "-F", "time=72h",
                            "-F", f"fileToUpload=@{path}", "https://litterbox.catbox.moe/resources/internals/api.php"],
                           capture_output=True, text=True, timeout=70)
        u = r.stdout.strip()
        return u if u.startswith("http") else None
    except Exception:
        return None

def export(wav, name):
    mp3 = name + ".mp3"
    # trim to <=4s, mono 22k 40kbps (small enough for base64-fallback to fit a log window)
    subprocess.run(["ffmpeg", "-y", "-t", "4.2", "-i", wav, "-ar", "22050", "-ac", "1", "-b:a", "40k", mp3], capture_output=True)
    url = upload_litterbox(mp3)
    if url:
        print(f"URL {name} {url}", flush=True)
        return
    # fallback: chunked base64 (200-char lines survive log truncation; small clips keep total fetchable)
    b = base64.b64encode(open(mp3, "rb").read()).decode()
    print(f"B64META {name} {len(b)}", flush=True)
    for i in range(0, len(b), 200):
        print(f"B64 {name} {i//200} {b[i:i+200]}", flush=True)

export("clip_her_real.wav", "her_real")
for i in range(len(AR)):
    export(f"synth_{i}.wav", f"synth_{i}")
    export(f"exemplar_{i}.wav", f"exemplar_{i}")
print("EXPORT_DONE", flush=True)
