# -*- coding: utf-8 -*-
"""metric-test kNN: PROPER (nearest her frames) vs RANDOM (random her frames=gibberish-but-her)."""
import os, glob, warnings, numpy as np, librosa, soundfile as sf, torch, torchaudio
warnings.filterwarnings("ignore")
dev = "cuda" if torch.cuda.is_available() else "cpu"
knn = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True, device=dev)
y, sr = librosa.load("her_audio.wav", sr=16000, mono=True)
iv = librosa.effects.split(y, top_db=30)
os.makedirs("her_chunks", exist_ok=True)
chunks = []; buf = []; cur = 0
for s, e in iv:
    buf.append(y[s:e]); cur += e - s
    if cur > 10*sr:
        chunks.append(np.concatenate(buf)); buf = []; cur = 0
if buf:
    chunks.append(np.concatenate(buf))
paths = []
for i, c in enumerate(chunks):
    p = f"her_chunks/{i}.wav"; sf.write(p, c, sr); paths.append(p)
pool = knn.get_matching_set(paths).to(dev)
voc = knn.hifigan
P = pool.shape[0]
os.makedirs("out_proper", exist_ok=True); os.makedirs("out_random", exist_ok=True)
torch.manual_seed(0)
for p in sorted(glob.glob("out_src/*.wav")):
    n = os.path.basename(p)
    q = knn.get_features(p).to(dev)
    out = knn.match(q, pool, topk=4); torchaudio.save(f"out_proper/{n}", out[None].cpu(), 16000)
    idx = torch.randint(0, P, (q.shape[0], 4), device=dev)
    rf = pool[idx].mean(1)
    with torch.no_grad():
        rw = voc(rf[None]).squeeze().cpu()
    torchaudio.save(f"out_random/{n}", rw[None], 16000)
    print("MT", n, flush=True)
print("MT_DONE", flush=True)
