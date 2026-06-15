# -*- coding: utf-8 -*-
"""rank-2: kNN-VC convert generic-Arabic queries into HER real frames (exemplar retrieval).
Output = stitched from her real WavLM frames -> should keep her fine identity (ECAPA) cross-lingually."""
import os, glob, warnings, numpy as np, librosa, soundfile as sf, torch, torchaudio
warnings.filterwarnings("ignore")
dev = "cuda" if torch.cuda.is_available() else "cpu"
knn_vc = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True, device=dev)
print("knn-vc loaded", flush=True)

# chunk her audio (~10s) BEFORE WavLM — full 329s in one pass OOMs (O(seq^2) attention bias)
y, sr = librosa.load("her_audio.wav", sr=16000, mono=True)
iv = librosa.effects.split(y, top_db=30)
os.makedirs("her_chunks", exist_ok=True)
chunks = []; buf = []; cur = 0
for s, e in iv:
    buf.append(y[s:e]); cur += e - s
    if cur > 10 * sr:
        chunks.append(np.concatenate(buf)); buf = []; cur = 0
if buf:
    chunks.append(np.concatenate(buf))
paths = []
for i, c in enumerate(chunks):
    p = f"her_chunks/{i}.wav"; sf.write(p, c, sr); paths.append(p)
print(f"her chunks: {len(paths)}", flush=True)
matching_set = knn_vc.get_matching_set(paths)
print(f"her matching set frames: {tuple(matching_set.shape)}", flush=True)

os.makedirs("out_knn_ar", exist_ok=True)
for p in sorted(glob.glob("out_ar_src/*.wav")):
    q = knn_vc.get_features(p)
    out = knn_vc.match(q, matching_set, topk=4)   # 16kHz waveform tensor
    torchaudio.save(f"out_knn_ar/{os.path.basename(p)}", out[None].cpu(), 16000)
    print("KNN_AR", os.path.basename(p), flush=True)
print("KNN_DONE", flush=True)
