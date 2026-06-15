# -*- coding: utf-8 -*-
"""rank-2 step2 kNN: POOL_EN (her real EN frames) vs POOL_AUG (+ her synthetic-Arabic frames).
Arms: A1 genQ->EN (baseline), A2 genQ->AUG (phoneme coverage), A3 herQ->AUG (fluent query+coverage)."""
import os, glob, warnings, numpy as np, librosa, soundfile as sf, torch, torchaudio
warnings.filterwarnings("ignore")
dev = "cuda" if torch.cuda.is_available() else "cpu"
knn_vc = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True, device=dev)
print("knn-vc loaded", flush=True)

# her real EN frames, chunked
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
en_paths = []
for i, c in enumerate(chunks):
    p = f"her_chunks/{i}.wav"; sf.write(p, c, sr); en_paths.append(p)
ar_pool_paths = sorted(glob.glob("out_ar_pool/*.wav"))   # her synthetic Arabic (disjoint content)
print(f"POOL_EN chunks={len(en_paths)}  AR aug clips={len(ar_pool_paths)}", flush=True)

POOL_EN = knn_vc.get_matching_set(en_paths)
POOL_AUG = knn_vc.get_matching_set(en_paths + ar_pool_paths)
print(f"POOL_EN frames={tuple(POOL_EN.shape)}  POOL_AUG frames={tuple(POOL_AUG.shape)}", flush=True)

def convert(query_dir, pool, outdir):
    os.makedirs(outdir, exist_ok=True)
    for p in sorted(glob.glob(f"{query_dir}/*.wav")):
        q = knn_vc.get_features(p)
        out = knn_vc.match(q, pool, topk=4)
        torchaudio.save(f"{outdir}/{os.path.basename(p)}", out[None].cpu(), 16000)
    print("ARM", outdir, flush=True)

convert("out_ar_genQ", POOL_EN,  "out_A1")   # generic query, EN-only pool (baseline ~0.569)
convert("out_ar_genQ", POOL_AUG, "out_A2")   # generic query, augmented pool
convert("out_ar_herQ", POOL_AUG, "out_A3")   # her-ish query, augmented pool
print("KNN_DONE", flush=True)
