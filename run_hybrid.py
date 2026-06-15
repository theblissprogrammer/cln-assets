# -*- coding: utf-8 -*-
"""HYBRID kNN: match her-expr-Arabic and generic-Arabic queries into her real-frame pool."""
import os, glob, warnings, numpy as np, librosa, soundfile as sf, torch, torchaudio
warnings.filterwarnings("ignore")
dev = "cuda" if torch.cuda.is_available() else "cpu"
knn_vc = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True, device=dev)
print("knn-vc loaded", flush=True)
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
pool = knn_vc.get_matching_set(paths)
print(f"pool frames {tuple(pool.shape)}", flush=True)
def convert(qdir, outdir):
    os.makedirs(outdir, exist_ok=True)
    for p in sorted(glob.glob(f"{qdir}/*.wav")):
        q = knn_vc.get_features(p)
        out = knn_vc.match(q, pool, topk=4)
        torchaudio.save(f"{outdir}/{os.path.basename(p)}", out[None].cpu(), 16000)
    print("ARM", outdir, flush=True)
convert("out_genAr", "out_knn_gen")     # exemplar, neutral emotion (generic query)
convert("out_exprAr", "out_knn_expr")   # HYBRID: her emotion (expr query) + her identity (pool)
print("KNN_DONE", flush=True)
