# -*- coding: utf-8 -*-
"""alpha-sweep: blend query WavLM features with her matched-frame features at alpha in [0,1],
vocode each via kNN-VC's prematched HiFiGAN. alpha=0 -> fluent synthesis end, alpha=1 -> her-exemplar end."""
import os, glob, warnings, numpy as np, torch, torchaudio
warnings.filterwarnings("ignore")
dev = "cuda" if torch.cuda.is_available() else "cpu"
knn_vc = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True, device=dev)
print("knn-vc loaded; attrs:", [a for a in dir(knn_vc) if not a.startswith("_")], flush=True)

import librosa, soundfile as sf
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
pool = knn_vc.get_matching_set(paths).to(dev)
print(f"pool {tuple(pool.shape)}", flush=True)

# vocoder handle
voc = getattr(knn_vc, "vocoder", None) or getattr(knn_vc, "hifigan", None)
print("vocoder attr found:", voc is not None, flush=True)

def matched_feats(q):
    qn = q / q.norm(dim=1, keepdim=True); pn = pool / pool.norm(dim=1, keepdim=True)
    sims = qn @ pn.T                       # [Q, P] cosine
    idx = sims.topk(4, dim=1).indices      # [Q, 4]
    return pool[idx].mean(1)               # [Q, D]

def vocode(feats):
    with torch.no_grad():
        out = voc(feats[None].to(dev)).squeeze().cpu()
    return out

ALPHAS = [0.0, 0.33, 0.5, 0.67, 1.0]
for a in ALPHAS:
    os.makedirs(f"out_a{int(a*100)}", exist_ok=True)
for p in sorted(glob.glob("out_exprAr/*.wav")):
    name = os.path.basename(p)
    q = knn_vc.get_features(p).to(dev)     # [T, D]
    m = matched_feats(q)
    for a in ALPHAS:
        blend = a * m + (1 - a) * q
        wav = vocode(blend)
        torchaudio.save(f"out_a{int(a*100)}/{name}", wav[None], 16000)
    print("ALPHASWEEP", name, flush=True)
print("KNN_DONE", flush=True)
