# -*- coding: utf-8 -*-
"""Transparent 24kHz exemplar render: match query in WavLM space against her real frames,
retrieve the corresponding HER REAL 24kHz Vocos-mel columns (topk-avg), Vocos-decode.
No training; her real full-band spectral content; transparent renderer."""
import os, glob, warnings, numpy as np, torch, librosa, soundfile as sf
import torch.nn.functional as F
warnings.filterwarnings("ignore")
dev = "cuda" if torch.cuda.is_available() else "cpu"
knn = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True, device=dev)
from vocos import Vocos
vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(dev)
print("loaded knn + vocos", flush=True)

MEL_RATE = 24000 / 256.0   # 93.75 Hz (vocos-mel-24khz hop=256)
WLM_RATE = 50.0            # WavLM L6 frame rate

def feats(path):  # WavLM features, NO vad trim (keep linear time alignment)
    return knn.get_features(path, vad_trigger_level=0).to(dev)
def mel_of(wav24):
    x = torch.tensor(wav24, dtype=torch.float32)[None].to(dev)
    return vocos.feature_extractor(x).squeeze(0)  # [100, M]

# build her pool: WavLM frames + aligned real mel columns
y, _ = librosa.load("her_audio.wav", sr=24000, mono=True)
iv = librosa.effects.split(y, top_db=30)
os.makedirs("her24", exist_ok=True)
chunks = []; buf = []; cur = 0
for s, e in iv:
    buf.append(y[s:e]); cur += e - s
    if cur > 10*24000:
        chunks.append(np.concatenate(buf)); buf = []; cur = 0
if buf:
    chunks.append(np.concatenate(buf))
POOL_W = []; POOL_MEL = []
for i, c in enumerate(chunks):
    p = f"her24/{i}.wav"; sf.write(p, c, 24000)
    mel = mel_of(c)                          # [100, M]
    fw = feats(p)                            # [T, 1024]
    T = fw.shape[0]; M = mel.shape[1]
    cols = np.clip((np.arange(T) / WLM_RATE * MEL_RATE).round().astype(int), 0, M-1)
    POOL_W.append(fw); POOL_MEL.append(mel[:, cols].T)   # [T, 100]
POOL_W = torch.cat(POOL_W, 0); POOL_MEL = torch.cat(POOL_MEL, 0)
print(f"pool W {tuple(POOL_W.shape)} MEL {tuple(POOL_MEL.shape)}", flush=True)
PWn = POOL_W / POOL_W.norm(dim=1, keepdim=True)

def render(qpath, outpath, topk=4):
    q = feats(qpath)
    qn = q / q.norm(dim=1, keepdim=True)
    sims = qn @ PWn.T
    idx = sims.topk(topk, dim=1).indices          # [Q, topk]
    mmel = POOL_MEL[idx].mean(1)                   # [Q, 100] @ 50Hz
    mel = mmel.T[None]                             # [1, 100, Q]
    newT = max(1, int(round(mmel.shape[0] * MEL_RATE / WLM_RATE)))
    mel = F.interpolate(mel, size=newT, mode="linear", align_corners=False)
    with torch.no_grad():
        wav = vocos.decode(mel).squeeze().cpu().numpy()
    sf.write(outpath, wav, 24000)

os.makedirs("out_vocos", exist_ok=True)
for p in sorted(glob.glob("out_exprAr/*.wav")):
    render(p, f"out_vocos/{os.path.basename(p)}")
    print("VOCOS", os.path.basename(p), flush=True)
print("RENDER_DONE", flush=True)
