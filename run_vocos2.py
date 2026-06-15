# -*- coding: utf-8 -*-
"""Debug the Vocos-render STATIC: it's frame-to-frame mel discontinuity from concatenation.
Sweep topk (more averaging) x temporal smoothing (kill jumps) to find clean + identity-preserving."""
import os, glob, warnings, numpy as np, torch, librosa, soundfile as sf
import torch.nn.functional as F
warnings.filterwarnings("ignore")
try:
    from scipy.ndimage import gaussian_filter1d
except Exception:
    def gaussian_filter1d(a, sigma, axis=0):
        if sigma <= 0: return a
        r = int(3*sigma); x = np.arange(-r, r+1); k = np.exp(-(x**2)/(2*sigma**2)); k /= k.sum()
        return np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), axis, a)

dev = "cuda" if torch.cuda.is_available() else "cpu"
knn = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True, device=dev)
from vocos import Vocos
vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(dev)
MEL_RATE = 24000/256.0; WLM_RATE = 50.0
def feats(p): return knn.get_features(p, vad_trigger_level=0).to(dev)
def mel_of(w): return vocos.feature_extractor(torch.tensor(w, dtype=torch.float32)[None].to(dev)).squeeze(0)

y, _ = librosa.load("her_audio.wav", sr=24000, mono=True)
iv = librosa.effects.split(y, top_db=30)
os.makedirs("her24", exist_ok=True)
chunks = []; buf = []; cur = 0
for s, e in iv:
    buf.append(y[s:e]); cur += e - s
    if cur > 10*24000:
        chunks.append(np.concatenate(buf)); buf = []; cur = 0
if buf: chunks.append(np.concatenate(buf))
POOL_W = []; POOL_MEL = []
for i, c in enumerate(chunks):
    p = f"her24/{i}.wav"; sf.write(p, c, 24000)
    mel = mel_of(c); fw = feats(p); T = fw.shape[0]; M = mel.shape[1]
    cols = np.clip((np.arange(T)/WLM_RATE*MEL_RATE).round().astype(int), 0, M-1)
    POOL_W.append(fw); POOL_MEL.append(mel[:, cols].T)
POOL_W = torch.cat(POOL_W, 0); POOL_MEL = torch.cat(POOL_MEL, 0)
PWn = POOL_W / POOL_W.norm(dim=1, keepdim=True)
print(f"pool {tuple(POOL_W.shape)}", flush=True)

def render(qpath, topk, sigma, outpath):
    q = feats(qpath); qn = q / q.norm(dim=1, keepdim=True)
    idx = (qn @ PWn.T).topk(topk, dim=1).indices
    mmel = POOL_MEL[idx].mean(1).cpu().numpy()          # [Q,100] @50Hz
    if sigma > 0: mmel = gaussian_filter1d(mmel, sigma, axis=0)
    mel = torch.tensor(mmel, device=dev, dtype=torch.float32).T[None]
    newT = max(1, int(round(mmel.shape[0]*MEL_RATE/WLM_RATE)))
    mel = F.interpolate(mel, size=newT, mode="linear", align_corners=False)
    with torch.no_grad():
        wav = vocos.decode(mel).squeeze().cpu().numpy()
    sf.write(outpath, wav, 24000)

queries = sorted(glob.glob("out_exprAr/*.wav"))
GRID = [(4, 0.0), (4, 1.5), (10, 0.0), (10, 1.5), (10, 3.0), (20, 2.0)]
for tk, sg in GRID:
    d = f"out_v_{tk}_{str(sg).replace('.','p')}"; os.makedirs(d, exist_ok=True)
    for qp in queries:
        render(qp, tk, sg, f"{d}/{os.path.basename(qp)}")
    print("RENDER", tk, sg, "->", d, flush=True)
print("RENDER_DONE", flush=True)
