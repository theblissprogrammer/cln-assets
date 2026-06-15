# -*- coding: utf-8 -*-
"""rank-2: kNN-VC convert generic-Arabic queries into HER real frames (exemplar retrieval).
Output = stitched from her real WavLM frames -> should keep her fine identity (ECAPA) cross-lingually."""
import os, glob, warnings, torch, torchaudio
warnings.filterwarnings("ignore")
dev = "cuda" if torch.cuda.is_available() else "cpu"
knn_vc = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True, device=dev)
print("knn-vc loaded", flush=True)

# her matching set = her real frames (the uncompressed exemplar pool)
matching_set = knn_vc.get_matching_set(["her_audio.wav"])
print(f"her matching set frames: {tuple(matching_set.shape)}", flush=True)

os.makedirs("out_knn_ar", exist_ok=True)
for p in sorted(glob.glob("out_ar_src/*.wav")):
    q = knn_vc.get_features(p)
    out = knn_vc.match(q, matching_set, topk=4)   # 16kHz waveform tensor
    torchaudio.save(f"out_knn_ar/{os.path.basename(p)}", out[None].cpu(), 16000)
    print("KNN_AR", os.path.basename(p), flush=True)
print("KNN_DONE", flush=True)
