#!/bin/bash
set -x; echo "=== KNN1 (rank-2: exemplar retrieval cross-lingual identity gate) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1; echo FFMPEG_DONE
pip install -q "numpy<2" resemblyzer speechbrain coqui-tts "transformers>=4.57,<5.0" torchaudio librosa soundfile 2>&1 | tail -2
export COQUI_TOS_AGREED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python gen_knn1.py
echo "=== KNN-VC CONVERT ==="
python run_knn1.py
echo "=== MEASURE ==="
python measure_knn1.py
echo "=== JOB DONE $(date) ==="
