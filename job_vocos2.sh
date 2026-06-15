#!/bin/bash
set -x; echo "=== VOCOS2 (debug static: smoothing/topk sweep) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg curl >/dev/null 2>&1; echo FFMPEG_DONE
pip install -q "numpy<2" resemblyzer speechbrain coqui-tts "transformers>=4.57,<5.0" torchaudio librosa soundfile vocos scipy 2>&1 | tail -1
export COQUI_TOS_AGREED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python gen_alpha.py
echo "=== VOCOS RENDER SWEEP ==="
python run_vocos2.py
echo "=== MEASURE ==="
python measure_vocos2.py
echo "=== JOB DONE $(date) ==="
