#!/bin/bash
set -x; echo "=== HYBRID (her identity[exemplar] + her emotion[expr query] + Arabic) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1; echo FFMPEG_DONE
pip install -q "numpy<2" resemblyzer speechbrain coqui-tts "transformers>=4.57,<5.0" torchaudio librosa soundfile openai-whisper jiwer 2>&1 | tail -1
export COQUI_TOS_AGREED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python gen_hybrid.py
echo "=== KNN ==="
python run_hybrid.py
echo "=== MEASURE ==="
python measure_hybrid.py
echo "=== JOB DONE $(date) ==="
