#!/bin/bash
set -x; echo "=== IMPOSTOR CALIB (definitive is-it-her) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1; echo FFMPEG_DONE
pip install -q "numpy<2" resemblyzer torchaudio librosa soundfile scikit-learn 2>&1 | tail -1
python impostor_calib.py
echo "=== JOB DONE $(date) ==="
