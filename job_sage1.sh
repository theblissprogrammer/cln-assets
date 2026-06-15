#!/bin/bash
set -x; echo "=== SAGE1 (STEP 1: frozen-source vs F0-transplant, ECAPA gate) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1; echo FFMPEG_DONE
pip install -q "numpy<2" resemblyzer speechbrain coqui-tts "transformers>=4.57,<5.0" torchaudio librosa soundfile praat-parselmouth pyworld 2>&1 | tail -2
export COQUI_TOS_AGREED=1
python gen_sage1.py
echo "=== ARMS (WORLD source manipulation) ==="
python arms_sage1.py
echo "=== MEASURE ==="
python measure_sage1.py
echo "=== JOB DONE $(date) ==="
