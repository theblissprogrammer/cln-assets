#!/bin/bash
set -x; echo "=== STAGE0 RIGOR (impostor control + ECAPA) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1; echo FFMPEG_DONE
pip install -q "numpy<2" resemblyzer speechbrain torchaudio librosa soundfile scipy 2>&1 | tail -1
python stage0_rigor.py
echo "=== JOB DONE $(date) ==="
