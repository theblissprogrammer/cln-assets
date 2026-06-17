#!/bin/bash
set -x; echo "=== IDIOLECT DEMO START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg curl >/dev/null 2>&1; echo FFMPEG_DONE
conda create -y -n cb python=3.11 2>&1 | tail -1
conda run -n cb pip install -q chatterbox-tts resemble-perth yt-dlp soundfile librosa "numpy<2" 2>&1 | tail -1
echo "CB_DEPS_DONE"
conda run -n cb python gen_idiolect_demo.py
echo "=== JOB DONE $(date) ==="
