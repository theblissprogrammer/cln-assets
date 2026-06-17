#!/bin/bash
set -x; echo "=== SPK2 GROUND-TRUTH ITERATION START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg curl build-essential >/dev/null 2>&1; echo FFMPEG_DONE
conda create -y -n cb python=3.11 2>&1 | tail -1
conda run -n cb pip install -q chatterbox-tts resemble-perth resemblyzer yt-dlp soundfile librosa praat-parselmouth "setuptools<81" "numpy<2" 2>&1 | tail -1
echo "CB_DEPS_DONE"
conda run -n cb python gen_s2_v2.py
echo "=== JOB DONE $(date) ==="
