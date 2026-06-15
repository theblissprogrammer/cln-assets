#!/bin/bash
set -x; echo "=== CHATTERBOX robustness demo JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg curl >/dev/null 2>&1; echo FFMPEG_DONE
conda create -y -n cb python=3.11 2>&1 | tail -1
conda run -n cb pip install -q chatterbox-tts resemble-perth soundfile librosa "numpy<2" 2>&1 | tail -2
echo "CB_DEPS_DONE"
conda run -n cb python gen_cb_demo.py
echo "=== MEASURE ==="
pip install -q "numpy<2" resemblyzer librosa soundfile 2>&1 | tail -1
python measure_cb_demo.py
echo "=== JOB DONE $(date) ==="
