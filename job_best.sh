#!/bin/bash
set -x; echo "=== CB FINAL BEST-RECIPE (delivery + MSA/Egyptian) START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg curl >/dev/null 2>&1; echo FFMPEG_DONE
conda create -y -n cb python=3.11 2>&1 | tail -1
conda run -n cb pip install -q chatterbox-tts resemble-perth soundfile librosa praat-parselmouth "numpy<2" 2>&1 | tail -1
echo "CB_DEPS_DONE"
conda run -n cb python gen_best.py
echo "=== MEASURE (base env) ==="
pip install -q "numpy<2" resemblyzer openai-whisper jiwer librosa soundfile praat-parselmouth 2>&1 | tail -1
python measure_best.py
echo "=== JOB DONE $(date) ==="
