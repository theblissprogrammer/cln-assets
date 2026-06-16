#!/bin/bash
set -x; echo "=== DELIVERY-ADAPTER BUILD (patch+train+gate) START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg curl >/dev/null 2>&1; echo FFMPEG_DONE
conda create -y -n cb python=3.11 2>&1 | tail -1
conda run -n cb pip install -q chatterbox-tts resemble-perth soundfile librosa praat-parselmouth "numpy<2" 2>&1 | tail -1
echo "CB_DEPS_DONE"
conda run -n cb python patch_chatterbox.py
echo "PATCH_STAGE_DONE"
conda run -n cb python train_delivery.py
echo "TRAIN_STAGE_DONE"
conda run -n cb python gate_delivery.py
echo "=== JOB DONE $(date) ==="
