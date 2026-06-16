#!/bin/bash
set -x; echo "=== MULTI-SPEAKER DELIVERY ADAPTER (train-once, zero-shot) START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg curl build-essential >/dev/null 2>&1; echo FFMPEG_DONE
pip install -q praat-parselmouth librosa soundfile "numpy<2" 2>&1 | tail -1
echo "BASE_DEPS_DONE"
python build_cremad_corpus.py
echo "CORPUS_STAGE_DONE"
conda create -y -n cb python=3.11 2>&1 | tail -1
conda run -n cb pip install -q chatterbox-tts resemble-perth soundfile librosa praat-parselmouth peft "numpy<2" 2>&1 | tail -1
echo "CB_DEPS_DONE"
conda run -n cb python patch_chatterbox.py
echo "PATCH_STAGE_DONE"
conda run -n cb python train_delivery_lora.py
echo "TRAIN_STAGE_DONE"
conda run -n cb python gate_multispk.py
echo "=== JOB DONE $(date) ==="
