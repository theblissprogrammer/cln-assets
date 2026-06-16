#!/bin/bash
set -x; echo "=== BIG-CORPUS DELIVERY-ADAPTER BUILD START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg curl >/dev/null 2>&1; echo FFMPEG_DONE
# STAGE 1: corpus build in BASE env (whisper+resemblyzer+yt-dlp; base image already has torch+numpy)
pip install -q resemblyzer openai-whisper yt-dlp praat-parselmouth librosa soundfile "numpy<2" 2>&1 | tail -1
echo "BASE_DEPS_DONE"
python -c "import resemblyzer, whisper, yt_dlp, parselmouth; print('base imports OK')"
python download_corpus.py
echo "CORPUS_STAGE_DONE"
# STAGE 2: train+gate in cb env (chatterbox)
conda create -y -n cb python=3.11 2>&1 | tail -1
conda run -n cb pip install -q chatterbox-tts resemble-perth soundfile librosa praat-parselmouth peft "numpy<2" 2>&1 | tail -1
echo "CB_DEPS_DONE"
conda run -n cb python patch_chatterbox.py
echo "PATCH_STAGE_DONE"
conda run -n cb python train_delivery_lora.py
echo "TRAIN_STAGE_DONE"
conda run -n cb python gate_delivery_lora.py
echo "=== JOB DONE $(date) ==="
