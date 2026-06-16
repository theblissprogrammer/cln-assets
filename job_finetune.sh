#!/bin/bash
set -x; echo "=== FINE-TUNE-ON-HER BUILD START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg curl build-essential >/dev/null 2>&1; echo FFMPEG_DONE
pip install -q resemblyzer openai-whisper yt-dlp praat-parselmouth librosa soundfile "numpy<2" 2>&1 | tail -1
echo "BASE_DEPS_DONE"
python -c "import resemblyzer, whisper, yt_dlp, parselmouth; print('base imports OK')"
python download_corpus.py
echo "CORPUS_STAGE_DONE"
conda create -y -n cb python=3.11 2>&1 | tail -1
conda run -n cb pip install -q chatterbox-tts resemble-perth resemblyzer soundfile librosa praat-parselmouth peft "numpy<2" 2>&1 | tail -1
echo "CB_DEPS_DONE"
conda run -n cb python -c "import chatterbox, resemblyzer; print('cb imports OK')"
conda run -n cb python train_finetune.py
echo "TRAIN_STAGE_DONE"
conda run -n cb python eval_finetune.py
echo "=== JOB DONE $(date) ==="
