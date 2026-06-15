#!/bin/bash
set -x; echo "=== CHATTERBOX bake-off (last off-the-shelf Arabic engine) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg curl >/dev/null 2>&1; echo FFMPEG_DONE
pip install -q chatterbox-tts resemblyzer speechbrain openai-whisper jiwer librosa soundfile 2>&1 | tail -3
echo "CB_DEPS_DONE"
python -c "import torch; print('TORCHCHECK', torch.__version__, 'cuda', torch.cuda.is_available())"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python gen_chatterbox.py
echo "=== MEASURE ==="
python measure_chatterbox.py
echo "=== JOB DONE $(date) ==="
