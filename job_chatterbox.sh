#!/bin/bash
set -x; echo "=== CHATTERBOX bake-off (isolated conda env) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg curl >/dev/null 2>&1; echo FFMPEG_DONE
# chatterbox in an ISOLATED conda env (pins conflict with base)
conda create -y -n cb python=3.11 2>&1 | tail -2
conda run -n cb pip install -q chatterbox-tts resemble-perth soundfile librosa "numpy<2" 2>&1 | tail -3
echo "CB_DEPS_DONE"
conda run -n cb python -c "import torch; print('TORCHCHECK', torch.__version__, 'cuda', torch.cuda.is_available())"
conda run -n cb python gen_chatterbox.py
echo "=== MEASURE (base env) ==="
pip install -q "numpy<2" resemblyzer speechbrain openai-whisper jiwer librosa soundfile 2>&1 | tail -1
python measure_chatterbox.py
echo "=== JOB DONE $(date) ==="
