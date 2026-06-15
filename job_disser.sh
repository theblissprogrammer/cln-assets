#!/bin/bash
set -x; echo "=== DISSER (where does emotion live: gpt-latent vs spk-emb, SER) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1; echo FFMPEG_DONE
pip install -q "numpy<2" resemblyzer coqui-tts "transformers>=4.57,<5.0" torchaudio librosa soundfile 2>&1 | tail -2
export COQUI_TOS_AGREED=1
python gen_disser.py
echo "=== MEASURE ==="
python measure_disser.py
echo "=== JOB DONE $(date) ==="
