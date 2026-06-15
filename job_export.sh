#!/bin/bash
set -x; echo "=== EXPORT (audio clips for ear) START $(date) ==="
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1; echo FFMPEG_DONE
pip install -q "numpy<2" coqui-tts "transformers>=4.57,<5.0" torchaudio librosa soundfile 2>&1 | tail -1
export COQUI_TOS_AGREED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python export_all.py
echo "=== JOB DONE $(date) ==="
