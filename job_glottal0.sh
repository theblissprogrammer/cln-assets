#!/bin/bash
set -x; echo "=== GLOTTAL0 (STEP 0: does her source drift cross-lingually?) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1; echo FFMPEG_DONE
pip install -q "numpy<2" resemblyzer speechbrain coqui-tts "transformers>=4.57,<5.0" torchaudio librosa soundfile praat-parselmouth 2>&1 | tail -2
export COQUI_TOS_AGREED=1
python gen_glottal0.py
echo "=== MEASURE ==="
python measure_glottal0.py
echo "=== JOB DONE $(date) ==="
