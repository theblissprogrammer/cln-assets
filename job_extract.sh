#!/bin/bash
set -x; echo "=== FULL EXTRACTION START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg curl build-essential >/dev/null 2>&1; echo FFMPEG_DONE
pip install -q resemblyzer openai-whisper yt-dlp praat-parselmouth librosa soundfile "setuptools<81" "numpy<2" 2>&1 | tail -1
echo "DEPS_DONE"
python extract_full.py
echo "=== uploading corpus ==="
tar czf corpus_audio.tar.gz corpus/*.wav 2>/dev/null
curl -s -F "reqtype=fileupload" -F "time=72h" -F "fileToUpload=@corpus/feature_table.json" https://litterbox.catbox.moe/resources/internals/api.php; echo
echo "=== JOB DONE $(date) ==="
