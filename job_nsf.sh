#!/bin/bash
set -x; echo "=== NSF-HiFiGAN F0-control de-risk START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
pip install -q praat-parselmouth resemblyzer librosa soundfile "numpy<2" 2>&1 | tail -1
echo "DEPS_DONE"
# nsf_hifigan modules (so-vits-svc) + checkpoint
git clone -q https://github.com/svc-develop-team/so-vits-svc.git so-vits-svc 2>&1 | tail -1
echo "REPO_DONE"
mkdir -p pretrain/nsf_hifigan dl
curl -sL -o dl/nsf.zip https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip
python -c "import zipfile; zipfile.ZipFile('dl/nsf.zip').extractall('dl')"
# locate model + config wherever they nested, copy flat into pretrain/nsf_hifigan/
MODEL=$(find dl -name 'model' -type f | head -1)
CONF=$(find dl -name 'config.json' | head -1)
echo "found model=$MODEL config=$CONF"
cp "$MODEL" pretrain/nsf_hifigan/model
cp "$CONF" pretrain/nsf_hifigan/config.json
ls -la pretrain/nsf_hifigan/
echo "CKPT_DONE"
python nsf_test.py
echo "=== JOB DONE $(date) ==="
