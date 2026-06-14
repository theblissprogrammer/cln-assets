#!/bin/bash
set -x; echo "=== JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1; echo FFMPEG_DONE
export ASSETS=$(pwd)
pip install -q resemblyzer 2>&1 | tail -1
cd /workspace 2>/dev/null || cd /root
git clone --depth 1 https://github.com/Plachtaa/seed-vc 2>&1 | tail -2
cd seed-vc && pip install -q -r requirements.txt 2>&1 | tail -5
cp $ASSETS/her_audio.wav $ASSETS/src_f5.wav $ASSETS/src_cb.wav $ASSETS/ref.wav .
ls -la her_audio.wav src_f5.wav src_cb.wav ref.wav
mkdir -p data
python -c "
import librosa,soundfile as sf,numpy as np
y,sr=librosa.load('her_audio.wav',sr=22050,mono=True)
iv=librosa.effects.split(y,top_db=30); buf=[];cur=0;i=0
for s,e in iv:
    buf.append(y[s:e]);cur+=e-s
    if cur>8*sr: sf.write(f'data/seg{i}.wav',np.concatenate(buf),sr);i+=1;buf=[];cur=0
print('SEGMENTS',i)
"
CFG=./configs/presets/config_dit_mel_seed_uvit_whisper_small_wavenet.yml
echo "=== ZERO-SHOT INFER ==="; mkdir -p out_zs
for s in src_f5 src_cb; do python inference.py --source $s.wav --target ref.wav --output out_zs --diffusion-steps 30 --length-adjust 1.0 --inference-cfg-rate 0.7 --f0-condition False --auto-f0-adjust False --semi-tone-shift 0 --config $CFG --fp16 True 2>&1 | tail -5; done
echo "=== TRAIN ==="
python train.py --config $CFG --dataset-dir data --run-name her --batch-size 2 --max-steps 500 --max-epochs 500 --save-every 500 --num-workers 0 2>&1 | tail -25
CKPT=$(ls -t runs/her/*.pth 2>/dev/null | head -1); echo "CKPT=$CKPT"
echo "=== FINETUNED INFER ==="; mkdir -p out_ft
for s in src_f5 src_cb; do python inference.py --source $s.wav --target ref.wav --output out_ft --diffusion-steps 30 --length-adjust 1.0 --inference-cfg-rate 0.7 --f0-condition False --auto-f0-adjust False --semi-tone-shift 0 --checkpoint "$CKPT" --config $CFG --fp16 True 2>&1 | tail -5; done
cp $ASSETS/measure.py .; echo "=== MEASURE ==="; python measure.py
echo "=== JOB DONE $(date) ==="
