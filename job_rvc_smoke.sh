#!/bin/bash
set -x; echo "=== RVC SMOKE JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg curl git >/dev/null 2>&1; echo FFMPEG_DONE
ASSETS=$(pwd)
cd /workspace 2>/dev/null || cd /root
git clone --depth 1 https://github.com/nakshatra-garg/rvc-no-gui rvc 2>&1 | tail -3
cd rvc
echo "=== PIP INSTALL REQUIREMENTS ==="
pip install -r requirements.txt 2>&1 | tail -8
echo "RVC_DEPS_DONE"
python -c "import torch; print('TORCHCHECK', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1
echo "=== SETUP (download prereq models) ==="
python pipeline.py setup 2>&1 | tail -12
echo "RVC_SETUP_DONE"
echo "=== POPULATE mute assets (rvc-no-gui setup leaves logs/mute incomplete) ==="
git clone --depth 1 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI /tmp/mainrvc 2>&1 | tail -1
for MUTEDIR in $(find /workspace/rvc -type d -path "*logs/mute" 2>/dev/null); do
  echo "copying mute into $MUTEDIR"; cp -rn /tmp/mainrvc/logs/mute/* "$MUTEDIR/" 2>/dev/null || true
done
# also ensure a logs/mute exists at the expected nested path even if find missed it
mkdir -p /workspace/rvc/RVC/logs 2>/dev/null; cp -rn /tmp/mainrvc/logs/mute /workspace/rvc/RVC/logs/ 2>/dev/null || true
echo "MUTE files present:"; find /workspace/rvc -path "*logs/mute*" -name "*.wav" 2>/dev/null | head; find /tmp/mainrvc/logs/mute -name "*.wav" 2>/dev/null | head
echo "=== TINY TRAIN (3 epochs, 30s) ==="
ffmpeg -y -t 30 -i "$ASSETS/her_audio.wav" -ar 48000 her30.wav 2>/dev/null
python pipeline.py train -m hertest -a her30.wav -e 3 -b 4 2>&1 | tail -25
echo "RVC_TRAIN_DONE"
echo "=== INFER (5s source) ==="
ffmpeg -y -ss 60 -t 5 -i "$ASSETS/her_audio.wav" -ar 48000 src5.wav 2>/dev/null
python pipeline.py infer -m hertest -i src5.wav -o out_test.wav --f0-method rmvpe 2>&1 | tail -12
echo "RVC_INFER_DONE output:"; ls -la out_test.wav 2>/dev/null || echo "NO OUTPUT"
echo "=== JOB DONE $(date) ==="
