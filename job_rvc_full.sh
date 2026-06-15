#!/bin/bash
set -x; echo "=== RVC FULL (train her 48k + cross-lingual infer, CLEAN render) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg curl git >/dev/null 2>&1; echo FFMPEG_DONE
ASSETS=$(pwd)
echo "=== 1) gen Arabic source (coqui, before RVC deps) ==="
pip install -q "numpy<2" coqui-tts "transformers>=4.57,<5.0" soundfile librosa 2>&1 | tail -1
export COQUI_TOS_AGREED=1
python gen_rvc_source.py 2>&1 | tail -3
echo "SRC files:"; ls /workspace/src 2>/dev/null
echo "=== 2) RVC setup ==="
cd /workspace
git clone --depth 1 https://github.com/nakshatra-garg/rvc-no-gui rvc 2>&1 | tail -2
cd rvc
pip install -q -r requirements.txt 2>&1 | tail -4
echo "RVC_DEPS_DONE"
python -c "import torch; print('TORCHCHECK', torch.__version__, 'cuda', torch.cuda.is_available())"
python pipeline.py setup 2>&1 | tail -3
git clone --depth 1 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI /tmp/mainrvc 2>&1 | tail -1
for MD in $(find /workspace/rvc -type d -path "*logs/mute" 2>/dev/null); do cp -rn /tmp/mainrvc/logs/mute/* "$MD/" 2>/dev/null; done
mkdir -p /workspace/rvc/RVC/logs; cp -rn /tmp/mainrvc/logs/mute /workspace/rvc/RVC/logs/ 2>/dev/null
echo "RVC_SETUP_DONE"
echo "=== 3) train her model (150 epochs) ==="
cp "$ASSETS/her_audio.wav" ./her_full.wav
python pipeline.py train -m her -a her_full.wav -e 150 -b 8 2>&1 | tail -12
echo "RVC_TRAIN_DONE"
echo "=== 4) cross-lingual infer ==="
mkdir -p /workspace/rvc/out_rvc
for f in /workspace/src/*.wav; do
  python pipeline.py infer -m her -i "$f" -o "/workspace/rvc/out_rvc/$(basename $f)" --f0-method rmvpe 2>&1 | tail -2
done
echo "RVC_INFER_DONE"
echo "=== locate outputs ==="
find /workspace/rvc /workspace -name "*.wav" -newer her_full.wav -not -path "*logs*" -not -path "*mute*" -not -path "*/src/*" -not -path "*sliced*" -not -path "*0_gt*" -not -path "*1_16k*" 2>/dev/null | head -20
echo "=== 5) upload clips (RVC outputs + 1 source) to litterbox ==="
upload(){ curl -s --max-time 60 -F reqtype=fileupload -F time=72h -F "fileToUpload=@$1" https://litterbox.catbox.moe/resources/internals/api.php; }
CANDS=$(find /workspace/rvc/out_rvc -name "*.wav" 2>/dev/null | sort | head -3)
[ -z "$CANDS" ] && CANDS=$(find /workspace/rvc -name "*.wav" -newer her_full.wav -not -path "*logs*" -not -path "*mute*" 2>/dev/null | grep -iE "out|result|infer|opt" | head -3)
for f in $CANDS; do
  ffmpeg -y -t 6 -i "$f" -ar 24000 -ac 1 -b:a 64k "${f%.wav}.mp3" 2>/dev/null
  echo "URL rvc_$(basename ${f%.wav}) $(upload ${f%.wav}.mp3)"
done
S=$(ls /workspace/src/0.wav 2>/dev/null); if [ -n "$S" ]; then ffmpeg -y -t 6 -i "$S" -ar 24000 -ac 1 -b:a 64k /workspace/src0.mp3 2>/dev/null; echo "URL src_arabic $(upload /workspace/src0.mp3)"; fi
echo "UPLOAD_DONE"
echo "=== JOB DONE $(date) ==="
