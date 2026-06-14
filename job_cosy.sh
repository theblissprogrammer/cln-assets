#!/bin/bash
set -x; echo "=== COSY JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg sox libsox-dev unzip >/dev/null 2>&1; echo FFMPEG_DONE
export ASSETS=$(pwd)
pip install -q "numpy<2" resemblyzer speechbrain faster-whisper 2>&1 | tail -2
cd /workspace 2>/dev/null || cd /root
git clone --recursive https://github.com/FunAudioLLM/CosyVoice 2>&1 | tail -3
cd CosyVoice
pip install -q -r requirements.txt 2>&1 | tail -5
pip install -q modelscope 2>&1 | tail -1
python -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')" 2>&1 | tail -3
cp $ASSETS/her_audio.wav $ASSETS/ref.wav $ASSETS/measure_v2.py .
mkdir -p out_cosy
echo "=== TRANSCRIBE REF ==="
REFTXT=$(python - <<'PY'
from faster_whisper import WhisperModel
m=WhisperModel("small", device="cuda", compute_type="float16")
segs,_=m.transcribe("ref.wav", language="en")
print(" ".join(s.text for s in segs).strip())
PY
)
echo "REFTXT=$REFTXT"
echo "=== COSY ZERO-SHOT INFER ==="
python - "$REFTXT" <<'PY'
import sys, torch, torchaudio, warnings; warnings.filterwarnings("ignore")
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
reftext=sys.argv[1] if len(sys.argv)>1 and sys.argv[1].strip() else "this is a reference recording of my voice."
cv=CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)
prompt=load_wav('ref.wav',16000)
sents=[("cosy_1","The morning light came slowly across the quiet harbor as the boats began to stir."),
       ("cosy_2","I never expected to find the answer hidden in such an ordinary place that day."),
       ("cosy_3","We should leave before the rain starts, or we will be stuck here all night long.")]
for name,txt in sents:
    try:
        for i,j in enumerate(cv.inference_zero_shot(txt, reftext, prompt, stream=False)):
            torchaudio.save(f'out_cosy/{name}.wav', j['tts_speech'], cv.sample_rate); break
        print("GEN", name, flush=True)
    except Exception as e:
        print("GENERR", name, repr(e), flush=True)
PY
echo "=== MEASURE ==="
python measure_v2.py out_cosy COSY-ZS
echo "=== JOB DONE $(date) ==="
