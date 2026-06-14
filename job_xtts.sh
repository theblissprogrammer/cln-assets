#!/bin/bash
set -x; echo "=== XTTS JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1; echo FFMPEG_DONE
export ASSETS=$(pwd)
pip install -q "numpy<2" resemblyzer speechbrain 2>&1 | tail -2
pip install -q coqui-tts 2>&1 | tail -3
pip install -q "transformers==4.46.2" 2>&1 | tail -2   # 5.x removed isin_mps_friendly that XTTS imports
python -c "import transformers; print('transformers', transformers.__version__)"
export COQUI_TOS_AGREED=1
mkdir -p out_xtts
echo "=== XTTS ZERO-SHOT INFER ==="
python - <<'PY'
import torch, warnings; warnings.filterwarnings("ignore")
from TTS.api import TTS
dev="cuda" if torch.cuda.is_available() else "cpu"
tts=TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(dev)
sents=[("xtts_1","The morning light came slowly across the quiet harbor as the boats began to stir."),
       ("xtts_2","I never expected to find the answer hidden in such an ordinary place that day."),
       ("xtts_3","We should leave before the rain starts, or we will be stuck here all night long.")]
for name,txt in sents:
    try:
        tts.tts_to_file(text=txt, speaker_wav="ref.wav", language="en", file_path=f"out_xtts/{name}.wav")
        print("GEN", name, flush=True)
    except Exception as e:
        print("GENERR", name, e, flush=True)
PY
echo "=== MEASURE ==="
python measure_v2.py out_xtts XTTS-ZS
echo "=== JOB DONE $(date) ==="
