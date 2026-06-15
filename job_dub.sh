#!/bin/bash
set -x; echo "=== DUB (TTS->VC cross-lingual) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1; echo FFMPEG_DONE
export ASSETS=$(pwd)
pip install -q "numpy<2" resemblyzer speechbrain 2>&1 | tail -1
echo "=== STAGE1: generic Arabic source via XTTS built-in speaker (NOT her) ==="
pip install -q coqui-tts "transformers>=4.57,<5.0" 2>&1 | tail -2
export COQUI_TOS_AGREED=1
mkdir -p out_src
python - <<'PY'
# -*- coding: utf-8 -*-
import torch, warnings; warnings.filterwarnings("ignore")
from TTS.api import TTS
tts=TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
ar=[("s1","صباح الخير، كيف حالك اليوم؟ أتمنى أن يكون يومك جميلاً."),
    ("s2","لم أتوقع أبداً أن أجد الإجابة في مكان عادي مثل هذا."),
    ("s3","علينا أن نغادر قبل أن يبدأ المطر، وإلا سنبقى هنا طوال الليل.")]
try:
    names=tts.synthesizer.tts_model.speaker_manager.speaker_names
    spk=names[0]; print("GENERIC SPK", spk, "of", len(names), flush=True)
    for sid,txt in ar:
        tts.tts_to_file(text=txt, speaker=spk, language="ar", file_path=f"out_src/{sid}.wav"); print("SRC",sid,flush=True)
except Exception as e:
    print("SRCERR", repr(e), flush=True)
PY
echo "=== STAGE2: Seed-VC zero-shot convert generic Arabic -> HER timbre ==="
cd /workspace 2>/dev/null || cd /root
git clone --depth 1 https://github.com/Plachtaa/seed-vc 2>&1 | tail -2
cd seed-vc && pip install -q -r requirements.txt 2>&1 | tail -5
pip install -q "numpy<2" 2>&1 | tail -1   # restore for resemblyzer
mkdir -p out_src out_dub
cp $ASSETS/out_src/*.wav out_src/ 2>/dev/null || cp /workspace/assets/out_src/*.wav out_src/ 2>/dev/null
cp $ASSETS/ref.wav $ASSETS/her_audio.wav $ASSETS/measure_v2.py . 2>/dev/null
ls -la out_src/
for s in s1 s2 s3; do
  python inference.py --source out_src/$s.wav --target ref.wav --output out_dub --diffusion-steps 30 --length-adjust 1.0 --inference-cfg-rate 0.7 --f0-condition False --auto-f0-adjust False 2>&1 | tail -3
done
ls -la out_dub/
echo "=== STAGE3: MEASURE vs her centroid ==="
echo "--- GENERIC SOURCE (control, should be LOW = not her) ---"; python measure_v2.py out_src DUB-SRC
echo "--- VC->HER Arabic (dubbing pipeline) ---";                 python measure_v2.py out_dub DUB-HER
echo "=== JOB DONE $(date) ==="
