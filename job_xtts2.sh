#!/bin/bash
set -x; echo "=== XTTS2 (full-ref) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1; echo FFMPEG_DONE
export ASSETS=$(pwd)
pip install -q "numpy<2" resemblyzer speechbrain 2>&1 | tail -2
pip install -q coqui-tts 2>&1 | tail -3
pip install -q "transformers>=4.57,<5.0" 2>&1 | tail -2
export COQUI_TOS_AGREED=1
echo "=== slice her into conditioning segments ==="
python - <<'PY'
import librosa, soundfile as sf, numpy as np, os
os.makedirs("hersegs",exist_ok=True)
y,sr=librosa.load("her_audio.wav",sr=24000,mono=True)
iv=librosa.effects.split(y,top_db=30)
segs=[]; buf=[]; cur=0
for s,e in iv:
    buf.append(y[s:e]); cur+=e-s
    if cur>11*sr:
        segs.append(np.concatenate(buf)); buf=[]; cur=0
    if len(segs)>=8: break
if buf and len(segs)<8: segs.append(np.concatenate(buf))
for i,s in enumerate(segs): sf.write(f"hersegs/h{i}.wav", s, sr)
print("HERSEGS", len(segs), flush=True)
PY
mkdir -p out_xtts2
echo "=== XTTS FULL-REF ZERO-SHOT ==="
python - <<'PY'
import glob, torch, warnings; warnings.filterwarnings("ignore")
from TTS.api import TTS
dev="cuda" if torch.cuda.is_available() else "cpu"
tts=TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(dev)
refs=sorted(glob.glob("hersegs/*.wav"))
print("using", len(refs), "reference segments", flush=True)
sents=[("xtts2_1","The morning light came slowly across the quiet harbor as the boats began to stir."),
       ("xtts2_2","I never expected to find the answer hidden in such an ordinary place that day."),
       ("xtts2_3","We should leave before the rain starts, or we will be stuck here all night long."),
       ("xtts2_4","She paused at the doorway, listening to the sound of the city waking up below."),
       ("xtts2_5","Honestly, I think we made the right choice, even if it did not feel that way then.")]
for name,txt in sents:
    try:
        tts.tts_to_file(text=txt, speaker_wav=refs, language="en", file_path=f"out_xtts2/{name}.wav")
        print("GEN", name, flush=True)
    except Exception as e:
        print("GENERR", name, repr(e), flush=True)
PY
echo "=== MEASURE ==="
python measure_v2.py out_xtts2 XTTS-FULLREF
echo "=== JOB DONE $(date) ==="
