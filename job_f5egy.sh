#!/bin/bash
set -x; echo "=== F5-EGYPTIAN smoke (authentic Egyptian dialect in her voice) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg curl git >/dev/null 2>&1; echo FFMPEG_DONE
pip install -q f5-tts huggingface_hub librosa soundfile openai-whisper "numpy<2" 2>&1 | tail -3
echo "F5_DEPS_DONE"
python -c "import torch; print('TORCHCHECK', torch.__version__, 'cuda', torch.cuda.is_available())"
python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil
m = hf_hub_download("MAdel121/f5-tts-egyptian-arabic", "model_5000.pt")
v = hf_hub_download("MAdel121/f5-tts-egyptian-arabic", "vocab.txt")
shutil.copy(m, "model_egy.pt"); shutil.copy(v, "vocab_egy.txt")
print("MODEL_DOWNLOADED")
PY
python - <<'PY'
import librosa, soundfile as sf, numpy as np, whisper, warnings
warnings.filterwarnings("ignore")
def loud(w):
    r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
y,_=librosa.load("her_audio.wav",sr=24000,mono=True); iv=librosa.effects.split(y,top_db=30)
seg=max([y[s:e] for s,e in iv],key=len)[:10*24000]
sf.write("her_ref.wav", loud(seg), 24000)
m=whisper.load_model("base"); txt=m.transcribe("her_ref.wav",language="en")["text"].strip()
open("ref_text.txt","w").write(txt); print("REF_TEXT:", txt[:90])
PY
echo "REF_READY"
REFTXT=$(cat ref_text.txt)
mkdir -p out_f5
echo "--- infer attempt 1: --model F5TTS_Base ---"
python -m f5_tts.infer.infer_cli --model F5TTS_Base --ckpt_file model_egy.pt --vocab_file vocab_egy.txt \
  --ref_audio her_ref.wav --ref_text "$REFTXT" \
  --gen_text "هنتقابل تاني يوم الخميس عشان نخلّص المراجعة." \
  --output_dir out_f5 2>&1 | tail -25
echo "F5_INFER_DONE"
echo "--- outputs ---"; find out_f5 -name "*.wav" 2>/dev/null | head; ls -la out_f5/ 2>/dev/null
W=$(find out_f5 -name "*.wav" 2>/dev/null | head -1)
if [ -n "$W" ]; then
  ffmpeg -y -t 7 -i "$W" -ar 24000 -ac 1 -b:a 64k f5egy.mp3 2>/dev/null
  echo "URL f5_egyptian $(curl -s --max-time 60 -F reqtype=fileupload -F time=72h -F fileToUpload=@f5egy.mp3 https://litterbox.catbox.moe/resources/internals/api.php)"
fi
echo "=== JOB DONE $(date) ==="
