#!/bin/bash
set -x; echo "=== 2STAGE (F5 authentic Egyptian -> kNN to her) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg curl git >/dev/null 2>&1; echo FFMPEG_DONE
pip install -q f5-tts huggingface_hub resemblyzer openai-whisper librosa soundfile "numpy<2" 2>&1 | tail -3
echo "DEPS_DONE"
python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil
shutil.copy(hf_hub_download("MAdel121/f5-tts-egyptian-arabic","model_5000.pt"),"model_egy.pt")
shutil.copy(hf_hub_download("MAdel121/f5-tts-egyptian-arabic","vocab.txt"),"vocab_egy.txt")
print("MODEL_DOWNLOADED")
PY
python - <<'PY'
import librosa, soundfile as sf, numpy as np, whisper, warnings; warnings.filterwarnings("ignore")
def loud(w): r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
y,_=librosa.load("her_audio.wav",sr=24000,mono=True); iv=librosa.effects.split(y,top_db=30)
seg=max([y[s:e] for s,e in iv],key=len)[:10*24000]; sf.write("her_ref.wav", loud(seg),24000)
m=whisper.load_model("base"); open("ref_text.txt","w").write(m.transcribe("her_ref.wav",language="en")["text"].strip())
print("REF_READY")
PY
REFTXT=$(cat ref_text.txt)
mkdir -p out_f5src tmpf5
i=0
for S in "هنتقابل تاني يوم الخميس عشان نخلّص المراجعة." "معظم الكراتين كانت متجهّزة خلاص ومستنية جنب الباب." "بصّت من الشباك مدة طويلة من غير ما تقول أي حاجة." "مشينا على طول البحر شوية وإحنا بنتكلم في حاجات عادية." "بعد كل اللي عدّى علينا، رجعنا تاني لنفس المكان."; do
  python -m f5_tts.infer.infer_cli --model F5TTS_Base --ckpt_file model_egy.pt --vocab_file vocab_egy.txt --ref_audio her_ref.wav --ref_text "$REFTXT" --gen_text "$S" --output_dir tmpf5 2>&1 | tail -2
  [ -f tmpf5/infer_cli_basic.wav ] && mv tmpf5/infer_cli_basic.wav out_f5src/$i.wav && echo "F5GEN $i"
  i=$((i+1))
done
echo "F5_GEN_DONE; sources:"; ls out_f5src/
echo "=== KNN to her + measure ==="
python run_2stage.py
echo "=== JOB DONE $(date) ==="
