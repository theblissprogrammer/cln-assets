#!/bin/bash
set -x; echo "=== XTTS-FT JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1; echo FFMPEG_DONE
export ASSETS=$(pwd)
pip install -q "numpy<2" resemblyzer speechbrain 2>&1 | tail -1
pip install -q coqui-tts 2>&1 | tail -2
pip install -q "transformers>=4.57,<5.0" faster-whisper 2>&1 | tail -2
export COQUI_TOS_AGREED=1
python -c "import TTS; print('coqui-tts', TTS.__version__)"

echo "=== STEP1: format/transcribe her_audio -> train/eval csv ==="
python - <<'PY'
import warnings; warnings.filterwarnings("ignore")
from TTS.demos.xtts_ft_demo.utils.formatter import format_audio_list
r = format_audio_list(["her_audio.wav"], target_language="en", out_path="ftdata",
                      buffer=0.2, eval_percentage=0.15, speaker_name="her", gradio_progress=None)
print("FORMAT_RET", r, flush=True)
import glob
print("CSVS", glob.glob("ftdata/*.csv"), flush=True)
for c in glob.glob("ftdata/*.csv"):
    print(c, "rows=", sum(1 for _ in open(c))-1, flush=True)
PY

echo "=== STEP2: train GPT ==="
python - <<'PY'
import warnings, glob, os; warnings.filterwarnings("ignore")
from TTS.demos.xtts_ft_demo.utils.gpt_train import train_gpt
tr=[c for c in glob.glob("ftdata/*train*.csv")] or glob.glob("ftdata/*.csv")
ev=[c for c in glob.glob("ftdata/*eval*.csv")] or tr
train_csv, eval_csv = tr[0], ev[0]
print("USING train=",train_csv,"eval=",eval_csv, flush=True)
r = train_gpt(custom_model="", version="v2.0.2", language="en",
              num_epochs=12, batch_size=3, grad_acumm=84,
              train_csv=train_csv, eval_csv=eval_csv,
              output_path=os.path.abspath("ftout"), max_audio_length=255995)
print("TRAIN_RET", r, flush=True)
PY

echo "=== STEP3: discover artifacts + inference ==="
python - <<'PY'
import warnings, glob, os, json; warnings.filterwarnings("ignore")
import torch, soundfile as sf
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

def newest(paths): return sorted([p for p in paths if os.path.exists(p)], key=os.path.getmtime)[-1] if paths else None
cfgs   = glob.glob("ftout/**/config.json", recursive=True)
config_path = newest(cfgs)
run_dir = os.path.dirname(config_path) if config_path else "ftout"
ckpt = newest(glob.glob(run_dir+"/best_model.pth")+glob.glob(run_dir+"/checkpoint_*.pth")+glob.glob(run_dir+"/*.pth")+glob.glob("ftout/**/*.pth", recursive=True))
vocab = (glob.glob(run_dir+"/vocab.json")+glob.glob("ftout/**/vocab.json", recursive=True)
         +glob.glob(os.path.expanduser("~/.local/share/tts/**/vocab.json"), recursive=True)
         +glob.glob("/root/.local/share/tts/**/vocab.json", recursive=True))
vocab = vocab[0] if vocab else None
print("ARTIFACTS config=",config_path,"ckpt=",ckpt,"vocab=",vocab, flush=True)
assert config_path and ckpt and vocab, "missing artifact"

cfg=XttsConfig(); cfg.load_json(config_path)
model=Xtts.init_from_config(cfg)
model.load_checkpoint(cfg, checkpoint_path=ckpt, vocab_path=vocab, use_deepspeed=False)
model.cuda()
refs=sorted(glob.glob("hersegs/*.wav"))
if not refs:
    import librosa, numpy as np
    os.makedirs("hersegs",exist_ok=True)
    y,sr=librosa.load("her_audio.wav",sr=24000,mono=True); iv=librosa.effects.split(y,top_db=30)
    segs=[];buf=[];cur=0
    for s,e in iv:
        buf.append(y[s:e]);cur+=e-s
        if cur>11*sr: segs.append(np.concatenate(buf));buf=[];cur=0
        if len(segs)>=8: break
    for i,s in enumerate(segs): sf.write(f"hersegs/h{i}.wav",s,sr)
    refs=sorted(glob.glob("hersegs/*.wav"))
gpt_lat, spk = model.get_conditioning_latents(audio_path=refs, max_ref_length=30, gpt_cond_len=30)
os.makedirs("out_xttsft",exist_ok=True)
sents=[("xttsft_1","The morning light came slowly across the quiet harbor as the boats began to stir."),
       ("xttsft_2","I never expected to find the answer hidden in such an ordinary place that day."),
       ("xttsft_3","We should leave before the rain starts, or we will be stuck here all night long."),
       ("xttsft_4","She paused at the doorway, listening to the sound of the city waking up below."),
       ("xttsft_5","Honestly, I think we made the right choice, even if it did not feel that way then.")]
for name,txt in sents:
    try:
        out=model.inference(txt,"en",gpt_lat,spk,temperature=0.7,enable_text_splitting=True)
        sf.write(f"out_xttsft/{name}.wav", out["wav"], 24000); print("GEN",name,flush=True)
    except Exception as e:
        print("GENERR",name,repr(e),flush=True)
PY
echo "=== MEASURE ==="
python measure_v2.py out_xttsft XTTS-FT
echo "=== JOB DONE $(date) ==="
