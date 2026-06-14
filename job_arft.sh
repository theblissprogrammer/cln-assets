#!/bin/bash
set -x; echo "=== AR-FT (FT vs base, cross-lingual) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1; echo FFMPEG_DONE
export ASSETS=$(pwd)
pip install -q "numpy<2" resemblyzer speechbrain 2>&1 | tail -1
pip install -q coqui-tts "transformers>=4.57,<5.0" faster-whisper pandas 2>&1 | tail -2
export COQUI_TOS_AGREED=1

echo "=== STEP1: format/transcribe ==="
python - <<'PY'
import warnings; warnings.filterwarnings("ignore")
from TTS.demos.xtts_ft_demo.utils.formatter import format_audio_list
r=format_audio_list(["her_audio.wav"],target_language="en",out_path="ftdata",buffer=0.2,eval_percentage=0.15,speaker_name="her",gradio_progress=None)
import glob; print("FORMAT_RET",r,glob.glob("ftdata/*.csv"),flush=True)
PY

echo "=== STEP2: train GPT (20 epochs for stronger adaptation) ==="
python - <<'PY'
import warnings, glob, os, inspect; warnings.filterwarnings("ignore")
from TTS.demos.xtts_ft_demo.utils.gpt_train import train_gpt
sig=inspect.signature(train_gpt)
tr=glob.glob("ftdata/*train*.csv") or glob.glob("ftdata/*.csv")
ev=glob.glob("ftdata/*eval*.csv") or tr
allkw=dict(custom_model="",version="v2.0.2",language="en",num_epochs=20,batch_size=3,grad_acumm=84,
           train_csv=os.path.abspath(tr[0]),eval_csv=os.path.abspath(ev[0]),
           output_path=os.path.abspath("ftout"),max_audio_length=255995)
kw={k:v for k,v in allkw.items() if k in sig.parameters}
print("TRAIN", train_gpt(**kw), flush=True)
PY

echo "=== top-4 refs ==="
python - <<'PY'
import librosa, soundfile as sf, numpy as np, os, warnings; warnings.filterwarnings("ignore")
from resemblyzer import VoiceEncoder, preprocess_wav
os.makedirs("hersegs",exist_ok=True)
y,sr=librosa.load("her_audio.wav",sr=24000,mono=True); iv=librosa.effects.split(y,top_db=30)
segs=[];buf=[];cur=0
for s,e in iv:
    buf.append(y[s:e]);cur+=e-s
    if cur>9*sr: segs.append(np.concatenate(buf));buf=[];cur=0
if buf: segs.append(np.concatenate(buf))
enc=VoiceEncoder(verbose=False)
def loud(w): r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
embs=[(lambda e:e/np.linalg.norm(e))(enc.embed_utterance(preprocess_wav(loud(s),source_sr=24000))) for s in segs]
c=np.mean(embs,0); c/=np.linalg.norm(c); sc=[float(e@c) for e in embs]; order=np.argsort(sc)[::-1]
for rank,i in enumerate(order[:4]): sf.write(f"hersegs/seg{rank:02d}.wav", loud(segs[i]),24000)
print("TOP4 done",flush=True)
PY

AR1="صباح الخير، كيف حالك اليوم؟ أتمنى أن يكون يومك جميلاً."
echo "=== STEP3: Arabic with FT model then BASE model (best-of-4) ==="
python - <<'PY'
# -*- coding: utf-8 -*-
import warnings, glob, os; warnings.filterwarnings("ignore")
import torch, soundfile as sf
SENTS=[("s1","صباح الخير، كيف حالك اليوم؟ أتمنى أن يكون يومك جميلاً."),
       ("s2","لم أتوقع أبداً أن أجد الإجابة في مكان عادي مثل هذا."),
       ("s3","علينا أن نغادر قبل أن يبدأ المطر، وإلا سنبقى هنا طوال الليل."),
       ("s4","بعد كل هذه السنوات، ما زال البيت القديم على حاله تماماً.")]
refs=sorted(glob.glob("hersegs/seg*.wav"))

# ---- FT model ----
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
def newest(ps): ps=[p for p in ps if os.path.exists(p)]; return sorted(ps,key=os.path.getmtime)[-1] if ps else None
base_cfg=glob.glob("ftout/**/XTTS_v2.0_original_model_files/config.json",recursive=True)
config_path=base_cfg[0]; base_dir=os.path.dirname(config_path); vocab=os.path.join(base_dir,"vocab.json")
ftck=[p for p in glob.glob("ftout/**/*.pth",recursive=True) if "original_model_files" not in p]
ckpt=newest([p for p in ftck if os.path.basename(p)=="best_model.pth"] or ftck)
print("FT ckpt",ckpt,flush=True)
cfg=XttsConfig(); cfg.load_json(config_path)
ft=Xtts.init_from_config(cfg); ft.load_checkpoint(cfg,checkpoint_path=ckpt,vocab_path=vocab,use_deepspeed=False); ft.cuda()
gl,sp=ft.get_conditioning_latents(audio_path=refs,max_ref_length=30,gpt_cond_len=30,gpt_cond_chunk_len=6)
os.makedirs("out_ar_ft",exist_ok=True)
for sid,txt in SENTS:
    for j in range(4):
        try:
            o=ft.inference(txt,"ar",gl,sp,temperature=0.75,enable_text_splitting=True)
            sf.write(f"out_ar_ft/{sid}_c{j}.wav",o["wav"],24000); print("GENFT",sid,j,flush=True)
        except Exception as e: print("GENFTERR",sid,j,repr(e),flush=True)
del ft; torch.cuda.empty_cache()

# ---- BASE model ----
from TTS.api import TTS
tts=TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
m=tts.synthesizer.tts_model
gl2,sp2=m.get_conditioning_latents(audio_path=refs,max_ref_length=30,gpt_cond_len=30,gpt_cond_chunk_len=6)
os.makedirs("out_ar_base",exist_ok=True)
for sid,txt in SENTS:
    for j in range(4):
        try:
            o=m.inference(txt,"ar",gl2,sp2,temperature=0.75,enable_text_splitting=True)
            sf.write(f"out_ar_base/{sid}_c{j}.wav",o["wav"],24000); print("GENBASE",sid,j,flush=True)
        except Exception as e: print("GENBASEERR",sid,j,repr(e),flush=True)
PY
echo "=== MEASURE: FT-Arabic vs BASE-Arabic (both vs her EN centroid) ==="
echo "--- FT ARABIC ---";   python measure_bon.py out_ar_ft
echo "--- BASE ARABIC ---"; python measure_bon.py out_ar_base
echo "=== JOB DONE $(date) ==="
