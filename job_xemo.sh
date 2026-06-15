#!/bin/bash
set -x; echo "=== XEMO (cross-lingual emotion transfer) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1; echo FFMPEG_DONE
export ASSETS=$(pwd)
pip install -q "numpy<2" resemblyzer speechbrain coqui-tts "transformers>=4.57,<5.0" torchaudio 2>&1 | tail -2
export COQUI_TOS_AGREED=1
echo "=== STEP1: her EXPRESSIVE vs CALM refs (by pitch range) ==="
python - <<'PY'
import librosa, soundfile as sf, numpy as np, os, warnings; warnings.filterwarnings("ignore")
def loud(w): r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
y,sr=librosa.load("her_audio.wav",sr=24000,mono=True); iv=librosa.effects.split(y,top_db=30)
segs=[];buf=[];cur=0
for s,e in iv:
    buf.append(y[s:e]);cur+=e-s
    if cur>8*sr: segs.append(np.concatenate(buf));buf=[];cur=0
if buf and cur>4*sr: segs.append(np.concatenate(buf))
rng=[]
for s in segs:
    w16=librosa.resample(s,orig_sr=24000,target_sr=16000)
    f0,_,_=librosa.pyin(w16,fmin=70,fmax=400,sr=16000); f0v=f0[~np.isnan(f0)]
    rng.append(12*np.log2((np.percentile(f0v,95)+1e-9)/(np.percentile(f0v,5)+1e-9)) if len(f0v)>5 else 0)
order=np.argsort(rng)
os.makedirs("ref_expr",exist_ok=True); os.makedirs("ref_calm",exist_ok=True)
for k,i in enumerate(order[::-1][:4]): sf.write(f"ref_expr/e{k}.wav", loud(segs[i]),24000)
for k,i in enumerate(order[:4]):       sf.write(f"ref_calm/c{k}.wav", loud(segs[i]),24000)
print("EXPR ranges",[round(rng[i],1) for i in order[::-1][:4]],"CALM",[round(rng[i],1) for i in order[:4]],flush=True)
PY
echo "=== STEP2: generate NEUTRAL Arabic with EXPRESSIVE vs CALM EN refs ==="
python - <<'PY'
# -*- coding: utf-8 -*-
import glob, torch, warnings, soundfile as sf, numpy as np, os; warnings.filterwarnings("ignore")
from TTS.api import TTS
tts=TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda" if torch.cuda.is_available() else "cpu")
m=tts.synthesizer.tts_model
AR=[("s1","سوف نلتقي مرة أخرى يوم الخميس لإنهاء المراجعة."),
    ("s2","معظم الصناديق كانت معبأة بالفعل وتنتظر بجانب الباب."),
    ("s3","مشينا على طول الشاطئ لبعض الوقت نتحدث عن أشياء عادية."),
    ("s4","بعد كل ما مررنا به، انتهى بنا الأمر في نفس المكان تماماً.")]
for tag,refdir,outdir in [("XEXPR","ref_expr","out_xexpr"),("XCALM","ref_calm","out_xcalm")]:
    os.makedirs(outdir,exist_ok=True)
    refs=sorted(glob.glob(f"{refdir}/*.wav"))
    gl,sp=m.get_conditioning_latents(audio_path=refs,max_ref_length=30,gpt_cond_len=30,gpt_cond_chunk_len=6)
    for sid,txt in AR:
        for j in range(3):
            o=m.inference(txt,"ar",gl,sp,temperature=0.75,enable_text_splitting=True)
            sf.write(f"{outdir}/{sid}_c{j}.wav", np.asarray(o["wav"]),24000)
        print("GEN",tag,sid,flush=True)
PY
echo "=== STEP3: does emotion transfer into Arabic? (prosody) + identity vs her EN centroid ==="
echo "--- EXPRESSIVE-ref Arabic ---"; python audio_analysis.py out_xexpr 2>/dev/null | grep "^MEAN"
echo "--- CALM-ref Arabic ---";       python audio_analysis.py out_xcalm 2>/dev/null | grep "^MEAN"
echo "=== identity (vs her EN centroid) ==="
python measure_v2.py out_xexpr XEXPR 2>/dev/null | grep MEASURE | tail -3
python measure_v2.py out_xcalm XCALM 2>/dev/null | grep MEASURE | tail -3
echo "=== JOB DONE $(date) ==="
