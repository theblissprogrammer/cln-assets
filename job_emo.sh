#!/bin/bash
set -x; echo "=== EMO (emotion controllability via reference) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1; echo FFMPEG_DONE
export ASSETS=$(pwd)
pip install -q "numpy<2" resemblyzer speechbrain coqui-tts "transformers>=4.57,<5.0" 2>&1 | tail -2
export COQUI_TOS_AGREED=1
echo "=== STEP1: split her segments into EXPRESSIVE vs CALM by pitch range ==="
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
    r=12*np.log2((np.percentile(f0v,95)+1e-9)/(np.percentile(f0v,5)+1e-9)) if len(f0v)>5 else 0
    rng.append(r)
order=np.argsort(rng)  # ascending pitch range
os.makedirs("ref_expr",exist_ok=True); os.makedirs("ref_calm",exist_ok=True)
for k,i in enumerate(order[::-1][:4]): sf.write(f"ref_expr/e{k}.wav", loud(segs[i]),24000)  # most expressive
for k,i in enumerate(order[:4]):       sf.write(f"ref_calm/c{k}.wav", loud(segs[i]),24000)  # calmest
print("EXPR ranges", [round(rng[i],1) for i in order[::-1][:4]], "CALM ranges", [round(rng[i],1) for i in order[:4]], flush=True)
PY
echo "=== STEP2: generate same sentences with EXPRESSIVE vs CALM refs ==="
python - <<'PY'
import glob, torch, warnings, soundfile as sf, numpy as np; warnings.filterwarnings("ignore")
from TTS.api import TTS
tts=TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda" if torch.cuda.is_available() else "cpu")
m=tts.synthesizer.tts_model
S=[("s1","I can't believe you actually did that, it's the best news I've heard all week!"),
   ("s2","Please, just tell me what happened, I need to understand all of it right now."),
   ("s3","We walked along the shore for a while, talking about nothing in particular."),
   ("s4","After everything we went through, somehow we ended up exactly where we started.")]
for tag,refdir,outdir in [("EXPR","ref_expr","out_expr"),("CALM","ref_calm","out_calm")]:
    import os; os.makedirs(outdir,exist_ok=True)
    refs=sorted(glob.glob(f"{refdir}/*.wav"))
    gl,sp=m.get_conditioning_latents(audio_path=refs,max_ref_length=30,gpt_cond_len=30,gpt_cond_chunk_len=6)
    for sid,txt in S:
        for j in range(3):
            o=m.inference(txt,"en",gl,sp,temperature=0.75,enable_text_splitting=True)
            sf.write(f"{outdir}/{sid}_c{j}.wav", np.asarray(o["wav"]),24000)
        print("GEN",tag,sid,flush=True)
PY
echo "=== STEP3: ANALYZE prosody/identity/naturalness (does expressive-ref -> more expressive output?) ==="
pip install -q torchaudio 2>&1 | tail -1
echo "--- EXPRESSIVE-conditioned output ---"; python audio_analysis.py out_expr 2>/dev/null | grep -E "^MEAN|ANALYZE"
echo "--- CALM-conditioned output ---";       python audio_analysis.py out_calm 2>/dev/null | grep -E "^MEAN|ANALYZE"
echo "--- her EXPRESSIVE refs (target) ---";   python audio_analysis.py ref_expr 2>/dev/null | grep -E "^MEAN"
echo "--- her CALM refs (target) ---";         python audio_analysis.py ref_calm 2>/dev/null | grep -E "^MEAN"
echo "=== identity check (both should stay her) ==="
python measure_v2.py out_expr EXPR 2>/dev/null | grep MEASURE | tail -4
python measure_v2.py out_calm CALM 2>/dev/null | grep MEASURE | tail -4
echo "=== JOB DONE $(date) ==="
