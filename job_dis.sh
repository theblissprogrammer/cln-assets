#!/bin/bash
set -x; echo "=== DIS (disentangle identity vs emotion conditioning) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1; echo FFMPEG_DONE
export ASSETS=$(pwd)
pip install -q "numpy<2" resemblyzer speechbrain coqui-tts "transformers>=4.57,<5.0" torchaudio 2>&1 | tail -2
export COQUI_TOS_AGREED=1
echo "=== STEP1: TOP-4 identity refs + HIGH-emotion clip ==="
python - <<'PY'
import librosa, soundfile as sf, numpy as np, os, warnings; warnings.filterwarnings("ignore")
from resemblyzer import VoiceEncoder, preprocess_wav
def loud(w): r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
y,sr=librosa.load("her_audio.wav",sr=24000,mono=True); iv=librosa.effects.split(y,top_db=30)
segs=[];buf=[];cur=0
for s,e in iv:
    buf.append(y[s:e]);cur+=e-s
    if cur>8*sr: segs.append(np.concatenate(buf));buf=[];cur=0
if buf and cur>5*sr: segs.append(np.concatenate(buf))
enc=VoiceEncoder(verbose=False)
embs=[(lambda e:e/np.linalg.norm(e))(enc.embed_utterance(preprocess_wav(loud(s),source_sr=24000))) for s in segs]
c=np.mean(embs,0); c/=np.linalg.norm(c); cen=[float(e@c) for e in embs]
rng=[]
for s in segs:
    w16=librosa.resample(s,orig_sr=24000,target_sr=16000); f0,_,_=librosa.pyin(w16,fmin=70,fmax=400,sr=16000); f0v=f0[~np.isnan(f0)]
    rng.append(12*np.log2((np.percentile(f0v,95)+1e-9)/(np.percentile(f0v,5)+1e-9)) if len(f0v)>5 else 0)
os.makedirs("ref_id",exist_ok=True)
for k,i in enumerate(np.argsort(cen)[::-1][:4]): sf.write(f"ref_id/i{k}.wav", loud(segs[i]),24000)   # most-central = identity
sf.write("emo_high.wav", loud(segs[int(np.argsort(rng)[-1])]),24000)                                   # most-expressive = emotion
print("identity refs + emo_high (f0range %.1f)"%max(rng), flush=True)
PY
echo "=== STEP2: 4 conditioning mixes ==="
python - <<'PY'
# -*- coding: utf-8 -*-
import glob, torch, warnings, soundfile as sf, numpy as np, os; warnings.filterwarnings("ignore")
from TTS.api import TTS
tts=TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda" if torch.cuda.is_available() else "cpu")
m=tts.synthesizer.tts_model
gl_id,sp_id   = m.get_conditioning_latents(audio_path=sorted(glob.glob("ref_id/*.wav")),max_ref_length=30,gpt_cond_len=30,gpt_cond_chunk_len=6)
gl_emo,sp_emo = m.get_conditioning_latents(audio_path=["emo_high.wav"],max_ref_length=30,gpt_cond_len=30,gpt_cond_chunk_len=6)
AR=[("s1","سوف نلتقي مرة أخرى يوم الخميس لإنهاء المراجعة."),
    ("s2","مشينا على طول الشاطئ لبعض الوقت نتحدث عن أشياء عادية."),
    ("s3","بعد كل ما مررنا به، انتهى بنا الأمر في نفس المكان تماماً.")]
# A=identity-only  B=emotion-only  C=DISENTANGLED(emo prosody + id timbre)  D=reverse
mixes={"A_id":(gl_id,sp_id),"B_emo":(gl_emo,sp_emo),"C_disent":(gl_emo,sp_id),"D_rev":(gl_id,sp_emo)}
for tag,(gl,sp) in mixes.items():
    outdir=f"out_{tag}"; os.makedirs(outdir,exist_ok=True)
    for sid,txt in AR:
        for j in range(2):
            o=m.inference(txt,"ar",gl,sp,temperature=0.75,enable_text_splitting=True)
            sf.write(f"{outdir}/{sid}_c{j}.wav", np.asarray(o["wav"]),24000)
    print("GEN",tag,flush=True)
PY
echo "=== STEP3: identity + emotion per mix (want C = high identity AND high emotion) ==="
for tag in A_id B_emo C_disent D_rev; do
  echo "--- $tag ---"; python audio_analysis.py out_$tag 2>/dev/null | grep "^MEAN"
  python measure_v2.py out_$tag $tag 2>/dev/null | grep -oE "resemblyzer=[0-9.]+ \(z[^)]*\) \| ecapa=[0-9.]+" | awk '{print "  id:",$0}' | tail -6
done
echo "=== JOB DONE $(date) ==="
