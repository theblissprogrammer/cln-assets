#!/bin/bash
set -x; echo "=== DIS2 (disentangle id vs emotion, COMPACT summary) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1; echo FFMPEG_DONE
export ASSETS=$(pwd)
pip install -q "numpy<2" resemblyzer speechbrain coqui-tts "transformers>=4.57,<5.0" torchaudio 2>&1 | tail -2
export COQUI_TOS_AGREED=1
python - <<'PY'
# -*- coding: utf-8 -*-
import librosa, numpy as np, os, glob, warnings, torch; warnings.filterwarnings("ignore")
from resemblyzer import VoiceEncoder, preprocess_wav
from speechbrain.inference.speaker import EncoderClassifier
from TTS.api import TTS
dev="cuda" if torch.cuda.is_available() else "cpu"
def loud(w): r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
y,sr=librosa.load("her_audio.wav",sr=24000,mono=True); iv=librosa.effects.split(y,top_db=30)
segs=[];buf=[];cur=0
for s,e in iv:
    buf.append(y[s:e]);cur+=e-s
    if cur>8*sr: segs.append(np.concatenate(buf));buf=[];cur=0
if buf and cur>5*sr: segs.append(np.concatenate(buf))
enc=VoiceEncoder(verbose=False)
embs=[(lambda e:e/np.linalg.norm(e))(enc.embed_utterance(preprocess_wav(loud(s),source_sr=24000))) for s in segs]
c=np.mean(embs,0); c/=np.linalg.norm(c); cen=[float(e@c) for e in embs]; herv=c
rng=[]
for s in segs:
    w16=librosa.resample(s,orig_sr=24000,target_sr=16000); f0,_,_=librosa.pyin(w16,fmin=70,fmax=400,sr=16000); f0v=f0[~np.isnan(f0)]
    rng.append(12*np.log2((np.percentile(f0v,95)+1e-9)/(np.percentile(f0v,5)+1e-9)) if len(f0v)>5 else 0)
eca=EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",savedir="/tmp/ecapa",run_opts={"device":dev})
def eemb(w16):
    with torch.no_grad(): v=eca.encode_batch(torch.tensor(w16).float().unsqueeze(0).to(dev)).squeeze().cpu().numpy()
    return v/np.linalg.norm(v)
Ee=np.array([eemb(loud(librosa.resample(s,orig_sr=24000,target_sr=16000))) for s in segs]); hereca=np.mean(Ee,0); hereca/=np.linalg.norm(hereca)
def f0range(w):
    f0,_,_=librosa.pyin(w,fmin=70,fmax=400,sr=16000); v=f0[~np.isnan(f0)]
    return (12*np.log2((np.percentile(v,95)+1e-9)/(np.percentile(v,5)+1e-9)),np.std(v)) if len(v)>5 else (0,0)
def score(w24):
    e=enc.embed_utterance(preprocess_wav(loud(w24),source_sr=24000)); e/=np.linalg.norm(e); rs=float(e@herv)
    w16=librosa.resample(w24,orig_sr=24000,target_sr=16000); ec=float(eemb(loud(w16))@hereca); fr,fs=f0range(w16)
    return rs,ec,fr,fs
tts=TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(dev); m=tts.synthesizer.tts_model
idrefs=[loud(segs[i]) for i in np.argsort(cen)[::-1][:4]]; import soundfile as sf; os.makedirs("r",exist_ok=True)
for k,w in enumerate(idrefs): sf.write(f"r/i{k}.wav",w,24000)
sf.write("r/emo.wav", loud(segs[int(np.argsort(rng)[-1])]),24000)
gl_id,sp_id=m.get_conditioning_latents(audio_path=sorted(glob.glob("r/i*.wav")),max_ref_length=30,gpt_cond_len=30,gpt_cond_chunk_len=6)
gl_emo,sp_emo=m.get_conditioning_latents(audio_path=["r/emo.wav"],max_ref_length=30,gpt_cond_len=30,gpt_cond_chunk_len=6)
print(f"REF emo-clip f0range={max(rng):.1f}",flush=True)
AR=["سوف نلتقي مرة أخرى يوم الخميس لإنهاء المراجعة.","مشينا على طول الشاطئ لبعض الوقت نتحدث عن أشياء عادية.","بعد كل ما مررنا به، انتهى بنا الأمر في نفس المكان تماماً."]
mixes={"A_id":(gl_id,sp_id),"B_emo":(gl_emo,sp_emo),"C_disent(emoGPT+idSPK)":(gl_emo,sp_id),"D_rev(idGPT+emoSPK)":(gl_id,sp_emo)}
print("\n===== SUMMARY (resem/ecapa=identity, f0range/f0std=emotion; want C high on ALL) =====",flush=True)
for tag,(gl,sp) in mixes.items():
    rs=[];ec=[];fr=[];fs=[]
    for txt in AR:
        for j in range(2):
            o=np.asarray(m.inference(txt,"ar",gl,sp,temperature=0.75,enable_text_splitting=True)["wav"])
            a,b,d,e=score(o); rs.append(a);ec.append(b);fr.append(d);fs.append(e)
    print(f"SUMMARY {tag:24s} resem={np.mean(rs):.3f} ecapa={np.mean(ec):.3f} f0range={np.mean(fr):.2f} f0std={np.mean(fs):.1f}",flush=True)
PY
echo "=== JOB DONE $(date) ==="
