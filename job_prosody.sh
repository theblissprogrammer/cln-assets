#!/bin/bash
set -x; echo "=== PROSODY (signal-level F0 emotion control, identity-preserving) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1; echo FFMPEG_DONE
export ASSETS=$(pwd)
pip install -q "numpy<2" resemblyzer speechbrain coqui-tts "transformers>=4.57,<5.0" torchaudio pyworld 2>&1 | tail -2
export COQUI_TOS_AGREED=1
python - <<'PY'
# -*- coding: utf-8 -*-
import librosa, numpy as np, os, glob, warnings, torch, pyworld as pw; warnings.filterwarnings("ignore")
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
herv=np.mean(embs,0); herv/=np.linalg.norm(herv); cen=[float(e@herv) for e in embs]
eca=EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",savedir="/tmp/ecapa",run_opts={"device":dev})
def eemb(w16):
    with torch.no_grad(): v=eca.encode_batch(torch.tensor(w16).float().unsqueeze(0).to(dev)).squeeze().cpu().numpy()
    return v/np.linalg.norm(v)
Ee=np.array([eemb(loud(librosa.resample(s,orig_sr=24000,target_sr=16000))) for s in segs]); hereca=np.mean(Ee,0); hereca/=np.linalg.norm(hereca)
# UTMOS
try:
    UT=torch.hub.load("tarepan/SpeechMOS:v1.2.0","utmos22_strong",trust_repo=True).eval()
except Exception as e: UT=None; print("UTMOS unavail",e)
def f0range(w16):
    f0,_,_=librosa.pyin(w16,fmin=70,fmax=400,sr=16000); v=f0[~np.isnan(f0)]
    return float(12*np.log2((np.percentile(v,95)+1e-9)/(np.percentile(v,5)+1e-9))) if len(v)>5 else 0.0
def score(w24):
    e=enc.embed_utterance(preprocess_wav(loud(w24),source_sr=24000)); e/=np.linalg.norm(e); rs=float(e@herv)
    w16=librosa.resample(w24,orig_sr=24000,target_sr=16000); ec=float(eemb(loud(w16))@hereca)
    ut=float(UT(torch.tensor(w16/(np.max(np.abs(w16))+1e-9)).float().unsqueeze(0),16000)) if UT is not None else 0
    return rs,ec,f0range(w16),ut
# WORLD F0-dynamics scaling: rescale log-f0 around its mean by factor k (k<1 flat, k>1 expressive); timbre (sp) + ap untouched
def f0_scale(w24, k):
    x=w24.astype(np.float64)
    f0,t=pw.harvest(x,24000); sp=pw.cheaptrick(x,f0,t,24000); ap=pw.d4c(x,f0,t,24000)
    vf=f0>0; lf=np.log(f0[vf]+1e-9); m=lf.mean()
    f0n=f0.copy(); f0n[vf]=np.exp(m+(lf-m)*k)
    return pw.synthesize(f0n,sp,ap,24000).astype(np.float32)
tts=TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(dev); m=tts.synthesizer.tts_model
import soundfile as sf
idrefs=[loud(segs[i]) for i in np.argsort(cen)[::-1][:4]]; os.makedirs("r",exist_ok=True)
for k,w in enumerate(idrefs): sf.write(f"r/i{k}.wav",w,24000)
gl,sp=m.get_conditioning_latents(audio_path=sorted(glob.glob("r/i*.wav")),max_ref_length=30,gpt_cond_len=30,gpt_cond_chunk_len=6)
AR=["سوف نلتقي مرة أخرى يوم الخميس لإنهاء المراجعة.","مشينا على طول الشاطئ لبعض الوقت نتحدث عن أشياء عادية.","بعد كل ما مررنا به، انتهى بنا الأمر في نفس المكان تماماً."]
bases=[np.asarray(m.inference(t,"ar",gl,sp,temperature=0.75,enable_text_splitting=True)["wav"]) for t in AR]
print("\n===== SUMMARY (want identity HELD across k while f0range scales; UTMOS not too hurt) =====",flush=True)
for tag,k in [("FLAT_k0.4",0.4),("BASE_k1.0",1.0),("EXPR_k1.7",1.7)]:
    rs=[];ec=[];fr=[];ut=[]
    for b in bases:
        w = b if abs(k-1.0)<1e-6 else f0_scale(b,k)
        a,c,d,e=score(w); rs.append(a);ec.append(c);fr.append(d);ut.append(e)
    print(f"SUMMARY {tag:12s} resem={np.mean(rs):.3f} ecapa={np.mean(ec):.3f} f0range={np.mean(fr):.2f} utmos={np.mean(ut):.2f}",flush=True)
PY
echo "=== JOB DONE $(date) ==="
