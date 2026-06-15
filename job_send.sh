#!/bin/bash
set -x; echo "=== SEND (regen best XTTS clones) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg zip >/dev/null 2>&1; echo TOOLS_DONE
export ASSETS=$(pwd)
pip install -q "numpy<2" resemblyzer speechbrain coqui-tts "transformers>=4.57,<5.0" 2>&1 | tail -2
export COQUI_TOS_AGREED=1
python - <<'PY'
import librosa, soundfile as sf, numpy as np, os, glob, warnings, torch; warnings.filterwarnings("ignore")
from resemblyzer import VoiceEncoder, preprocess_wav
from speechbrain.inference.speaker import EncoderClassifier
from TTS.api import TTS
def loud(w): r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
dev="cuda" if torch.cuda.is_available() else "cpu"
# her segments
y,sr=librosa.load("her_audio.wav",sr=24000,mono=True); iv=librosa.effects.split(y,top_db=30)
segs=[];buf=[];cur=0
for s,e in iv:
    buf.append(y[s:e]);cur+=e-s
    if cur>9*sr: segs.append(np.concatenate(buf));buf=[];cur=0
if buf: segs.append(np.concatenate(buf))
# resemblyzer centroid + ref selection
enc=VoiceEncoder(verbose=False)
embs=[(lambda e:e/np.linalg.norm(e))(enc.embed_utterance(preprocess_wav(loud(s),source_sr=24000))) for s in segs]
c=np.mean(embs,0); c/=np.linalg.norm(c); sc=[float(e@c) for e in embs]; order=np.argsort(sc)[::-1]
os.makedirs("hersegs",exist_ok=True)
for rank,i in enumerate(order[:4]): sf.write(f"hersegs/seg{rank:02d}.wav", loud(segs[i]),24000)
herv=np.mean(embs,0); herv/=np.linalg.norm(herv)
# ecapa centroid
eca=EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",savedir="/tmp/ecapa",run_opts={"device":dev})
def eemb(w16):
    with torch.no_grad(): v=eca.encode_batch(torch.tensor(w16).float().unsqueeze(0).to(dev)).squeeze().cpu().numpy()
    return v/np.linalg.norm(v)
Ee=np.array([eemb(loud(s16)) for s16 in [librosa.resample(s,orig_sr=24000,target_sr=16000) for s in segs]])
hereca=np.mean(Ee,0); hereca/=np.linalg.norm(hereca)
def rsim(w): e=enc.embed_utterance(preprocess_wav(loud(w),source_sr=24000)); e/=np.linalg.norm(e); return float(e@herv)
def esim(w): w16=librosa.resample(w,orig_sr=24000,target_sr=16000); return float(eemb(loud(w16))@hereca)
# generate
tts=TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(dev); m=tts.synthesizer.tts_model
refs=sorted(glob.glob("hersegs/seg*.wav"))
gl,sp=m.get_conditioning_latents(audio_path=refs,max_ref_length=30,gpt_cond_len=30,gpt_cond_chunk_len=6)
sents=[("o1","The morning light came slowly across the quiet harbor as the boats began to stir."),
       ("o2","I never expected to find the answer hidden in such an ordinary place that day."),
       ("o3","She paused at the doorway, listening to the sound of the city waking up below."),
       ("o4","After all these years, the old house still smelled exactly the way I remembered."),
       ("o5","We should leave before the rain starts, or we will be stuck here all night long.")]
os.makedirs("bundle",exist_ok=True)
for sid,txt in sents:
    best=None
    for j in range(5):
        o=m.inference(txt,"en",gl,sp,temperature=0.75,enable_text_splitting=True)
        w=np.asarray(o["wav"]); r=rsim(w); e=esim(w); s=r+e
        if best is None or s>best[0]: best=(s,w.copy(),r,e)
    sf.write(f"bundle/ours_XTTS_{sid}.wav", best[1], 24000)
    print(f"BEST {sid} resemblyzer={best[2]:.3f} ecapa={best[3]:.3f}", flush=True)
sf.write("bundle/her_REAL_reference.wav", librosa.load("ref.wav",sr=24000,mono=True)[0], 24000)
print("BUNDLE", glob.glob("bundle/*.wav"), flush=True)
PY
cd bundle && zip -q ours.zip *.wav && cd ..
echo "=== UPLOAD ==="
URL=$(curl -s -F "reqtype=fileupload" -F "fileToUpload=@bundle/ours.zip" https://catbox.moe/user/api.php)
echo "BUNDLE_URL=$URL"
echo "=== JOB DONE $(date) ==="
