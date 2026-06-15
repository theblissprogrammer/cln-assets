#!/bin/bash
set -x; echo "=== PUSH1 (best-of-10 + revocode feasibility) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1; echo FFMPEG_DONE
export ASSETS=$(pwd)
pip install -q "numpy<2" resemblyzer speechbrain vocos 2>&1 | tail -1
pip install -q descript-audio-codec 2>&1 | tail -1 || echo dac_optional
pip install -q coqui-tts "transformers>=4.57,<5.0" 2>&1 | tail -2
export COQUI_TOS_AGREED=1
python - <<'PY'
import librosa, soundfile as sf, numpy as np, os, glob, warnings, torch; warnings.filterwarnings("ignore")
from resemblyzer import VoiceEncoder, preprocess_wav
from TTS.api import TTS
dev="cuda" if torch.cuda.is_available() else "cpu"
def loud(w): r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
y,sr=librosa.load("her_audio.wav",sr=24000,mono=True); iv=librosa.effects.split(y,top_db=30)
segs=[];buf=[];cur=0
for s,e in iv:
    buf.append(y[s:e]);cur+=e-s
    if cur>9*sr: segs.append(np.concatenate(buf));buf=[];cur=0
if buf: segs.append(np.concatenate(buf))
enc=VoiceEncoder(verbose=False)
embs=[(lambda e:e/np.linalg.norm(e))(enc.embed_utterance(preprocess_wav(loud(s),source_sr=24000))) for s in segs]
c=np.mean(embs,0); c/=np.linalg.norm(c); sc=[float(e@c) for e in embs]; order=np.argsort(sc)[::-1]
os.makedirs("hersegs",exist_ok=True)
for rank,i in enumerate(order[:4]): sf.write(f"hersegs/seg{rank:02d}.wav", loud(segs[i]),24000)
tts=TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(dev); m=tts.synthesizer.tts_model
refs=sorted(glob.glob("hersegs/seg*.wav"))
gl,sp=m.get_conditioning_latents(audio_path=refs,max_ref_length=30,gpt_cond_len=30,gpt_cond_chunk_len=6)
S=[("s1","The morning light came slowly across the quiet harbor as the boats began to stir."),
   ("s2","I never expected to find the answer hidden in such an ordinary place that day."),
   ("s3","She paused at the doorway, listening to the sound of the city waking up below."),
   ("s4","After all these years, the old house still smelled exactly the way I remembered."),
   ("s5","We should leave before the rain starts, or we will be stuck here all night long.")]
os.makedirs("out_raw",exist_ok=True)
for sid,txt in S:
    for j in range(10):
        o=m.inference(txt,"en",gl,sp,temperature=0.75,enable_text_splitting=True)
        sf.write(f"out_raw/{sid}_c{j}.wav", np.asarray(o["wav"]),24000)
    print("GEN",sid,flush=True)
del tts,m; torch.cuda.empty_cache()
# ---- generic re-vocode (feasibility: is the vocoder in the identity path?) ----
os.makedirs("out_vocos",exist_ok=True); os.makedirs("out_dac",exist_ok=True)
try:
    from vocos import Vocos
    vo=Vocos.from_pretrained("charactr/vocos-mel-24khz").to(dev)
    for p in glob.glob("out_raw/*.wav"):
        w,_=librosa.load(p,sr=24000,mono=True); t=torch.tensor(w).float().unsqueeze(0).to(dev)
        with torch.no_grad(): rec=vo.decode(vo.feature_extractor(t)).squeeze().cpu().numpy()
        sf.write(f"out_vocos/{os.path.basename(p)}", rec, 24000)
    print("VOCOS_REVOC_DONE",flush=True)
except Exception as e: print("VOCOS_ERR",repr(e),flush=True)
try:
    import dac; from audiotools import AudioSignal
    md=dac.DAC.load(dac.utils.download(model_type="24khz")).to(dev)
    for p in glob.glob("out_raw/*.wav"):
        w,_=librosa.load(p,sr=24000,mono=True)
        sig=AudioSignal(w.astype(np.float32),24000).to(dev); x=md.preprocess(sig.audio_data,sig.sample_rate)
        with torch.no_grad(): z,_,_,_,_=md.encode(x); rec=md.decode(z)
        sf.write(f"out_dac/{os.path.basename(p)}", rec.squeeze().detach().cpu().numpy(),24000)
    print("DAC_REVOC_DONE",flush=True)
except Exception as e: print("DAC_ERR",repr(e),flush=True)
PY
echo "=== MEASURE (best-of-10 each) ==="
echo "--- RAW XTTS (best-of-10) ---";  python measure_bon.py out_raw
echo "--- VOCOS re-vocode ---";        python measure_bon.py out_vocos
echo "--- DAC re-vocode ---";          python measure_bon.py out_dac
echo "=== JOB DONE $(date) ==="
