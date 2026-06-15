#!/bin/bash
set -x; echo "=== ROBUST (content/expressivity stress) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1; echo FFMPEG_DONE
export ASSETS=$(pwd)
pip install -q "numpy<2" resemblyzer speechbrain 2>&1 | tail -1
pip install -q coqui-tts "transformers>=4.57,<5.0" 2>&1 | tail -2
export COQUI_TOS_AGREED=1
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
mkdir -p out_rob
echo "=== diverse-content English (best-of-3) ==="
python - <<'PY'
import glob, torch, warnings, soundfile as sf; warnings.filterwarnings("ignore")
from TTS.api import TTS
tts=TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
m=tts.synthesizer.tts_model; refs=sorted(glob.glob("hersegs/seg*.wav"))
gl,sp=m.get_conditioning_latents(audio_path=refs,max_ref_length=30,gpt_cond_len=30,gpt_cond_chunk_len=6)
S=[ # neutral
   ("s01n","The committee will meet again on Thursday to finish the review."),
   ("s02n","Most of the boxes were already packed and waiting by the door."),
   ("s03n","She kept a small notebook where she wrote down everything she noticed."),
   # questions
   ("s04q","Are you absolutely sure this is the right address for the meeting?"),
   ("s05q","Why would anyone leave the keys in the door overnight like that?"),
   ("s06q","Could we possibly reschedule to next week if the weather gets worse?"),
   # numbers/dates
   ("s07d","The flight departs at 6:45 on March 3rd and arrives around noon."),
   ("s08d","We sold 1,284 units in the first quarter, up 17 percent from last year."),
   ("s09d","Call me back at 555 0192 sometime after 8 in the evening."),
   # emphatic / excited
   ("s10e","That is absolutely incredible, I cannot believe we actually won!"),
   ("s11e","No, no, you have to listen to me right now, this is really important!"),
   ("s12e","Stop everything, this changes absolutely everything we planned today!"),
   # long / complex / somber
   ("s13l","After the long drive through the empty hills, she finally reached the old house where she had grown up, and for a moment she just stood there, remembering."),
   ("s14l","Although the report was technically complete, everyone in the room understood that the real questions, the difficult ones, had not even been asked yet."),
   ("s15l","He spoke quietly, almost as if he were afraid the words themselves might break something fragile that had taken years to build.")]
for sid,txt in S:
    for j in range(3):
        try:
            o=m.inference(txt,"en",gl,sp,temperature=0.75,enable_text_splitting=True)
            sf.write(f"out_rob/{sid}_c{j}.wav",o["wav"],24000); print("GEN",sid,j,flush=True)
        except Exception as e: print("GENERR",sid,j,repr(e),flush=True)
PY
echo "=== MEASURE (per-sentence best-of-3) ==="
python measure_bon.py out_rob
echo "=== JOB DONE $(date) ==="
