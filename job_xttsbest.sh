#!/bin/bash
set -x; echo "=== XTTS-BEST (ref-selection) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1; echo FFMPEG_DONE
export ASSETS=$(pwd)
pip install -q "numpy<2" resemblyzer speechbrain 2>&1 | tail -1
pip install -q coqui-tts "transformers>=4.57,<5.0" 2>&1 | tail -2
export COQUI_TOS_AGREED=1
echo "=== slice + rank her segments by centroid proximity ==="
python - <<'PY'
import librosa, soundfile as sf, numpy as np, os, warnings; warnings.filterwarnings("ignore")
from resemblyzer import VoiceEncoder, preprocess_wav
os.makedirs("hersegs",exist_ok=True)
y,sr=librosa.load("her_audio.wav",sr=24000,mono=True)
iv=librosa.effects.split(y,top_db=30)
segs=[]; buf=[]; cur=0
for s,e in iv:
    buf.append(y[s:e]); cur+=e-s
    if cur>9*sr: segs.append(np.concatenate(buf)); buf=[]; cur=0
if buf: segs.append(np.concatenate(buf))
enc=VoiceEncoder(verbose=False)
def loud(w):
    r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
embs=[enc.embed_utterance(preprocess_wav(loud(s),source_sr=24000)) for s in segs]
embs=[e/np.linalg.norm(e) for e in embs]
c=np.mean(embs,0); c/=np.linalg.norm(c)
scores=[float(e@c) for e in embs]
order=np.argsort(scores)[::-1]
print("SEG SCORES", [round(scores[i],3) for i in order[:12]], flush=True)
for rank,i in enumerate(order):
    sf.write(f"hersegs/seg{rank:02d}_{scores[i]:.3f}.wav", loud(segs[i]), 24000)
print("NSEGS", len(segs), flush=True)
PY
mkdir -p out_all out_top
echo "=== XTTS infer: ALL vs TOP-central refs ==="
python - <<'PY'
import glob, torch, warnings; warnings.filterwarnings("ignore")
from TTS.api import TTS
tts=TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda" if torch.cuda.is_available() else "cpu")
m=tts.synthesizer.tts_model
allsegs=sorted(glob.glob("hersegs/seg*.wav"))
top=allsegs[:4]            # 4 most-central segments
allr=allsegs[:8]           # up to 8 segments
import soundfile as sf
sents=[("s1","The morning light came slowly across the quiet harbor as the boats began to stir."),
       ("s2","I never expected to find the answer hidden in such an ordinary place that day."),
       ("s3","We should leave before the rain starts, or we will be stuck here all night long."),
       ("s4","She paused at the doorway, listening to the sound of the city waking up below."),
       ("s5","Honestly, I think we made the right choice, even if it did not feel that way then."),
       ("s6","There is something about late autumn that makes everything feel a little slower."),
       ("s7","Could you tell me whether the train to the coast still leaves from platform nine?"),
       ("s8","After all these years, the old house still smelled exactly the way I remembered.")]
for tag,refs,outdir in [("ALL",allr,"out_all"),("TOP",top,"out_top")]:
    gpt_lat, spk = m.get_conditioning_latents(audio_path=refs, max_ref_length=30, gpt_cond_len=30, gpt_cond_chunk_len=6)
    for name,txt in sents:
        try:
            out=m.inference(txt,"en",gpt_lat,spk,temperature=0.7,enable_text_splitting=True)
            sf.write(f"{outdir}/{name}.wav", out["wav"], 24000); print("GEN",tag,name,flush=True)
        except Exception as e:
            print("GENERR",tag,name,repr(e),flush=True)
PY
echo "=== MEASURE ==="
python measure_v2.py out_all XTTS-ALL
python measure_v2.py out_top XTTS-TOP
echo "=== JOB DONE $(date) ==="
