#!/bin/bash
set -x; echo "=== XLINGUAL (her voice -> Arabic) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1; echo FFMPEG_DONE
export ASSETS=$(pwd)
pip install -q "numpy<2" resemblyzer speechbrain 2>&1 | tail -1
pip install -q coqui-tts "transformers>=4.57,<5.0" 2>&1 | tail -2
export COQUI_TOS_AGREED=1
echo "=== rank her segments, keep top-4 (English conditioning) ==="
python - <<'PY'
import librosa, soundfile as sf, numpy as np, os, warnings; warnings.filterwarnings("ignore")
from resemblyzer import VoiceEncoder, preprocess_wav
os.makedirs("hersegs",exist_ok=True)
y,sr=librosa.load("her_audio.wav",sr=24000,mono=True); iv=librosa.effects.split(y,top_db=30)
segs=[]; buf=[]; cur=0
for s,e in iv:
    buf.append(y[s:e]); cur+=e-s
    if cur>9*sr: segs.append(np.concatenate(buf)); buf=[]; cur=0
if buf: segs.append(np.concatenate(buf))
enc=VoiceEncoder(verbose=False)
def loud(w): r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
embs=[(lambda e:e/np.linalg.norm(e))(enc.embed_utterance(preprocess_wav(loud(s),source_sr=24000))) for s in segs]
c=np.mean(embs,0); c/=np.linalg.norm(c); sc=[float(e@c) for e in embs]
order=np.argsort(sc)[::-1]
for rank,i in enumerate(order[:4]): sf.write(f"hersegs/seg{rank:02d}.wav", loud(segs[i]), 24000)
print("TOP4 scores", [round(sc[i],3) for i in order[:4]], flush=True)
PY
mkdir -p out_ar out_en
echo "=== XTTS cross-lingual: Arabic (+English control), best-of-3, TOP-4 EN refs ==="
python - <<'PY'
# -*- coding: utf-8 -*-
import glob, torch, warnings, soundfile as sf; warnings.filterwarnings("ignore")
from TTS.api import TTS
tts=TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda" if torch.cuda.is_available() else "cpu")
m=tts.synthesizer.tts_model
refs=sorted(glob.glob("hersegs/seg*.wav"))
gpt_lat, spk = m.get_conditioning_latents(audio_path=refs, max_ref_length=30, gpt_cond_len=30, gpt_cond_chunk_len=6)
ar=[("s1","صباح الخير، كيف حالك اليوم؟ أتمنى أن يكون يومك جميلاً."),
    ("s2","لم أتوقع أبداً أن أجد الإجابة في مكان عادي مثل هذا."),
    ("s3","علينا أن نغادر قبل أن يبدأ المطر، وإلا سنبقى هنا طوال الليل."),
    ("s4","بعد كل هذه السنوات، ما زال البيت القديم على حاله تماماً."),
    ("s5","هل يمكنك أن تخبرني إن كان القطار لا يزال يغادر من الرصيف التاسع؟")]
en=[("s1","The morning light came slowly across the quiet harbor."),
    ("s2","I never expected to find the answer in such an ordinary place.")]
for lang,sents,outdir in [("ar",ar,"out_ar"),("en",en,"out_en")]:
    for sid,txt in sents:
        for j in range(3):
            try:
                out=m.inference(txt,lang,gpt_lat,spk,temperature=0.75,enable_text_splitting=True)
                sf.write(f"{outdir}/{sid}_c{j}.wav", out["wav"], 24000); print("GEN",lang,sid,j,flush=True)
            except Exception as e:
                print("GENERR",lang,sid,j,repr(e),flush=True)
PY
echo "=== MEASURE (clone vs her ENGLISH centroid = cross-lingual identity preservation) ==="
echo "--- ARABIC ---"; python measure_bon.py out_ar
echo "--- ENGLISH CONTROL ---"; python measure_bon.py out_en
echo "=== JOB DONE $(date) ==="
