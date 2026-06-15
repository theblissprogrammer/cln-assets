#!/bin/bash
set -x; echo "=== VOC (her-fine-tuned vocoder) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1; echo FFMPEG_DONE
export ASSETS=$(pwd)
pip install -q "numpy<2" resemblyzer speechbrain vocos auraloss 2>&1 | tail -1
pip install -q coqui-tts "transformers>=4.57,<5.0" 2>&1 | tail -2
export COQUI_TOS_AGREED=1
echo "=== STEP1: generate XTTS clones (5x5) ==="
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
    for j in range(5):
        o=m.inference(txt,"en",gl,sp,temperature=0.75,enable_text_splitting=True)
        sf.write(f"out_raw/{sid}_c{j}.wav", np.asarray(o["wav"]),24000)
    print("GEN",sid,flush=True)
PY
echo "=== STEP2: fine-tune Vocos on her ==="
python - <<'PY'
import librosa, numpy as np, torch, glob, os, warnings, random; warnings.filterwarnings("ignore")
import soundfile as sf
from vocos import Vocos
import auraloss
dev="cuda" if torch.cuda.is_available() else "cpu"
def loud(w): r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
# her chunks (1s) from voiced regions
y,sr=librosa.load("her_audio.wav",sr=24000,mono=True); iv=librosa.effects.split(y,top_db=30)
voiced=np.concatenate([y[s:e] for s,e in iv])
CH=24000; chunks=[voiced[i:i+CH] for i in range(0,len(voiced)-CH,CH//2)]
chunks=[loud(c) for c in chunks if len(c)==CH]
print("her chunks", len(chunks), flush=True)
vo=Vocos.from_pretrained("charactr/vocos-mel-24khz").to(dev)
import copy
gen_vo=Vocos.from_pretrained("charactr/vocos-mel-24khz").to(dev); gen_vo.eval()  # frozen generic copy
for p in vo.parameters(): p.requires_grad=False
for p in list(vo.backbone.parameters())+list(vo.head.parameters()): p.requires_grad=True
opt=torch.optim.AdamW([p for p in vo.parameters() if p.requires_grad], lr=2e-4, betas=(0.8,0.99))
mrstft=auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=[512,1024,2048],hop_sizes=[128,256,512],win_lengths=[512,1024,2048]).to(dev)
fe=vo.feature_extractor
data=torch.tensor(np.stack(chunks)).float().to(dev)
N=data.shape[0]; STEPS=3000; B=16
vo.train()
for step in range(STEPS):
    idx=torch.randint(0,N,(B,))
    wav=data[idx]
    mel=fe(wav)
    recon=vo.decode(mel)
    L=min(recon.shape[-1],wav.shape[-1])
    rec=recon[...,:L]; tgt=wav[...,:L]
    loss=mrstft(rec.unsqueeze(1),tgt.unsqueeze(1))
    mel_l1=torch.nn.functional.l1_loss(fe(rec),fe(tgt))
    total=loss+mel_l1
    opt.zero_grad(); total.backward(); opt.step()
    if step%500==0: print(f"voc step {step} mrstft={loss.item():.3f} mel={mel_l1.item():.3f}",flush=True)
vo.eval()
# re-vocode XTTS output through HER vocos and GENERIC vocos
os.makedirs("out_hvoc",exist_ok=True); os.makedirs("out_gvoc",exist_ok=True)
def revoc(model,p,outdir):
    w,_=librosa.load(p,sr=24000,mono=True); t=torch.tensor(w).float().unsqueeze(0).to(dev)
    with torch.no_grad(): rec=model.decode(model.feature_extractor(t)).squeeze().cpu().numpy()
    sf.write(f"{outdir}/{os.path.basename(p)}", rec, 24000)
for p in glob.glob("out_raw/*.wav"):
    revoc(vo,p,"out_hvoc"); revoc(gen_vo,p,"out_gvoc")
# sanity: her real seg through her-vocos
print("REVOC_DONE",flush=True)
PY
echo "=== MEASURE: raw vs generic-vocos vs HER-vocos (best-of-5) ==="
echo "--- RAW XTTS ---";        python measure_bon.py out_raw
echo "--- GENERIC VOCOS ---";   python measure_bon.py out_gvoc
echo "--- HER VOCOS ---";       python measure_bon.py out_hvoc
echo "=== JOB DONE $(date) ==="
