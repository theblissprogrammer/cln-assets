#!/bin/bash
set -x; echo "=== VOCSAMPLE (her-vocoder naturalness + A/B bundle) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg zip >/dev/null 2>&1; echo TOOLS_DONE
export ASSETS=$(pwd)
pip install -q "numpy<2" resemblyzer speechbrain vocos auraloss coqui-tts "transformers>=4.57,<5.0" 2>&1 | tail -2
export COQUI_TOS_AGREED=1
python - <<'PY'
import librosa, soundfile as sf, numpy as np, os, glob, warnings, torch, random; warnings.filterwarnings("ignore")
from resemblyzer import VoiceEncoder, preprocess_wav
from speechbrain.inference.speaker import EncoderClassifier
from TTS.api import TTS
from vocos import Vocos
import auraloss
dev="cuda" if torch.cuda.is_available() else "cpu"
def loud(w): r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
# her segments + refs
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
herv=np.mean(embs,0); herv/=np.linalg.norm(herv)
eca=EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",savedir="/tmp/ecapa",run_opts={"device":dev})
def eemb(w16):
    with torch.no_grad(): v=eca.encode_batch(torch.tensor(w16).float().unsqueeze(0).to(dev)).squeeze().cpu().numpy()
    return v/np.linalg.norm(v)
Ee=np.array([eemb(loud(librosa.resample(s,orig_sr=24000,target_sr=16000))) for s in segs]); hereca=np.mean(Ee,0); hereca/=np.linalg.norm(hereca)
def rsim(w): e=enc.embed_utterance(preprocess_wav(loud(w),source_sr=24000)); e/=np.linalg.norm(e); return float(e@herv)
def esim(w): return float(eemb(loud(librosa.resample(w,orig_sr=24000,target_sr=16000)))@hereca)
# naturalness proxies: HF energy fractions + spectral tilt
def natur(w):
    w=w/(np.max(np.abs(w))+1e-9)
    S=np.abs(librosa.stft(w,n_fft=2048))**2; f=librosa.fft_frequencies(sr=24000,n_fft=2048)
    tot=S.sum()+1e-9
    e48=S[(f>=4000)&(f<8000)].sum()/tot; e812=S[(f>=8000)&(f<12000)].sum()/tot
    # spectral tilt: slope of log-power vs log-freq (rough)
    band=(f>200)&(f<10000); lp=10*np.log10(S[band].mean(1)+1e-9); lf=np.log10(f[band]+1e-9)
    tilt=np.polyfit(lf,lp,1)[0]
    return e48*100, e812*100, tilt
# her-real reference naturalness (avg over her segs)
hr=[natur(s) for s in segs[:8]]; hr=np.mean(hr,0)
print(f"NATUR her-REAL: HF4-8={hr[0]:.2f}% HF8-12={hr[1]:.2f}% tilt={hr[2]:.1f}", flush=True)
# XTTS gen
tts=TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(dev); m=tts.synthesizer.tts_model
refs=sorted(glob.glob("hersegs/seg*.wav"))
gl,sp=m.get_conditioning_latents(audio_path=refs,max_ref_length=30,gpt_cond_len=30,gpt_cond_chunk_len=6)
S=[("s1","The morning light came slowly across the quiet harbor as the boats began to stir."),
   ("s2","I never expected to find the answer hidden in such an ordinary place that day."),
   ("s3","She paused at the doorway, listening to the sound of the city waking up below."),
   ("s4","After all these years, the old house still smelled exactly the way I remembered.")]
takes={}
for sid,txt in S:
    takes[sid]=[np.asarray(m.inference(txt,"en",gl,sp,temperature=0.75,enable_text_splitting=True)["wav"]) for _ in range(6)]
    print("GEN",sid,flush=True)
del tts,m; torch.cuda.empty_cache()
# fine-tune her Vocos
vo=Vocos.from_pretrained("charactr/vocos-mel-24khz").to(dev); fe=vo.feature_extractor
voiced=np.concatenate([y[s:e] for s,e in iv]); CH=24000
chunks=[loud(voiced[i:i+CH]) for i in range(0,len(voiced)-CH,CH//2) if len(voiced[i:i+CH])==CH]
for p in vo.parameters(): p.requires_grad_(True)
opt=torch.optim.AdamW(vo.parameters(),lr=2e-4,betas=(0.8,0.99))
mrstft=auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=[512,1024,2048],hop_sizes=[128,256,512],win_lengths=[512,1024,2048]).to(dev)
data=torch.tensor(np.stack(chunks)).float().to(dev); N=data.shape[0]
vo.train()
for step in range(4000):
    wav=data[torch.randint(0,N,(16,))]; mel=fe(wav); recon=vo.head(vo.backbone(mel))
    L=min(recon.shape[-1],wav.shape[-1]); loss=mrstft(recon[...,:L].unsqueeze(1),wav[...,:L].unsqueeze(1))+torch.nn.functional.l1_loss(fe(recon[...,:L]),fe(wav[...,:L]))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(vo.parameters(),1.0); opt.step()
    if step%1000==0: print(f"voc {step} loss={loss.item():.3f}",flush=True)
vo.eval()
def hervoc(w):
    t=torch.tensor(w).float().unsqueeze(0).to(dev)
    with torch.no_grad(): return np.nan_to_num(vo.head(vo.backbone(fe(t))).squeeze().cpu().numpy())
# pick best take per sentence (raw), make her-vocoder version, save both + naturalness
os.makedirs("bundle",exist_ok=True)
for sid in takes:
    scored=[(rsim(w)+esim(w), w) for w in takes[sid]]
    best=max(scored,key=lambda t:t[0])[1]
    hv=hervoc(best)
    sf.write(f"bundle/RAW_{sid}.wav", best, 24000); sf.write(f"bundle/HERVOC_{sid}.wav", hv, 24000)
    nr=natur(best); nh=natur(hv)
    print(f"{sid} RAW r={rsim(best):.3f} e={esim(best):.3f} HF4-8={nr[0]:.2f} HF8-12={nr[1]:.2f} tilt={nr[2]:.1f} | HERVOC r={rsim(hv):.3f} e={esim(hv):.3f} HF4-8={nh[0]:.2f} HF8-12={nh[1]:.2f} tilt={nh[2]:.1f}", flush=True)
sf.write("bundle/her_REAL_reference.wav", librosa.load("ref.wav",sr=24000,mono=True)[0], 24000)
print("BUNDLE_READY", flush=True)
PY
cd bundle && zip -q ab.zip *.wav && cd ..
URL=$(curl -s -F "reqtype=fileupload" -F "fileToUpload=@bundle/ab.zip" https://catbox.moe/user/api.php)
echo "BUNDLE_URL=$URL"
echo "=== JOB DONE $(date) ==="
