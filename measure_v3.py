import numpy as np, librosa, glob, os, sys, warnings
warnings.filterwarnings("ignore")
from resemblyzer import VoiceEncoder, preprocess_wav
OUTDIR = sys.argv[1] if len(sys.argv)>1 else "out"
TAG    = sys.argv[2] if len(sys.argv)>2 else "GEN"
def loud(w):
    r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
y,sr=librosa.load('her_audio.wav',sr=16000,mono=True)
iv=librosa.effects.split(y,top_db=30)
segs=[y[s:e] for s,e in iv if (e-s)>1.5*16000]

# resemblyzer
enc=VoiceEncoder(verbose=False)
E=np.array([(lambda e:e/np.linalg.norm(e))(enc.embed_utterance(preprocess_wav(loud(s),source_sr=16000))) for s in segs])
def loo_stats(M):
    v=[]
    for i in range(len(M)):
        c=np.mean(np.delete(M,i,0),0); c/=np.linalg.norm(c); v.append(float(M[i]@c))
    return np.mean(v),np.std(v)
R_MEAN,R_STD=loo_stats(E); herv=np.mean(E,0); herv/=np.linalg.norm(herv)
def rsim(p):
    w,s=librosa.load(p,sr=24000,mono=True); e=enc.embed_utterance(preprocess_wav(loud(w),source_sr=24000)); e/=np.linalg.norm(e); return float(e@herv)
import torch
dev="cuda" if torch.cuda.is_available() else "cpu"
# ECAPA
from speechbrain.inference.speaker import EncoderClassifier
eca=EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",savedir="/tmp/ecapa",run_opts={"device":dev})
def e_emb(w16):
    with torch.no_grad(): v=eca.encode_batch(torch.tensor(w16).float().unsqueeze(0).to(dev)).squeeze().cpu().numpy()
    return v/np.linalg.norm(v)
Ee=np.array([e_emb(loud(s)) for s in segs]); EC_MEAN,EC_STD=loo_stats(Ee); hereca=np.mean(Ee,0); hereca/=np.linalg.norm(hereca)
def esim(p):
    w,s=librosa.load(p,sr=16000,mono=True); return float(e_emb(loud(w))@hereca)
# WavLM-SV (third, stricter encoder)
wavlm=None
try:
    from transformers import AutoFeatureExtractor, WavLMForXVector
    wfe=AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    wm=WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()
    def w_emb(w16):
        i=wfe([w16],sampling_rate=16000,return_tensors="pt",padding=True)
        with torch.no_grad(): v=wm(**{k:t.to(dev) for k,t in i.items()}).embeddings.squeeze().cpu().numpy()
        return v/np.linalg.norm(v)
    Ew=np.array([w_emb(loud(s)) for s in segs]); W_MEAN,W_STD=loo_stats(Ew); herw=np.mean(Ew,0); herw/=np.linalg.norm(herw)
    def wsim(p):
        w,s=librosa.load(p,sr=16000,mono=True); return float(w_emb(loud(w))@herw)
    wavlm=True
    print(f"[real] resemblyzer {R_MEAN:.3f}±{R_STD:.3f} | ecapa {EC_MEAN:.3f}±{EC_STD:.3f} | wavlm {W_MEAN:.3f}±{W_STD:.3f}", flush=True)
except Exception as e:
    print(f"[wavlm] UNAVAIL {e}", flush=True)
    print(f"[real] resemblyzer {R_MEAN:.3f}±{R_STD:.3f} | ecapa {EC_MEAN:.3f}±{EC_STD:.3f}", flush=True)
for p in sorted(glob.glob(f"{OUTDIR}/*.wav")):
    r=rsim(p); e=esim(p)
    line=f"MEASURE3 {TAG} {os.path.basename(p)} -> resemblyzer={r:.3f}(z{(r-R_MEAN)/R_STD:+.2f}) ecapa={e:.3f}(z{(e-EC_MEAN)/EC_STD:+.2f})"
    if wavlm:
        wv=wsim(p); line+=f" wavlm={wv:.3f}(z{(wv-W_MEAN)/W_STD:+.2f})"
    print(line, flush=True)
print("[measure_v3] DONE", flush=True)
