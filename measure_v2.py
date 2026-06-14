import numpy as np, librosa, glob, os, sys, warnings
warnings.filterwarnings("ignore")
from resemblyzer import VoiceEncoder, preprocess_wav

OUTDIR = sys.argv[1] if len(sys.argv)>1 else "out"
TAG    = sys.argv[2] if len(sys.argv)>2 else "GEN"

def loud(w):
    rms=np.sqrt(np.mean(w**2))+1e-9
    return w*(10**(-23/20)/rms)

# ---------- her segments ----------
y,sr=librosa.load('her_audio.wav',sr=16000,mono=True)
iv=librosa.effects.split(y,top_db=30)
segs=[y[s:e] for s,e in iv if (e-s)>1.5*16000]
print(f"[measure] {len(segs)} her segments", flush=True)

# ---------- resemblyzer ----------
enc=VoiceEncoder(verbose=False)
E=np.array([ (lambda e: e/np.linalg.norm(e))(enc.embed_utterance(preprocess_wav(loud(s),source_sr=16000))) for s in segs])
loo=[]
for i in range(len(E)):
    c=np.mean(np.delete(E,i,0),0); c/=np.linalg.norm(c); loo.append(float(E[i]@c))
loo=np.array(loo); R_MEAN,R_STD=loo.mean(),loo.std()
herv=np.mean(E,0); herv/=np.linalg.norm(herv)
print(f"[resemblyzer] real held-out: mean={R_MEAN:.3f} std={R_STD:.3f} p50={np.percentile(loo,50):.3f} p90={np.percentile(loo,90):.3f}", flush=True)
def rsim(p):
    w,s=librosa.load(p,sr=24000,mono=True)
    e=enc.embed_utterance(preprocess_wav(loud(w),source_sr=24000)); e/=np.linalg.norm(e); return float(e@herv)

# ---------- ECAPA anchor (optional) ----------
ecapa=None
try:
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    dev="cuda" if torch.cuda.is_available() else "cpu"
    ecapa=EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/tmp/ecapa", run_opts={"device":dev})
    def eemb(w16):
        import torch as T
        with T.no_grad():
            v=ecapa.encode_batch(T.tensor(w16).float().unsqueeze(0).to(dev)).squeeze().cpu().numpy()
        return v/np.linalg.norm(v)
    Ee=np.array([eemb(loud(s)) for s in segs])
    eloo=[]
    for i in range(len(Ee)):
        c=np.mean(np.delete(Ee,i,0),0); c/=np.linalg.norm(c); eloo.append(float(Ee[i]@c))
    eloo=np.array(eloo); EC_MEAN,EC_STD=eloo.mean(),eloo.std()
    hereca=np.mean(Ee,0); hereca/=np.linalg.norm(hereca)
    print(f"[ecapa] real held-out: mean={EC_MEAN:.3f} std={EC_STD:.3f}", flush=True)
    def esim(p):
        w,s=librosa.load(p,sr=16000,mono=True); v=eemb(loud(w)); return float(v@hereca)
except Exception as e:
    print(f"[ecapa] UNAVAILABLE: {e}", flush=True)

# ---------- score outputs ----------
files=sorted(glob.glob(f"{OUTDIR}/*.wav"))
print(f"[measure] scoring {len(files)} files in {OUTDIR}", flush=True)
for p in files:
    try:
        r=rsim(p); zr=(r-R_MEAN)/R_STD; pct=(loo<r).mean()*100
        line=f"MEASURE {TAG} {os.path.basename(p)} -> resemblyzer={r:.3f} (z={zr:+.2f} p{pct:.0f})"
        if ecapa is not None:
            try:
                ec=esim(p); zc=(ec-EC_MEAN)/EC_STD
                line+=f" | ecapa={ec:.3f} (z={zc:+.2f})"
            except Exception as e: line+=f" | ecapa_err={e}"
        print(line, flush=True)
    except Exception as e:
        print(f"MEASURE {TAG} {os.path.basename(p)} ERROR {e}", flush=True)
print("[measure] DONE", flush=True)
