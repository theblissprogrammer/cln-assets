import numpy as np, librosa, glob, os, sys, re, warnings
warnings.filterwarnings("ignore")
from resemblyzer import VoiceEncoder, preprocess_wav
OUTDIR = sys.argv[1] if len(sys.argv)>1 else "out"

def loud(w):
    r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)

y,sr=librosa.load('her_audio.wav',sr=16000,mono=True)
iv=librosa.effects.split(y,top_db=30)
segs=[y[s:e] for s,e in iv if (e-s)>1.5*16000]
enc=VoiceEncoder(verbose=False)
E=np.array([ (lambda e:e/np.linalg.norm(e))(enc.embed_utterance(preprocess_wav(loud(s),source_sr=16000))) for s in segs])
loo=[]
for i in range(len(E)):
    c=np.mean(np.delete(E,i,0),0); c/=np.linalg.norm(c); loo.append(float(E[i]@c))
loo=np.array(loo); R_MEAN,R_STD=loo.mean(),loo.std()
herv=np.mean(E,0); herv/=np.linalg.norm(herv)
def rsim(p):
    w,s=librosa.load(p,sr=24000,mono=True); e=enc.embed_utterance(preprocess_wav(loud(w),source_sr=24000)); e/=np.linalg.norm(e); return float(e@herv)
import torch
from speechbrain.inference.speaker import EncoderClassifier
dev="cuda" if torch.cuda.is_available() else "cpu"
ec=EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",savedir="/tmp/ecapa",run_opts={"device":dev})
def eemb(w16):
    with torch.no_grad(): v=ec.encode_batch(torch.tensor(w16).float().unsqueeze(0).to(dev)).squeeze().cpu().numpy()
    return v/np.linalg.norm(v)
Ee=np.array([eemb(loud(s)) for s in segs])
eloo=[]
for i in range(len(Ee)):
    c=np.mean(np.delete(Ee,i,0),0); c/=np.linalg.norm(c); eloo.append(float(Ee[i]@c))
eloo=np.array(eloo); EC_MEAN,EC_STD=eloo.mean(),eloo.std()
hereca=np.mean(Ee,0); hereca/=np.linalg.norm(hereca)
def esim(p):
    w,s=librosa.load(p,sr=16000,mono=True); v=eemb(loud(w)); return float(v@hereca)
print(f"[real] resemblyzer mean={R_MEAN:.3f} std={R_STD:.3f} | ecapa mean={EC_MEAN:.3f} std={EC_STD:.3f}", flush=True)

# group candidates by sentence id: filename like s3_c2.wav -> group s3
groups={}
for p in sorted(glob.glob(f"{OUTDIR}/*.wav")):
    g=re.match(r"(s\d+)", os.path.basename(p))
    if not g: continue
    groups.setdefault(g.group(1),[]).append(p)
sel_by_e_rep_r=[]; sel_by_r_rep_e=[]; all_r=[]; all_e=[]
for g in sorted(groups):
    cand=[(p, rsim(p), esim(p)) for p in groups[g]]
    for _,r,e in cand: all_r.append(r); all_e.append(e)
    # pick best by ECAPA, report its resemblyzer (cross-validated)
    be=max(cand,key=lambda t:t[2]); sel_by_e_rep_r.append(be[1])
    # pick best by resemblyzer, report its ECAPA
    br=max(cand,key=lambda t:t[1]); sel_by_r_rep_e.append(br[2])
    print(f"BON {g}: n={len(cand)} | best-by-ecapa -> resemblyzer={be[1]:.3f} ecapa={be[2]:.3f} | best-by-resem -> resemblyzer={br[1]:.3f} ecapa={br[2]:.3f}", flush=True)
print(f"BON SUMMARY: all-candidate avg resemblyzer={np.mean(all_r):.3f} ecapa={np.mean(all_e):.3f}", flush=True)
print(f"BON SUMMARY: select-by-ecapa -> resemblyzer avg={np.mean(sel_by_e_rep_r):.3f} (real {R_MEAN:.3f})", flush=True)
print(f"BON SUMMARY: select-by-resem -> ecapa avg={np.mean(sel_by_r_rep_e):.3f} (real {EC_MEAN:.3f})", flush=True)
print("[measure_bon] DONE", flush=True)
