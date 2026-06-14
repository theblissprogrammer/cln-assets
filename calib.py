import numpy as np, librosa, warnings
warnings.filterwarnings("ignore")
from resemblyzer import VoiceEncoder, preprocess_wav
enc=VoiceEncoder(verbose=False)

y,sr=librosa.load('her_audio.wav',sr=16000,mono=True)
iv=librosa.effects.split(y,top_db=30)
# all reasonably long segments
segs=[y[s:e] for s,e in iv if (e-s)>1.5*16000]
print(f"n real segments: {len(segs)}", flush=True)
E=[enc.embed_utterance(preprocess_wav(s,source_sr=16000)) for s in segs]
E=np.array([e/np.linalg.norm(e) for e in E])

# proper leave-one-out: each seg vs centroid of all OTHERS
loo=[]
for i in range(len(E)):
    c=np.mean(np.delete(E,i,0),0); c/=np.linalg.norm(c)
    loo.append(float(E[i]@c))
loo=np.array(loo)
print(f"LEAVE-ONE-OUT real held-out clip -> her centroid:", flush=True)
print(f"  mean={loo.mean():.3f} std={loo.std():.3f} min={loo.min():.3f} max={loo.max():.3f}", flush=True)
for p in [10,25,50,75,90]:
    print(f"  p{p}={np.percentile(loo,p):.3f}", flush=True)

# full centroid for scoring clones
herv=np.mean(E,0); herv/=np.linalg.norm(herv)
def simfile(p, loud=False):
    w,s=librosa.load(p,sr=24000,mono=True)
    if loud:
        rms=np.sqrt(np.mean(w**2))+1e-9; w=w*(10**(-23/20)/rms)
    e=enc.embed_utterance(preprocess_wav(w,source_sr=24000)); e/=np.linalg.norm(e); return float(e@herv)
print("\nCLONES vs full centroid:", flush=True)
for p in ["src_f5.wav","src_cb.wav","ref.wav"]:
    print(f"  {p:12s} raw={simfile(p):.3f}  loudnorm={simfile(p,True):.3f}", flush=True)

# where does the clone sit in the real held-out distribution? (z-score / percentile)
f5=simfile("src_f5.wav",True)
pct=(loo<f5).mean()*100
print(f"\nloudnorm F5 clone = {f5:.3f}  ->  percentile {pct:.0f}% of real held-out clips; z={(f5-loo.mean())/loo.std():+.2f}", flush=True)
print("DONE", flush=True)
