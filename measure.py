import numpy as np, librosa, glob, os, warnings
warnings.filterwarnings("ignore")
from resemblyzer import VoiceEncoder, preprocess_wav
enc=VoiceEncoder(verbose=False)
y,sr=librosa.load('her_audio.m4a',sr=16000,mono=True)
iv=librosa.effects.split(y,top_db=30)
segs=[y[s:e] for s,e in iv if (e-s)>2*16000][:8]
embs=[enc.embed_utterance(preprocess_wav(s,source_sr=16000)) for s in segs]; embs=[e/np.linalg.norm(e) for e in embs]
herv=np.mean(embs,0); herv/=np.linalg.norm(herv)
print("MEASURE her-floor(real seg->her):", round(float(np.mean([e@herv for e in embs])),3))
def sim(p):
    e=enc.embed_utterance(preprocess_wav(p)); e/=np.linalg.norm(e); return float(e@herv)
for tag,d in [("ZEROSHOT","out_zs"),("FINETUNED","out_ft")]:
    for p in sorted(glob.glob(f"{d}/*.wav")):
        print(f"MEASURE {tag} {os.path.basename(p)} -> her: {sim(p):.3f}",flush=True)
