# -*- coding: utf-8 -*-
"""alpha-sweep measure: identity (resem/ecapa->HER) + Arabic WER per alpha. Find the Pareto point."""
import glob, os, json, re, warnings, numpy as np, librosa
warnings.filterwarnings("ignore")
from resemblyzer import VoiceEncoder, preprocess_wav
def loud(w):
    r = np.sqrt(np.mean(w**2)) + 1e-9
    return w * (10**(-23/20) / r)
enc = VoiceEncoder(verbose=False)
def emb(w16):
    e = enc.embed_utterance(preprocess_wav(loud(w16), source_sr=16000)); return e / np.linalg.norm(e)
y16, _ = librosa.load("her_audio.wav", sr=16000, mono=True)
iv = librosa.effects.split(y16, top_db=30)
hs = [y16[s:e] for s, e in iv if (e - s) > 1.5*16000]
herv = np.mean([emb(s) for s in hs], 0); herv /= np.linalg.norm(herv)
ecapa = None
try:
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="/tmp/ecapa", run_opts={"device": dev})
    def eemb(w16):
        import torch as T
        with T.no_grad():
            v = ec.encode_batch(T.tensor(w16).float().unsqueeze(0).to(dev)).squeeze().cpu().numpy()
        return v / np.linalg.norm(v)
    hereca = np.mean([eemb(loud(s)) for s in hs], 0); hereca /= np.linalg.norm(hereca)
    def esim(p):
        w, _ = librosa.load(p, sr=16000, mono=True); return float(eemb(loud(w)) @ hereca)
    ecapa = True
except Exception as e:
    print("ecapa unavail", str(e)[:40], flush=True)
def rsim(p):
    w, _ = librosa.load(p, sr=16000, mono=True); return float(emb(w) @ herv)
import whisper, jiwer
asr = whisper.load_model("medium")
def norm_ar(t):
    t = re.sub(r'[ً-ْٰـ]', '', t).replace('أ','ا').replace('إ','ا').replace('آ','ا').replace('ى','ي').replace('ة','ه').replace('ؤ','و').replace('ئ','ي')
    return re.sub(r'\s+', ' ', re.sub(r'[^ء-ي\s]', ' ', t)).strip()
def wer_of(p, ref):
    try: hyp = asr.transcribe(p, language='ar', fp16=False)['text']
    except Exception: return 1.0
    r, h = norm_ar(ref), norm_ar(hyp); return float(jiwer.wer(r, h)) if r else 1.0
TEXTS = json.load(open("texts.json"))
def grp(d):
    fs = sorted(glob.glob(f"{d}/*.wav"), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    R = [rsim(p) for p in fs]; E = [esim(p) for p in fs] if ecapa else [float('nan')]*len(fs)
    W = [wer_of(p, TEXTS[int(os.path.splitext(os.path.basename(p))[0])]) for p in fs]
    return float(np.mean(R)), float(np.nanmean(E)), float(np.mean(W))
print("===== SUMMARY (alpha sweep: 0=fluent synthesis end, 1=her exemplar end; find high-id + fluent) =====", flush=True)
for a in [0, 33, 50, 67, 100]:
    d = f"out_a{a}"
    if not glob.glob(f"{d}/*.wav"):
        continue
    r, e, w = grp(d)
    print(f"SUMMARY alpha={a/100:.2f} resemHER={r:.3f} ecapaHER={e:.3f} WER={w:.3f}", flush=True)
print("MEASURE_DONE", flush=True)
