# -*- coding: utf-8 -*-
"""rank-2 step2 measure: identity (resem/ecapa->HER) + Arabic intelligibility (Whisper WER)
for synthesis bar (out_ar_herQ) vs kNN arms A1/A2/A3. Win: kNN matches synth WER, beats synth ecapa."""
import glob, os, json, re, warnings, numpy as np, librosa
warnings.filterwarnings("ignore")
from resemblyzer import VoiceEncoder, preprocess_wav

def loud(w):
    r = np.sqrt(np.mean(w**2)) + 1e-9
    return w * (10**(-23/20) / r)

# identity encoders
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
    print("ecapa unavail:", str(e)[:60], flush=True)
def rsim(p):
    w, _ = librosa.load(p, sr=24000, mono=True); return float(emb(librosa.resample(w, orig_sr=24000, target_sr=16000)) @ herv)

# Arabic ASR (Whisper) for intelligibility
import whisper, jiwer
asr = whisper.load_model("medium")
def norm_ar(t):
    t = re.sub(r'[ً-ْٰـ]', '', t)
    t = t.replace('أ','ا').replace('إ','ا').replace('آ','ا').replace('ى','ي').replace('ة','ه').replace('ؤ','و').replace('ئ','ي')
    t = re.sub(r'[^ء-ي\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()
def wer_of(p, ref):
    try:
        hyp = asr.transcribe(p, language='ar', fp16=False)['text']
    except Exception:
        return 1.0
    r, h = norm_ar(ref), norm_ar(hyp)
    return float(jiwer.wer(r, h)) if r else 1.0
TEXTS = json.load(open("texts.json"))

def grp(d):
    fs = sorted(glob.glob(f"{d}/*.wav"), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    R = [rsim(p) for p in fs]
    E = [esim(p) for p in fs] if ecapa else [float('nan')]*len(fs)
    W = [wer_of(p, TEXTS[int(os.path.splitext(os.path.basename(p))[0])]) for p in fs]
    return float(np.mean(R)), float(np.nanmean(E)), float(np.mean(W))

print("===== SUMMARY (rank-2 step2: identity AND Arabic fluency; win=match synth WER + beat synth ecapa) =====", flush=True)
labels = [("out_ar_herQ", "SYNTH(XTTS-AR)"), ("out_A1", "kNN genQ POOL_EN"),
          ("out_A2", "kNN genQ POOL_AUG"), ("out_A3", "kNN herQ POOL_AUG")]
res = {}
for d, lab in labels:
    if not glob.glob(f"{d}/*.wav"):
        continue
    r, e, w = grp(d); res[lab] = (r, e, w)
    print(f"SUMMARY {lab:18s} resemHER={r:.3f} ecapaHER={e:.3f} WER={w:.3f}", flush=True)
print("SUMMARY NOTE lower WER=more intelligible Arabic; synth WER is the fluency bar; ecapa>0.529 beats synth identity", flush=True)
print("MEASURE_DONE", flush=True)
