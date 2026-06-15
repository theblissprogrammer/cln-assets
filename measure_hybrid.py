# -*- coding: utf-8 -*-
"""HYBRID measure: identity (resem/ecapa->HER) + emotion (SER arousal) + fluency (Whisper WER)
for synth-expr-Arabic (emotion bar) / kNN gen (exemplar neutral) / kNN expr (HYBRID)."""
import glob, os, json, re, warnings, numpy as np, librosa
warnings.filterwarnings("ignore")
from ser_lib import load_ser
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
    print("ecapa unavail:", str(e)[:50], flush=True)
def rsim(p):
    w, _ = librosa.load(p, sr=24000, mono=True); return float(emb(librosa.resample(w, orig_sr=24000, target_sr=16000)) @ herv)

ser = load_ser()
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

# her expressive reference arousal (emotion target)
her_ar = np.mean([ser(loud(librosa.resample(s.astype(np.float64), orig_sr=16000, target_sr=16000)))[0] for s in hs[:12]])

def grp(d):
    fs = sorted(glob.glob(f"{d}/*.wav"), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    R = [rsim(p) for p in fs]; E = [esim(p) for p in fs] if ecapa else [float('nan')]*len(fs)
    A = [ser(loud(librosa.load(p, sr=16000, mono=True)[0]))[0] for p in fs]
    W = [wer_of(p, TEXTS[int(os.path.splitext(os.path.basename(p))[0])]) for p in fs]
    return float(np.mean(R)), float(np.nanmean(E)), float(np.mean(A)), float(np.mean(W))

print("===== SUMMARY (HYBRID: her identity + her emotion + fluent Arabic?) =====", flush=True)
print(f"SUMMARY HER_real_arousal_ref={her_ar:.3f}", flush=True)
for d, lab in [("out_exprAr", "SYNTH-expr (emotion bar)"), ("out_knn_gen", "kNN gen (exemplar neutral)"), ("out_knn_expr", "kNN expr (HYBRID)")]:
    if not glob.glob(f"{d}/*.wav"):
        continue
    r, e, a, w = grp(d)
    print(f"SUMMARY {lab:28s} resemHER={r:.3f} ecapaHER={e:.3f} arousal={a:.3f} WER={w:.3f}", flush=True)
print("SUMMARY WANT hybrid: ecapaHER~0.56 (>synth) AND arousal~her_ref (emotion) AND WER low (fluent)", flush=True)
print("MEASURE_DONE", flush=True)
