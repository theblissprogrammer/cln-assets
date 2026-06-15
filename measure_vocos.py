# -*- coding: utf-8 -*-
"""Measure the 24kHz Vocos exemplar render: resem/ecapa->HER + Arabic WER, vs the 16kHz baseline.
Export clips to litterbox for Ahmed's ear."""
import glob, os, json, re, subprocess, warnings, numpy as np, librosa
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
    w, _ = librosa.load(p, sr=24000, mono=True); return float(emb(librosa.resample(w, orig_sr=24000, target_sr=16000)) @ herv)
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
print("===== SUMMARY (24kHz Vocos exemplar render vs 16kHz baseline) =====", flush=True)
rv, ev, wv = grp("out_vocos")
print(f"SUMMARY VOCOS_24k    resemHER={rv:.3f} ecapaHER={ev:.3f} WER={wv:.3f}", flush=True)
print(f"SUMMARY BASELINE_16k alpha1 resemHER~0.880 ecapaHER~0.613 WER~0.195 (the bar)", flush=True)
rs, es, ws = grp("out_exprAr")
print(f"SUMMARY SYNTH_expr   resemHER={rs:.3f} ecapaHER={es:.3f} WER={ws:.3f}", flush=True)

def upload(path):
    try:
        r = subprocess.run(["curl", "-s", "--max-time", "60", "-F", "reqtype=fileupload", "-F", "time=72h",
                            "-F", f"fileToUpload=@{path}", "https://litterbox.catbox.moe/resources/internals/api.php"],
                           capture_output=True, text=True, timeout=70)
        return r.stdout.strip() if r.stdout.strip().startswith("http") else None
    except Exception:
        return None
for nm, src in [("vocos24_arabic", "out_vocos/3.wav"), ("synth_arabic", "out_exprAr/3.wav")]:
    if os.path.exists(src):
        subprocess.run(["ffmpeg", "-y", "-t", "5", "-i", src, "-ar", "24000", "-ac", "1", "-b:a", "64k", nm+".mp3"], capture_output=True)
        u = upload(nm+".mp3")
        print(f"URL {nm} {u}", flush=True)
print("MEASURE_DONE", flush=True)
