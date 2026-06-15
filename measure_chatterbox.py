# -*- coding: utf-8 -*-
"""Measure Chatterbox arms: resem/ecapa->HER + UTMOS + WER vs XTTS baseline. Export best to litterbox."""
import glob, os, json, re, subprocess, warnings, numpy as np, librosa, torch
warnings.filterwarnings("ignore")
from resemblyzer import VoiceEncoder, preprocess_wav
def loud(w):
    r = np.sqrt(np.mean(w**2)) + 1e-9
    return w * (10**(-23/20) / r)
enc = VoiceEncoder(verbose=False)
def emb(w16):
    e = enc.embed_utterance(preprocess_wav(loud(w16), source_sr=16000)); return e/np.linalg.norm(e)
y16, _ = librosa.load("her_audio.wav", sr=16000, mono=True)
iv = librosa.effects.split(y16, top_db=30)
hs = [y16[s:e] for s, e in iv if (e-s) > 1.5*16000]
herv = np.mean([emb(s) for s in hs], 0); herv /= np.linalg.norm(herv)
from speechbrain.inference.speaker import EncoderClassifier
dev = "cuda" if torch.cuda.is_available() else "cpu"
ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="/tmp/ecapa", run_opts={"device": dev})
def eemb(w16):
    with torch.no_grad():
        v = ec.encode_batch(torch.tensor(w16).float().unsqueeze(0).to(dev)).squeeze().cpu().numpy()
    return v/np.linalg.norm(v)
hereca = np.mean([eemb(loud(s)) for s in hs], 0); hereca /= np.linalg.norm(hereca)
ut = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True); ut.eval()
def utmos(p):
    w, _ = librosa.load(p, sr=16000, mono=True)
    with torch.no_grad(): return float(ut(torch.tensor(w).float().unsqueeze(0), 16000))
def rsim(p):
    w, _ = librosa.load(p, sr=24000, mono=True); return float(emb(librosa.resample(w, orig_sr=24000, target_sr=16000)) @ herv)
def esim(p):
    w, _ = librosa.load(p, sr=16000, mono=True); return float(eemb(loud(w)) @ hereca)
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
    if not fs: return None
    R = [rsim(p) for p in fs]; E = [esim(p) for p in fs]; U = [utmos(p) for p in fs]
    W = [wer_of(p, TEXTS[int(os.path.splitext(os.path.basename(p))[0])]) for p in fs]
    return float(np.mean(R)), float(np.mean(E)), float(np.mean(U)), float(np.mean(W))
print("===== SUMMARY (Chatterbox-ML Arabic vs XTTS baseline resem0.860/ecapa0.516/UTMOS2.92/WER0.02) =====", flush=True)
results = {}
for d in sorted(glob.glob("out_cb_*")):
    g = grp(d)
    if g: results[d] = g; print(f"SUMMARY {d:14s} resemHER={g[0]:.3f} ecapaHER={g[1]:.3f} UTMOS={g[2]:.2f} WER={g[3]:.3f}", flush=True)
def upload(p):
    try:
        r = subprocess.run(["curl","-s","--max-time","60","-F","reqtype=fileupload","-F","time=72h","-F",f"fileToUpload=@{p}","https://litterbox.catbox.moe/resources/internals/api.php"], capture_output=True, text=True, timeout=70)
        return r.stdout.strip() if r.stdout.strip().startswith("http") else None
    except Exception: return None
if results:
    best = max(results, key=lambda d: results[d][0] - results[d][3])  # high resem, low WER
    src = sorted(glob.glob(f"{best}/*.wav"))[0]
    subprocess.run(["ffmpeg","-y","-t","6","-i",src,"-ar","24000","-ac","1","-b:a","64k","cb_best.mp3"], capture_output=True)
    print(f"SUMMARY BEST={best} {results[best]}", flush=True)
    print(f"URL chatterbox_best {upload('cb_best.mp3')}", flush=True)
print("MEASURE_DONE", flush=True)
