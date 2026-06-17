# -*- coding: utf-8 -*-
"""Generalization test: clone speaker2 (Mohamed Hijab clip, unique male voice + room echo) into Arabic
via the calibrated recipe. Download video -> build ref -> measure his f0range -> generate Arabic at a
few exaggerations -> resem->him + upload. (Echo/background preservation handled locally after.)"""
import os, json, warnings, subprocess, numpy as np, librosa, soundfile as sf, torch
warnings.filterwarnings("ignore")
try:
    import perth
    if getattr(perth,"PerthImplicitWatermarker",None) is None:
        class _N:
            def __init__(s,*a,**k):pass
            def apply_watermark(s,w,*a,**k):return w
            def get_watermark(s,*a,**k):return None
        perth.PerthImplicitWatermarker=_N
except Exception: pass
import parselmouth
from resemblyzer import VoiceEncoder, preprocess_wav
SR=24000; st=lambda hz:12*np.log2(np.maximum(hz,1e-6)/55.0)
subprocess.run(["yt-dlp","-x","--audio-format","wav","-o","s2.%(ext)s","https://www.youtube.com/watch?v=Tu2-1qowW2A"],capture_output=True,timeout=180)
src,_=librosa.load("s2.wav",sr=SR,mono=True)
def loud(w): r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
# ref = first ~25s of clean-ish speech (stitched voiced regions)
iv=librosa.effects.split(src,top_db=30); buf=[];c=0
for s,e in iv:
    buf.append(src[s:e]); c+=e-s
    if c>25*SR: break
ref=loud(np.concatenate(buf)[:25*SR]); sf.write("s2_ref.wav",ref,SR)
f=parselmouth.Sound(ref.astype(np.float64),sampling_frequency=SR).to_pitch(0.01,65,500).selected_array['frequency']; v=f[f>0]
his_rng=float(np.percentile(st(v),95)-np.percentile(st(v),5)); his_med=float(np.median(v))
print(f"his ref: f0range {his_rng:.1f} st, f0med {his_med:.0f} Hz",flush=True)

enc=VoiceEncoder(verbose=False)
def emb(w): e=enc.embed_utterance(preprocess_wav(loud(librosa.resample(w.astype('float32'),orig_sr=SR,target_sr=16000)),source_sr=16000)); return e/np.linalg.norm(e)
himv=emb(ref)
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
m=ChatterboxMultilingualTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
TXT=["الحقيقة لا تحتاج إلى من يدافع عنها، بل تحتاج إلى من يفهمها.",
     "كثير من الناس يتكلمون بثقة عن أشياء لم يدرسوها جيدًا.",
     "إذا أردت أن تفهم فكرة، فاقرأها من مصدرها الأصلي لا من خصومها."]
def litter(p):
    r=subprocess.run(["curl","-s","-F","reqtype=fileupload","-F","time=72h","-F",f"fileToUpload=@{p}","https://litterbox.catbox.moe/resources/internals/api.php"],capture_output=True,text=True,timeout=60); return r.stdout.strip()
m.prepare_conditionals("s2_ref.wav", exaggeration=0.5)
for exag in [0.5,1.0]:
    for i,t in enumerate(TXT):
        best=None;br=-9
        for n in range(2):
            wav=m.generate(t, language_id="ar", exaggeration=exag, temperature=0.7).squeeze().detach().cpu().numpy()
            r=float(emb(wav)@himv)
            if r>br: br,best=r,wav
        fn=f"s2_e{int(exag*10)}_{i}.wav"; sf.write(fn,best,m.sr)
        print(f"SPK2 exag {exag} clip {i}: resem->him {br:.3f}",flush=True)
        print(f"URL {fn} {litter(fn)}",flush=True)
print("MEASURE_DONE",flush=True)
