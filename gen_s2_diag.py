# -*- coding: utf-8 -*-
"""DIAGNOSTIC: is the cross-lingual gap caused by REFERENCE-DELIVERY-LANGUAGE-MISMATCH?
Clone speaker2 into Arabic conditioned on (a) his ENGLISH ref [what we do] vs (b) his real ARABIC ref
[ideal L2 delivery]. Score both vs his real-AR centroid + delivery. If AR-ref >> EN-ref, the gap is
largely that we import L1 delivery into L2. (AR-ref isn't a deployable method for a new speaker — it's
the diagnostic that proves the cause + shows the achievable ceiling with correct delivery.)"""
import os, warnings, subprocess, numpy as np, librosa, soundfile as sf, torch
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
for url,out in [("https://www.youtube.com/watch?v=Tu2-1qowW2A","en"),("https://www.youtube.com/shorts/rRiHIdz8BiY","ar")]:
    subprocess.run(["yt-dlp","-x","--audio-format","wav","-o",f"{out}.%(ext)s",url],capture_output=True,timeout=180)
enc=VoiceEncoder(verbose=False)
def loud(w): r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
def emb(w,sr=SR): w16=librosa.resample(w.astype('float32'),orig_sr=sr,target_sr=16000) if sr!=16000 else w; e=enc.embed_utterance(preprocess_wav(loud(w16),source_sr=16000)); return e/np.linalg.norm(e)
def centroid(path,maxn=40):
    w,_=librosa.load(path,sr=16000,mono=True); iv=librosa.effects.split(w,top_db=30)
    E=[emb(w[s:e],16000) for s,e in iv if (e-s)>1.5*16000][:maxn]; c=np.mean(E,0); return c/np.linalg.norm(c)
def deliv(w):
    f=parselmouth.Sound(w.astype(np.float64),sampling_frequency=SR).to_pitch(0.01,65,500).selected_array['frequency']; v=f[f>0]
    return (float(np.median(v)), float(np.percentile(st(v),95)-np.percentile(st(v),5)), float(np.mean(np.abs(np.diff(st(f[f>0]))))/0.01))
cAR=centroid("ar.wav")
def mkref(path,name,sec=20):
    w,_=librosa.load(path,sr=SR,mono=True); iv=librosa.effects.split(w,top_db=30); buf=[];c=0
    for s,e in iv:
        buf.append(w[s:e]); c+=e-s
        if c>sec*SR: break
    sf.write(name, loud(np.concatenate(buf)[:sec*SR]), SR)
mkref("en.wav","enref.wav"); mkref("ar.wav","arref.wav")
print(f"GROUND TRUTH real AR: f0med 180 range 19.4 dyn 104 | ceiling 0.805",flush=True)
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
m=ChatterboxMultilingualTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
TXT=["لماذا لا يوجد علم سعودي فوق المسجد الحرام؟","الحقيقة لا تحتاج إلى من يدافع عنها بل إلى من يفهمها.","كثير من الناس يتكلمون بثقة عن أشياء لم يدرسوها."]
def litter(p):
    r=subprocess.run(["curl","-s","-F","reqtype=fileupload","-F","time=72h","-F",f"fileToUpload=@{p}","https://litterbox.catbox.moe/resources/internals/api.php"],capture_output=True,text=True,timeout=60); return r.stdout.strip()
for refname,ref in [("EN_ref","enref.wav"),("AR_ref","arref.wav")]:
    m.prepare_conditionals(ref, exaggeration=1.0)
    rs=[];ds=[];clips=[]
    for t in TXT:
        w=m.generate(t, language_id="ar", exaggeration=1.0, temperature=0.7).squeeze().detach().cpu().numpy()
        rs.append(float(emb(w)@cAR)); ds.append(deliv(w)); clips.append(w)
    dm=np.mean(ds,0)
    print(f"DIAG {refname}: resem->realAR {np.mean(rs):.3f} | f0med {dm[0]:.0f} range {dm[1]:.1f} dyn {dm[2]:.0f}",flush=True)
    sf.write(f"diag_{refname}.wav", clips[0], m.sr); print(f"URL {refname} {litter(f'diag_{refname}.wav')}",flush=True)
print("MEASURE_DONE",flush=True)
