# -*- coding: utf-8 -*-
"""Bilingual GROUND-TRUTH iteration: clone speaker2 EN->Arabic, score vs his REAL Arabic (not his EN).
Diagnosis: his real AR is very animated (range 19.4, dyn 104) but clone@exag0.5 was flat (10.3, 165Hz).
Sweep exaggeration to match his real-AR delivery + maximize resem vs his REAL AR centroid."""
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
def emb(w,sr=SR): e=enc.embed_utterance(preprocess_wav(loud(librosa.resample(w.astype('float32'),orig_sr=sr,target_sr=16000) if sr!=16000 else w),source_sr=16000)); return e/np.linalg.norm(e)
def centroid(path,maxn=40):
    w,_=librosa.load(path,sr=16000,mono=True); iv=librosa.effects.split(w,top_db=30)
    E=[emb(w[s:e],16000) for s,e in iv if (e-s)>1.5*16000][:maxn]; c=np.mean(E,0); return c/np.linalg.norm(c)
def deliv(w):
    f=parselmouth.Sound(w.astype(np.float64),sampling_frequency=SR).to_pitch(0.01,65,500).selected_array['frequency']; v=f[f>0]
    return (float(np.median(v)), float(np.percentile(st(v),95)-np.percentile(st(v),5)), float(np.mean(np.abs(np.diff(st(f[f>0]))))/0.01))
cAR=centroid("ar.wav")              # GROUND TRUTH centroid (his real Arabic)
hm,hr,hd=deliv(librosa.load("ar.wav",sr=SR,mono=True)[0])
print(f"GROUND TRUTH his real AR: resem-centroid built | f0med {hm:.0f} range {hr:.1f} dyn {hd:.0f}",flush=True)
# EN ref (25s)
en,_=librosa.load("en.wav",sr=SR,mono=True); iv=librosa.effects.split(en,top_db=30); buf=[];c=0
for s,e in iv:
    buf.append(en[s:e]); c+=e-s
    if c>25*SR: break
sf.write("enref.wav", loud(np.concatenate(buf)[:25*SR]), SR)
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
m=ChatterboxMultilingualTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
TXT=["لماذا لا يوجد علم سعودي فوق المسجد الحرام؟","الحقيقة لا تحتاج إلى من يدافع عنها بل إلى من يفهمها.","كثير من الناس يتكلمون بثقة عن أشياء لم يدرسوها."]
def litter(p):
    r=subprocess.run(["curl","-s","-F","reqtype=fileupload","-F","time=72h","-F",f"fileToUpload=@{p}","https://litterbox.catbox.moe/resources/internals/api.php"],capture_output=True,text=True,timeout=60); return r.stdout.strip()
m.prepare_conditionals("enref.wav", exaggeration=0.5)
print(f"\n{'exag':>5s} {'resem->REAL_AR':>14s} {'f0med':>6s} {'range':>6s} {'dyn':>5s}  (target 0.805 ceiling / {hm:.0f}/{hr:.1f}/{hd:.0f})",flush=True)
best=None
for exag in [0.5,1.0,1.5,2.0]:
    rs=[];ds=[];clips=[]
    for i,t in enumerate(TXT):
        w=m.generate(t, language_id="ar", exaggeration=exag, temperature=0.7).squeeze().detach().cpu().numpy()
        rs.append(float(emb(w)@cAR)); ds.append(deliv(w)); clips.append(w)
    dm=np.mean(ds,0); R=np.mean(rs)
    print(f"SWEEP {exag:5.1f} {R:14.3f} {dm[0]:6.0f} {dm[1]:6.1f} {dm[2]:5.0f}",flush=True)
    sf.write(f"s2v2_e{int(exag*10)}.wav", clips[0], m.sr)
    print(f"URL e{int(exag*10)} {litter(f's2v2_e{int(exag*10)}.wav')}",flush=True)
print("MEASURE_DONE",flush=True)
