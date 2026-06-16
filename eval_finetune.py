# -*- coding: utf-8 -*-
"""Eval: does fine-tuning on her hours make the clone's DELIVERY more-her (EN + cross-lingual AR),
identity held + clean? Compare BASE (zero-shot) vs FT (LoRA) on her_delivery_prior + resem + UTMOS."""
import os, json, warnings, numpy as np, librosa, soundfile as sf, torch
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
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
dev="cuda"; st=lambda hz:12*np.log2(np.maximum(hz,1e-6)/55.0)
prior=json.load(open("train/her_delivery_prior.json"))
enc=VoiceEncoder(verbose=False)
def loud(w): r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
def remb(w,sr): w16=librosa.resample(w.astype(np.float32),orig_sr=sr,target_sr=16000); e=enc.embed_utterance(preprocess_wav(loud(w16),source_sr=16000)); return e/np.linalg.norm(e)
hy,_=librosa.load("her_audio.wav",sr=16000,mono=True); iv=librosa.effects.split(hy,top_db=30)
herv=np.mean([remb(hy[s:e],16000) for s,e in iv if (e-s)>1.5*16000],0); herv/=np.linalg.norm(herv)
UT=torch.hub.load("tarepan/SpeechMOS:v1.2.0","utmos22_strong",trust_repo=True); UT.eval()
def feats(w,sr):
    f=parselmouth.Sound(w.astype(np.float64),sampling_frequency=sr).to_pitch(0.01,65,500).selected_array['frequency']; v=f[f>0]
    rng=float(np.percentile(st(v),95)-np.percentile(st(v),5)) if len(v)>10 else 0
    dyn=float(np.mean(np.abs(np.diff(st(f[f>0]))))/0.01) if (f>0).sum()>5 else 0
    rt=librosa.feature.rms(y=w.astype(np.float32),frame_length=int(0.025*sr),hop_length=int(0.01*sr))[0]
    n=min(len(f),len(rt)); mk=f[:n]>0
    cpl=float(np.corrcoef(st(f[:n][mk]),20*np.log10(rt[:n][mk]+1e-6))[0,1]) if mk.sum()>10 else 0
    r=float(remb(w,sr)@herv)
    w16=librosa.resample(w.astype(np.float32),orig_sr=sr,target_sr=16000)
    with torch.no_grad(): u=float(UT(torch.from_numpy(w16)[None],16000))
    return rng,dyn,cpl,r,u

EN=["Despite everything that happened, she never lost her faith.","He looked at her for a long time and then quietly walked away.","After all those years, they finally returned to the same place."]
AR=["وقفت عند النافذة تنظر إلى الشارع الفارغ ولم تقل شيئًا.","بعد كل ما مررنا به عدنا أخيرًا إلى المكان نفسه.","نظر إليها طويلًا ثم انصرف في صمت."]

m=ChatterboxMultilingualTTS.from_pretrained(device=dev)
def run(tag):
    print(f"\n=== {tag} ===  (her prior: f0range {prior['f0range']:.1f} f0dyn {prior['f0dyn']:.1f} coupling {prior['coupling']:.2f})",flush=True)
    for lang,texts in [("EN",EN),("AR",AR)]:
        R=[]
        for t in texts:
            wav=m.generate(t, language_id=lang.lower(), audio_prompt_path="her_audio.wav", exaggeration=0.7, temperature=0.7)
            R.append(feats(wav.squeeze().detach().cpu().numpy(), m.sr))
        R=np.array(R); mr=R.mean(0)
        print(f"  {lang}: f0range {mr[0]:.1f} (her {prior['f0range']:.1f}) | f0dyn {mr[1]:.0f} | coupling {mr[2]:+.2f} (her {prior['coupling']:.2f}) | resem {mr[3]:.3f} | UTMOS {mr[4]:.2f}",flush=True)
run("BASE (zero-shot)")
from peft import PeftModel
m.t3.tfmr=PeftModel.from_pretrained(m.t3.tfmr,"train/lora_ft"); m.t3.tfmr=m.t3.tfmr.merge_and_unload()
print("FT LoRA merged",flush=True)
run("FT (fine-tuned on her hours)")
print("MEASURE_DONE",flush=True)
