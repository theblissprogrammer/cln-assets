# -*- coding: utf-8 -*-
"""FINAL best-recipe generation: the calibrated delivery operator = her-EXPRESSIVE ref + exaggeration
{0.5,1.0} on flowing Arabic, BOTH MSA and EGYPTIAN (Ahmed's dubbing requirement). Multiple passages x
takes so averaged delivery_sim is stable. Upload all takes for local delivery-space measurement."""
import os, json, warnings, numpy as np, librosa, soundfile as sf, torch, subprocess
warnings.filterwarnings("ignore")
try:
    import perth
    if getattr(perth,"PerthImplicitWatermarker",None) is None:
        class _N:
            def __init__(s,*a,**k):pass
            def apply_watermark(s,wav,*a,**k):return wav
            def get_watermark(s,*a,**k):return None
        perth.PerthImplicitWatermarker=_N
except Exception: pass
def loud(w): r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
SR0=24000
import parselmouth
y,_=librosa.load("her_audio.wav",sr=SR0,mono=True); iv=librosa.effects.split(y,top_db=30)
def f0range(seg):
    f=parselmouth.Sound(seg.astype(np.float64),sampling_frequency=SR0).to_pitch(0.01,100,500).selected_array['frequency']; v=f[f>0]
    return (np.percentile(12*np.log2(v/55),95)-np.percentile(12*np.log2(v/55),5)) if len(v)>20 else 0
segs=[(s,e,y[s:e]) for s,e in iv if (e-s)>1.0*SR0]; segs.sort(key=lambda t:-f0range(t[2]))
ebuf=[];ec=0
for s,e,sg in segs:
    ebuf.append(sg);ec+=len(sg)
    if ec>30*SR0: break
sf.write("her_ref_expr.wav", loud(np.concatenate(ebuf)[:30*SR0]), SR0); print("REF_DONE",flush=True)

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
model=ChatterboxMultilingualTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
SR=getattr(model,"sr",24000)
MSA=["وقفت عند النافذة تنظر إلى الشارع الفارغ. لم تقل شيئًا لوقت طويل. ثم التفتت إليّ وابتسمت ابتسامة خفيفة.",
     "بعد كل ما مررنا به، عدنا أخيرًا إلى المكان نفسه. تذكرت كيف كانت الأمور من قبل. وضحكت حين رأيت أن لا شيء قد تغيّر."]
EGY=["وقفت عند الشباك بتبصّ على الشارع الفاضي. مقالتش حاجة لمدة طويلة. وبعدين بصّتلي وابتسمت ابتسامة خفيفة.",
     "بعد كل اللي عدّى علينا، رجعنا تاني لنفس المكان. افتكرت كانت الدنيا عاملة إزاي زمان. وضحكت لما لقيت محدش اتغيّر."]
json.dump({"MSA":MSA,"EGY":EGY},open("best_texts.json","w"),ensure_ascii=False)
N=3
def gen(tag, texts):
    for exag in [0.5,1.0]:
        d=f"best_{tag}_e{int(exag*10)}"; os.makedirs(d,exist_ok=True)
        for i,t in enumerate(texts):
            for n in range(N):
                try:
                    wav=model.generate(t, language_id="ar", audio_prompt_path="her_ref_expr.wav", cfg_weight=0.5, exaggeration=exag)
                    sf.write(f"{d}/p{i}_t{n}.wav", wav.squeeze().detach().cpu().numpy(), SR)
                except Exception as ex: print("GEN_ERR",tag,exag,i,n,str(ex)[:60],flush=True)
        print("DONE",tag,exag,flush=True)
gen("msa",MSA); gen("egy",EGY)
print("GEN_DONE",flush=True)
