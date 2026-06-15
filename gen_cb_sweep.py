# -*- coding: utf-8 -*-
"""GENERATION-TIME delivery sweep: exaggeration x reference(std vs her-most-expressive), on FLOWING
multi-phrase Arabic (so rhythm/pause channels are present + WER measurable). The constructive test of
whether the engine can GENERATE her delivery (post-hoc editing proven limited)."""
import os, json, warnings, numpy as np, librosa, soundfile as sf, torch
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
y,_=librosa.load("her_audio.wav",sr=SR0,mono=True); iv=librosa.effects.split(y,top_db=30)
# standard 30s ref
buf=[];cur=0
for s,e in iv:
    buf.append(y[s:e]);cur+=e-s
    if cur>30*SR0: break
sf.write("her_ref_std.wav", loud(np.concatenate(buf)[:30*SR0]), SR0)
# expressive 30s ref: segments with highest f0-range
import parselmouth
def f0range(seg):
    f=parselmouth.Sound(seg.astype(np.float64),sampling_frequency=SR0).to_pitch(0.01,100,500).selected_array['frequency']
    v=f[f>0]
    return (np.percentile(12*np.log2(v/55),95)-np.percentile(12*np.log2(v/55),5)) if len(v)>20 else 0
segs=[(s,e,y[s:e]) for s,e in iv if (e-s)>1.0*SR0]
segs.sort(key=lambda t:-f0range(t[2]))
ebuf=[];ec=0
for s,e,sg in segs:
    ebuf.append(sg);ec+=len(sg)
    if ec>30*SR0: break
sf.write("her_ref_expr.wav", loud(np.concatenate(ebuf)[:30*SR0]), SR0)
print("REFS_DONE",flush=True)

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
model=ChatterboxMultilingualTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
SR=getattr(model,"sr",24000)
PASS=[
 "وقفت عند النافذة تنظر إلى الشارع الفارغ. لم تقل شيئًا لوقت طويل. ثم التفتت إليّ وابتسمت ابتسامة خفيفة.",
 "مشينا على طول الشاطئ ونحن نتحدث عن أشياء بسيطة. كان الهواء باردًا قليلًا في ذلك المساء. شعرت أن كل شيء سيكون على ما يرام.",
 "بعد كل ما مررنا به، عدنا أخيرًا إلى المكان نفسه. تذكرت كيف كانت الأمور من قبل. وضحكت حين رأيت أن لا شيء قد تغيّر.",
]
json.dump(PASS,open("passages.json","w"),ensure_ascii=False)
N=3
for ref in ["std","expr"]:
    for exag in [0.5,1.0,1.5,2.0]:
        d=f"sw_{ref}_e{int(exag*10)}"; os.makedirs(d,exist_ok=True)
        for i,t in enumerate(PASS):
            for n in range(N):
                try:
                    wav=model.generate(t, language_id="ar", audio_prompt_path=f"her_ref_{ref}.wav",
                                       cfg_weight=0.5, exaggeration=exag)
                    sf.write(f"{d}/p{i}_t{n}.wav", wav.squeeze().detach().cpu().numpy(), SR)
                except Exception as ex:
                    print("GEN_ERR",ref,exag,i,n,str(ex)[:80],flush=True)
        print("DONE_SETTING",ref,exag,flush=True)
print("GEN_DONE",flush=True)
