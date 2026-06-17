# -*- coding: utf-8 -*-
"""SMART idiolect injection (fix: too many + random). Learned from his real AR: rate 4.2/min (~1 per
14s), markers ONLY at sentence/clause boundaries — "طيب"=topic transition (start of a new point),
"شوف"=introduce an emphatic point, triple "شوف شوف شوف" ONCE for the single strongest emphasis.
Never mid-phrase. Algorithmic placer matches his rate to the dub's duration."""
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
RATE_PER_MIN=4.2; SEC_PER_WORD=0.40   # his measured rate; ~speaking rate
def smart_inject(sentences):
    # estimate duration, target marker count = rate * minutes
    words=sum(len(s.split()) for s in sentences); dur_s=words*SEC_PER_WORD
    target=max(1,int(round(RATE_PER_MIN*dur_s/60)))
    # candidate slots = sentence starts (boundaries only). transitions (idx>=1) -> طيب; idx0 / emphatic -> شوف
    out=list(sentences); n=len(sentences)
    # choose ~target boundary positions spread out
    slots=list(range(n)); placed=0; used=set()
    # 1) one emphatic triple-shof on the LAST sentence (strong close), if target>=2
    order=[]
    if n>=1: order.append((0,'شوف'))                 # open
    for i in range(1,n): order.append((i,'طيب'))     # transitions
    # spread: take evenly spaced from order up to target
    pick=[order[int(round(k*(len(order)-1)/max(target-1,1)))] for k in range(min(target,len(order)))] if order else []
    seen=set()
    for idx,mk in pick:
        if idx in seen: continue
        seen.add(idx)
        # last sentence + emphatic -> triple shof once
        marker= 'شوف شوف' if (idx==n-1 and mk=='شوف' and placed==0 and n>=3) else mk
        out[idx]=f"{marker}، "+out[idx]
        placed+=1
    return ' '.join(out), placed, target
PASSAGE=["الموضوع ده مهم جدًا وعايز أوضحه كويس.",
         "الحقيقة ما تحتاجش حد يدافع عنها بالعافية.",
         "هي تحتاج بس حد يفهمها صح من غير تعصب.",
         "لكن في ناس كتير بتتكلم بثقة عن حاجات ما درسوهاش.",
         "بيقروا عنوان وبيبنوا عليه رأي كامل.",
         "وبعدين لما تسألهم في التفاصيل ما يعرفوش.",
         "ده بالظبط اللي بيخرب أي نقاش جاد.",
         "احنا محتاجين نرجع للمصادر الأصلية بنفسنا."]
plain=' '.join(PASSAGE); idio,placed,tg=smart_inject(PASSAGE)
print(f"SMART placed {placed} markers (target {tg} at his {RATE_PER_MIN}/min):",flush=True)
print(" PLAIN:",plain,flush=True); print(" IDIO :",idio,flush=True)
subprocess.run(["yt-dlp","-x","--audio-format","wav","-o","en.%(ext)s","https://www.youtube.com/watch?v=Tu2-1qowW2A"],capture_output=True,timeout=180)
def loud(w): r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
en,_=librosa.load("en.wav",sr=24000,mono=True); iv=librosa.effects.split(en,top_db=30); buf=[];c=0
for s,e in iv:
    buf.append(en[s:e]); c+=e-s
    if c>25*24000: break
sf.write("enref.wav", loud(np.concatenate(buf)[:25*24000]), 24000)
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
m=ChatterboxMultilingualTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
def litter(p):
    r=subprocess.run(["curl","-s","-F","reqtype=fileupload","-F","time=72h","-F",f"fileToUpload=@{p}","https://litterbox.catbox.moe/resources/internals/api.php"],capture_output=True,text=True,timeout=60); return r.stdout.strip()
m.prepare_conditionals("enref.wav", exaggeration=1.0)
for tag,txt in [("plain",plain),("smart",idio)]:
    w=m.generate(txt, language_id="ar", exaggeration=1.0, temperature=0.7).squeeze().detach().cpu().numpy()
    fn=f"smart_{tag}.wav"; sf.write(fn,w,m.sr); print(f"URL {tag} {litter(fn)}",flush=True)
print("MEASURE_DONE",flush=True)
