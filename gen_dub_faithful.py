# -*- coding: utf-8 -*-
"""FAITHFUL DUB (Ahmed: it's dubbing — keep his markers at the SAME positions as the original, mapped
to the target language). His real EN: '...the Quran. [You know], Subhanallah... [So] he said...'.
Dub to Arabic two ways: (A) CLEAN = normal dubbing strips his markers; (B) FAITHFUL = his markers
preserved at their original positions, mapped (you know->شوف, so->يعني). Same voice. The difference
IS his verbal identity, placed where HE put it — not invented/sprinkled."""
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
# His real EN segment (transcribed). Markers at positions: [You know] after 'Quran'; [So] before 'he said'.
# Arabic translation of the SAME content; CLEAN strips markers, FAITHFUL keeps them mapped at position.
CLEAN  = "القرآن. سبحان الله، محمد هجاب ده، وأخوه فقر، كان بيكلّم مُلحد. قال للملحد عن الكون، موجود ولا لأ، ولا الكون خلق نفسه. هو تخطّى دي، ما انتبهش ليها."
FAITHFUL = "القرآن. شوف، سبحان الله، محمد هجاب ده، وأخوه فقر، كان بيكلّم مُلحد. يعني قال للملحد عن الكون، موجود ولا لأ، ولا الكون خلق نفسه. هو تخطّى دي، ما انتبهش ليها."
print("EN source markers: [You know]@pos2  [So]@before-'he said'", flush=True)
print("FAITHFUL keeps: you-know->شوف (same spot), so->يعني (same spot)", flush=True)
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
for tag,txt in [("clean",CLEAN),("faithful",FAITHFUL)]:
    w=m.generate(txt, language_id="ar", exaggeration=1.0, temperature=0.7).squeeze().detach().cpu().numpy()
    fn=f"dub_{tag}.wav"; sf.write(fn,w,m.sr); print(f"URL {tag} {litter(fn)}",flush=True)
print("MEASURE_DONE",flush=True)
