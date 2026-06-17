# -*- coding: utf-8 -*-
"""Extract a speaker's IDIOLECT / VERBAL FINGERPRINT (Ahmed: the meaningless tokens that are pure
identity — 'ah', pauses, 'shof', 'you know what I mean', thinking habits). From transcript+timestamps:
filler rate, discourse markers, pause-to-think pattern, repetitions. Compare his EN vs AR to test if
the idiolect is a CONSISTENT cross-lingual identity marker (like the voice-quality grain)."""
import json, re, numpy as np
from collections import Counter
EN_FILL=set("uh um ah er hmm mhm erm uhh ahh umm eh".split())
EN_MARK=["you know","i mean","kind of","sort of","you know what i mean","i guess","or whatever","and stuff"]
EN_MARK1=set("like so right actually basically well okay literally anyway obviously honestly".split())
AR_FILL=set("اه آه ااه امم ايه اه،".split())
AR_MARK=["يعني","شوف","شوفي","طيب","والله","خلاص","بقى","يا اخي","الحقيقة","ماشي","تمام","ع فكرة"]
def norm(w): return re.sub(r"[^\w؀-ۿ']","",w.lower())
def analyze(path,lang):
    d=json.load(open(path)); words=d["words"]; text=d["text"]; toks=[norm(w["w"]) for w in words]
    dur=words[-1]["e"]-words[0]["s"] if words else 1; mins=dur/60
    FILL=EN_FILL if lang=="en" else AR_FILL; MARK=EN_MARK if lang=="en" else AR_MARK; MARK1=EN_MARK1 if lang=="en" else set()
    nf=sum(1 for t in toks if t in FILL)
    nm1=sum(1 for t in toks if t in MARK1)
    low=text.lower()
    nm=sum(low.count(m) for m in MARK)
    # pauses (gaps between words)
    gaps=[words[i]["s"]-words[i-1]["e"] for i in range(1,len(words))]; gaps=np.array([g for g in gaps if g>0])
    short=((gaps>0.3)&(gaps<=0.7)).sum(); think=(gaps>0.7).sum()
    # repetitions (immediate word repeats)
    reps=sum(1 for i in range(1,len(toks)) if toks[i]==toks[i-1] and toks[i])
    # top markers actually used
    used=Counter()
    for t in toks:
        if t in FILL: used[t]+=1
    for m in MARK:
        c=low.count(m)
        if c: used[m]=c
    if lang=="en":
        for t in toks:
            if t in MARK1: used[t]+=1
    print(f"  [{lang}] {len(words)} words, {dur:.0f}s")
    print(f"    fillers: {nf} ({nf/mins:.1f}/min) | discourse-markers: {nm+nm1} ({(nm+nm1)/mins:.1f}/min) | repeats: {reps} ({reps/mins:.1f}/min)")
    print(f"    pauses: short(0.3-0.7s) {short} ({short/mins:.1f}/min) | THINK(>0.7s) {think} ({think/mins:.1f}/min) | mean-gap {gaps.mean():.2f}s")
    print(f"    top tics: {dict(used.most_common(8))}")
    return dict(fill_min=nf/mins, mark_min=(nm+nm1)/mins, rep_min=reps/mins, think_min=think/mins, meangap=float(gaps.mean()))
print("IDIOLECT / VERBAL FINGERPRINT (is it consistent EN vs AR = a cross-lingual identity marker?)")
en=analyze("speaker2/transcript_en.json","en")
ar=analyze("speaker2/transcript_ar.json","ar")
print("\nCROSS-LINGUAL CONSISTENCY (rates should be similar if idiolect is identity):")
for k,lbl in [("fill_min","filler/min"),("mark_min","marker/min"),("rep_min","repeat/min"),("think_min","think-pause/min"),("meangap","mean-gap")]:
    print(f"  {lbl:16s} EN {en[k]:.2f}  AR {ar[k]:.2f}")
print("DONE")
