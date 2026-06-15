# -*- coding: utf-8 -*-
"""
GATE-4 step 2: does her word-level F0 MELODY become PREDICTABLE with LINGUISTIC slot features?
(phrase-level audio-only features gave F0 R^2 ~ 0; the research says F0 attaches to linguistic
structure -- stress/focus/content-word. Test it.)

word features (language-AGNOSTIC roles, transfer EN->AR):
  is_content (content vs function word), n_syll, word_dur, position_in_phrase,
  is_phrase_final, is_phrase_initial, is_utt_final, rel_dur
word VALUE (her melody): word_f0_st_rel (mean F0 st minus phrase register), word_f0_slope, word_energy_rel
"""
import json, numpy as np, librosa, warnings
warnings.filterwarnings("ignore")
import parselmouth
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

SR=24000
FUNC=set("a an the of to in on at by for with from as is are was were be been being am "
         "and or but nor so yet if then than that this these those it its he she they we you i "
         "his her their our your my me him them us do does did have has had will would can could "
         "shall should may might must not no n't 'll 're 've 's 'd 'm to into onto out up down off "
         "about over under again just very too also only".split())
def is_content(w):
    t=''.join(c for c in w.lower() if c.isalpha() or c=="'")
    return 0.0 if (t in FUNC or len(t)<=1) else 1.0
def nsyll(w):
    t=''.join(c for c in w.lower() if c.isalpha())
    if not t: return 1
    groups=0; prev=False
    for c in t:
        v=c in "aeiouy"
        if v and not prev: groups+=1
        prev=v
    return max(groups,1)

words=json.load(open("/tmp/her_words.json"))
words=[w for w in words if w['e']>w['s'] and w['w']]
# F0 track
y,_=librosa.load("her_audio.wav",sr=SR,mono=True)
f0=parselmouth.Sound(y.astype(np.float64),sampling_frequency=SR).to_pitch(0.01,100,500).selected_array['frequency']
rms=librosa.feature.rms(y=y.astype(np.float32),frame_length=1024,hop_length=256)[0]
st=lambda hz:12*np.log2(np.maximum(hz,1e-6)/55.0)
def f0_win(s,e):
    a=int(s/0.01); b=max(int(e/0.01),a+1); seg=f0[a:b]; v=seg>0
    return seg[v] if v.sum()>=2 else None
def eng_win(s,e):
    a=int(s*SR/256); b=max(int(e*SR/256),a+1); return np.mean(20*np.log10(rms[a:b]+1e-6))

# group into phrases (gap>0.3) + utterances (gap>0.5)
phr=[]; cur=[words[0]]; utt_break=[False]
for i in range(1,len(words)):
    gap=words[i]['s']-words[i-1]['e']
    if gap>0.3:
        phr.append(cur); cur=[words[i]]
    else:
        cur.append(words[i])
phr.append(cur)

# utterance-final detection: phrase followed by gap>0.5 (or last)
X=[];Y=[]
utt_eng=np.mean(20*np.log10(rms+1e-6))
for pi,ph in enumerate(phr):
    durs=[w['e']-w['s'] for w in ph]; mdur=np.mean(durs)
    # phrase register = median F0 over phrase
    allf=[]
    for w in ph:
        fw=f0_win(w['s'],w['e'])
        if fw is not None: allf+=list(fw)
    if len(allf)<5: continue
    reg=np.median(st(np.array(allf)))
    n=len(ph)
    last_gap = (phr[pi+1][0]['s']-ph[-1]['e']) if pi+1<len(phr) else 9.9
    for k,w in enumerate(ph):
        fw=f0_win(w['s'],w['e'])
        if fw is None: continue
        sv=st(fw)
        f0rel=np.mean(sv)-reg
        slope=np.polyfit(np.linspace(0,1,len(sv)),sv,1)[0] if len(sv)>=3 else 0.0
        erel=eng_win(w['s'],w['e'])-utt_eng
        feat=[is_content(w['w']), nsyll(w['w']), w['e']-w['s'], k/max(n-1,1),
              float(k==n-1), float(k==0), float(k==n-1 and last_gap>0.5), (w['e']-w['s'])/(mdur+1e-9)]
        X.append(feat); Y.append([f0rel, slope, erel])
X=np.array(X); Y=np.array(Y)
print(f"her words used: {len(X)}  (content frac {np.mean(X[:,0]):.2f})",flush=True)

CH=["word_f0_rel(st)","word_f0_slope","word_energy_rel"]
def r2(y,yh): ss=np.sum((y-y.mean())**2); return 1-np.sum((y-yh)**2)/(ss+1e-12)
kf=KFold(5,shuffle=True,random_state=0)
print(f"\n{'channel':18s} {'R2(5fold)':>10s}   {'feature importances (|beta| z-space)':s}",flush=True)
for j,ch in enumerate(CH):
    preds=np.zeros(len(X))
    betas=[]
    for tr,te in kf.split(X):
        sc=StandardScaler().fit(X[tr]); m=Ridge(1.0).fit(sc.transform(X[tr]),Y[tr,j])
        preds[te]=m.predict(sc.transform(X[te])); betas.append(m.coef_)
    R=r2(Y[:,j],preds); b=np.mean(np.abs(betas),0)
    fn=["content","nsyll","dur","pos","final","init","uttfinal","reldur"]
    top=sorted(zip(fn,b),key=lambda z:-z[1])[:4]
    print(f"{ch:18s} {R:10.3f}   {', '.join(f'{n}={v:.2f}' for n,v in top)}",flush=True)

# headline: word_f0_rel predictability (the MELODY) -- contrast with the phrase-audio-only ~0
preds=np.zeros(len(X))
for tr,te in kf.split(X):
    sc=StandardScaler().fit(X[tr]); m=Ridge(1.0).fit(sc.transform(X[tr]),Y[tr,0]); preds[te]=m.predict(sc.transform(X[te]))
R0=r2(Y[:,0],preds)
print(f"\nSUMMARY word_f0_melody_R2={R0:.3f}  (phrase-audio-only was ~0.00 -> linguistic features {'UNLOCK' if R0>0.08 else 'do NOT unlock'} her melody)",flush=True)
print("MEASURE_DONE",flush=True)
