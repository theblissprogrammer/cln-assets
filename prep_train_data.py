# -*- coding: utf-8 -*-
"""Build the training corpus for the delivery-conditioned TTS (substrate-agnostic):
her audio -> utterance clips + transcript + per-utterance DELIVERY conditioning features
(F0 range/dynamism/coupling/declination, rate). Plus her speaker references. Writes train/ + manifest.csv."""
import os, json, warnings, numpy as np, librosa, soundfile as sf, csv
warnings.filterwarnings("ignore")
import parselmouth
SR=24000
st=lambda hz:12*np.log2(np.maximum(hz,1e-6)/55.0)
os.makedirs("train/wavs",exist_ok=True)

y,_=librosa.load("her_audio.wav",sr=SR,mono=True)
words=[w for w in json.load(open("/tmp/her_words.json")) if w['e']>w['s'] and w['w'].strip()]
f0_full=parselmouth.Sound(y.astype(np.float64),sampling_frequency=SR).to_pitch(0.01,100,500).selected_array['frequency']
rms_full=librosa.feature.rms(y=y.astype(np.float32),frame_length=int(0.025*SR),hop_length=int(0.01*SR))[0]

# group words into utterances by gaps > 0.45s, target 2-12s
utts=[]; cur=[words[0]]
for i in range(1,len(words)):
    gap=words[i]['s']-words[i-1]['e']
    dur=words[i-1]['e']-cur[0]['s']
    if gap>0.45 and dur>1.2:
        utts.append(cur); cur=[words[i]]
    else:
        cur.append(words[i])
utts.append(cur)

def deliv_feats(s,e):
    a=int(s/0.01); b=int(e/0.01); seg=f0_full[a:b]; v=seg>0
    ra=int(s*SR/(0.01*SR)); rb=int(e*SR/(0.01*SR))
    if v.sum()<10: return None
    sv=st(seg[v])
    f0range=float(np.percentile(sv,95)-np.percentile(sv,5))
    f0dyn=float(np.mean(np.abs(np.diff(st(seg[seg>0]))))/0.01)
    f0med=float(np.median(seg[v]))
    # coupling: corr of f0 and energy over voiced frames
    rr=rms_full[a:b]; n=min(len(seg),len(rr)); m=seg[:n]>0
    fec=float(np.corrcoef(st(seg[:n][m]),20*np.log10(rr[:n][m]+1e-6))[0,1]) if m.sum()>10 else 0.0
    # declination: slope over the utterance voiced frames
    vi=np.where(v)[0]; declin=float(np.polyfit(vi*0.01, sv,1)[0]) if len(vi)>15 else 0.0
    rate=float(len(seg[v])/((e-s)+1e-9))
    return dict(f0range=f0range,f0dyn=f0dyn,f0med=f0med,coupling=fec,declination=declin,voiced_rate=rate)

rows=[]
for k,u in enumerate(utts):
    s=u[0]['s']; e=u[-1]['e']; dur=e-s
    if dur<1.2 or dur>13: continue
    seg=y[int(s*SR):int(e*SR)]
    if len(seg)<int(1.0*SR): continue
    df=deliv_feats(s,e)
    if df is None: continue
    text=' '.join(w['w'].strip() for w in u).strip()
    fn=f"train/wavs/utt_{k:04d}.wav"
    sf.write(fn, seg, SR)
    rows.append(dict(wav=fn, dur=round(dur,2), text=text, **{kk:round(vv,3) for kk,vv in df.items()}))

with open("train/manifest.csv","w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# corpus-level delivery stats (her conditioning prior = what the model must reproduce)
arr=lambda k:np.array([r[k] for r in rows])
print(f"utterances: {len(rows)}  total {sum(r['dur'] for r in rows):.0f}s",flush=True)
print(f"text sample: {rows[0]['text'][:70]}",flush=True)
for k in ['f0range','f0dyn','f0med','coupling','declination','voiced_rate']:
    print(f"  HER {k:12s} median {np.median(arr(k)):6.2f}  p10-p90 [{np.percentile(arr(k),10):.2f},{np.percentile(arr(k),90):.2f}]",flush=True)
json.dump({k:float(np.median(arr(k))) for k in ['f0range','f0dyn','f0med','coupling','declination','voiced_rate']},
          open("train/her_delivery_prior.json","w"))
print("DONE",flush=True)
