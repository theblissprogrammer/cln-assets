# -*- coding: utf-8 -*-
"""Build the EXPANDED training corpus from her YouTube channel (Veiled Stories) to break the 102s
data limiter. Download N same-narrator videos -> verify same-speaker (resem>0.70) -> Whisper word-ts
-> segment into utterances + delivery features -> big train/manifest.csv. Clean speech (no music bed)."""
import os, json, csv, subprocess, warnings, numpy as np, librosa, soundfile as sf, torch
warnings.filterwarnings("ignore")
import parselmouth
from resemblyzer import VoiceEncoder, preprocess_wav
import whisper
SR=24000; st=lambda hz:12*np.log2(np.maximum(hz,1e-6)/55.0)
os.makedirs("train/wavs",exist_ok=True); os.makedirs("dl_au",exist_ok=True)

VIDS=["LkjlKy8Xl2o","5s9Oin0-j_o","yhQ2se0eZ-k","WwR7lLR6rE0","Id21b7QtyDw","oJzLbHWTTOU",
      "Gep9UIP4jtE","OwvOfoIciYQ","wILBC1ZmJac","T-6zyG29dy4","jxAJ4--9l_E","SxqFUpeCwbM",
      "1f1uk0w4gzc","ia0Yq8dWXpc","5rRAdTPygo0","LxcMZ-iKDEw","miFFfNwQzfw","oJ_placeholder"]
VIDS=[v for v in VIDS if v!="oJ_placeholder"]

enc=VoiceEncoder(verbose=False)
def loud(w): r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
def emb16(w): e=enc.embed_utterance(preprocess_wav(loud(w),source_sr=16000)); return e/np.linalg.norm(e)
# her anchor centroid from original clean clip
hy,_=librosa.load("her_audio.wav",sr=16000,mono=True); iv=librosa.effects.split(hy,top_db=30)
herv=np.mean([emb16(hy[s:e]) for s,e in iv if (e-s)>1.5*16000],0); herv/=np.linalg.norm(herv)

asr=whisper.load_model("small")
def delivery_feats(f0seg, rmsseg, s, e):
    v=f0seg>0
    if v.sum()<10: return None
    sv=st(f0seg[v])
    f0range=float(np.percentile(sv,95)-np.percentile(sv,5))
    f0dyn=float(np.mean(np.abs(np.diff(st(f0seg[f0seg>0]))))/0.01)
    f0med=float(np.median(f0seg[v]))
    n=min(len(f0seg),len(rmsseg)); m=f0seg[:n]>0
    fec=float(np.corrcoef(st(f0seg[:n][m]),20*np.log10(rmsseg[:n][m]+1e-6))[0,1]) if m.sum()>10 else 0.0
    vi=np.where(v)[0]; declin=float(np.polyfit(vi*0.01,sv,1)[0]) if len(vi)>15 else 0.0
    return dict(f0range=f0range,f0dyn=f0dyn,f0med=f0med,coupling=fec,declination=declin,
                voiced_rate=float(len(f0seg[v])/((e-s)+1e-9)))

rows=[]; kept=0
# seed with original utterances already in manifest? rebuild fresh from all videos incl original
for vi,vid in enumerate(VIDS):
    wavp=f"dl_au/{vid}.wav"
    if not os.path.exists(wavp):
        subprocess.run(["yt-dlp","-x","--audio-format","wav","-q","-o",f"dl_au/{vid}.%(ext)s",
                        f"https://www.youtube.com/watch?v={vid}"],capture_output=True,timeout=300)
    if not os.path.exists(wavp): print("DL_FAIL",vid,flush=True); continue
    w16,_=librosa.load(wavp,sr=16000,mono=True)
    iv2=librosa.effects.split(w16,top_db=30); segs=[w16[s:e] for s,e in iv2 if (e-s)>1.5*16000]
    sim=np.mean([float(emb16(s)@herv) for s in segs[:20]]) if segs else 0
    if sim<0.70: print(f"SKIP {vid} resem {sim:.2f} (not her/too noisy)",flush=True); continue
    kept+=1
    # full-res for features + tokens
    w,_=librosa.load(wavp,sr=SR,mono=True)
    f0=parselmouth.Sound(w.astype(np.float64),sampling_frequency=SR).to_pitch(0.01,100,500).selected_array['frequency']
    rms=librosa.feature.rms(y=w.astype(np.float32),frame_length=int(0.025*SR),hop_length=int(0.01*SR))[0]
    tr=asr.transcribe(wavp, language="en", word_timestamps=True)
    words=[]
    for sg in tr["segments"]:
        for wd in sg.get("words",[]):
            if wd["end"]>wd["start"]: words.append({"w":wd["word"].strip(),"s":wd["start"],"e":wd["end"]})
    if len(words)<5: continue
    # group into utterances
    utt=[]; cur=[words[0]]
    for i in range(1,len(words)):
        if words[i]["s"]-words[i-1]["e"]>0.45 and (words[i-1]["e"]-cur[0]["s"])>1.2:
            utt.append(cur); cur=[words[i]]
        else: cur.append(words[i])
    utt.append(cur)
    for k,u in enumerate(utt):
        s=u[0]["s"]; e=u[-1]["e"]
        if e-s<1.2 or e-s>13: continue
        a=int(s/0.01); b=int(e/0.01)
        df=delivery_feats(f0[a:b], rms[a:b], s, e)
        if df is None: continue
        text=' '.join(x["w"] for x in u).strip()
        if len(text)<4: continue
        fn=f"train/wavs/{vid}_{k:03d}.wav"
        sf.write(fn, w[int(s*SR):int(e*SR)], SR)
        rows.append(dict(wav=fn,dur=round(e-s,2),text=text,**{kk:round(vv,3) for kk,vv in df.items()}))
    print(f"VID {vid} resem {sim:.2f} -> {len([r for r in rows if vid in r['wav']])} utts (total {len(rows)})",flush=True)

with open("train/manifest.csv","w",newline="") as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
arr=lambda k:np.array([r[k] for r in rows])
print(f"\nCORPUS: {kept} videos kept, {len(rows)} utterances, {sum(r['dur'] for r in rows)/60:.1f} min total",flush=True)
print(f"  f0range median {np.median(arr('f0range')):.1f} p10-p90 [{np.percentile(arr('f0range'),10):.1f},{np.percentile(arr('f0range'),90):.1f}]  (variation = trainable signal)",flush=True)
print("CORPUS_DONE",flush=True)
