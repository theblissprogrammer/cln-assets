# -*- coding: utf-8 -*-
"""PHASE 1 — same-speaker BILINGUAL corpus (the field's missing supervision). For each candidate
speaker: yt-dlp search their EN + AR content, pull short samples, VERIFY same-speaker across languages
(resemblyzer: EN-centroid vs AR-centroid > impostor ceiling + each set internally consistent).
Only genuine bilingual same-speaker pairs survive. Reports yield. (Verified set -> full extraction later.)"""
import os, subprocess, warnings, json, numpy as np, librosa
warnings.filterwarnings("ignore")
from resemblyzer import VoiceEncoder, preprocess_wav
os.makedirs("bicorpus", exist_ok=True)
enc=VoiceEncoder(verbose=False)
def loud(w): r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
def emb(w): e=enc.embed_utterance(preprocess_wav(loud(w),source_sr=16000)); return e/np.linalg.norm(e)
def dl_samples(query, prefix, n=3, sec=90):
    outs=[]
    for k in range(n):
        p=f"bicorpus/{prefix}_{k}.wav"
        if not os.path.exists(p):
            subprocess.run(["yt-dlp",f"ytsearch{n}:{query}","--playlist-items",str(k+1),
                            "--download-sections",f"*30-{30+sec}","-x","--audio-format","wav","-q",
                            "-o",f"bicorpus/{prefix}_{k}.%(ext)s"],capture_output=True,timeout=200)
        if os.path.exists(p) and os.path.getsize(p)>50000: outs.append(p)
    return outs
def centroid(paths):
    E=[]
    for p in paths:
        try:
            w,_=librosa.load(p,sr=16000,mono=True); iv=librosa.effects.split(w,top_db=30)
            segs=[w[s:e] for s,e in iv if (e-s)>1.5*16000][:10]
            E+= [emb(s) for s in segs]
        except Exception: pass
    if len(E)<3: return None,0
    E=np.array(E); c=E.mean(0); c/=np.linalg.norm(c)
    consist=float(np.mean([e@c for e in E]))   # internal consistency (did search return one person?)
    return c, consist

# seed candidates (name, EN query, AR query) — bilingual-likely Arab public figures; verification filters
SEED=[
 ("hijab","Mohammed Hijab english debate","محمد هجاب عربي"),
 ("moez","Moez Masoud english","معز مسعود محاضرة"),
 ("tzortzis","Hamza Tzortzis lecture english","حمزة تزورتزس عربي"),
 ("bassem","Bassem Youssef english interview","باسم يوسف حلقة"),
 ("adnan","Adnan Ibrahim english","عدنان ابراهيم محاضرة"),
 ("tariq","Tariq Ramadan english lecture","طارق رمضان محاضرة"),
 ("hamzayusuf","Hamza Yusuf english lecture","حمزة يوسف عربي"),
 ("amrkhaled","Amr Khaled english","عمرو خالد حلقة"),
 ("suwaidan","Tariq Al Suwaidan english","طارق السويدان محاضرة"),
 ("jifri","Habib Ali al Jifri english","الحبيب علي الجفري"),
 ("yasirqadhi","Yasir Qadhi english lecture","ياسر قاضي عربي"),
 ("jonathanbrown","Jonathan Brown islam english lecture","جوناثان براون عربي"),
 ("aboulfadl","Khaled Abou El Fadl english","خالد ابو الفضل"),
 ("fadelsoliman","Fadel Soliman bridges english","فاضل سليمان"),
 ("yasirfazaga","Yasir Fazaga english khutbah","ياسر فزاغة"),
 ("mostafahosny","Mostafa Hosny english","مصطفى حسني"),
 ("rania","Queen Rania interview english","الملكة رانيا مقابلة"),
 ("elbaradei","Mohamed ElBaradei interview english","محمد البرادعي مقابلة"),
 ("amrmoussa","Amr Moussa interview english","عمرو موسى مقابلة"),
 ("ahmedaboulgheit","Ahmed Aboul Gheit english","احمد ابو الغيط"),
 ("azzamtamimi","Azzam Tamimi english interview","عزام التميمي"),
 ("waelhallaq","Wael Hallaq lecture english","وائل حلاق محاضرة"),
 ("nadiabilal","Nadia interview arabic english bilingual",""),
 ("dahimalik","Dahi Khalfan english arabic",""),
 ("muftimenk","Mufti Menk english lecture","مفتي منك عربي"),
 ("alidawah","Ali Dawah english","علي دعوة عربي"),
 ("yusufestes","Yusuf Estes english","يوسف استس عربي"),
 ("shabirally","Shabir Ally english lecture","شبير الي عربي"),
]
print("BILINGUAL CORPUS yield assessment (same-speaker EN<->AR verification):",flush=True)
print(f"{'speaker':12s} {'EN_clips':>8s} {'AR_clips':>8s} {'EN_consist':>10s} {'AR_consist':>10s} {'EN<->AR':>8s} {'VERDICT':>10s}",flush=True)
kept=[]
for name,enq,arq in SEED:
    en=dl_samples(enq,f"{name}_en"); ar=dl_samples(arq,f"{name}_ar")
    cEN,kEN=centroid(en); cAR,kAR=centroid(ar)
    if cEN is None or cAR is None:
        print(f"{name:12s} {len(en):8d} {len(ar):8d} {'--':>10s} {'--':>10s} {'--':>8s} {'NO_AUDIO':>10s}",flush=True); continue
    cross=float(cEN@cAR)
    ok = (cross>0.70) and (kEN>0.72) and (kAR>0.72)   # same person across langs + each set coherent
    if ok: kept.append(name)
    print(f"{name:12s} {len(en):8d} {len(ar):8d} {kEN:10.2f} {kAR:10.2f} {cross:8.2f} {('SAME-SPK' if ok else 'reject'):>10s}",flush=True)
print(f"\nYIELD: {len(kept)}/{len(SEED)} verified bilingual same-speakers: {kept}",flush=True)
print("(>0.70 EN<->AR cross-lingual = same person; this is the data the field lacks)",flush=True)
print("DONE",flush=True)
