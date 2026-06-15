# -*- coding: utf-8 -*-
"""
DELIVERY OPERATOR v2 -- HER-CONTOUR-BANK TRANSPLANT (the real "transfer her way of talking").
Don't synthesize her melody from features (proven not learnable). Instead bank her REAL intonation
contours and re-impose them, run-by-run, onto the Arabic clone via PSOLA -- decoupled from timbre
(Chatterbox supplies that). Like the kNN timbre-exemplar, but for MELODY.
  - her bank: every voiced run's normalized shape (register+range removed) + range + phrase-final tag.
  - per Arabic run: pick a her-contour matched by length + phrase-position; de-normalize to clip's
    LOCAL register (keep WHO) + HER typical range (fixes dynamism); PSOLA-impose.
GUARDS: IMPOSTOR-contour-bank control MUST NOT raise delivery-sim->her (specificity, anti-Goodhart);
resemblyzer held (identity); UTMOS clean; (WER later).
"""
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, librosa, soundfile as sf, torch
import parselmouth
from parselmouth.praat import call
from resemblyzer import VoiceEncoder, preprocess_wav
import deliv_core as dc
SR=24000
st=lambda hz:12*np.log2(np.maximum(hz,1e-6)/55.0)

enc=VoiceEncoder(verbose=False)
def loud(w): r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
def remb(w):
    w16=librosa.resample(w.astype(np.float32),orig_sr=SR,target_sr=16000)
    e=enc.embed_utterance(preprocess_wav(loud(w16),source_sr=16000)); return e/np.linalg.norm(e)
hy,_=librosa.load("her_audio.wav",sr=SR,mono=True)
ivh=librosa.effects.split(hy,top_db=30); herv=np.mean([remb(hy[s:e]) for s,e in ivh if (e-s)>1.5*SR],0); herv/=np.linalg.norm(herv)
def resem(w): return float(remb(w)@herv)
UT=torch.hub.load("tarepan/SpeechMOS:v1.2.0","utmos22_strong",trust_repo=True); UT.eval()
def utmos(w):
    w16=librosa.resample(w.astype(np.float32),orig_sr=SR,target_sr=16000)
    with torch.no_grad(): return round(float(UT(torch.from_numpy(w16)[None],16000)),3)

def contour_bank(w):
    """list of dicts: shape (len-norm 64, register+range removed), nframes, range_st, final(bool)."""
    f0=parselmouth.Sound(w.astype(np.float64),sampling_frequency=SR).to_pitch(0.01,100,500).selected_array['frequency']
    iv=librosa.effects.split(w.astype(np.float32),top_db=30)
    bank=[]
    for j,(s,e) in enumerate(iv):
        a=int(s/(0.01*SR)); b=int(e/(0.01*SR)); seg=f0[a:b]; v=seg>0
        if v.sum()<12: continue
        c=st(seg[v]); rng=np.percentile(c,90)-np.percentile(c,10)
        cc=c-np.mean(c); s_=np.std(cc); cc=cc/s_ if s_>1e-6 else cc
        cc=np.interp(np.linspace(0,1,64),np.linspace(0,1,len(cc)),cc)
        gap_after=(iv[j+1][0]-e)/SR if j+1<len(iv) else 9.9
        bank.append(dict(shape=cc, nfr=int(v.sum()), rng=float(rng), final=gap_after>0.4))
    return bank

HER_BANK=contour_bank(hy)
her_ranges=np.array([b['rng'] for b in HER_BANK]); HER_RANGE=float(np.median(her_ranges))
print(f"her contour bank: {len(HER_BANK)} runs, median range {HER_RANGE:.1f} st",flush=True)

def pick(bank, nfr, final, rng_state):
    cand=[b for b in bank if b['final']==final] or bank
    # prefer similar length
    cand=sorted(cand, key=lambda b:abs(b['nfr']-nfr))[:max(8,len(cand)//4)]
    return cand[rng_state.randint(len(cand))]

def transplant(w, bank, target_range=None, seed=0):
    rs=np.random.RandomState(seed)
    snd=parselmouth.Sound(w.astype(np.float64),sampling_frequency=SR)
    manip=call(snd,"To Manipulation",0.01,75,500)
    pt=call(manip,"Extract pitch tier")
    iv=librosa.effects.split(w.astype(np.float32),top_db=30)
    f0=snd.to_pitch(0.01,100,500).selected_array['frequency']
    for j,(s,e) in enumerate(iv):
        ts,te=s/SR,e/SR
        a=int(s/(0.01*SR)); b=int(e/(0.01*SR)); seg=f0[a:b]; vmask=seg>0
        if vmask.sum()<12: continue
        reg=np.median(st(seg[vmask]))                      # keep clip LOCAL register (WHO)
        gap_after=(iv[j+1][0]-e)/SR if j+1<len(iv) else 9.9
        donor=pick(bank, int(vmask.sum()), gap_after>0.4, rs)
        rng = target_range if target_range else donor['rng']
        # voiced-frame times within this run
        vt=[(a+i)*0.01 for i in range(len(seg)) if seg[i]>0]
        shp=np.interp(np.linspace(0,1,len(vt)), np.linspace(0,1,64), donor['shape'])
        for t,sh in zip(vt,shp):
            new_st = reg + rng*sh
            call(pt,"Remove points between", t-0.004, t+0.004)
            call(pt,"Add point", t, float(np.clip(55*2**(new_st/12),60,600)))
    call([manip,pt],"Replace pitch tier")
    return np.array(call(manip,"Get resynthesis (overlap-add)").values[0])

# impostor bank (specificity control): pool a few female impostors' contours
IMP_BANK=[]
for f in sorted(os.listdir("imp2"))[:8]:
    w,_=librosa.load(f"imp2/{f}",sr=SR,mono=True)
    f0=parselmouth.Sound(w.astype(np.float64),sampling_frequency=SR).to_pitch(0.01,100,500).selected_array['frequency']
    vd=f0[f0>0]
    if len(vd)==0 or np.median(vd)<165: continue
    IMP_BANK+=contour_bank(w)
print(f"impostor contour bank: {len(IMP_BANK)} runs",flush=True)

S=dc.her_space()
print(f"\nher delivery self-sim {S['her_self']:.3f} | generic impostor {S['imp_sim']:.3f}  (her/not-her ~0.22)",flush=True)
clips=["ex5_0","ex5_1","ex12_0","ex12_1","egy3_0","egy5_0"]
print(f"\n{'arm':18s} {'dlv_sim->her':>12s} {'UTMOS':>7s} {'resem->her':>11s}",flush=True)
def runarm(name, fn):
    ds=[];um=[];rs=[]
    for c in clips:
        w,_=librosa.load(f"cb_ar/{c}.mp3",sr=SR,mono=True)
        out=np.nan_to_num(fn(w)) if fn else w
        ds.append(dc.delivery_sim(out)); um.append(utmos(out)); rs.append(resem(out))
        if c=="ex12_0" and fn: sf.write(f"/tmp/opv2_{name}.wav", out/(np.max(np.abs(out))+1e-9)*0.95, SR)
    print(f"{name:18s} {np.nanmean(ds):12.3f} {np.mean(um):7.3f} {np.mean(rs):11.3f}",flush=True)
    return np.nanmean(ds)
runarm("orig", None)
d_her=runarm("HER_transplant", lambda w: transplant(w, HER_BANK, HER_RANGE, seed=1))
d_imp=runarm("IMPOSTOR_ctrl", lambda w: transplant(w, IMP_BANK, HER_RANGE, seed=1))
print(f"\nSUMMARY her_transplant_dlv={d_her:.3f} vs impostor_ctrl={d_imp:.3f} vs orig (specificity: her should beat impostor)",flush=True)
print("MEASURE_DONE",flush=True)
