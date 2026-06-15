# -*- coding: utf-8 -*-
"""
DELIVERY OPERATOR v3 -- clean PSOLA contour transplant (tooling fixed: PSOLA is FAITHFUL, r=0.997).
GATE-1 FLOOR (same-language, validates the whole loop): her segment -> flatten (kill contour) ->
re-impose HER contour bank -> does delivery-sim->her RECOVER? If yes, machinery sound.
Then CROSS-LINGUAL: CB-Arabic orig vs HER-transplant vs IMPOSTOR-control (specificity).
Clean imposition: keep original pitch-point density; per point, new_st = run_register + target_range*donor_shape(pos).
"""
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, librosa, soundfile as sf, torch
import parselmouth
from parselmouth.praat import call
from resemblyzer import VoiceEncoder, preprocess_wav
import deliv_core as dc
SR=24000
st=lambda hz:12*np.log2(np.maximum(hz,1e-6)/55.0)
hz=lambda s:55.0*2**(np.asarray(s)/12)

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

def runs_of(w):
    return [(s,e) for s,e in librosa.effects.split(w.astype(np.float32),top_db=30)]
def f0_of(w):
    return parselmouth.Sound(w.astype(np.float64),sampling_frequency=SR).to_pitch(0.01,100,500).selected_array['frequency']

def contour_bank(w):
    f0=f0_of(w); iv=runs_of(w); bank=[]
    for j,(s,e) in enumerate(iv):
        a=int(s/(0.01*SR)); b=int(e/(0.01*SR)); seg=f0[a:b]; v=seg>0
        if v.sum()<12: continue
        c=st(seg[v]); rng=np.percentile(c,90)-np.percentile(c,10)
        cc=c-np.mean(c); sd=np.std(cc); cc=cc/sd if sd>1e-6 else cc
        cc=np.interp(np.linspace(0,1,64),np.linspace(0,1,len(cc)),cc)
        gap=(iv[j+1][0]-e)/SR if j+1<len(iv) else 9.9
        bank.append(dict(shape=cc,nfr=int(v.sum()),rng=float(rng),final=gap>0.4))
    return bank

def _edit(w, shape_fn, target_range):
    """shape_fn(run_idx, pos)->normalized shape value; target_range in st; keep run register."""
    snd=parselmouth.Sound(w.astype(np.float64),sampling_frequency=SR)
    manip=call(snd,"To Manipulation",0.01,75,500); pt=call(manip,"Extract pitch tier")
    n=call(pt,"Get number of points")
    if n<3: return w
    pts=[(call(pt,"Get time from index",i),call(pt,"Get value at index",i)) for i in range(1,n+1)]
    iv=runs_of(w); f0=f0_of(w)
    regs={}
    for ri,(s,e) in enumerate(iv):
        a=int(s/(0.01*SR)); b=int(e/(0.01*SR)); seg=f0[a:b]; v=seg>0
        regs[ri]=np.median(st(seg[v])) if v.sum()>=2 else None
    def run_of(t):
        for ri,(s,e) in enumerate(iv):
            if s/SR-0.02<=t<=e/SR+0.02: return ri,(s/SR,e/SR)
        return None,None
    call(pt,"Remove points between",0,1e9)
    for t,v in pts:
        ri,span=run_of(t)
        if ri is None or regs.get(ri) is None:
            call(pt,"Add point",t,float(v)); continue
        ts,te=span; pos=np.clip((t-ts)/(te-ts+1e-9),0,1)
        new_st=regs[ri]+target_range*shape_fn(ri,pos)
        call(pt,"Add point",t,float(np.clip(hz(new_st),60,600)))
    call([manip,pt],"Replace pitch tier")
    return np.array(call(manip,"Get resynthesis (overlap-add)").values[0])

def flatten(w):
    return _edit(w, lambda ri,pos:0.0, 0.0)

def transplant(w, bank, target_range, seed=1):
    rs=np.random.RandomState(seed); iv=runs_of(w); f0=f0_of(w)
    donors={}
    for ri,(s,e) in enumerate(iv):
        a=int(s/(0.01*SR)); b=int(e/(0.01*SR)); seg=f0[a:b]; v=seg>0
        gap=(iv[ri+1][0]-e)/SR if ri+1<len(iv) else 9.9
        cand=[d for d in bank if d['final']==(gap>0.4)] or bank
        cand=sorted(cand,key=lambda d:abs(d['nfr']-int(v.sum())))[:max(8,len(cand)//4)]
        donors[ri]=cand[rs.randint(len(cand))]['shape']
    return _edit(w, lambda ri,pos: float(np.interp(pos,np.linspace(0,1,64),donors[ri])), target_range)

HER_BANK=contour_bank(hy); HER_RANGE=float(np.median([d['rng'] for d in HER_BANK]))
IMP_BANK=[]
for f in sorted(os.listdir("imp2"))[:10]:
    w,_=librosa.load(f"imp2/{f}",sr=SR,mono=True); vd=f0_of(w); vd=vd[vd>0]
    if len(vd)==0 or np.median(vd)<165: continue
    IMP_BANK+=contour_bank(w)
S=dc.her_space()
print(f"her bank {len(HER_BANK)} runs, range {HER_RANGE:.1f}st | imp bank {len(IMP_BANK)} | her_self {S['her_self']:.3f} generic {S['imp_sim']:.3f} (thr~0.22)",flush=True)

# ---------- GATE-1 FLOOR (same-language) ----------
print("\n=== GATE-1 FLOOR (her English: baseline -> flatten -> re-impose HER) ===",flush=True)
hseg=[hy[s:e] for s,e in ivh if (e-s)>6*SR][:5]
for tag,fn in [("her_orig",None),("her_flat",flatten),("her_reHER",lambda x:transplant(x,HER_BANK,HER_RANGE))]:
    ds=[];um=[]
    for sgmt in hseg:
        out=sgmt if fn is None else np.nan_to_num(fn(sgmt))
        ds.append(dc.delivery_sim(out)); um.append(utmos(out))
    print(f"  {tag:10s} dlv_sim->her {np.nanmean(ds):.3f}  UTMOS {np.mean(um):.3f}",flush=True)

# ---------- CROSS-LINGUAL ----------
print("\n=== CROSS-LINGUAL (CB-Arabic: orig vs HER-transplant vs IMPOSTOR-ctrl) ===",flush=True)
clips=["ex5_0","ex5_1","ex12_0","ex12_1","egy3_0","egy5_0"]
for tag,fn in [("orig",None),("HER_transplant",lambda x:transplant(x,HER_BANK,HER_RANGE)),
               ("IMPOSTOR_ctrl",lambda x:transplant(x,IMP_BANK,HER_RANGE))]:
    ds=[];um=[];rs=[]
    for c in clips:
        w,_=librosa.load(f"cb_ar/{c}.mp3",sr=SR,mono=True)
        out=w if fn is None else np.nan_to_num(fn(w))
        ds.append(dc.delivery_sim(out)); um.append(utmos(out)); rs.append(resem(out))
        if c=="ex12_0" and fn: sf.write(f"/tmp/opv3_{tag}.wav", out/(np.max(np.abs(out))+1e-9)*0.95, SR)
    print(f"  {tag:16s} dlv_sim->her {np.nanmean(ds):.3f}  UTMOS {np.mean(um):.3f}  resem->her {np.mean(rs):.3f}",flush=True)
print("MEASURE_DONE",flush=True)
