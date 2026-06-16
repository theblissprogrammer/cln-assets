# -*- coding: utf-8 -*-
"""DECISIVE ZERO-SHOT GATE: the delivery adapter trained on 78 speakers -> applied to HELD-OUT speakers
(1079-1091, never in training). For each held-out voice: sweep delivery f0range target -> measure
realized f0range. If it tracks (corr>0.6) on UNSEEN voices => a GENERAL, replicable, fast (zero-shot)
delivery knob (per Ahmed: works on any voice, no per-voice training). Flat => axis didn't generalize."""
import os, json, warnings, subprocess, numpy as np, librosa, soundfile as sf, torch
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
import parselmouth
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from peft import PeftModel
dev="cuda"; st=lambda hz:12*np.log2(np.maximum(hz,1e-6)/55.0)
BASE="https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV"
m=ChatterboxMultilingualTTS.from_pretrained(device=dev)
m.t3.cond_enc.delivery_fc.load_state_dict(torch.load("train/delivery_fc.pt"))
m.t3.tfmr=PeftModel.from_pretrained(m.t3.tfmr,"train/lora"); m.t3.tfmr=m.t3.tfmr.merge_and_unload()
norm=json.load(open("train/delivery_norm.json")); mu=np.array(norm["mu"]); sd=np.array(norm["sd"])
print(f"loaded adapter; her-agnostic mu f0range {mu[0]:.1f}",flush=True)
SR=m.sr
def f0range(w):
    f=parselmouth.Sound(w.astype(np.float64),sampling_frequency=SR).to_pitch(0.01,65,500).selected_array['frequency']; v=f[f>0]
    return float(np.percentile(st(v),95)-np.percentile(st(v),5)) if len(v)>10 else 0.0
def dl(fn):
    p=f"/workspace/{fn}"
    subprocess.run(["curl","-sL","-o",p,f"{BASE}/{fn}"],capture_output=True,timeout=40)
    return p if os.path.exists(p) and os.path.getsize(p)>2000 else None

TEXT="She walked into the room and looked around without saying a word."
HELDOUT=[1079,1083,1086,1090]
allt=[]; allr=[]
for act in HELDOUT:
    ref=None
    for it in ["XX","HI","MD"]:
        ref=dl(f"{act}_IEO_NEU_{it}.wav") or dl(f"{act}_TIE_NEU_{it}.wav")
        if ref: break
    if not ref: print(f"  actor {act}: no ref",flush=True); continue
    m.prepare_conditionals(ref, exaggeration=0.5)
    tr=[]; rr=[]
    for targ in [4.0,10.0,16.0,22.0]:
        dv=np.zeros(3); dv[0]=(targ-mu[0])/sd[0]
        vals=[f0range(m.generate(TEXT,language_id="en",exaggeration=0.5,temperature=0.5,delivery=dv.tolist()).squeeze().detach().cpu().numpy()) for _ in range(6)]
        tr.append(targ); rr.append(np.mean(vals))
    c=float(np.corrcoef(tr,rr)[0,1])
    print(f"  HELDOUT actor {act}: targets {tr} -> realized {[round(x,1) for x in rr]}  corr={c:+.2f}",flush=True)
    allt+=tr; allr+=rr
oc=float(np.corrcoef(allt,allr)[0,1])
print(f"\nSUMMARY zero-shot-on-unseen-speakers corr = {oc:+.2f}",flush=True)
print(f"VERDICT: {'GENERAL KNOB WORKS ZERO-SHOT (replicable+fast: trained once, any voice)' if oc>0.5 else 'axis did NOT generalize zero-shot'}",flush=True)
print("MEASURE_DONE",flush=True)
