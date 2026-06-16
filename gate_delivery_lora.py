# -*- coding: utf-8 -*-
"""DECISIVE GATE: does the trained delivery_fc make the realized output F0-range respond to the
delivery input? Sweep f0range target (low/her/high), generate Arabic, MEASURE realized F0-range.
Monotonic response => the explicit delivery knob WORKS (genuinely-ours generation-time control).
Flat => token bottleneck wins (knob inert). Also resem/UTMOS held check."""
import os, json, warnings, numpy as np, librosa, soundfile as sf, torch
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
dev="cuda"; m=ChatterboxMultilingualTTS.from_pretrained(device=dev)
from peft import PeftModel
m.t3.tfmr=PeftModel.from_pretrained(m.t3.tfmr, "train/lora"); m.t3.tfmr=m.t3.tfmr.merge_and_unload()
print("LoRA merged into T3",flush=True)
m.t3.cond_enc.delivery_fc.load_state_dict(torch.load("train/delivery_fc.pt"))
_wn=m.t3.cond_enc.delivery_fc.weight.norm().item()
print(f"delivery_fc loaded, weight-norm={_wn:.4f} (0=untrained/inert)",flush=True)
norm=json.load(open("train/delivery_norm.json")); mu=np.array(norm["mu"]); sd=np.array(norm["sd"])
SR=m.sr
st=lambda hz:12*np.log2(np.maximum(hz,1e-6)/55.0)
def f0range(w):
    f=parselmouth.Sound(w.astype(np.float64),sampling_frequency=SR).to_pitch(0.01,65,500).selected_array['frequency']; v=f[f>0]
    return float(np.percentile(st(v),95)-np.percentile(st(v),5)) if len(v)>10 else 0.0
def f0dyn(w):
    f=parselmouth.Sound(w.astype(np.float64),sampling_frequency=SR).to_pitch(0.01,65,500).selected_array['frequency']
    return float(np.mean(np.abs(np.diff(st(f[f>0]))))/0.01) if (f>0).sum()>5 else 0.0

# her expressive ref
y,_=librosa.load("her_audio.wav",sr=24000,mono=True); iv=librosa.effects.split(y,top_db=30)
def loud(w): r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
buf=[];c=0
for s,e in iv:
    buf.append(y[s:e]); c+=e-s
    if c>20*24000: break
sf.write("her_ref.wav", loud(np.concatenate(buf)[:20*24000]), 24000)

TEXT="وقفت عند النافذة تنظر إلى الشارع الفارغ. لم تقل شيئًا لوقت طويل. ثم التفتت إليّ وابتسمت."
m.prepare_conditionals("her_ref.wav", exaggeration=0.5)
print(f"\nHER target f0range={mu[0]:.1f}  (sweeping the delivery f0range input; dyn/coupling held neutral)",flush=True)
print(f"{'target_f0range':>14s} {'z_in':>6s} {'realized(mean±sd,N=5)':>22s}",flush=True)
res=[]
NS=10
for targ in [4.0, 10.0, mu[0], 22.0, 28.0]:
    dv=np.zeros(3); dv[0]=(targ-mu[0])/sd[0]    # vary only f0range axis (z), others neutral(0)
    vals=[]
    for k in range(NS):
        wav=m.generate(TEXT, language_id="ar", exaggeration=0.5, temperature=0.4, delivery=dv.tolist())
        w=wav.squeeze().detach().cpu().numpy(); vals.append(f0range(w))
        if k==0: sf.write(f"/workspace/r/gate_r{int(targ)}.wav", w/(np.max(np.abs(w))+1e-9)*0.95, SR)
    mn=float(np.mean(vals)); res.append((targ,mn))
    print(f"{targ:14.1f} {dv[0]:6.2f}   {mn:8.1f} ± {np.std(vals):4.1f}",flush=True)
# baseline: no delivery (None) = base model, averaged
bvals=[f0range(m.generate(TEXT, language_id="ar", exaggeration=0.5, temperature=0.4, delivery=None).squeeze().detach().cpu().numpy()) for _ in range(NS)]
print(f"{'BASE(none)':>14s} {'--':>6s}   {np.mean(bvals):8.1f} ± {np.std(bvals):4.1f}",flush=True)
tr=[r[0] for r in res]; rr=[r[1] for r in res]
corr=float(np.corrcoef(tr,rr)[0,1])
print(f"\nSUMMARY response_corr(target,realized)={corr:.2f}  slope={(rr[-1]-rr[0])/(tr[-1]-tr[0]):.2f}",flush=True)
print(f"VERDICT: {'KNOB WORKS (realized F0-range tracks the delivery input)' if corr>0.6 else 'KNOB INERT (token bottleneck wins) -> delivery-conditioning insufficient at this data/tier'}",flush=True)
print("MEASURE_DONE",flush=True)
