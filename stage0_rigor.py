# -*- coding: utf-8 -*-
"""Stage-0 RIGOR: double-dissociation on HER + an IMPOSTOR, on resemblyzer AND ECAPA.
If VTL-warp collapses self-identity and constriction-shift preserves it for BOTH speakers on BOTH
encoders, the dissociation is a real property of the source-filter axis, not a metric/speaker artifact."""
import warnings, numpy as np, librosa, torch, torchaudio
warnings.filterwarnings("ignore")
from resemblyzer import VoiceEncoder, preprocess_wav
from speechbrain.inference.speaker import EncoderClassifier
from source_lpc import analyze, synth, mcadams, shift_formant
SR=16000
def loud(w):
    r=np.sqrt(np.mean(w**2))+1e-9; return w*(10**(-23/20)/r)
renc=VoiceEncoder(verbose=False)
def remb(w): e=renc.embed_utterance(preprocess_wav(loud(w),source_sr=SR)); return e/np.linalg.norm(e)
dev="cuda" if torch.cuda.is_available() else "cpu"
ec=EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",savedir="/tmp/ecapa",run_opts={"device":dev})
def eemb(w):
    with torch.no_grad(): v=ec.encode_batch(torch.tensor(loud(w)).float().unsqueeze(0).to(dev)).squeeze().cpu().numpy()
    return v/np.linalg.norm(v)
# her
y,_=librosa.load("her_audio.wav",sr=SR,mono=True); iv=librosa.effects.split(y,top_db=30)
her_segs=[y[s:e] for s,e in iv if (e-s)>3*SR][:6]
her_all=[y[s:e] for s,e in iv if (e-s)>1.5*SR]
her_rv=np.mean([remb(s) for s in her_all],0); her_rv/=np.linalg.norm(her_rv)
her_ev=np.mean([eemb(s) for s in her_all],0); her_ev/=np.linalg.norm(her_ev)
# impostor: one LibriSpeech speaker
ds=torchaudio.datasets.LIBRISPEECH(".",url="dev-clean",download=True)
spk_clips={}
for i in range(len(ds)):
    it=ds[i]; w,sr,sp=it[0].squeeze().numpy(),it[1],it[3]
    w=librosa.resample(w,orig_sr=sr,target_sr=SR)
    spk_clips.setdefault(sp,[]).append(w)
    if len(spk_clips)>3 and sum(len(v) for v in spk_clips.values())>40: break
imp_sp=max(spk_clips,key=lambda k:len(spk_clips[k])); imp=spk_clips[imp_sp]
imp_segs=[w for w in imp if len(w)>3*SR][:6] or imp[:6]
imp_all=[w for w in imp if len(w)>1.5*SR][:20]
imp_rv=np.mean([remb(s) for s in imp_all],0); imp_rv/=np.linalg.norm(imp_rv)
imp_ev=np.mean([eemb(s) for s in imp_all],0); imp_ev/=np.linalg.norm(imp_ev)
print(f"her segs {len(her_segs)} | impostor {imp_sp} segs {len(imp_segs)}",flush=True)
ARMS={"recon":None,"VTLwarp_0.80":lambda a:mcadams(a,0.80),"VTLwarp_1.20":lambda a:mcadams(a,1.20),"constr_F2_1.2":lambda a:shift_formant(a,2,1.2),"constr_F1_1.2":lambda a:shift_formant(a,1,1.2)}
def run(segs,rv,ev,label):
    print(f"--- {label} ---",flush=True)
    for arm,mod in ARMS.items():
        R=[];E=[]
        for sg in segs:
            ys=np.nan_to_num(synth(analyze(sg,SR,18),mod))
            R.append(float(remb(ys)@rv)); E.append(float(eemb(ys)@ev))
        print(f"SUMMARY {label} {arm:14s} resem={np.mean(R):.3f} ecapa={np.mean(E):.3f}",flush=True)
run(her_segs,her_rv,her_ev,"HER")
run(imp_segs,imp_rv,imp_ev,"IMPOSTOR")
print("READ: double-dissociation is REAL if for BOTH speakers, BOTH encoders: VTLwarp << recon AND constriction ~ recon.",flush=True)
print("MEASURE_DONE",flush=True)
