# -*- coding: utf-8 -*-
"""DE-RISK the genuinely-ours build: does a TRAINED source-filter vocoder (NSF-HiFiGAN, mel+F0)
F0-scale a Chatterbox-Arabic clip CLEANLY (range up, resem/UTMOS held) where post-hoc PSOLA taxed
identity/quality? mel carries her timbre (fixed); F0 drives the source branch (scaled) -> no formant warp.
Compares: original | NSF copy-synth (tax baseline) | NSF F0-range-scaled | PSOLA F0-range-scaled."""
import sys, os, glob, warnings, numpy as np, librosa, soundfile as sf, torch
warnings.filterwarnings("ignore")
sys.path.append("so-vits-svc")
from vdecoder.nsf_hifigan.nvSTFT import STFT
from vdecoder.nsf_hifigan.models import load_model
import parselmouth
from parselmouth.praat import call

SR=44100
dev="cuda" if torch.cuda.is_available() else "cpu"
gen,h=load_model("pretrain/nsf_hifigan/model", device=dev)
print("NSF loaded: sr",h.sampling_rate,"mels",h.num_mels,"hop",h.hop_size,flush=True)
stft=STFT(h.sampling_rate,h.num_mels,h.n_fft,h.win_size,h.hop_size,h.fmin,h.fmax)
st=lambda hz:12*np.log2(np.maximum(hz,1e-6)/55.0)

def get_mel(w):
    t=torch.from_numpy(w).float().unsqueeze(0).to(dev)
    return stft.get_mel(t)   # [1, num_mels, T]

def get_f0(w, T):
    # parselmouth pitch -> align to mel frames
    snd=parselmouth.Sound(w.astype(np.float64),sampling_frequency=SR)
    p=snd.to_pitch(h.hop_size/SR, 65, 500); f=p.selected_array['frequency']
    f=np.interp(np.linspace(0,1,T), np.linspace(0,1,len(f)), f)
    # fill unvoiced by interpolation (NSF wants continuous-ish f0; 0 = unvoiced source)
    return f

def render(mel, f0):
    f0t=torch.from_numpy(f0).float().unsqueeze(0).to(dev)
    with torch.no_grad():
        y=gen(mel, f0t)
    return y.squeeze().cpu().numpy()

def f0_scale(f0, scale):
    v=f0>0; out=f0.copy()
    if v.sum()<5: return out
    med=np.median(st(f0[v]))
    out[v]=55*2**((med+(st(f0[v])-med)*scale)/12)
    return out

def rng(w):
    f=parselmouth.Sound(w.astype(np.float64),sampling_frequency=SR).to_pitch(0.01,65,500).selected_array['frequency']; vv=f[f>0]
    return float(np.percentile(st(vv),95)-np.percentile(st(vv),5)) if len(vv)>10 else 0

def psola_scale(w, scale):
    snd=parselmouth.Sound(w.astype(np.float64),sampling_frequency=SR)
    manip=call(snd,"To Manipulation",0.01,75,500); pt=call(manip,"Extract pitch tier")
    n=call(pt,"Get number of points"); reg=call(pt,"Get mean (curve)",0,0)
    pts=[(call(pt,"Get time from index",i),call(pt,"Get value at index",i)) for i in range(1,n+1)]
    call(pt,"Remove points between",0,1e9)
    for t,vv in pts: call(pt,"Add point",t,float(np.clip(55*2**((st(reg)+(st(vv)-st(reg))*scale)/12),60,600)))
    call([manip,pt],"Replace pitch tier")
    return np.array(call(manip,"Get resynthesis (overlap-add)").values[0])

# measurement
from resemblyzer import VoiceEncoder, preprocess_wav
import importlib.util
enc=VoiceEncoder(verbose=False)
def loud(x): r=np.sqrt(np.mean(x**2))+1e-9; return x*(10**(-23/20)/r)
def remb(w):
    w16=librosa.resample(w.astype(np.float32),orig_sr=SR,target_sr=16000)
    e=enc.embed_utterance(preprocess_wav(loud(w16),source_sr=16000)); return e/np.linalg.norm(e)
hy,_=librosa.load("her_audio.wav",sr=SR,mono=True);
import librosa as L
ivh=L.effects.split(hy,top_db=30); herv=np.mean([remb(hy[s:e]) for s,e in ivh if (e-s)>1.5*SR],0); herv/=np.linalg.norm(herv)
def resem(w): return float(remb(w)@herv)
UT=torch.hub.load("tarepan/SpeechMOS:v1.2.0","utmos22_strong",trust_repo=True); UT.eval()
def utmos(w):
    w16=librosa.resample(w.astype(np.float32),orig_sr=SR,target_sr=16000)
    with torch.no_grad(): return round(float(UT(torch.from_numpy(w16)[None],16000)),3)

print(f"\n{'clip/arm':22s} {'f0range':>8s} {'resem':>7s} {'UTMOS':>7s}",flush=True)
for f in sorted(glob.glob("cb_ar/ex12_0.mp3"))+sorted(glob.glob("best_clips/msa_e1.0.wav"))[:1]:
    w,_=librosa.load(f,sr=SR,mono=True)
    name=f.split("/")[-1][:12]
    mel=get_mel(w); T=mel.shape[-1]; f0=get_f0(w,T)
    nsf_cs=render(mel,f0)                          # copy-synth (vocoder tax baseline)
    nsf_sc=render(mel,f0_scale(f0,1.6))            # F0-range scaled via NSF (clean?)
    ps_sc=np.nan_to_num(psola_scale(w,1.6))        # F0-range scaled via PSOLA (baseline)
    for tag,sig in [("orig",w),("NSF_copysynth",nsf_cs),("NSF_f0x1.6",nsf_sc),("PSOLA_f0x1.6",ps_sc)]:
        sig=np.nan_to_num(sig)
        print(f"{name+'/'+tag:22s} {rng(sig):8.1f} {resem(sig):7.3f} {utmos(sig):7.3f}",flush=True)
        sf.write(f"/workspace/r/out_{name}_{tag}.wav", sig/(np.max(np.abs(sig))+1e-9)*0.95, SR)
print("MEASURE_DONE",flush=True)
