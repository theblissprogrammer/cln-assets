import numpy as np, librosa, soundfile as sf, scipy.signal as ss, warnings, os
warnings.filterwarnings("ignore")
from resemblyzer import VoiceEncoder, preprocess_wav
enc=VoiceEncoder(verbose=False)

# her centroid from full audio (same recipe as on-box measure.py)
y,sr=librosa.load('her_audio.wav',sr=16000,mono=True)
iv=librosa.effects.split(y,top_db=30)
segs=[y[s:e] for s,e in iv if (e-s)>2*16000][:8]
embs=[enc.embed_utterance(preprocess_wav(s,source_sr=16000)) for s in segs]
embs=[e/np.linalg.norm(e) for e in embs]
herv=np.mean(embs,0); herv/=np.linalg.norm(herv)
floor=float(np.mean([e@herv for e in embs]))
print(f"her-floor(real seg->her): {floor:.3f}", flush=True)

def simwav(wav, src_sr):
    e=enc.embed_utterance(preprocess_wav(wav, source_sr=src_sr)); e/=np.linalg.norm(e); return float(e@herv)
def simfile(p):
    w,s=librosa.load(p, sr=None, mono=True); return simwav(w, s)

def lowpass(wav, sr_in, cutoff):
    sos=ss.butter(8, cutoff/(sr_in/2), btype='low', output='sos')
    return ss.sosfiltfilt(sos, wav)

def loudnorm(wav):  # simple RMS normalize to -23 dBFS-ish
    rms=np.sqrt(np.mean(wav**2))+1e-9
    return wav*(10**(-23/20)/rms)

print("\n=== BASELINE ===", flush=True)
for p in ["src_f5.wav","src_cb.wav"]:
    print(f"{p:14s} -> her: {simfile(p):.3f}", flush=True)

print("\n=== 16kHz-BLINDNESS TEST (low-pass src_f5; if score ~unchanged at 8k, >8kHz is invisible to metric) ===", flush=True)
w,s=librosa.load("src_f5.wav", sr=24000, mono=True)
print(f"src_f5 full(24k)   -> her: {simwav(w,24000):.3f}", flush=True)
for cut in [8000,7000,6000,5000,4000]:
    wl=lowpass(w,24000,cut)
    print(f"src_f5 LP@{cut:5d}Hz -> her: {simwav(wl,24000):.3f}", flush=True)

print("\n=== CHANNEL CONTROL (loudness-normalize) ===", flush=True)
w,s=librosa.load("src_f5.wav", sr=24000, mono=True)
print(f"src_f5 raw         -> her: {simwav(w,24000):.3f}", flush=True)
print(f"src_f5 loudnorm    -> her: {simwav(loudnorm(w),24000):.3f}", flush=True)

print("\n=== COPY-SYNTHESIS CEILING (her seg resample round-trips; how high can a re-render of HER OWN audio score?) ===", flush=True)
# take a held-out her chunk (not in the 8 centroid segs ideally, but ok for ceiling estimate)
yfull,_=librosa.load('her_audio.wav',sr=24000,mono=True)
ivf=librosa.effects.split(yfull,top_db=30)
hsegs=[yfull[a:b] for a,b in ivf if (b-a)>3*24000]
if hsegs:
    hs=hsegs[len(hsegs)//2]  # middle segment
    print(f"her seg direct(24k)     -> her: {simwav(hs,24000):.3f}", flush=True)
    # 24k -> 16k -> 24k round trip (cheap re-render proxy)
    r16=librosa.resample(hs,orig_sr=24000,target_sr=16000)
    r24=librosa.resample(r16,orig_sr=16000,target_sr=24000)
    print(f"her seg 24->16->24      -> her: {simwav(r24,24000):.3f}", flush=True)
    # low-pass her seg at 8k (what the metric actually keeps)
    print(f"her seg LP@8000         -> her: {simwav(lowpass(hs,24000,8000),24000):.3f}", flush=True)
print("\nDONE", flush=True)
