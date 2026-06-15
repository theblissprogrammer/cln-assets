# -*- coding: utf-8 -*-
"""Detect her REAL breath events (inhales) in her English audio. Breath = unvoiced + noise-like
(high spectral flatness) + mid-low energy (above silence, below speech) + 0.15-0.6s, typically in
inter-phrase gaps. Extract + rank; these are authentic-grain identity events for cross-lingual splicing."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, librosa, soundfile as sf
import parselmouth
SR=24000
def detect_breaths(w, topn=12):
    hop=int(0.010*SR); fl=int(0.025*SR)
    rms=librosa.feature.rms(y=w.astype(np.float32),frame_length=fl,hop_length=hop)[0]
    edb=20*np.log10(rms+1e-6)
    flat=librosa.feature.spectral_flatness(y=w.astype(np.float32),n_fft=fl,hop_length=hop)[0]
    cen=librosa.feature.spectral_centroid(y=w.astype(np.float32),sr=SR,n_fft=fl,hop_length=hop)[0]
    f0=parselmouth.Sound(w.astype(np.float64),sampling_frequency=SR).to_pitch(0.01,100,500).selected_array['frequency']
    n=min(len(edb),len(flat),len(f0),len(cen))
    speech_floor=np.percentile(edb,85)            # speech level
    sil=np.percentile(edb,15)                      # silence
    lo=sil+6; hi=speech_floor-8                    # breath energy band
    cand=np.zeros(n,bool)
    for i in range(n):
        cand[i]= (f0[i]<=0) and (flat[i]>0.08) and (lo<edb[i]<hi) and (cen[i]>1500)
    # group consecutive
    ev=[]; i=0
    while i<n:
        if cand[i]:
            j=i
            while j<n and cand[j]: j+=1
            dur=(j-i)*0.010
            if 0.14<=dur<=0.65:
                s=int(i*hop); e=int(j*hop)
                seg=w[s:e]
                ev.append(dict(s=s,e=e,dur=dur,
                               rms=float(np.mean(edb[i:j])),
                               flat=float(np.mean(flat[i:j])),
                               cen=float(np.mean(cen[i:j])),
                               wav=seg))
            i=j
        else: i+=1
    # rank by breath-likeness: high flatness, mid energy, decent duration
    ev.sort(key=lambda d:-(d['flat']*1.0 + min(d['dur'],0.4)*1.5))
    return ev[:topn]

if __name__=="__main__":
    w,_=librosa.load("her_audio.wav",sr=SR,mono=True)
    ev=detect_breaths(w)
    print(f"detected {len(ev)} breath candidates:")
    for k,d in enumerate(ev):
        print(f"  b{k:2d} t={d['s']/SR:6.1f}s dur={d['dur']:.2f}s rms={d['rms']:.1f}dB flat={d['flat']:.2f} cen={d['cen']:.0f}Hz")
        sf.write(f"/tmp/breath_{k}.wav", d['wav']/(np.max(np.abs(d['wav']))+1e-9)*0.7, SR)
    # also dump a 'silence' control and a 'speech' sample stats for sanity
    print("MEASURE_DONE")
