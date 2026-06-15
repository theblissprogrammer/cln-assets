# -*- coding: utf-8 -*-
"""
DELIVERY FINGERPRINT extractor — the formalization of Ahmed's "way of talking".

Identity, decomposed (Ahmed's two-layer thesis):
  LAYER 1 = the finite language-specific LETTER/PHONEME sounds  -> fused with timbre, REGENERATE per language
  LAYER 2 = the DELIVERY FINGERPRINT = "the way she talks"      -> language-INVARIANT, TRANSFER cross-lingually

This module measures LAYER 2 as a vector of mostly-language-invariant features, grouped:

  G1 VOICE QUALITY / GLOTTAL SOURCE  (the grain of her voice; language-invariant)
       hnr cpps h1h2 tilt jitter shimmer
  G2 F0 DYNAMICS  (how she moves pitch; SHAPE language-invariant, register part-anatomy)
       f0_med f0_range_st f0_iqr_st f0_dyn_st_s declination_st_s voiced_frac
  G3 TIMING / RHYTHM  (her rhythm + pausing; partly language-BOUND -> flagged)
       artic_rate pause_rate pause_mean pause_frac rhythm_irreg
  G4 ENERGY / EMPHASIS DYNAMICS  (how she stresses; language-invariant)
       rms_range_db rms_std_db emphasis_peak f0_energy_corr

Every feature isolated in try/except so one failure never kills a row. Praat(parselmouth)+librosa+numpy.
"""
import numpy as np, librosa, warnings
warnings.filterwarnings("ignore")
import parselmouth
from parselmouth.praat import call

SR = 24000

# language-invariant subset used for SPEAKER-DISCRIMINATION (drop the language-bound timing feats)
INVARIANT = ["hnr","cpps","h1h2","tilt","jitter","shimmer",
             "f0_range_st","f0_iqr_st","f0_dyn_st_s","declination_st_s",
             "rms_range_db","rms_std_db","emphasis_peak","f0_energy_corr"]
TIMING    = ["artic_rate","pause_rate","pause_mean","pause_frac","rhythm_irreg"]  # language-bound, report separately
ALLFEATS  = ["hnr","cpps","h1h2","tilt","jitter","shimmer",
             "f0_med","f0_range_st","f0_iqr_st","f0_dyn_st_s","declination_st_s","voiced_frac",
             "artic_rate","pause_rate","pause_mean","pause_frac","rhythm_irreg",
             "rms_range_db","rms_std_db","emphasis_peak","f0_energy_corr"]

def _safe(d, k, fn):
    try:
        v = fn()
        d[k] = float(v) if v is not None and np.isfinite(v) else float('nan')
    except Exception:
        d[k] = float('nan')

def _h1h2(w, sr, f0):
    n = int(0.04 * sr); vals = []
    pitch_t = f0['t']; pitch_f = f0['f']
    for t, f in zip(pitch_t, pitch_f):
        if f <= 0: continue
        c = int(t * sr); a = c - n // 2; b = c + n // 2
        if a < 0 or b >= len(w): continue
        seg = w[a:b] * np.hanning(b - a)
        S = np.abs(np.fft.rfft(seg, n=8192)); freq = np.fft.rfftfreq(8192, 1 / sr)
        def peak(fc):
            m = (freq >= 0.8 * fc) & (freq <= 1.2 * fc)
            return S[m].max() if m.any() else 1e-9
        vals.append(20 * np.log10((peak(f) + 1e-9) / (peak(2 * f) + 1e-9)))
    return float(np.mean(vals)) if vals else float('nan')

def fingerprint(wav_or_arr, sr=SR):
    if isinstance(wav_or_arr, str):
        w, sr = librosa.load(wav_or_arr, sr=SR, mono=True)
    else:
        w = np.asarray(wav_or_arr, dtype=np.float64);
    w = w / (np.max(np.abs(w)) + 1e-9)
    snd = parselmouth.Sound(w, sampling_frequency=sr)
    o = {}

    # --- pitch track (shared) ---
    pitch = snd.to_pitch(0.01, 100, 500)
    f0v = pitch.selected_array['frequency']; tv = pitch.xs()
    f0 = {'t': tv, 'f': f0v}
    voiced = f0v[f0v > 0]
    st = lambda hz: 12*np.log2(np.maximum(hz,1e-6)/55.0)  # semitones re 55Hz

    # ===== G1 VOICE QUALITY / GLOTTAL SOURCE =====
    _safe(o,'hnr',  lambda: call(snd.to_harmonicity_cc(0.01,100,0.1,1.0), "Get mean", 0, 0))
    _safe(o,'cpps', lambda: call(call(snd,"To PowerCepstrogram",60,0.002,5000,50),
                                 "Get CPPS", False,0.01,0.001,60,330,0.05,"Parabolic",0.001,0,"Straight","Robust"))
    _safe(o,'h1h2', lambda: _h1h2(w, sr, f0))
    def _tilt():
        S = np.abs(librosa.stft(w.astype(np.float32), n_fft=2048))**2
        f = librosa.fft_frequencies(sr=sr, n_fft=2048); band = (f>200)&(f<8000)
        lp = 10*np.log10(S[band].mean(1)+1e-9)
        return np.polyfit(np.log10(f[band]), lp, 1)[0]
    _safe(o,'tilt', _tilt)
    pp = call(snd, "To PointProcess (periodic, cc)", 100, 500)
    _safe(o,'jitter',  lambda: call(pp, "Get jitter (local)", 0,0,0.0001,0.02,1.3)*100)   # %
    _safe(o,'shimmer', lambda: call([snd,pp], "Get shimmer (local)", 0,0,0.0001,0.02,1.3,1.6)*100)  # %

    # ===== G2 F0 DYNAMICS =====
    _safe(o,'f0_med',     lambda: np.median(voiced) if len(voiced) else np.nan)
    _safe(o,'f0_range_st',lambda: (np.percentile(st(voiced),95)-np.percentile(st(voiced),5)) if len(voiced)>10 else np.nan)
    _safe(o,'f0_iqr_st',  lambda: (np.percentile(st(voiced),75)-np.percentile(st(voiced),25)) if len(voiced)>10 else np.nan)
    def _dyn():
        s = st(f0v.copy()); m = f0v>0
        d = np.abs(np.diff(s[m]))           # st change between consecutive voiced frames
        return np.mean(d)/0.01              # st per second (frame hop 10ms)
    _safe(o,'f0_dyn_st_s', _dyn)
    def _declin():
        # mean within-voiced-run slope (st/s): her tendency to drift pitch down across a phrase
        m = f0v>0; runs=[]; cur=[]
        for i,v in enumerate(m):
            if v: cur.append(i)
            elif cur: runs.append(cur); cur=[]
        if cur: runs.append(cur)
        sl=[]
        for r in runs:
            if len(r)<15: continue
            y = st(f0v[r]); x = np.array(r)*0.01
            sl.append(np.polyfit(x,y,1)[0])
        return np.mean(sl) if sl else np.nan
    _safe(o,'declination_st_s', _declin)
    _safe(o,'voiced_frac', lambda: float(np.mean(f0v>0)))

    # ===== G3 TIMING / RHYTHM (language-bound; report separately) =====
    iv = librosa.effects.split(w.astype(np.float32), top_db=30)
    dur = len(w)/sr
    segdur = np.array([(e-s)/sr for s,e in iv]) if len(iv) else np.array([])
    gaps = []
    for i in range(1,len(iv)):
        gaps.append((iv[i][0]-iv[i-1][1])/sr)
    gaps = np.array([g for g in gaps if g>0.05])
    _safe(o,'artic_rate', lambda: len(iv)/dur if dur>0 else np.nan)         # voiced segs / s
    _safe(o,'pause_rate', lambda: len(gaps)/dur if dur>0 else np.nan)
    _safe(o,'pause_mean', lambda: float(np.mean(gaps)) if len(gaps) else 0.0)
    _safe(o,'pause_frac', lambda: float(np.sum(gaps)/dur) if dur>0 else np.nan)
    _safe(o,'rhythm_irreg', lambda: float(np.std(segdur)/(np.mean(segdur)+1e-9)) if len(segdur)>2 else np.nan)

    # ===== G4 ENERGY / EMPHASIS DYNAMICS =====
    rms = librosa.feature.rms(y=w.astype(np.float32), frame_length=1024, hop_length=256)[0]
    rms_db = 20*np.log10(rms+1e-6); rdb = rms_db[rms_db > rms_db.max()-40]  # ignore deep silence
    _safe(o,'rms_range_db', lambda: float(np.percentile(rdb,95)-np.percentile(rdb,5)) if len(rdb)>10 else np.nan)
    _safe(o,'rms_std_db',   lambda: float(np.std(rdb)) if len(rdb)>2 else np.nan)
    def _emph():
        from scipy.signal import find_peaks
        env = rms/(rms.max()+1e-9)
        pk,_ = find_peaks(env, height=0.4, distance=10)
        if len(pk)<2: return np.nan
        prom = env[pk]
        return float(np.mean(prom)/ (np.mean(env)+1e-9))   # how much peaks stand above mean
    _safe(o,'emphasis_peak', _emph)
    def _fecorr():
        # couple pitch & loudness? corr of f0 and energy over voiced frames (emphasis style)
        hop = 0.01
        rms_t = librosa.feature.rms(y=w.astype(np.float32), frame_length=int(0.025*sr), hop_length=int(hop*sr))[0]
        n = min(len(f0v), len(rms_t)); m = f0v[:n]>0
        if m.sum()<10: return np.nan
        a = st(f0v[:n][m]); b = 20*np.log10(rms_t[:n][m]+1e-6)
        return float(np.corrcoef(a,b)[0,1])
    _safe(o,'f0_energy_corr', _fecorr)
    return o


if __name__ == "__main__":
    import sys
    fp = fingerprint(sys.argv[1])
    for k in ALLFEATS:
        print(f"{k:18s} {fp.get(k, float('nan')):.3f}")
