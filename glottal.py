"""Glottal / voice-quality (SOURCE channel) features: H1-H2, HNR, CPPS, spectral tilt, F0.
Robust for high-F0 female voices. parselmouth (Praat) + numpy. Each feature isolated in try/except
so one failure never kills the row. Used to test whether her SOURCE drifts cross-lingually."""
import numpy as np, librosa, warnings
warnings.filterwarnings("ignore")
import parselmouth
from parselmouth.praat import call

def _h1h2(w, sr, f0_floor=100, f0_ceil=500):
    """Mean H1-H2 (dB) over voiced frames: amp at F0 minus amp at 2*F0 (open-quotient/breathiness proxy)."""
    snd = parselmouth.Sound(w, sampling_frequency=sr)
    pitch = snd.to_pitch(0.01, f0_floor, f0_ceil)
    f0 = pitch.selected_array['frequency']; times = pitch.xs()
    n = int(0.04 * sr); vals = []
    for t, f in zip(times, f0):
        if f <= 0:
            continue
        c = int(t * sr); a = c - n // 2; b = c + n // 2
        if a < 0 or b >= len(w):
            continue
        seg = w[a:b] * np.hanning(b - a)
        S = np.abs(np.fft.rfft(seg, n=8192)); freq = np.fft.rfftfreq(8192, 1 / sr)
        def peak(fc):
            m = (freq >= 0.8 * fc) & (freq <= 1.2 * fc)
            return S[m].max() if m.any() else 1e-9
        vals.append(20 * np.log10((peak(f) + 1e-9) / (peak(2 * f) + 1e-9)))
    return float(np.mean(vals)) if vals else float('nan')

def glottal_feats(wav_or_arr, sr=24000):
    if isinstance(wav_or_arr, str):
        w, sr = librosa.load(wav_or_arr, sr=24000, mono=True)
    else:
        w = wav_or_arr.astype(np.float64)
    w = w / (np.max(np.abs(w)) + 1e-9)
    snd = parselmouth.Sound(w, sampling_frequency=sr)
    o = {}
    try:
        harm = snd.to_harmonicity_cc(0.01, 100, 0.1, 1.0)
        o['hnr'] = round(float(call(harm, "Get mean", 0, 0)), 2)
    except Exception:
        o['hnr'] = float('nan')
    try:
        pc = call(snd, "To PowerCepstrogram", 60, 0.002, 5000, 50)
        o['cpps'] = round(float(call(pc, "Get CPPS", False, 0.01, 0.001, 60, 330, 0.05,
                                     "Parabolic", 0.001, 0, "Straight", "Robust")), 2)
    except Exception:
        o['cpps'] = float('nan')
    try:
        o['h1h2'] = round(_h1h2(w, sr), 2)
    except Exception:
        o['h1h2'] = float('nan')
    try:
        S = np.abs(librosa.stft(w.astype(np.float32), n_fft=2048)) ** 2
        f = librosa.fft_frequencies(sr=sr, n_fft=2048); band = (f > 200) & (f < 8000)
        lp = 10 * np.log10(S[band].mean(1) + 1e-9)
        o['tilt'] = round(float(np.polyfit(np.log10(f[band]), lp, 1)[0]), 1)
    except Exception:
        o['tilt'] = float('nan')
    try:
        pitch = snd.to_pitch(0.01, 100, 500); f0 = pitch.selected_array['frequency']; f0 = f0[f0 > 0]
        o['f0_mean'] = round(float(np.mean(f0)), 1) if len(f0) else float('nan')
        o['f0_std'] = round(float(np.std(f0)), 1) if len(f0) else float('nan')
    except Exception:
        o['f0_mean'] = o['f0_std'] = float('nan')
    return o
