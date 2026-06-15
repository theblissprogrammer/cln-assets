"""WORLD source/filter manipulation for SAGE: freeze + re-impose her glottal source
(spectral tilt + aperiodicity/HNR) and transplant her F0 register, onto Arabic content.
All arms share the WORLD analysis-synthesis path so the vocoder artifact cancels in A/B/C deltas."""
import numpy as np, pyworld as pw

def decompose(x, sr):
    x = np.ascontiguousarray(x.astype(np.float64))
    f0, t = pw.harvest(x, sr)
    f0 = pw.stonemask(x, f0, t, sr)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)
    return f0, sp, ap

def synth(f0, sp, ap, sr):
    return pw.synthesize(np.ascontiguousarray(f0),
                         np.ascontiguousarray(sp),
                         np.ascontiguousarray(ap), sr)

def transplant_f0(f0, her_logmean, her_logstd):
    """Map voiced log-F0 to her register (mean+std). Her identity register, language-independent."""
    f0n = f0.copy(); v = f0 > 0
    if v.sum() < 2:
        return f0n
    lf = np.log(f0[v]); z = (lf - lf.mean()) / (lf.std() + 1e-9)
    f0n[v] = np.exp(z * her_logstd + her_logmean)
    return f0n

def tilt_correct(sp, sr, slope_delta_db_per_decade):
    """Shift overall spectral slope by slope_delta (dB/decade) — re-imposes her source tilt,
    preserves Arabic formant PEAKS (pivots about mean log-freq)."""
    nfreq = sp.shape[1]
    freqs = np.linspace(1.0, sr / 2.0, nfreq)
    logf = np.log10(freqs)
    gain_db = slope_delta_db_per_decade * (logf - logf.mean())
    gain = 10.0 ** (gain_db / 10.0)   # power-domain gain
    return sp * gain[None, :]

def ap_match(ap, hnr_delta_db):
    """her_hnr - utt_hnr; positive => her less breathy => scale aperiodicity DOWN (more periodic)."""
    factor = float(np.clip(10.0 ** (-hnr_delta_db / 20.0), 0.25, 4.0))
    return np.clip(ap * factor, 1e-6, 1.0 - 1e-6)
