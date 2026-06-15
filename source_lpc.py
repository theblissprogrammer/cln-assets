"""LPC source/filter primitives for the Stage-0 double-dissociation probe.
Residual r = inverse-filter(x) through the all-pole H; re-filtering r through a MODIFIED H
edits ONLY the filter while her literal excitation/residual rides through unchanged.
  - mcadams(a, alpha): pole-angle warp ~ vocal-tract-length / formant warp (the identity axis)
  - shift_formant(a, k, factor): move the k-th formant pole (a constriction/phoneme edit)
"""
import numpy as np, scipy.signal as sps, librosa

def analyze(x, sr, order=18, frame_ms=25.0, hop_ms=6.25):
    fl = int(frame_ms*sr/1000); hl = int(hop_ms*sr/1000)
    win = np.hanning(fl)
    x = np.asarray(x, dtype=np.float64)
    xp = np.concatenate([np.zeros(fl), x, np.zeros(2*fl)])
    nfr = 1 + (len(xp)-fl)//hl
    A=[]; E=[]
    for i in range(nfr):
        seg = xp[i*hl:i*hl+fl]*win
        try:
            a = librosa.lpc(seg + 1e-12, order=order)
        except Exception:
            a = np.r_[1.0, np.zeros(order)]
        e = sps.lfilter(a, [1.0], seg)         # whitened residual
        A.append(a); E.append(e)
    return dict(A=A, E=E, win=win, hl=hl, fl=fl, total=len(xp), pad=fl, xlen=len(x))

def synth(P, modfn=None):
    A,E,win,hl,fl,total,pad,xlen = P["A"],P["E"],P["win"],P["hl"],P["fl"],P["total"],P["pad"],P["xlen"]
    out=np.zeros(total); norm=np.zeros(total)
    for i,(a,e) in enumerate(zip(A,E)):
        am = a
        if modfn is not None:
            try: am = modfn(a)
            except Exception: am = a
        try:
            fr = sps.lfilter([1.0], am, e)
            if not np.all(np.isfinite(fr)): fr = sps.lfilter([1.0], a, e)
        except Exception:
            fr = sps.lfilter([1.0], a, e)
        s=i*hl; out[s:s+fl]+=fr*win; norm[s:s+fl]+=win**2
    norm[norm<1e-8]=1e-8
    y=out/norm
    return y[pad:pad+xlen]

def _poly_from_poles(p):
    a = np.real(np.poly(p))
    return a

def mcadams(a, alpha):
    """Warp pole angles theta -> sign(theta)*|theta|^alpha. alpha!=1 shifts formants (VTL-ish)."""
    p = np.roots(a)
    out=[]
    for z in p:
        r=np.abs(z); th=np.angle(z)
        if abs(th) > 1e-6 and r < 1.0:
            nth = np.sign(th)*(abs(th)**alpha)
            out.append(r*np.exp(1j*nth))
        else:
            out.append(z)
    a2 = _poly_from_poles(np.array(out))
    return a2 if len(a2)==len(a) else a

def shift_formant(a, k, factor):
    """Multiply the freq of the k-th lowest formant pole-pair by `factor` (a constriction edit)."""
    p = np.roots(a)
    pos = sorted([z for z in p if np.imag(z) > 1e-4 and np.abs(z) < 1.0], key=lambda z: abs(np.angle(z)))
    if len(pos) < k:
        return a
    z = pos[k-1]; r=np.abs(z); th=abs(np.angle(z))
    nth = float(np.clip(th*factor, 0.01, np.pi*0.98))
    keep=[q for q in p if not (np.isclose(q, z) or np.isclose(q, np.conj(z)))]
    keep += [r*np.exp(1j*nth), r*np.exp(-1j*nth)]
    a2 = _poly_from_poles(np.array(keep))
    return a2 if len(a2)==len(a) else a
