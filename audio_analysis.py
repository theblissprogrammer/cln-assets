"""Audio analysis toolkit — judge naturalness/air/emotion by measurement, not ear.
Usage: python audio_analysis.py <dir-or-wav> [label]   (prints per-file + group means)
Metrics:
  utmos      = neural no-reference naturalness MOS (1-5, higher=more natural)  [the 'ear']
  stoi/pesq  = SQUIM no-ref intelligibility/quality estimates (optional)
  hf4_8/hf8_12 = % spectral energy in 4-8k / 8-12k bands (the 'air')
  tilt       = spectral tilt (dB/decade; less negative = brighter)
  flatness   = spectral flatness (buzziness/noise proxy; lower=tonal)
  f0_std/f0_range_st = pitch dynamics (expressiveness/emotion; higher=more expressive)
  energy_std_db = loudness dynamics (expressiveness)
"""
import sys, glob, os, json, numpy as np, librosa, warnings
warnings.filterwarnings("ignore")
import torch
torch.set_num_threads(1)

UT=None
try:
    UT=torch.hub.load("tarepan/SpeechMOS:v1.2.0","utmos22_strong",trust_repo=True); UT.eval()
    print("[tool] UTMOS loaded",flush=True)
except Exception as e: print("[tool] UTMOS unavail:",str(e)[:80],flush=True)
SQ=None
try:
    from torchaudio.pipelines import SQUIM_OBJECTIVE
    SQ=SQUIM_OBJECTIVE.get_model(); SQ.eval()
    print("[tool] SQUIM loaded",flush=True)
except Exception as e: print("[tool] SQUIM unavail:",str(e)[:80],flush=True)

SER=None
try:
    import torch.nn as nn
    from transformers import Wav2Vec2Processor
    from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel
    class _Head(nn.Module):
        def __init__(s,c): super().__init__(); s.dense=nn.Linear(c.hidden_size,c.hidden_size); s.dropout=nn.Dropout(c.final_dropout); s.out_proj=nn.Linear(c.hidden_size,c.num_labels)
        def forward(s,x): x=s.dropout(x); x=torch.tanh(s.dense(x)); x=s.dropout(x); return s.out_proj(x)
    class _Emo(Wav2Vec2PreTrainedModel):
        def __init__(s,c): super().__init__(c); s.wav2vec2=Wav2Vec2Model(c); s.classifier=_Head(c); s.init_weights()
        def forward(s,x): h=s.wav2vec2(x)[0].mean(1); return s.classifier(h)
    _SERNAME="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    _proc=Wav2Vec2Processor.from_pretrained(_SERNAME); _ser=_Emo.from_pretrained(_SERNAME).eval()
    def _ser_run(w16):
        x=_proc(w16,sampling_rate=16000,return_tensors="pt").input_values
        with torch.no_grad(): v=_ser(x)[0].numpy()
        return float(v[0]),float(v[2])  # arousal, valence (0-1)
    SER=_ser_run; print("[tool] SER (arousal/valence) loaded",flush=True)
except Exception as e: print("[tool] SER unavail:",str(e)[:80],flush=True)

def analyze(p):
    o={}
    w,_=librosa.load(p,sr=16000,mono=True); w=w/(np.max(np.abs(w))+1e-9)
    t=torch.tensor(w).float().unsqueeze(0)
    if SER is not None:
        try: o["arousal"],o["valence"]=[round(x,3) for x in SER(w)]
        except Exception as e: o["ser_err"]=str(e)[:30]
    if UT is not None:
        try:
            with torch.no_grad(): o["utmos"]=round(float(UT(t,16000)),3)
        except Exception as e: o["utmos_err"]=str(e)[:30]
    if SQ is not None:
        try:
            with torch.no_grad(): st,pe,si=SQ(t)
            o["stoi"]=round(float(st),3); o["pesq"]=round(float(pe),3)
        except Exception as e: o["squim_err"]=str(e)[:30]
    w24,_=librosa.load(p,sr=24000,mono=True); w24=w24/(np.max(np.abs(w24))+1e-9)
    S=np.abs(librosa.stft(w24,n_fft=2048))**2; f=librosa.fft_frequencies(sr=24000,n_fft=2048); tot=S.sum()+1e-9
    o["hf4_8"]=round(float(S[(f>=4000)&(f<8000)].sum()/tot*100),3)
    o["hf8_12"]=round(float(S[(f>=8000)&(f<12000)].sum()/tot*100),3)
    band=(f>200)&(f<10000); lp=10*np.log10(S[band].mean(1)+1e-9)
    o["tilt"]=round(float(np.polyfit(np.log10(f[band]),lp,1)[0]),1)
    o["flatness"]=round(float(np.mean(librosa.feature.spectral_flatness(y=w24))),4)
    f0,_,_=librosa.pyin(w,fmin=70,fmax=400,sr=16000)
    f0v=f0[~np.isnan(f0)]
    if len(f0v)>5:
        o["f0_std"]=round(float(np.std(f0v)),1)
        o["f0_range_st"]=round(float(12*np.log2((np.percentile(f0v,95)+1e-9)/(np.percentile(f0v,5)+1e-9))),1)
    rms=librosa.feature.rms(y=w)[0]; o["energy_std_db"]=round(float(np.std(20*np.log10(rms+1e-9))),1)
    return o

if __name__=="__main__":
    for d in sys.argv[1:]:
        files=[d] if d.endswith(".wav") else sorted(glob.glob(f"{d}/*.wav"))
        rows=[]
        for p in files:
            a=analyze(p); rows.append(a)
            print("ANALYZE", os.path.basename(p), json.dumps(a), flush=True)
        if len(rows)>1:
            keys=set().union(*[r.keys() for r in rows]); keys={k for k in keys if not k.endswith("_err")}
            mean={k:round(float(np.mean([r[k] for r in rows if k in r])),3) for k in keys}
            print(f"MEAN[{d}]", json.dumps(mean), flush=True)
