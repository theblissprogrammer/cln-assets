#!/bin/bash
set -x; echo "=== CEILING (copy-synth) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1; echo FFMPEG_DONE
export ASSETS=$(pwd)
pip install -q "numpy<2" resemblyzer speechbrain vocos 2>&1 | tail -2
pip install -q descript-audio-codec 2>&1 | tail -2 || echo "dac optional"
echo "=== copy-synthesize her own segments through neutral vocoders ==="
python - <<'PY'
import librosa, soundfile as sf, numpy as np, torch, os, warnings; warnings.filterwarnings("ignore")
dev="cuda" if torch.cuda.is_available() else "cpu"
y,sr=librosa.load("her_audio.wav",sr=24000,mono=True); iv=librosa.effects.split(y,top_db=30)
segs=[y[a:b] for a,b in iv if (b-a)>3*24000][:8]
os.makedirs("out_vocos",exist_ok=True); os.makedirs("out_dac",exist_ok=True); os.makedirs("out_real",exist_ok=True)
for i,s in enumerate(segs): sf.write(f"out_real/r{i}.wav", s, 24000)
# Vocos mel-24k copy-synthesis
try:
    from vocos import Vocos
    vo=Vocos.from_pretrained("charactr/vocos-mel-24khz").to(dev)
    for i,s in enumerate(segs):
        w=torch.tensor(s).float().unsqueeze(0).to(dev)
        with torch.no_grad():
            mel=vo.feature_extractor(w); rec=vo.decode(mel)
        sf.write(f"out_vocos/v{i}.wav", rec.squeeze().cpu().numpy(), 24000)
        print("VOCOS",i,flush=True)
except Exception as e: print("VOCOS_ERR", repr(e), flush=True)
# DAC 24kHz copy-synthesis (high bitrate)
try:
    import dac
    from audiotools import AudioSignal
    m=dac.DAC.load(dac.utils.download(model_type="24khz")).to(dev)
    for i,s in enumerate(segs):
        sig=AudioSignal(s.astype(np.float32), 24000).to(dev)
        x=m.preprocess(sig.audio_data, sig.sample_rate)
        with torch.no_grad():
            z,_,_,_,_=m.encode(x); rec=m.decode(z)
        sf.write(f"out_dac/d{i}.wav", rec.squeeze().detach().cpu().numpy(), 24000)
        print("DAC",i,flush=True)
except Exception as e: print("DAC_ERR", repr(e), flush=True)
print("COPYSYNTH_DONE", flush=True)
PY
echo "=== MEASURE ceilings vs her ==="
python measure_v2.py out_real REAL-HELDOUT
python measure_v2.py out_vocos VOCOS-CEIL
python measure_v2.py out_dac DAC-CEIL
echo "=== JOB DONE $(date) ==="
