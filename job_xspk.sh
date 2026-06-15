#!/bin/bash
set -x; echo "=== XSPK (cross-speaker dub: donor emotion + her identity, SER) JOB START $(date) ==="
nvidia-smi --query-gpu=name --format=csv,noheader || true
apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1; echo FFMPEG_DONE
pip install -q "numpy<2" resemblyzer coqui-tts "transformers>=4.57,<5.0" torchaudio librosa soundfile 2>&1 | tail -2
export COQUI_TOS_AGREED=1
mkdir -p donor_raw
BASE="https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV"
for f in 1001_IEO_ANG_HI 1001_IEO_ANG_MD 1001_IEO_ANG_LO 1001_TIE_ANG_XX 1001_IOM_ANG_XX 1001_IWW_ANG_XX 1001_TAI_ANG_XX 1001_MTI_ANG_XX 1001_IWL_ANG_XX 1001_ITH_ANG_XX 1001_DFA_ANG_XX 1001_ITS_ANG_XX 1001_TSI_ANG_XX 1001_WSI_ANG_XX; do
  curl -sfL "$BASE/$f.wav" -o "donor_raw/$f.wav" && echo "got $f" || { rm -f "donor_raw/$f.wav"; echo "miss $f"; }
done
find donor_raw -name "*.wav" -size -2k -delete 2>/dev/null   # drop LFS pointers / errors
echo "donor clips downloaded: $(ls donor_raw/*.wav 2>/dev/null | wc -l)"
python gen_xspk.py
echo "=== MEASURE ==="
python measure_xspk.py
echo "=== JOB DONE $(date) ==="
