"""Shared SER loader — audeering MSP-dim wav2vec2 (arousal, dominance, valence in 0-1)."""
import torch, torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

class _Head(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.dense = nn.Linear(c.hidden_size, c.hidden_size)
        s.dropout = nn.Dropout(c.final_dropout)
        s.out_proj = nn.Linear(c.hidden_size, c.num_labels)
    def forward(s, x):
        x = s.dropout(x); x = torch.tanh(s.dense(x)); x = s.dropout(x)
        return s.out_proj(x)

class _Emo(Wav2Vec2PreTrainedModel):
    def __init__(s, c):
        super().__init__(c); s.wav2vec2 = Wav2Vec2Model(c); s.classifier = _Head(c); s.init_weights()
    def forward(s, x):
        h = s.wav2vec2(x)[0].mean(1); return s.classifier(h)

_N = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

def load_ser():
    proc = Wav2Vec2Processor.from_pretrained(_N)
    mdl = _Emo.from_pretrained(_N).eval()
    def ser(w16):
        x = proc(w16, sampling_rate=16000, return_tensors="pt").input_values
        with torch.no_grad():
            v = mdl(x)[0].numpy()
        return float(v[0]), float(v[2])  # arousal, valence
    return ser
