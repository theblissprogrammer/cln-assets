# -*- coding: utf-8 -*-
"""Fine-tune Chatterbox-T3 (LoRA) ON HER hours of speech to BAKE IN her way-of-talking via TRAINING
(not a control dial — the dial was exhaustively falsified). Standard next-token speech CE on her big
corpus, speaker-conditioned. No delivery_fc. Save LoRA. Eval (separate) = does her delivery become
more-her (EN + cross-lingual AR) than zero-shot, identity held."""
import os, csv, warnings, numpy as np, librosa, torch, torch.nn.functional as F, random
warnings.filterwarnings("ignore")
try:
    import perth
    if getattr(perth,"PerthImplicitWatermarker",None) is None:
        class _N:
            def __init__(s,*a,**k):pass
            def apply_watermark(s,w,*a,**k):return w
            def get_watermark(s,*a,**k):return None
        perth.PerthImplicitWatermarker=_N
except Exception: pass
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, punc_norm
from chatterbox.models.t3.modules.cond_enc import T3Cond
from peft import LoraConfig, get_peft_model
S3_SR=16000; dev="cuda"
m=ChatterboxMultilingualTTS.from_pretrained(device=dev); t3=m.t3
lconf=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                 target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
t3.tfmr=get_peft_model(t3.tfmr, lconf)
print("LoRA on T3:", sum(p.numel() for p in t3.parameters() if p.requires_grad), "trainable params", flush=True)

rows=list(csv.DictReader(open("train/manifest.csv")))
data=[]
for r in rows:
    try:
        wav,_=librosa.load(r["wav"], sr=S3_SR, mono=True)
        sp,_=m.s3gen.tokenizer.forward([wav]); sp=torch.atleast_2d(sp).to(dev).long()
        if sp.size(1)<8: continue
        tt=m.tokenizer.text_to_tokens(punc_norm(r["text"]), language_id="en").to(dev)
        tt=F.pad(tt,(1,0),value=t3.hp.start_text_token); tt=F.pad(tt,(0,1),value=t3.hp.stop_text_token)
        ve=torch.from_numpy(m.ve.embeds_from_wavs([wav], sample_rate=S3_SR)).mean(0,keepdim=True).to(dev)
        cpt,_=m.s3gen.tokenizer.forward([wav[:6*S3_SR]], max_len=t3.hp.speech_cond_prompt_len)
        cpt=torch.atleast_2d(cpt).to(dev).long()
        data.append((tt,sp,ve,cpt))
    except Exception as e:
        print("skip:",str(e)[:60],flush=True)
print(f"prepared {len(data)} utterances", flush=True)
trainables=[p for p in t3.parameters() if p.requires_grad]
opt=torch.optim.AdamW(trainables, lr=1e-4); t3.train()
order=list(range(len(data))); random.seed(0)
STEPS=6000
for step in range(STEPS):
    if step%len(data)==0: random.shuffle(order)
    tt,sp,ve,cpt=data[order[step%len(data)]]
    cond=T3Cond(speaker_emb=ve, cond_prompt_speech_tokens=cpt, emotion_adv=0.5*torch.ones(1,1,1,device=dev)).to(device=dev)
    out=t3.forward(t3_cond=cond, text_tokens=tt, text_token_lens=torch.tensor([tt.size(1)],device=dev),
                   speech_tokens=sp, speech_token_lens=torch.tensor([sp.size(1)],device=dev), training=True)
    ls=F.cross_entropy(out.speech_logits[:, :-1, :].transpose(1,2), sp[:, 1:])
    opt.zero_grad(); ls.backward(); opt.step()
    if step%200==0: print(f"step {step} loss_speech {ls.item():.4f}", flush=True)
t3.tfmr.save_pretrained("train/lora_ft")
print("TRAIN_DONE", flush=True)
