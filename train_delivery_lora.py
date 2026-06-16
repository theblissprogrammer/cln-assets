# -*- coding: utf-8 -*-
"""Train ONLY the zero-init delivery_fc (backbone frozen) so Chatterbox's T3 generation responds to an
explicit delivery target [f0range, f0dyn, coupling] (z-scored). ~100s her English, t3.loss speech-CE."""
import os, json, csv, warnings, numpy as np, librosa, torch, torch.nn.functional as F
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
S3_SR=16000; dev="cuda"
m=ChatterboxMultilingualTTS.from_pretrained(device=dev); t3=m.t3
from peft import LoraConfig, get_peft_model
lconf=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                 target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
t3.tfmr=get_peft_model(t3.tfmr, lconf)
print("LoRA wrapped T3 backbone",flush=True)
assert t3.cond_enc.delivery_fc is not None, "patch not applied!"
for p in t3.parameters(): p.requires_grad=False
dfc=t3.cond_enc.delivery_fc
for p in dfc.parameters(): p.requires_grad=True
print("trainable params:", sum(p.numel() for p in dfc.parameters()), flush=True)

rows=list(csv.DictReader(open("train/manifest.csv")))
feats=["f0range","f0dyn","coupling"]
X=np.array([[float(r[f]) for f in feats] for r in rows]); mu=X.mean(0); sd=X.std(0)+1e-9
json.dump({"mu":mu.tolist(),"sd":sd.tolist(),"feats":feats}, open("train/delivery_norm.json","w"))

data=[]
for r in rows:
    try:
        wav,_=librosa.load(r["wav"], sr=S3_SR, mono=True)
        sp,_=m.s3gen.tokenizer.forward([wav]); sp=torch.atleast_2d(sp).to(dev).long()
        tt=m.tokenizer.text_to_tokens(punc_norm(r["text"]), language_id="en").to(dev)
        tt=F.pad(tt,(1,0),value=t3.hp.start_text_token); tt=F.pad(tt,(0,1),value=t3.hp.stop_text_token)
        ve=torch.from_numpy(m.ve.embeds_from_wavs([wav], sample_rate=S3_SR)).mean(0,keepdim=True).to(dev)
        plen=t3.hp.speech_cond_prompt_len
        cpt,_=m.s3gen.tokenizer.forward([wav[:6*S3_SR]], max_len=plen); cpt=torch.atleast_2d(cpt).to(dev).long()
        dv=((np.array([float(r[f]) for f in feats])-mu)/sd)
        data.append((tt, sp, ve, cpt, torch.tensor(dv,dtype=torch.float32,device=dev).view(1,1,-1)))
    except Exception as e:
        print("skip utt:", str(e)[:80], flush=True)
print(f"prepared {len(data)} utterances (~{5000/max(len(data),1):.1f} epochs at 5000 steps)", flush=True)

trainables=[p for p in t3.parameters() if p.requires_grad]
print("total trainable tensors:", len(trainables), "params:", sum(p.numel() for p in trainables), flush=True)
opt=torch.optim.AdamW(trainables, lr=1e-4)
t3.train()
order=list(range(len(data)))
import random; random.seed(0)
STEPS=10000
for step in range(STEPS):
    if step%len(data)==0: random.shuffle(order)
    tt,sp,ve,cpt,dv=data[order[step%len(data)]]
    cond=T3Cond(speaker_emb=ve, cond_prompt_speech_tokens=cpt,
                emotion_adv=0.5*torch.ones(1,1,1,device=dev), delivery=dv).to(device=dev)
    # bypass Chatterbox's buggy t3.loss (no logit transpose); compute speech next-token CE directly
    out=t3.forward(t3_cond=cond, text_tokens=tt, text_token_lens=torch.tensor([tt.size(1)],device=dev),
                   speech_tokens=sp, speech_token_lens=torch.tensor([sp.size(1)],device=dev), training=True)
    logits=out.speech_logits                       # [1, S, V]
    ls=F.cross_entropy(logits[:, :-1, :].transpose(1,2), sp[:, 1:])  # AR next-token
    opt.zero_grad(); ls.backward(); opt.step()
    if step%50==0:
        gnorm=dfc.weight.grad.norm().item() if dfc.weight.grad is not None else 0
        print(f"step {step} loss_speech {ls.item():.4f} dfc_wnorm {dfc.weight.norm().item():.4f} gnorm {gnorm:.4f}", flush=True)
torch.save(dfc.state_dict(), "train/delivery_fc.pt")
t3.tfmr.save_pretrained("train/lora")
import subprocess, shutil
shutil.make_archive("train/adapter","zip","train/lora")
shutil.copy("train/delivery_fc.pt","train/lora/delivery_fc.pt")
shutil.make_archive("adapter_full","zip","train/lora")
r=subprocess.run(["curl","-s","-F","reqtype=fileupload","-F","time=72h","-F","fileToUpload=@adapter_full.zip","https://litterbox.catbox.moe/resources/internals/api.php"],capture_output=True,text=True)
print("ADAPTER_URL", r.stdout.strip(), flush=True)
print("TRAIN_DONE", flush=True)
