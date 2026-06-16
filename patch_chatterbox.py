# -*- coding: utf-8 -*-
"""Patch installed Chatterbox to add an explicit DELIVERY-conditioning input (zero-init -> base-safe).
Extends the proven emotion_adv conditioning path from scalar to scalar + delivery-vector.
Edits: t3_config.py (delivery_dim), cond_enc.py (T3Cond.delivery + zero-init delivery_fc + forward add),
mtl_tts.py (thread delivery through generate)."""
import chatterbox, os
base=os.path.dirname(chatterbox.__file__)

# 1) t3_config.py
p=base+"/models/t3/modules/t3_config.py"; s=open(p).read()
if "delivery_dim" not in s:
    s=s.replace("self.emotion_adv = True", "self.emotion_adv = True\n        self.delivery_dim = 3")
    open(p,"w").write(s); print("patched t3_config")

# 2) cond_enc.py
p=base+"/models/t3/modules/cond_enc.py"; s=open(p).read()
if "delivery" not in s:
    s=s.replace("    emotion_adv: Optional[Tensor] = 0.5",
                "    emotion_adv: Optional[Tensor] = 0.5\n    delivery: Optional[Tensor] = None")
    s=s.replace(
"""        self.emotion_adv_fc = None
        if hp.emotion_adv:
            self.emotion_adv_fc = nn.Linear(1, hp.n_channels, bias=False)""",
"""        self.emotion_adv_fc = None
        if hp.emotion_adv:
            self.emotion_adv_fc = nn.Linear(1, hp.n_channels, bias=False)
        self.delivery_fc = None
        if hp.emotion_adv and getattr(hp, 'delivery_dim', 0):
            self.delivery_fc = nn.Linear(hp.delivery_dim, hp.n_channels, bias=False)
            nn.init.zeros_(self.delivery_fc.weight)""")
    s=s.replace(
"""            cond_emotion_adv = self.emotion_adv_fc(cond.emotion_adv.view(-1, 1, 1))""",
"""            cond_emotion_adv = self.emotion_adv_fc(cond.emotion_adv.view(-1, 1, 1))
            if getattr(self, 'delivery_fc', None) is not None and getattr(cond, 'delivery', None) is not None:
                _B = cond_emotion_adv.shape[0]
                cond_emotion_adv = cond_emotion_adv + self.delivery_fc(cond.delivery.view(_B, 1, -1))""")
    open(p,"w").write(s); print("patched cond_enc")

# 3) mtl_tts.py: add delivery kwarg to generate + set on cond before inference
p=base+"/mtl_tts.py"; s=open(p).read()
if "delivery=None," not in s:
    s=s.replace("        top_p=1.0,\n    ):", "        top_p=1.0,\n        delivery=None,\n    ):", 1)
    s=s.replace("        # Norm and tokenize text",
"""        if delivery is not None:
            self.conds.t3.delivery = torch.tensor(delivery, dtype=torch.float32, device=self.device).view(1, 1, -1)
        # Norm and tokenize text""", 1)
    open(p,"w").write(s); print("patched mtl_tts")
print("PATCH_DONE")
