# -*- coding: utf-8 -*-
"""Bulletproof the foundation: is DELIVERY (timbre-free) speaker-discriminative under a proper
leave-one-impostor-speaker-out protocol, and WHICH channel groups carry it (robustness to the
read-vs-conversational confound)? Logistic regression, ROC-AUC + EER, per feature-group ablation."""
import os, numpy as np, librosa, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import deliv_core as dc
SR=24000
# delivery_vector dims: [f0_range,f0_iqr,f0_dyn,declin,declin_sd, dct1..5, rms_std, fecorr, rhythm, artic, voiced_frac]
IDX={"f0dyn":[0,1,2,3,4],"shape":[5,6,7,8,9],"energy":[10,11],"rhythm":[12,13,14]}
GROUPS={"F0dyn_only":["f0dyn"],"shape_only":["shape"],"energy_only":["energy"],"rhythm_only":["rhythm"],
        "F0dyn+energy":["f0dyn","energy"],"ALL":list(IDX)}

hy,_=librosa.load("her_audio.wav",sr=SR,mono=True)
HER=[v for v in (dc.delivery_vector(s) for s in dc.segs_of(hy,4.0,4.0,40)) if v is not None]
IMP={}  # per-speaker segment vectors
for f in sorted(os.listdir("imp2")):
    w,_=librosa.load(f"imp2/{f}",sr=SR,mono=True); vd=dc.f0_track(w); vd=vd[vd>0]
    if len(vd)==0 or np.median(vd)<165: continue
    vs=[v for v in (dc.delivery_vector(s) for s in dc.segs_of(w,4.0,3.0,6)) if v is not None]
    if len(vs)>=2: IMP[f]=vs
print(f"her segs {len(HER)} | female impostor speakers {len(IMP)}",flush=True)

HER=np.array(HER)
imp_spk=list(IMP)
def cols(groups):
    c=[]; [c.extend(IDX[g]) for g in groups]; return sorted(c)

print(f"\n{'feature group':14s} {'AUC':>6s} {'EER':>6s}   (leave-one-impostor-speaker-out)",flush=True)
for gname,groups in GROUPS.items():
    c=cols(groups)
    scores=[];labels=[]
    # LOSO over impostor speakers; her split in half each fold (train/test)
    rs=np.random.RandomState(0)
    for held in imp_spk:
        tr_imp=np.vstack([np.array(IMP[s])[:,c] for s in imp_spk if s!=held])
        te_imp=np.array(IMP[held])[:,c]
        idx=rs.permutation(len(HER)); half=len(idx)//2
        tr_her=HER[idx[:half]][:,c]; te_her=HER[idx[half:]][:,c]
        sc=StandardScaler().fit(np.vstack([tr_her,tr_imp]))
        X=sc.transform(np.vstack([tr_her,tr_imp])); ytr=[1]*len(tr_her)+[0]*len(tr_imp)
        clf=LogisticRegression(max_iter=1000,C=0.5).fit(X,ytr)
        Xte=sc.transform(np.vstack([te_her,te_imp])); yte=[1]*len(te_her)+[0]*len(te_imp)
        p=clf.predict_proba(Xte)[:,1]
        scores+=list(p); labels+=yte
    auc=roc_auc_score(labels,scores)
    fpr,tpr,_=roc_curve(labels,scores); fnr=1-tpr; k=int(np.nanargmin(np.abs(fnr-fpr))); eer=(fpr[k]+fnr[k])/2
    print(f"{gname:14s} {auc:6.3f} {eer*100:5.1f}%",flush=True)
print("\nVERDICT: delivery channels that independently discriminate her = the ones the operator must transfer.",flush=True)
print("MEASURE_DONE",flush=True)
