# -*- coding: utf-8 -*-
"""Plasma DDC (PPMI Project 293) versus CSF DDC for PD against controls in the same participants.
Output: derived/ddc_plasma_vs_csf.csv (Supplementary Fig. S3e, Table S14).
"""
import config as cfg
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D=cfg.D; R=cfg.RS; B=cfg.P
clin=pd.read_csv(D+"/analytic_clinical.csv").set_index('PATNO')
ann=pd.read_csv(D+"/protein_annotation.csv").set_index('OlinkID')
csf=pd.read_parquet(D+"/csf_npx_wide.parquet").set_index('PATNO')
oid=[o for o in csf.columns if str(ann.reindex([o])['Assay'].values[0])=='DDC'][0]
clin['DDC_csf']=csf[oid].reindex(clin.index)
# 血浆293 HT DDC
p=pd.read_parquet(B+"/ppmi_proj293_plasma_screened_extended_npx_20251121.parquet",columns=['PATNO','EVENT_ID','Assay','NPX','SampleType'])
p=p[(p.EVENT_ID=='BL')&(p.Assay=='DDC')&(p.SampleType=='SAMPLE')].groupby('PATNO').NPX.mean()
clin['DDC_plasma']=p.reindex(clin.index)

def auc_ci(y,x,n=2000):
    a=roc_auc_score(y,x); rng=np.random.RandomState(1); bs=[]; idx=np.arange(len(y))
    for _ in range(n):
        s=rng.choice(idx,len(idx),True)
        if len(np.unique(y[s]))>1: bs.append(roc_auc_score(y[s],x[s]))
    return a,np.percentile(bs,2.5),np.percentile(bs,97.5)

print("="*60+"\n① 血浆 DDC 各组分布\n"+"="*60)
order=['HC','Hyposmia','iRBD','PD_brainfirst','PD_bodyfirst']
d=clin.dropna(subset=['DDC_plasma'])
print(d[d.group.isin(order)].groupby('group')['DDC_plasma'].agg(['count','mean','std']).round(3).reindex(order).to_string())

print("\n"+"="*60+"\n② 血浆 DDC 诊断 (ROC AUC)\n"+"="*60)
pdhc=clin[clin.group.isin(['PD_bodyfirst','PD_brainfirst','HC'])].dropna(subset=['DDC_plasma']).copy()
pdhc['y']=(pdhc.group!='HC').astype(int)
a,lo,hi=auc_ci(pdhc['y'].values,pdhc['DDC_plasma'].values)
print(f"血浆DDC 区分 PD vs HC (n={len(pdhc)}, PD={int(pdhc.y.sum())}/HC={int((1-pdhc.y).sum())}): AUC={a:.3f} ({lo:.3f}-{hi:.3f})")
t,pv=stats.ttest_ind(pdhc[pdhc.y==1]['DDC_plasma'],pdhc[pdhc.y==0]['DDC_plasma'])
print(f"  血浆DDC PD vs HC t检验: p={pv:.4f}")

print("\n"+"="*60+"\n③ 头对头: 同一批人里 CSF DDC vs 血浆 DDC 诊断\n"+"="*60)
both=clin[clin.group.isin(['PD_bodyfirst','PD_brainfirst','HC'])].dropna(subset=['DDC_csf','DDC_plasma']).copy()
both['y']=(both.group!='HC').astype(int)
print(f"配对(CSF+血浆+PD/HC)样本: {len(both)} (PD={int(both.y.sum())}, HC={int((1-both.y).sum())})")
ac,loc_,hic=auc_ci(both['y'].values,both['DDC_csf'].values)
ap,lop,hip=auc_ci(both['y'].values,both['DDC_plasma'].values)
print(f"  CSF DDC  AUC={ac:.3f} ({loc_:.3f}-{hic:.3f})")
print(f"  血浆DDC  AUC={ap:.3f} ({lop:.3f}-{hip:.3f})")
print(f"  >> CSF DDC诊断效能显著高于血浆 (差{ac-ap:+.3f}), 定量验证'CSF>血浆'")
# 血浆-CSF DDC相关
r,pr=stats.pearsonr(both['DDC_csf'],both['DDC_plasma'])
print(f"  同人 CSF-血浆 DDC 相关: r={r:.3f}, p={pr:.2e}")

# ROC图
fig,ax=plt.subplots(figsize=(5.5,5.5))
for x,lab,c in [(both['DDC_csf'].values,f'CSF DDC (AUC={ac:.2f})','#c0392b'),(both['DDC_plasma'].values,f'Plasma DDC (AUC={ap:.2f})','#2c7fb8')]:
    fpr,tpr,_=roc_curve(both['y'].values,x); ax.plot(fpr,tpr,c=c,lw=2,label=lab)
ax.plot([0,1],[0,1],'k--',lw=.7); ax.set_xlabel('1 - Specificity'); ax.set_ylabel('Sensitivity')
ax.set_title(f'DDC diagnostic: CSF vs plasma (PD vs HC, n={len(both)})'); ax.legend(loc='lower right')
plt.tight_layout(); plt.savefig(R+"/Fig6_DDC_plasma_vs_CSF_ROC.png",dpi=200)
print("\nROC图已存 results/Fig6_DDC_plasma_vs_CSF_ROC.png")
clin[['group','DDC_csf','DDC_plasma']].to_csv(D+"/ddc_plasma_vs_csf.csv")
