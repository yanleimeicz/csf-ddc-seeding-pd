# -*- coding: utf-8 -*-
"""Discovery set: CSF DDC by clinical group, NSD stage, striatal binding ratio and progression.
Output: derived/ddc_deepdive.csv (Fig. 1, Supplementary Fig. S3, Table S14).
"""
import config as cfg
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import roc_auc_score
from lifelines import CoxPHFitter
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D=cfg.D; R=cfg.RS; CUR=cfg.CURATED
wide=pd.read_parquet(D+"/csf_npx_wide.parquet").set_index('PATNO')
ann=pd.read_csv(D+"/protein_annotation.csv").set_index('OlinkID')
clin=pd.read_csv(D+"/analytic_clinical.csv").set_index('PATNO')
oid=[o for o in wide.columns if str(ann.reindex([o])['Assay'].values[0])=='DDC'][0]
clin['DDC']=wide[oid].reindex(clin.index)
def nsd_num(v):
    s=str(v); return 0.0 if s.startswith('Not') else (float(s[0]) if s[0].isdigit() else np.nan)
clin['NSD_num']=clin['NSD_STAGE'].map(nsd_num)

d=clin.dropna(subset=['DDC'])
print("="*60+"\n① DDC 各组分布 (CSF, Explore HT)\n"+"="*60)
order=['HC','Hyposmia','iRBD','PD_brainfirst','PD_bodyfirst']
print(d[d.group.isin(order)].groupby('group')['DDC'].agg(['count','mean','std']).round(3).reindex(order).to_string())
groups=[d[d.group==g]['DDC'].dropna() for g in order]
H,pk=stats.kruskal(*groups); print(f"Kruskal-Wallis across groups: H={H:.1f}, p={pk:.2e}")

print("\n"+"="*60+"\n② DDC 诊断效能 (ROC AUC)\n"+"="*60)
def auc_ci(y,x,n=1000):
    a=roc_auc_score(y,x); rng=np.random.RandomState(42); bs=[]
    idx=np.arange(len(y))
    for _ in range(n):
        s=rng.choice(idx,len(idx),replace=True)
        if len(np.unique(y[s]))<2: continue
        bs.append(roc_auc_score(y[s],x[s]))
    return a,np.percentile(bs,2.5),np.percentile(bs,97.5)
# NSD阳性 vs 阴性
dd=clin.dropna(subset=['DDC','NSD_Status'])
a,lo,hi=auc_ci(dd['NSD_Status'].values.astype(int),dd['DDC'].values)
print(f"DDC 区分 NSD阳性 vs 阴性 (n={len(dd)}): AUC={a:.3f} (95%CI {lo:.3f}-{hi:.3f})")
# 突触核蛋白病(iRBD+Hyposmia+PD) vs HC
dd2=clin[clin.group.isin(order)].dropna(subset=['DDC']).copy()
dd2['syn']=(dd2.group!='HC').astype(int)
a2,lo2,hi2=auc_ci(dd2['syn'].values,dd2['DDC'].values)
print(f"DDC 区分 突触核蛋白病 vs HC (n={len(dd2)}): AUC={a2:.3f} (95%CI {lo2:.3f}-{hi2:.3f})")

print("\n"+"="*60+"\n③ DDC vs 多巴胺能影像(DaTscan SBR)\n"+"="*60)
cur=pd.read_excel(CUR,sheet_name='20260511'); cbl=cur[cur.EVENT_ID=='BL'].set_index('PATNO')
for reg in ['MIA_PUTAMEN_L','MIA_PUTAMEN_R','MIA_CAUDATE_L','MIA_CAUDATE_R']:
    clin[reg]=cbl[reg].reindex(clin.index)
clin['striatum_mean']=clin[['MIA_PUTAMEN_L','MIA_PUTAMEN_R','MIA_CAUDATE_L','MIA_CAUDATE_R']].mean(1)
dd3=clin.dropna(subset=['DDC','striatum_mean','age','SEX'])
f=smf.ols('DDC ~ striatum_mean + age + SEX',data=dd3).fit()
print(f"DDC ~ 纹状体SBR均值 (n={len(dd3)}): beta={f.params['striatum_mean']:+.3f}, p={f.pvalues['striatum_mean']:.2e}")
print(" (负=多巴胺能越低DDC越高, 符合代偿性AADC上调假说)" if f.params['striatum_mean']<0 else "")

print("\n"+"="*60+"\n④ DDC 预后: 进展速率 & iRBD转化\n"+"="*60)
mo=pd.read_csv(D+"/module5_score_outcomes.csv").set_index('PATNO')
clin=clin.join(mo[['slope_moca','slope_scopa','slope_updrs3_score']])
for v,nm in [('slope_moca','MoCA'),('slope_scopa','SCOPA'),('slope_updrs3_score','UPDRS3')]:
    dv=clin.dropna(subset=['DDC',v,'age','SEX'])
    fv=smf.ols(f'{v} ~ DDC + age + SEX',data=dv).fit()
    print(f"  {nm}年变化率 ~ DDC (n={len(dv)}): beta={fv.params['DDC']:+.4f}, p={fv.pvalues['DDC']:.4f}")
# iRBD转化 Cox
full=cur
irbd=clin[clin.group=='iRBD'].index
rows=[]
for pid,g in full[full.PATNO.isin(irbd)].groupby('PATNO'):
    g=g.sort_values('YEAR'); conv=g[g.PRIMDIAG==1]
    rows.append((pid, conv['YEAR'].iloc[0] if len(conv) else g['YEAR'].max(), 1 if len(conv) else 0))
surv=pd.DataFrame(rows,columns=['PATNO','time','event']).set_index('PATNO').join(clin[['DDC','age','SEX']]).dropna()
surv=surv[surv.time>0]
try:
    cph=CoxPHFitter().fit(surv[['time','event','DDC','age','SEX']],'time','event')
    h=cph.summary.loc['DDC']
    print(f"  iRBD转化(n={len(surv)},事件{int(surv.event.sum())}): DDC HR={np.exp(h['coef']):.2f} (95%CI {np.exp(h['coef']-1.96*h['se(coef)']):.2f}-{np.exp(h['coef']+1.96*h['se(coef)']):.2f}), p={h['p']:.4f}")
except Exception as e: print("  Cox失败:",repr(e)[:100])

# ---- 图: DDC多面板 ----
fig,ax=plt.subplots(1,3,figsize=(14,4.2))
d[d.group.isin(order)].boxplot('DDC',by='group',ax=ax[0]); ax[0].set_title('DDC by group'); ax[0].set_xlabel(''); plt.sca(ax[0]); plt.xticks(rotation=30,fontsize=7)
st=sorted(clin['NSD_num'].dropna().unique())
ax[1].errorbar(st,[clin[clin.NSD_num==s]['DDC'].mean() for s in st],yerr=[clin[clin.NSD_num==s]['DDC'].sem() for s in st],marker='o',capsize=3)
ax[1].set_title('DDC across NSD stage'); ax[1].set_xlabel('NSD stage'); ax[1].set_ylabel('DDC NPX')
ax[2].scatter(clin['striatum_mean'],clin['DDC'],s=8,alpha=.4); ax[2].set_xlabel('Striatal SBR (mean)'); ax[2].set_ylabel('DDC NPX'); ax[2].set_title('DDC vs dopaminergic imaging')
plt.suptitle('DDC as a dual biomarker of onset-mode and progression (PPMI CSF)')
plt.tight_layout(); plt.savefig(R+"/Fig5_DDC_deepdive.png",dpi=200)
print("\nDDC多面板图已存 results/Fig5_DDC_deepdive.png")
clin[['group','DDC','NSD_num','striatum_mean']].to_csv(D+"/ddc_deepdive.csv")
