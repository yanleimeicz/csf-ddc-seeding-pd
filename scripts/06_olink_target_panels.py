# -*- coding: utf-8 -*-
"""Independent Olink Target 96 panels (PPMI Project 9000, CSF inflammation and cardiometabolic panels).
Output: derived/validation_chemokines_INF.csv (Supplementary Fig. S2d, Table S10).
"""
import config as cfg
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
D=cfg.D; B=cfg.P
clin=pd.read_csv(D+"/analytic_clinical.csv")
def nsd_num(v):
    s=str(v)
    return 0.0 if s.startswith('Not') else (float(s[0]) if s[0].isdigit() else np.nan)
clin['NSD_num']=clin['NSD_STAGE'].map(nsd_num)
clin=clin.set_index('PATNO')

def load_panel(fn):
    d=pd.read_csv(fn, usecols=['PATNO','EVENT_ID','ASSAY','NPX','QC_WARNING'])
    d=d[d.EVENT_ID=='BL']
    d['PATNO']=d['PATNO'].astype(str).str.replace('PPMI-','',regex=False).astype(int)  # 归一化ID
    w=d.groupby(['PATNO','ASSAY'],as_index=False).NPX.mean().pivot(index='PATNO',columns='ASSAY',values='NPX')
    return w

print("="*60+"\n验证1: 炎症面板(独立Olink Target) — 趋化因子@身体优先\n"+"="*60)
inf=load_panel(B+"/PPMI_Project_9000_CSF_INF_NPX_02Jul2026.csv")
m=inf.join(clin[['group','age','SEX']])
bf={'iRBD':1,'PD_bodyfirst':1,'Hyposmia':0,'PD_brainfirst':0}
m['bodyfirst']=m['group'].map(bf)
prod=m[m.group.isin(['iRBD','Hyposmia'])].copy()
print(f"炎症面板 iRBD={ (prod.group=='iRBD').sum() }, Hyposmia={ (prod.group=='Hyposmia').sum() }")
chemo=[c for c in inf.columns if any(k in str(c).upper() for k in ['CXCL','CCL'])]
print(f"面板中趋化因子数: {len(chemo)}")
# 趋化因子综合评分(mean z) iRBD vs Hyposmia
Z=prod[chemo].apply(lambda c:(c-c.mean())/c.std())
prod['chemo_score']=Z.mean(1)
f=smf.ols('chemo_score ~ C(group,Treatment("Hyposmia")) + age + SEX',data=prod.dropna(subset=['chemo_score','age','SEX'])).fit()
coef=[c for c in f.params.index if 'iRBD' in c][0]
print(f">> 趋化因子综合评分 iRBD vs Hyposmia: beta={f.params[coef]:+.3f}, p={f.pvalues[coef]:.4f} (正=身体优先↑, 验证发现)")
# 逐趋化因子
rows=[]
for c in chemo:
    d=prod[[c,'group','age','SEX']].dropna()
    if len(d)<30:continue
    d.columns=['y','group','age','SEX']
    ff=smf.ols('y ~ C(group,Treatment("Hyposmia")) + age + SEX',data=d).fit()
    cc=[x for x in ff.params.index if 'iRBD' in x][0]
    rows.append((c,ff.params[cc],ff.pvalues[cc]))
r=pd.DataFrame(rows,columns=['chemokine','beta_iRBD','p']); r['fdr']=multipletests(r['p'],method='fdr_bh')[1]
up=((r.beta_iRBD>0)&(r.p<0.05)).sum()
print(f">> 逐趋化因子: {len(r)}个中 {up}个在身体优先显著↑(p<0.05); FDR<0.05: {(r.fdr<0.05).sum()}")
print(r[r.p<0.05].sort_values('p').head(10).round(4).to_string(index=False))
r.to_csv(D+"/validation_chemokines_INF.csv",index=False)

print("\n"+"="*60+"\n验证2: Cardio面板(独立Olink) — DDC@身体优先 & NSD分期\n"+"="*60)
car=load_panel(B+"/PPMI_Project_9000_CSF_Cardio_NPX_02Jul2026.csv")
if 'DDC' in car.columns:
    m2=car[['DDC']].join(clin[['group','age','SEX','NSD_num','rem']])
    prod2=m2[m2.group.isin(['iRBD','Hyposmia'])].dropna(subset=['DDC','age','SEX'])
    print(f"Cardio面板 DDC: iRBD={ (prod2.group=='iRBD').sum() }, Hyposmia={ (prod2.group=='Hyposmia').sum() }")
    f2=smf.ols('DDC ~ C(group,Treatment("Hyposmia")) + age + SEX',data=prod2).fit()
    cc=[x for x in f2.params.index if 'iRBD' in x][0]
    print(f">> DDC iRBD vs Hyposmia: beta={f2.params[cc]:+.3f}, p={f2.pvalues[cc]:.4f} (验证发现)")
    # DDC vs NSD分期(全体)
    d3=m2.dropna(subset=['DDC','NSD_num','age','SEX'])
    f3=smf.ols('DDC ~ NSD_num + age + SEX',data=d3).fit()
    print(f">> DDC ~ NSD分期 (n={len(d3)}): beta={f3.params['NSD_num']:+.3f}, p={f3.pvalues['NSD_num']:.2e} (验证进展轨迹)")
    # DDC vs RBDSQ
    d4=m2.dropna(subset=['DDC','rem','age','SEX'])
    f4=smf.ols('DDC ~ rem + age + SEX',data=d4).fit()
    print(f">> DDC ~ RBDSQ (n={len(d4)}): beta={f4.params['rem']:+.4f}, p={f4.pvalues['rem']:.2e}")
else:
    print("Cardio面板未找到DDC")
print("\n验证结果已保存 derived/validation_*.csv")
