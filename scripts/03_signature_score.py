# -*- coding: utf-8 -*-
"""Discovery set: composite RBD-signature score and its relation to progression slopes and phenoconversion.
Output: derived/module5_score_outcomes.csv (Supplementary Fig. S3).
"""
import config as cfg
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from scipy import stats
from lifelines import CoxPHFitter
D=cfg.D; R=cfg.RS
CUR=cfg.CURATED
wide=pd.read_parquet(D+"/csf_npx_wide.parquet").set_index('PATNO')
clin=pd.read_csv(D+"/analytic_clinical.csv").set_index('PATNO').reindex(wide.index)
sigr=pd.read_csv(D+"/sig_rem.csv").set_index('OlinkID')
sig=sigr[sigr.fdr<0.05]

# ---- 1. 构建评分: 55个RBD信号蛋白, 权重=beta_z (标准化NPX加权和) ----
common=[o for o in sig.index if o in wide.columns]
Z=wide[common].apply(lambda c:(c-c.mean())/c.std())
w=sig.loc[common,'beta_z'].values
score=pd.Series(np.nansum(Z.values*w,axis=1), index=wide.index, name='RBDscore')
score=(score-score.mean())/score.std()
clin['RBDscore']=score
# 构念效度: iRBD应高于Hyposmia
print("评分构念效度(均值):")
print(clin.groupby('group')['RBDscore'].mean().round(3).to_string())
bf={'iRBD':1,'PD_bodyfirst':1,'Hyposmia':0,'PD_brainfirst':0}
clin['bodyfirst']=clin['group'].map(bf)
import statsmodels.formula.api as smf0
sub=clin.dropna(subset=['bodyfirst','RBDscore','age','SEX']).copy()
f0=smf0.ols('RBDscore ~ bodyfirst + age + SEX', data=sub).fit()
print(f"身体优先 vs 大脑优先 评分差异(校正年龄性别): beta={f0.params['bodyfirst']:+.3f}, p={f0.pvalues['bodyfirst']:.3f}")

# ---- 2. 纵向进展: 每人各结局年斜率 ----
print("\n读取全随访计算进展速率...")
full=pd.read_excel(CUR, sheet_name='20260511')
def slopes(var):
    s={}
    for pid,g in full[['PATNO','YEAR',var]].dropna().groupby('PATNO'):
        if g[var].notna().sum()>=2 and g['YEAR'].nunique()>=2:
            s[pid]=np.polyfit(g['YEAR'],g[var],1)[0]
    return pd.Series(s)
import statsmodels.formula.api as smf
for var,nm in [('moca','MoCA'),('scopa','SCOPA_AUT'),('updrs3_score','UPDRS3')]:
    sl=slopes(var); clin[f'slope_{var}']=clin.index.map(sl)
    d=clin.rename(columns={var:'baseval'})[[f'slope_{var}','RBDscore','age','SEX','baseval']].dropna()
    d=d.rename(columns={f'slope_{var}':'slope'})
    fit=smf.ols('slope ~ RBDscore + age + SEX + baseval', data=d).fit()
    b=fit.params['RBDscore']; p=fit.pvalues['RBDscore']
    print(f"  {nm}年变化率 ~ RBD评分 (n={len(d)}): beta={b:+.4f}, p={p:.4f}  (负=评分高者{nm}降更快)")

# ---- 3. 表型转化 Cox (iRBD组) ----
print("\niRBD组 表型转化(->PD) Cox分析:")
irbd=clin[clin.group=='iRBD'].index
fu=full[full.PATNO.isin(irbd)]
rows=[]
for pid,g in fu.groupby('PATNO'):
    g=g.sort_values('YEAR')
    conv=g[g.PRIMDIAG==1]
    if len(conv): time=conv['YEAR'].iloc[0]; ev=1
    else: time=g['YEAR'].max(); ev=0
    rows.append((pid,time,ev))
surv=pd.DataFrame(rows,columns=['PATNO','time','event']).set_index('PATNO')
surv=surv.join(clin[['RBDscore','age','SEX']]).dropna()
surv=surv[surv.time>0]
print(f"  iRBD可分析: {len(surv)}, 转化事件: {int(surv.event.sum())}")
try:
    cph=CoxPHFitter().fit(surv[['time','event','RBDscore','age','SEX']],'time','event')
    hr=cph.summary.loc['RBDscore']
    print(f"  RBD评分 HR={np.exp(hr['coef']):.2f} (95%CI {np.exp(hr['coef']-1.96*hr['se(coef)']):.2f}-{np.exp(hr['coef']+1.96*hr['se(coef)']):.2f}), p={hr['p']:.4f}")
except Exception as e:
    print("  Cox失败:",repr(e)[:120])
clin[['group','RBDscore','slope_moca','slope_scopa','slope_updrs3_score']].to_csv(D+"/module5_score_outcomes.csv")
print("\n评分与结局已存 derived/module5_score_outcomes.csv")
