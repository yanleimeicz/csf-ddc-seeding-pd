# -*- coding: utf-8 -*-
"""Discovery set: association of each CSF protein with prodromal-symptom channels (RBDSQ, UPSIT, SCOPA-AUT,
SCOPA-AUT gastrointestinal) and with the body-first axis; vectorised OLS NPX ~ channel + age + sex, BH-FDR.
Input: 00_build_discovery_dataset.py outputs. Output: derived/sig_*.csv (sig_rem.csv = RBD signature, Table S3, Fig. 2).
"""
import config as cfg
import os, numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
D=cfg.D
wide=pd.read_parquet(D+"/csf_npx_wide.parquet").set_index('PATNO')
clin=pd.read_csv(D+"/analytic_clinical.csv").set_index('PATNO').reindex(wide.index)
ann=pd.read_csv(D+"/protein_annotation.csv").set_index('OlinkID')
prot=wide.columns.tolist()
X=wide.values.astype(float)
med=np.nanmedian(X,0); ii=np.where(np.isnan(X)); X[ii]=np.take(med,ii[1])  # 中位填补
genes=ann.reindex(prot)['Assay'].values

def vec_ols(channel, name, extra_adj=True):
    """向量化: 每个蛋白 y ~ channel+age+sex, 返回beta,p (channel项)"""
    c=clin[channel].values.astype(float)
    age=clin['age'].values.astype(float); sex=clin['SEX'].values.astype(float)
    m=~(np.isnan(c)|np.isnan(age)|np.isnan(sex))
    Xd=np.column_stack([np.ones(m.sum()),c[m],age[m],sex[m]]) if extra_adj else np.column_stack([np.ones(m.sum()),c[m]])
    Y=X[m]                                   # n×P
    XtXinv=np.linalg.inv(Xd.T@Xd)
    B=XtXinv@Xd.T@Y                          # k×P
    resid=Y-Xd@B; dof=m.sum()-Xd.shape[1]
    s2=(resid**2).sum(0)/dof                 # P
    se=np.sqrt(s2*XtXinv[1,1])
    beta=B[1]; t=beta/se; p=2*stats.t.sf(np.abs(t),dof)
    # 标准化beta
    betaz=beta*np.nanstd(c[m])/Y.std(0)
    fdr=multipletests(p,method='fdr_bh')[1]
    r=pd.DataFrame({'OlinkID':prot,'gene':genes,'beta_z':betaz,'p':p,'fdr':fdr}).set_index('OlinkID')
    nsig=(fdr<0.05).sum()
    print(f"\n=== 通道 {name} (n={m.sum()}) : FDR<0.05 蛋白 {nsig} ===")
    print(r.sort_values('p').head(12)[['gene','beta_z','p','fdr']].round(4).to_string())
    r.to_csv(D+f"/sig_{channel}.csv")
    return r

# 身体优先轴(二元)
bf={'iRBD':1,'PD_bodyfirst':1,'Hyposmia':0,'PD_brainfirst':0}
clin['bodyfirst']=clin['group'].map(bf)

res={}
res['RBDSQ']       = vec_ols('rem','RBDSQ (身体优先↑)')
res['UPSIT']       = vec_ols('upsit','UPSIT (值低=嗅觉差)')
res['SCOPA_AUT']   = vec_ols('scopa','SCOPA自主神经')
res['SCOPA_GI']    = vec_ols('scopa_gi','SCOPA胃肠')
res['bodyfirst']   = vec_ols('bodyfirst','身体优先轴(iRBD/PDbody vs Hyposmia/PDbrain)')

# ---- 通道特异性 vs 共享 ----
sig={k:set(v.index[v.fdr<0.05]) for k,v in res.items()}
print("\n"+"="*60+"\n通道信号交集 (FDR<0.05 蛋白数)")
for k in sig: print(f"  {k}: {len(sig[k])}")
bfcore = sig['RBDSQ'] & sig['SCOPA_AUT']            # RBD+自主神经共同 = 身体优先核心
print(f"\n身体优先核心(RBDSQ∩SCOPA自主): {len(bfcore)}")
# 方向一致(都随身体优先升高)的核心蛋白
core=[]
for o in bfcore:
    if res['RBDSQ'].loc[o,'beta_z']>0 and res['SCOPA_AUT'].loc[o,'beta_z']>0:
        core.append((genes[prot.index(o)], round(res['bodyfirst'].loc[o,'beta_z'],3), round(res['bodyfirst'].loc[o,'fdr'],4)))
core=sorted(core,key=lambda x:-x[1])
print(f"其中方向一致(身体优先↑)蛋白 {len(core)} 个, top20:")
for g,b,q in core[:20]: print(f"   {g:12s} bodyfirstβ={b:+.3f} fdr={q}")
pd.DataFrame(core,columns=['gene','bodyfirst_beta','bodyfirst_fdr']).to_csv(D+"/bodyfirst_core_proteins.csv",index=False)
print("\n信号表已保存 derived/sig_*.csv 与 bodyfirst_core_proteins.csv")
