# -*- coding: utf-8 -*-
"""Discovery set: pre-ranked GSEA of the iRBD-versus-hyposmia contrast (Supplementary Fig. S1, Table S9) and protein
trajectories across NSD stages (Table S12).
Output: derived/gsea_iRBD_vs_Hyposmia.csv, derived/nsd_stage_trajectory.csv. Requires internet (gseapy gene-set libraries).
"""
import config as cfg
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D=cfg.D; R=cfg.RS
wide=pd.read_parquet(D+"/csf_npx_wide.parquet").set_index('PATNO')
clin=pd.read_csv(D+"/analytic_clinical.csv").set_index('PATNO').reindex(wide.index)
ann=pd.read_csv(D+"/protein_annotation.csv").set_index('OlinkID')
prot=wide.columns.tolist(); genes=ann.reindex(prot)['Assay'].values
X=wide.values.astype(float); med=np.nanmedian(X,0); ii=np.where(np.isnan(X)); X[ii]=np.take(med,ii[1])

print("="*60); print("模块3: 轴差异 + GSEA prerank"); print("="*60)
# ---- GSEA prerank on iRBD vs Hyposmia ----
try:
    import gseapy as gp
    d1=pd.read_csv(D+"/diff_iRBD_vs_Hyposmia.csv")
    d1=d1.dropna(subset=['gene','beta_bodyfirst'])
    d1['g']=d1['gene'].astype(str).str.split('_').str[0]
    rnk=d1.groupby('g')['beta_bodyfirst'].mean().sort_values(ascending=False)
    pre=gp.prerank(rnk=rnk.reset_index(), gene_sets=['GO_Biological_Process_2021','Reactome_2022'],
                   min_size=5,max_size=500,permutation_num=1000,seed=42,outdir=None,threads=8)   # 1,000 gene-set permutations (as reported)
    r=pre.res2d.copy(); r['NES']=r['NES'].astype(float); r['FDR q-val']=r['FDR q-val'].astype(float)
    print("\niRBD(身体优先) vs Hyposmia — GSEA 富集通路 (NES>0=身体优先↑):")
    top=r.reindex(r['NES'].abs().sort_values(ascending=False).index).head(12)
    for _,x in top.iterrows():
        print(f"  NES={x['NES']:+.2f} FDR={x['FDR q-val']:.3f} | {str(x['Term'])[:60]}")
    r.to_csv(D+"/gsea_iRBD_vs_Hyposmia.csv",index=False)
except Exception as e:
    print("GSEA失败:",repr(e)[:150])

print("\n"+"="*60); print("模块4: 沿NSD分期(0->6)的蛋白轨迹"); print("="*60)
def nsd_num(v):
    s=str(v)
    if s.startswith('Not'): return 0.0
    if s[0].isdigit(): return float(s[0])
    return np.nan
clin['NSD_num']=clin['NSD_STAGE'].map(nsd_num)
t=clin['NSD_num'].values; age=clin['age'].values.astype(float); sex=clin['SEX'].values.astype(float)
m=~(np.isnan(t)|np.isnan(age)|np.isnan(sex))
Xd=np.column_stack([np.ones(m.sum()),t[m],age[m],sex[m]]); Y=X[m]
XtXi=np.linalg.inv(Xd.T@Xd); B=XtXi@Xd.T@Y
resid=Y-Xd@B; dof=m.sum()-4; s2=(resid**2).sum(0)/dof; se=np.sqrt(s2*XtXi[1,1])
beta=B[1]; p=2*stats.t.sf(np.abs(beta/se),dof); fdr=multipletests(p,method='fdr_bh')[1]
traj=pd.DataFrame({'OlinkID':prot,'gene':genes,'beta_stage':beta*t[m].std()/Y.std(0),'p':p,'fdr':fdr}).set_index('OlinkID')
print(f"NSD分期(n={m.sum()}): FDR<0.05 蛋白 {(fdr<0.05).sum()}")
print("\n随NSD分期升高↑ Top12:")
print(traj[traj.beta_stage>0].sort_values('p').head(12)[['gene','beta_stage','fdr']].round(4).to_string())
print("\n随NSD分期降低↓ Top12:")
print(traj[traj.beta_stage<0].sort_values('p').head(12)[['gene','beta_stage','fdr']].round(4).to_string())
traj.to_csv(D+"/nsd_stage_trajectory.csv")

# RBD信号蛋白沿NSD分期轨迹图
rbd=pd.read_csv(D+"/sig_rem.csv"); rbdsig=rbd[rbd.fdr<0.05]
key=['DDC','KLK6','C3','CXCL13','TGFB1','CNP']
stages=sorted([s for s in clin['NSD_num'].dropna().unique()])
fig,axes=plt.subplots(2,3,figsize=(12,7))
for ax,g in zip(axes.ravel(),key):
    oid=[o for o in prot if str(ann.reindex([o])['Assay'].values[0])==g]
    if not oid: ax.set_visible(False); continue
    vals=wide[oid[0]].reindex(clin.index)
    means=[vals[clin['NSD_num']==s].mean() for s in stages]
    sems=[vals[clin['NSD_num']==s].sem() for s in stages]
    ax.errorbar(stages,means,yerr=sems,marker='o',capsize=3)
    ax.set_title(g); ax.set_xlabel('NSD stage'); ax.set_ylabel('CSF NPX')
plt.suptitle('Key RBD-signature proteins across NSD biological stage (0-6)')
plt.tight_layout(); plt.savefig(R+"/Fig4_NSD_trajectory.png",dpi=200)
print("\nNSD轨迹图已存 results/Fig4_NSD_trajectory.png")
