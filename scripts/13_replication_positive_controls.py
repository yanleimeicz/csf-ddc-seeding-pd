# -*- coding: utf-8 -*-
"""Platform positive controls, Project 314 against Project 277: proportion below LOD and cross-set consistency of
age, sex, PD, SAA and RBDSQ effects. Output: derived/p314_posctrl_*.csv (Fig. 3, Table S7).
"""
import config as cfg
import sys, warnings
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
B=cfg.P; D=cfg.D; OUT=cfg.RS
LOG=open(OUT+"/p314_positive_controls_log.txt","w",encoding="utf-8")
def P(*a):
    s=" ".join(str(x) for x in a); print(s); LOG.write(s+"\n"); LOG.flush()

# 1) LOD
cohort733=set(pd.read_parquet(D+"/csf_npx_wide.parquet",columns=['PATNO'])['PATNO'].astype(int))
lod={}
for k,f in [('277',B+"/PPMI_Project_277_CSF_Screened_NPX_20260130.parquet"),('314',B+"/ppmi_proj314_csf_screened_extended_npx_20260116.parquet")]:
    t=pq.read_table(f,columns=['PATNO','EVENT_ID','SampleType','SampleQC','AssayType','OlinkID','NPX','LOD'])
    t=t.filter(pc.and_(pc.and_(pc.equal(t['SampleType'],'SAMPLE'),pc.equal(t['EVENT_ID'],'BL')),pc.equal(t['AssayType'],'assay'))).to_pandas()
    t=t[(t.SampleQC=='PASS')&t.PATNO.notna()]; t['PATNO']=t.PATNO.astype(int)
    t=t[t.PATNO.isin(cohort733)] if k=='277' else t[~t.PATNO.isin(cohort733)]
    lod[k]=(t.NPX<t.LOD).groupby(t.OlinkID).mean()
L=pd.DataFrame(lod).dropna(); L.to_csv(D+"/p314_posctrl_lod.csv")
P("1) 每蛋白低于 LOD 比例 (基线): 中位 277 = %.1f%%, 314 = %.1f%%; >50%% 低于 LOD 的蛋白: 277 %d, 314 %d; 两队列 LOD 比例 ρ=%.2f"
  % (100*L['277'].median(),100*L['314'].median(),(L['277']>0.5).sum(),(L['314']>0.5).sum(),stats.spearmanr(L['277'],L['314'])[0]))

# 2) 全蛋白组 β
ann=pd.read_csv(D+"/protein_annotation.csv").set_index('OlinkID')
def load(which):
    if which=='277':
        w=pd.read_parquet(D+"/csf_npx_wide.parquet").set_index('PATNO'); w.index=w.index.astype(int)
    else:
        w=pd.read_parquet(D+"/p314_wide_BL_new.parquet").set_index('PATNO')
        w=w.loc[:,w.isna().mean()<=0.2]; w=w.loc[w.isna().mean(1)<=0.3]
    return w
W={k:load(k) for k in ['277','314']}
shared=sorted(set(W['277'].columns)&set(W['314'].columns)); P(f"共有蛋白 {len(shared)}")
clin=pd.read_csv(D+"/analytic_clinical.csv").set_index('PATNO')
GEN=['LRRK2','GBA','LRRK2 + GBA','SNCA','PRKN','PINK1','PRKN + RBD']
def ols_vec(Y,Xd,j=1):
    XtXinv=np.linalg.pinv(Xd.T@Xd); Bh=XtXinv@Xd.T@Y; res=Y-Xd@Bh; dof=Y.shape[0]-Xd.shape[1]
    s2=(res**2).sum(0)/dof; se=np.sqrt(s2*XtXinv[j,j]); b=Bh[j]; return b,se,2*stats.t.sf(np.abs(b/se),dof)
out={}
for k,w in W.items():
    w=w[shared]; c=clin.reindex(w.index)
    keep=(~c.subgroup.isin(GEN))&c.group0.isin(['HC','Hyposmia','iRBD','PD'])&c.age.notna()&c.SEX.notna()
    w=w[keep.values]; c=c[keep.values]
    X=w.values.astype(float); med=np.nanmedian(X,0); ii=np.where(np.isnan(X)); X[ii]=np.take(med,ii[1]); X=(X-X.mean(0))/X.std(0)
    age=c.age.values; sex=c.SEX.values.astype(float); one=np.ones(len(c))
    za=(age-age.mean())/age.std(); zs=(sex-sex.mean())/sex.std()
    r={}
    r['age']=ols_vec(X,np.column_stack([one,za,sex]))
    r['sex']=ols_vec(X,np.column_stack([one,zs,age]))
    m=c.group0.isin(['PD','HC']).values; y=(c.group0[m]=='PD').astype(float).values
    r['PD_vs_HC']=ols_vec(X[m],np.column_stack([one[m],y,age[m],sex[m]]))
    m=c.CSFSAA.isin([0,1]).values
    r['SAA']=ols_vec(X[m],np.column_stack([one[m],c.CSFSAA[m].values.astype(float),age[m],sex[m]]))
    m=c.rem.notna().values; zr=(c.rem[m]-c.rem[m].mean())/c.rem[m].std()
    r['RBDSQ']=ols_vec(X[m],np.column_stack([one[m],zr.values,age[m],sex[m]]))
    out[k]=r; P(f"  {k}: n={len(c)} ({c.group0.value_counts().to_dict()})")
rows=[]; allb=[]
P("\n2) 全蛋白组 β 一致性 (蛋白已 z 化; 年龄/性别/RBDSQ 为标准化 β, PD/SAA 为 SD 差)")
P(f"  {'exposure':10s} | FDR<0.05: 277 / 314 | ρ(β) 全部 | 发现 FDR 显著蛋白中: n, 复现 P<0.05 同向 %, 复现/发现 β 斜率")
for e in ['age','sex','PD_vs_HC','SAA','RBDSQ']:
    b1,s1,p1=out['277'][e]; b2,s2,p2=out['314'][e]; f1=multipletests(p1,method='fdr_bh')[1]; f2=multipletests(p2,method='fdr_bh')[1]
    sig=f1<0.05; rho=stats.spearmanr(b1,b2)[0]
    rep=((p2[sig]<0.05)&(np.sign(b2[sig])==np.sign(b1[sig]))).mean() if sig.sum() else np.nan
    sl=(b2[sig]*b1[sig]).sum()/(b1[sig]**2).sum() if sig.sum() else np.nan
    rows.append(dict(exposure=e,fdr_277=int(sig.sum()),fdr_314=int((f2<0.05).sum()),rho_all=rho,n_disc_sig=int(sig.sum()),rep_rate=rep,slope=sl))
    P(f"  {e:10s} | {int(sig.sum()):5d} / {int((f2<0.05).sum()):5d} | ρ={rho:+.2f} | n={int(sig.sum()):4d}, 复现 {100*rep if sig.sum() else float('nan'):.0f}%, 斜率 {sl:.2f}")
    allb.append(pd.DataFrame({'OlinkID':shared,'exposure':e,'beta_277':b1,'p_277':p1,'fdr_277':f1,'beta_314':b2,'p_314':p2,'fdr_314':f2}))
pd.DataFrame(rows).to_csv(D+"/p314_posctrl_summary.csv",index=False)
pd.concat(allb).to_csv(D+"/p314_posctrl_betas.csv",index=False)
# 3) 发现 FDR 显著且 314 中低于 LOD 比例低 (<20%) 的蛋白中, PD vs HC 复现率 —— 排除检测灵敏度解释
b1,_,p1=out['277']['PD_vs_HC']; b2,_,p2=out['314']['PD_vs_HC']; f1=multipletests(p1,method='fdr_bh')[1]
lo=L.reindex(shared)['314'].values<0.2; sig=(f1<0.05)&lo
P(f"\n3) PD vs HC, 发现 FDR 显著且 314 中 <20% 低于 LOD 的蛋白 n={sig.sum()}: 复现同向 P<0.05 {100*((p2[sig]<0.05)&(np.sign(b2[sig])==np.sign(b1[sig]))).mean():.0f}%, 斜率 {(b2[sig]*b1[sig]).sum()/(b1[sig]**2).sum():.2f}")
rem55=pd.read_csv(D+"/sig_rem.csv"); i55=set(rem55[rem55.fdr<0.05].OlinkID)
m55=np.array([s in i55 for s in shared])
P(f"   55 RBD 蛋白在 314 中低于 LOD 比例中位 {100*L.reindex(np.array(shared)[m55])['314'].median():.1f}% (277: {100*L.reindex(np.array(shared)[m55])['277'].median():.1f}%)")
P("done"); LOG.close()
