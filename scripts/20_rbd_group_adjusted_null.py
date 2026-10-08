# -*- coding: utf-8 -*-
"""RBD signature with clinical-group adjustment: Freedman-Lane and within-group permutation, injection simulation
and reweighting. Output: derived/r4A_*.csv (Fig. 3, Table S7).
"""
import config as cfg
import sys, warnings, json
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
D=cfg.D; OUT=cfg.RS; NJ=30
LOG=open(OUT+"/r4A_log.txt","w",encoding="utf-8")
def P(*a):
    s=" ".join(str(x) for x in a); print(s); LOG.write(s+"\n"); LOG.flush()
GEN=['LRRK2','GBA','LRRK2 + GBA','SNCA','PRKN','PINK1','PRKN + RBD']
clin=pd.read_csv(D+"/analytic_clinical.csv").set_index('PATNO')
sig=pd.read_csv(D+"/sig_rem.csv").set_index('OlinkID'); s55=set(sig[sig.fdr<0.05].index)
def prep(wf,filt):
    w=pd.read_parquet(wf).set_index('PATNO'); w.index=w.index.astype(int); w=w.loc[:,w.isna().mean()<=0.2]; w=w.loc[w.isna().mean(1)<=0.3]
    c=clin.reindex(w.index); m=filt(c)&c.rem.notna()&c.age.notna()&c.SEX.notna(); return w[m.values],c[m.values]
wd,cd=prep(D+"/csf_npx_wide.parquet",lambda c:c.group0.isin(['HC','Hyposmia','iRBD','PD']))
wr,cr=prep(D+"/p314_wide_BL_new.parquet",lambda c:(~c.subgroup.isin(GEN))&c.group0.isin(['HC','Hyposmia','iRBD','PD']))
sh=[c for c in wd.columns if c in wr.columns]; i55=np.array([c in s55 for c in sh])
def imp(w):
    X=w[sh].values.astype(float); med=np.nanmedian(X,0); ii=np.where(np.isnan(X)); X[ii]=np.take(med,ii[1]); return X
Xd=imp(wd); Xr=imp(wr); Yd=Xd/Xd[(cd.group0=='HC').values].std(0); Yr=Xr/Xr[(cr.group0=='HC').values].std(0)
G=['Hyposmia','iRBD','PD']
def cov(c,g4=True): return np.column_stack([np.ones(len(c)),c.age.values,c.SEX.values.astype(float)]+([(c.group0.values==g).astype(float) for g in G] if g4 else []))
def fit(Y,rem,Z):
    X=np.column_stack([Z[:,:1],rem,Z[:,1:]]); XtXi=np.linalg.pinv(X.T@X); B=XtXi@X.T@Y; e=Y-X@B; dof=len(Y)-X.shape[1]
    se=np.sqrt((e**2).sum(0)/dof*XtXi[1,1]); return B[1],se,2*stats.t.sf(np.abs(B[1]/se),dof),e
# 发现队列组别校正每分效应 (及其 z) 与方向
Zd=cov(cd); bdg,sedg,pdg,_=fit(Yd,cd.rem.values.astype(float),Zd); bdf,sedf,pdf_,_=fit(Yd,cd.rem.values.astype(float),cov(cd,False))
sgn=np.sign(bdf[i55])
Zr=cov(cr); rem=cr.rem.values.astype(float)
Zs=(Yr-Yr.mean(0))/Yr.std(0); score=(Zs[:,i55]*sgn).mean(1)
Yt=np.column_stack([Yr[:,i55],score])        # 55 蛋白 + 复合分数
b_obs,_,p_obs,_=fit(Yt,rem,Zr); same_obs=int((np.sign(b_obs[:55])==sgn).sum()); sc_obs=b_obs[55]
P(f"复现 组别校正: 同向 {same_obs}/55, 复合分数 β={sc_obs:+.4f}/分 (参数 P={p_obs[55]:.3f})")
# --- 1) Freedman–Lane 与组内置换
Hred=Zr@np.linalg.pinv(Zr.T@Zr)@Zr.T; Fit0=Hred@Yt; Res0=Yt-Fit0
grp=cr.group0.values
def fl(seed):
    r=np.random.default_rng(seed); Ys=Fit0+Res0[r.permutation(len(Yt))]; b,_,_,_=fit(Ys,rem,Zr); return (np.sign(b[:55])==sgn).sum(),b[55]
def wg(seed):
    r=np.random.default_rng(seed); rp=rem.copy()
    for g in np.unique(grp):
        ix=np.where(grp==g)[0]; rp[ix]=rem[r.permutation(ix)]
    b,_,_,_=fit(Yt,rp,Zr); return (np.sign(b[:55])==sgn).sum(),b[55]
rows=[]
for nm,fn in [('Freedman–Lane',fl),('within-group permutation',wg)]:
    nul=np.array(Parallel(n_jobs=NJ)(delayed(fn)(k) for k in range(2000)))
    ps=(1+(nul[:,0]>=same_obs).sum())/2001; pc=(1+(np.abs(nul[:,1])>=abs(sc_obs)).sum())/2001
    rows.append(dict(method=nm,same_dir=same_obs,p_same=ps,score_beta=sc_obs,p_score=pc,null_same_median=np.median(nul[:,0])))
    P(f"  {nm:26s}: 同向 P={ps:.3f} (零分布中位 {np.median(nul[:,0]):.0f}) | 复合分数 P={pc:.3f}")
# 无约束置换(原方法)作对照
def un(seed):
    r=np.random.default_rng(seed); b,_,_,_=fit(Yt,rem[r.permutation(len(rem))],Zr); return (np.sign(b[:55])==sgn).sum(),b[55]
nul=np.array(Parallel(n_jobs=NJ)(delayed(un)(k) for k in range(2000)))
rows.append(dict(method='unrestricted permutation (previous)',same_dir=same_obs,p_same=(1+(nul[:,0]>=same_obs).sum())/2001,score_beta=sc_obs,p_score=(1+(np.abs(nul[:,1])>=abs(sc_obs)).sum())/2001))
P(f"  unrestricted (previous)   : 同向 P={rows[-1]['p_same']:.3f} | 复合分数 P={rows[-1]['p_score']:.3f}")
pd.DataFrame(rows).to_csv(D+"/r4A_group_adjusted_permutation.csv",index=False)
# --- 2) 残差化注入模拟
z=bdg/sedg; q=multipletests(pdg,method='fdr_bh')[1]; z_f=np.sign(z)*stats.norm.isf(np.clip(q,1e-300,1)/2); bdg_f=bdg*np.where(z!=0,z_f/z,0)
P(f"发现队列组别校正: 55 蛋白每分|β|中位 {np.median(np.abs(bdg[i55])):.4f}; FIQT 后 {np.median(np.abs(bdg_f[i55])):.4f}")
Ya=Yr[:,i55]; X=np.column_stack([Zr[:,:1],rem,Zr[:,1:]]); B=np.linalg.pinv(X.T@X)@X.T@Ya; E=Ya-X@B
mu0=X@B-np.outer(rem,B[1])                    # 去掉真实 RBDSQ 效应后的拟合值
def inj(eff,seed):
    r=np.random.default_rng(seed); Ys=mu0+np.outer(rem,eff)+E[r.integers(0,len(E),len(E))]
    b,se,p,_=fit(Ys,rem,Zr); return int((p<0.05).sum())
sim=[]
for lab,eff in [('discovery group-adjusted, uncorrected',bdg[i55]),('discovery group-adjusted, FIQT-corrected',bdg_f[i55])]:
    out=np.array(Parallel(n_jobs=NJ)(delayed(inj)(eff,k) for k in range(1000)))
    sim.append(dict(effect=lab,exp_nominal=out.mean(),lo=np.percentile(out,2.5),hi=np.percentile(out,97.5),obs_nominal=int((p_obs[:55]<0.05).sum())))
    P(f"  注入 {lab:42s}: 期望名义 {out.mean():.1f} ({np.percentile(out,2.5):.0f}–{np.percentile(out,97.5):.0f}) | 观察 {int((p_obs[:55]<0.05).sum())}")
pd.DataFrame(sim).to_csv(D+"/r4A_injection_group_adjusted.csv",index=False)
# --- 3) 组别 × SAA 重加权 + sandwich SE
def cell(c): return np.where(c.group0=='HC','HC',c.group0+np.where(c.CSFSAA==1,'+','-'))
cdc=pd.Series(cell(cd)); crc=pd.Series(cell(cr)); pdisc=cdc.value_counts(normalize=True); prep_=crc.value_counts(normalize=True)
w=crc.map(lambda k: pdisc.get(k,0)/prep_[k]).values
P(f"单元: 发现 {pdisc.round(3).to_dict()} | 复现 {prep_.round(3).to_dict()} | 权重>0 的复现人数 {(w>0).sum()}")
keep=w>0; Xw=np.column_stack([np.ones(keep.sum()),rem[keep],cr.age.values[keep],cr.SEX.values[keep].astype(float)]); W=w[keep]
Y=Yr[keep]; A=np.linalg.pinv(Xw.T@(Xw*W[:,None])); Bw=A@(Xw*W[:,None]).T@Y; e=Y-Xw@Bw
a=(A@(Xw*W[:,None]).T)[1]; n=keep.sum(); k_=Xw.shape[1]
se=np.sqrt(((a**2)[:,None]*(e**2)).sum(0)*n/(n-k_)); pz=2*stats.norm.sf(np.abs(Bw[1]/se)); fdr=multipletests(pz,method='fdr_bh')[1]
sl=(Bw[1][i55]*bdf[i55]).sum()/(bdf[i55]**2).sum()
res=dict(n_weighted=int(n),fdr_all=int((fdr<0.05).sum()),fdr55=int((fdr[i55]<0.05).sum()),nom55=int((pz[i55]<0.05).sum()),slope=sl)
P(f"  重加权(组别×SAA, sandwich SE): 全蛋白组 FDR {res['fdr_all']}; 55 中 FDR {res['fdr55']}, 名义 {res['nom55']}; 每分斜率 {sl:.2f}")
json.dump(res,open(D+"/r4A_reweighted.json","w"))
P("done"); LOG.close()
