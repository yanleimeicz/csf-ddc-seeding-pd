# -*- coding: utf-8 -*-
"""RBD signature replication on a per-RBDSQ-point scale, winner's-curse (FIQT) corrected expectations, injection
simulations, reweighting and permutation tests. Output: derived/r3C_*.csv (Fig. 3, Table S7).
"""
import config as cfg
import sys, warnings, json
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
D=cfg.D; OUT=cfg.RS; NJ=30; NSIM=1000
LOG=open(OUT+"/r3C_log.txt","w",encoding="utf-8")
def P(*a):
    s=" ".join(str(x) for x in a); print(s); LOG.write(s+"\n"); LOG.flush()
def H(t): P("\n"+"="*78+"\n"+t+"\n"+"="*78)
GEN=['LRRK2','GBA','LRRK2 + GBA','SNCA','PRKN','PINK1','PRKN + RBD']
clin=pd.read_csv(D+"/analytic_clinical.csv").set_index('PATNO')
sig=pd.read_csv(D+"/sig_rem.csv").set_index('OlinkID'); s55=list(sig[sig.fdr<0.05].index)
def prep(wf,filt):
    w=pd.read_parquet(wf).set_index('PATNO'); w.index=w.index.astype(int)
    w=w.loc[:,w.isna().mean()<=0.2]; w=w.loc[w.isna().mean(1)<=0.3]
    c=clin.reindex(w.index); m=filt(c)&c.rem.notna()&c.age.notna()&c.SEX.notna(); w=w[m.values]; c=c[m.values]
    return w,c
wd,cd=prep(D+"/csf_npx_wide.parquet",lambda c: c.group0.isin(['HC','Hyposmia','iRBD','PD']))
wr,cr=prep(D+"/p314_wide_BL_new.parquet",lambda c: (~c.subgroup.isin(GEN))&c.group0.isin(['HC','Hyposmia','iRBD','PD']))
shared=[c for c in wd.columns if c in wr.columns]; wd=wd[shared]; wr=wr[shared]; i55=np.array([c in set(s55) for c in shared])
def imp(w):
    X=w.values.astype(float); med=np.nanmedian(X,0); ii=np.where(np.isnan(X)); X[ii]=np.take(med,ii[1]); return X
Xd=imp(wd); Xr=imp(wr)
sd_hc_d=Xd[(cd.group0=='HC').values].std(0); sd_hc_r=Xr[(cr.group0=='HC').values].std(0)
Yd=Xd/sd_hc_d; Yr=Xr/sd_hc_r          # 蛋白以各自队列 HC 的 SD 标度 (单位: HC-SD)
def ols(Y,Xm,w=None,j=1):
    if w is not None: sw=np.sqrt(w)[:,None]; Y=Y*sw; Xm=Xm*sw
    XtXi=np.linalg.pinv(Xm.T@Xm); Bh=XtXi@Xm.T@Y; res=Y-Xm@Bh; dof=Y.shape[0]-Xm.shape[1]
    se=np.sqrt((res**2).sum(0)/dof*XtXi[j,j]); return Bh[j],se,2*stats.t.sf(np.abs(Bh[j]/se),dof)
def design(c,extra=None):
    cols=[np.ones(len(c)),c.rem.values.astype(float),c.age.values.astype(float),c.SEX.values.astype(float)]
    if extra=='g4': cols+=[(c.group0==g).astype(float).values for g in ['Hyposmia','iRBD','PD']]
    if extra=='g5': cols+=[(c.group.astype(str)==g).astype(float).values for g in ['Hyposmia','iRBD','PD_brainfirst','PD_bodyfirst']]
    return np.column_stack(cols)
strata={'Full':lambda c: np.ones(len(c),bool),'Within SAA+':lambda c:(c.CSFSAA==1).values,'Within PD':lambda c:(c.group0=='PD').values}
H("1-2. 每分效应 (HC-SD / RBDSQ 分) 与各分层 RBDSQ SD")
rows=[]; B={}
for coh,(Y,c) in {'Discovery':(Yd,cd),'Replication':(Yr,cr)}.items():
    for sn,f in strata.items():
        m=f(c)
        for ex in ([None,'g4','g5'] if sn=='Full' else [None]):
            b,se,p=ols(Y[m],design(c[m],ex)); fdr=multipletests(p,method='fdr_bh')[1]; key=(coh,sn,ex); B[key]=(b,se,p,fdr)
            rows.append(dict(cohort=coh,stratum=sn,adjust=ex or 'age,sex',n=int(m.sum()),sd_RBDSQ=c.rem[m].std(),fdr_hits=int((fdr<0.05).sum()),
                             of55_fdr=int((fdr[i55]<0.05).sum()),of55_nominal=int((p[i55]<0.05).sum()),median_abs_b55=np.median(np.abs(b[i55]))))
            P(f"  {coh:11s} {sn:11s} {str(ex or 'age,sex'):8s} n={int(m.sum()):4d} SD(RBDSQ)={c.rem[m].std():.2f} | FDR {int((fdr<0.05).sum()):3d} | 55中 FDR {int((fdr[i55]<0.05).sum()):2d} 名义 {int((p[i55]<0.05).sum()):2d} | 55 每分|β|中位 {np.median(np.abs(b[i55])):.4f}")
bfull=B[('Discovery','Full',None)][0][i55]
def slope(a,b): return (a*b).sum()/(b*b).sum()
H("   每分 β 比值 (相对发现队列全队列; 过原点斜率)")
for key in [('Discovery','Full','g4'),('Discovery','Full','g5'),('Discovery','Within SAA+',None),('Discovery','Within PD',None),
            ('Replication','Full',None),('Replication','Full','g4'),('Replication','Within SAA+',None),('Replication','Within PD',None)]:
    s=slope(B[key][0][i55],bfull); rows.append(dict(cohort=key[0],stratum=key[1],adjust=key[2] or 'age,sex',per_point_slope_vs_disc_full=s)); P(f"  {key[0]:11s} {key[1]:11s} {str(key[2] or 'age,sex'):8s}: 每分 β 斜率 = {s:.2f}")
pd.DataFrame(rows).to_csv(D+"/r3C_per_point_models.csv",index=False)

H("3. 期望命中数: 注入每分效应 (原始 / FIQT 赢家诅咒校正 / 发现队列组别校正后效应)")
bd,sed,pd_,_=B[('Discovery','Full',None)]; z=bd/sed; q=multipletests(pd_,method='fdr_bh')[1]
z_fiqt=np.sign(z)*stats.norm.isf(np.clip(q,1e-300,1)/2); b_fiqt=bd*np.where(z!=0,z_fiqt/z,0)
b_g4=B[('Discovery','Full','g4')][0]
P(f"  55 蛋白发现队列每分 β 中位 |{np.median(np.abs(bd[i55])):.4f}|; FIQT 校正后 |{np.median(np.abs(b_fiqt[i55])):.4f}| (收缩 {np.median(np.abs(b_fiqt[i55])/np.abs(bd[i55])):.2f}); 组别校正后 |{np.median(np.abs(b_g4[i55])):.4f}|")
def sim_once(Y,c,m,eff,ex,seed):
    r=np.random.default_rng(seed); Xm=design(c[m],ex); Ym=Y[m]
    Bh=np.linalg.pinv(Xm.T@Xm)@Xm.T@Ym; Yres=Ym-Xm[:,[1]]@Bh[[1]]
    perm=r.permutation(m.sum()); rp=c.rem.values[m][perm]; Xs=Xm.copy(); Xs[:,1]=rp
    Ys=Yres.copy(); Ys[:,i55]+=np.outer(rp-rp.mean(),eff[i55])
    b,se,p=ols(Ys,Xs); fdr=multipletests(p,method='fdr_bh')[1]; return int((fdr[i55]<0.05).sum()),int((p[i55]<0.05).sum())
srows=[]
for lab,eff,ex in [('uncorrected per-point',bd,None),('FIQT-corrected per-point',b_fiqt,None),('group-adjusted (4) per-point, tested with group adjustment',b_g4,'g4')]:
    for sn,f in strata.items():
        if ex and sn!='Full': continue
        m=f(cr); out=np.array(Parallel(n_jobs=NJ)(delayed(sim_once)(Yr,cr,m,eff,ex,k) for k in range(NSIM)))
        key=('Replication',sn,ex); obs_f=int((B[key][3][i55]<0.05).sum()); obs_n=int((B[key][2][i55]<0.05).sum())
        srows.append(dict(effect=lab,stratum=sn,exp_fdr=out[:,0].mean(),exp_fdr_lo=np.percentile(out[:,0],2.5),exp_fdr_hi=np.percentile(out[:,0],97.5),exp_nominal=out[:,1].mean(),
                          exp_nom_lo=np.percentile(out[:,1],2.5),exp_nom_hi=np.percentile(out[:,1],97.5),obs_fdr=obs_f,obs_nominal=obs_n))
        P(f"  {lab:58s} {sn:11s}: 期望 FDR {out[:,0].mean():5.1f} ({np.percentile(out[:,0],2.5):.0f}–{np.percentile(out[:,0],97.5):.0f}), 期望名义 {out[:,1].mean():5.1f} ({np.percentile(out[:,1],2.5):.0f}–{np.percentile(out[:,1],97.5):.0f}) | 观察 FDR {obs_f}, 名义 {obs_n}")
pd.DataFrame(srows).to_csv(D+"/r3C_expected_hits.csv",index=False)

H("4. 按发现队列组成重加权 (WLS) 的复现")
pd_d=cd.group0.value_counts(normalize=True); pd_r=cr.group0.value_counts(normalize=True); wts=cr.group0.map(pd_d/pd_r).values
b,se,p=ols(Yr,design(cr),w=wts); fdr=multipletests(p,method='fdr_bh')[1]
P(f"  组成: 发现 {pd_d.round(2).to_dict()} | 复现 {pd_r.round(2).to_dict()}")
P(f"  重加权后: FDR {int((fdr<0.05).sum())}; 55 中 FDR {int((fdr[i55]<0.05).sum())}, 名义 {int((p[i55]<0.05).sum())}; 每分 β 斜率 vs 发现 {slope(b[i55],bfull):.2f}")
json.dump(dict(fdr=int((fdr<0.05).sum()),fdr55=int((fdr[i55]<0.05).sum()),nom55=int((p[i55]<0.05).sum()),slope=slope(b[i55],bfull)),open(D+"/r3C_reweighted.json","w"))

H("5. 置换检验 (1,000 次; 保留蛋白相关结构): 55 蛋白同向数与复合分数")
sgn=np.sign(bd[i55]); Zr=(Yr-Yr.mean(0))/Yr.std(0); score=(Zr[:,i55]*sgn).mean(1)
prow=[]
for sn,f in strata.items():
    for ex in ([None,'g4'] if sn=='Full' else [None]):
        m=f(cr); Xm=design(cr[m],ex)
        def stat(rem):
            Xs=Xm.copy(); Xs[:,1]=rem; b,_,_=ols(Yr[m][:,i55],Xs); bs,_,_=ols(score[m][:,None],Xs); return (np.sign(b)==sgn).sum(), bs[0]
        obs=stat(cr.rem.values[m])
        null=np.array(Parallel(n_jobs=NJ)(delayed(lambda s: stat(np.random.default_rng(s).permutation(cr.rem.values[m])))(k) for k in range(NSIM)))
        p_same=(1+(null[:,0]>=obs[0]).sum())/(NSIM+1); p_score=(1+(np.abs(null[:,1])>=abs(obs[1])).sum())/(NSIM+1)
        prow.append(dict(stratum=sn,adjust=ex or 'age,sex',same_dir=int(obs[0]),perm_p_same=p_same,score_beta=obs[1],perm_p_score=p_score,null_same_median=np.median(null[:,0])))
        P(f"  {sn:11s} {str(ex or 'age,sex'):8s}: 同向 {int(obs[0])}/55 (置换 P={p_same:.3f}; 零分布中位 {np.median(null[:,0]):.0f}) | 复合分数 β={obs[1]:+.4f}/分 (置换 P={p_score:.3f})")
pd.DataFrame(prow).to_csv(D+"/r3C_permutation.csv",index=False)
P("\ndone"); LOG.close()
