# -*- coding: utf-8 -*-
"""Replication set: RBD-anchored signature (55 proteins) under the five discovery models, whole-proteome consistency,
composite score and power. Output: derived/p314_wide_BL_new.parquet, derived/p314_rbd_*.csv.
"""
import config as cfg
import os, sys, warnings
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc
from scipy import stats
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
B=cfg.P; D=cfg.D; OUT=cfg.RS
NJ=30; NSIM=1000
LOG=open(OUT+"/p314_rbd_confounding_log.txt","w",encoding="utf-8")
def P(*a):
    s=" ".join(str(x) for x in a); print(s); LOG.write(s+"\n"); LOG.flush()
def H(t): P("\n"+"="*78+"\n"+t+"\n"+"="*78)

# ---------------------------------------------------------------- A) 宽矩阵
WF=D+"/p314_wide_BL_new.parquet"
cohort733=set(pd.read_parquet(D+"/csf_npx_wide.parquet",columns=['PATNO'])['PATNO'].astype(int))
if not os.path.exists(WF):
    t=pq.read_table(B+"/ppmi_proj314_csf_screened_extended_npx_20260116.parquet",
        columns=['PATNO','EVENT_ID','SampleType','SampleQC','AssayType','OlinkID','NPX','LAB_QC_FLAG'])
    t=t.filter(pc.and_(pc.and_(pc.equal(t['SampleType'],'SAMPLE'),pc.equal(t['EVENT_ID'],'BL')),pc.equal(t['AssayType'],'assay'))).to_pandas()
    t=t[t.PATNO.notna()]; t['PATNO']=t.PATNO.astype(int)
    bad=t.LAB_QC_FLAG.fillna('').astype(str).str.contains('6|7')
    t.loc[(t.SampleQC!='PASS')|bad,'NPX']=np.nan          # 未通过 QC 的数据点按缺失处理
    t=t[~t.PATNO.isin(cohort733)]
    w=t.pivot_table(index='PATNO',columns='OlinkID',values='NPX',aggfunc='mean')
    w.reset_index().to_parquet(WF,index=False)
w=pd.read_parquet(WF).set_index('PATNO')
H("A) 314 基线宽矩阵 (733 之外)")
P(f"原始 {w.shape[0]} 人 × {w.shape[1]} 个检测")
pm=w.isna().mean(0); w=w.loc[:,pm<=0.20]; sm=w.isna().mean(1); w=w.loc[sm<=0.30]
P(f"QC 后 {w.shape[0]} 人 × {w.shape[1]} 个检测 (蛋白缺失>20% 剔除 {int((pm>0.20).sum())}; 样本缺失>30% 剔除 {int((sm>0.30).sum())})")

ann=pd.read_csv(D+"/protein_annotation.csv").set_index('OlinkID')
disc=pd.read_csv(D+"/sig_rem.csv").set_index('OlinkID')               # 发现队列全队列 RBDSQ 结果
att=pd.read_csv(D+"/rev2_beta_attenuation_55.csv").set_index('gene')
shared=[c for c in w.columns if c in disc.index]
w=w[shared]; P(f"与发现队列共有蛋白 {len(shared)} / 发现 {len(disc)}")
X=w.values.astype(float); med=np.nanmedian(X,0); ii=np.where(np.isnan(X)); X[ii]=np.take(med,ii[1])
genes=ann.reindex(shared)['Assay'].astype(str).values
i55=(disc.reindex(shared).fdr<0.05).values; P(f"55 蛋白中在 314 可用: {i55.sum()}")
b_disc=disc.reindex(shared).beta_z.values

clin=pd.read_csv(D+"/analytic_clinical.csv").set_index('PATNO').reindex(w.index)
plate=pd.read_csv(D+"/p314_rep_baseline_analytic.csv").set_index('PATNO').Plate.reindex(w.index)
GEN=['LRRK2','GBA','LRRK2 + GBA','SNCA','PRKN','PINK1','PRKN + RBD']
spor=(~clin.subgroup.isin(GEN))&clin.group0.isin(['HC','Hyposmia','iRBD','PD'])&clin.rem.notna()
clin['NSD_num']=clin.NSD_STAGE.map(lambda v: 0.0 if str(v).startswith('Not') else (float(str(v)[0]) if str(v)[:1].isdigit() else np.nan))
full=spor.values; saa=full&(clin.CSFSAA==1).values; pdm=full&(clin.group0=='PD').values
P(f"散发分析集 {full.sum()} (HC {int((full&(clin.group0=='HC').values).sum())}, 嗅觉减退 {int((full&(clin.group0=='Hyposmia').values).sum())}, "
  f"iRBD {int((full&(clin.group0=='iRBD').values).sum())}, PD {pdm.sum()}); SAA+ {saa.sum()}; RBDSQ SD 全/SAA+/PD = "
  f"{clin.rem[full].std():.2f}/{clin.rem[saa].std():.2f}/{clin.rem[pdm].std():.2f}")

# ---------------------------------------------------------------- B) 模型
def ols_vec(Y,Xd,j=1):
    XtXinv=np.linalg.pinv(Xd.T@Xd); Bh=XtXinv@Xd.T@Y; res=Y-Xd@Bh; dof=Y.shape[0]-Xd.shape[1]
    s2=(res**2).sum(0)/dof; se=np.sqrt(s2*XtXinv[j,j]); b=Bh[j]; return b,se,2*stats.t.sf(np.abs(b/se),dof)
def design(mask,extra=None):
    sub=clin[mask]; c=sub.rem.values.astype(float)
    cols=[np.ones(len(sub)),c,sub.age.values.astype(float),sub.SEX.values.astype(float)]
    if extra is not None: cols+=extra(sub)
    Xd=np.column_stack(cols); ok=~np.isnan(Xd).any(1); return Xd[ok],np.where(mask)[0][ok],c[ok]
g5=lambda s:[(s.group.astype(str)==k).astype(float).values for k in ['Hyposmia','iRBD','PD_brainfirst','PD_bodyfirst']]
g4=lambda s:[(s.group0.astype(str)==k).astype(float).values for k in ['Hyposmia','iRBD','PD']]
sn=lambda s:[s.CSFSAA.astype(float).where(s.CSFSAA.isin([0,1])).values,s.NSD_num.astype(float).values]
def platefx(s):
    pl=plate.reindex(s.index).astype(str); lv=sorted(pl.unique())[1:]; return [(pl==k).astype(float).values for k in lv]
MODELS=[('Full cohort, age+sex',full,None),('Full cohort, age+sex + plate',full,platefx),
        ('Full cohort + group (5 groups, as discovery)',full,g5),('Full cohort + group (4 groups, PD combined)',full,g4),
        ('Full cohort + SAA status + NSD stage',full,sn),('Within SAA+',saa,None),('Within PD',pdm,None)]
DISC={'Full cohort, age+sex':(55,55,55),'Full cohort + group (5 groups, as discovery)':(0,0,19),'Full cohort + SAA status + NSD stage':(2,2,51),'Within SAA+':(4,4,55),'Within PD':(0,0,6)}
H("B) 全蛋白组 RBDSQ 关联 (标准化 β); 括号内为发现队列 [FDR 全蛋白组 / 55 中 FDR / 55 中名义]")
res={}; rows=[]
for lab,mask,ex in MODELS:
    Xd,idx,c=design(mask,ex); b,se,p=ols_vec(X[idx],Xd); fdr=multipletests(p,method='fdr_bh')[1]
    sdY=X[idx].std(0); bz=b*c.std()/sdY; sez=se*c.std()/sdY
    same=np.sign(bz[i55])==np.sign(b_disc[i55]); nsame=int(same.sum()); bp=stats.binomtest(nsame,int(i55.sum()),0.5,alternative='greater').pvalue
    nom_same=int(((p[i55]<0.05)&same).sum())
    r_all=stats.spearmanr(bz,b_disc)[0]
    res[lab]=pd.DataFrame({'OlinkID':shared,'gene':genes,'in55':i55,'beta_z':bz,'se_z':sez,'p':p,'fdr':fdr,'beta_z_discovery':b_disc})
    d=DISC.get(lab,('-','-','-'))
    rows.append(dict(model=lab,n=len(idx),sd_rem=c.std(),sig_all=int((fdr<0.05).sum()),sig_in55=int((fdr[i55]<0.05).sum()),nominal_in55=int((p[i55]<0.05).sum()),
                     nominal_same_dir_in55=nom_same,same_direction_in55=nsame,sign_test_p=bp,median_abs_beta55=np.median(np.abs(bz[i55])),
                     spearman_beta_vs_discovery_all=r_all,disc_sig_all=d[0],disc_sig_in55=d[1],disc_nominal_in55=d[2]))
    P(f"  {lab:46s} n={len(idx):4d} | FDR 全蛋白组 {int((fdr<0.05).sum()):3d} [{d[0]}] | 55 中 FDR {int((fdr[i55]<0.05).sum()):2d} [{d[1]}] | 55 中名义 {int((p[i55]<0.05).sum()):2d} [{d[2]}] (同向 {nom_same})"
      f" | 同向 {nsame}/55 (符号检验 P={bp:.1e}) | |β| 中位 {np.median(np.abs(bz[i55])):.3f} | 全蛋白组 β 与发现 ρ={r_all:+.2f}")
tab=pd.DataFrame(rows); tab.to_csv(D+"/p314_rbd_models.csv",index=False)
pd.concat([v.assign(model=k) for k,v in res.items()]).to_csv(D+"/p314_rbd_proteome_betas.csv",index=False)

# 55 蛋白 β 衰减 (复现队列内部): 调整 group 后 / SAA+ 内 / PD 内, 相对全队列 β —— 用跨 55 蛋白的过原点回归斜率 (比中位比值稳健)
H("B2) 55 蛋白 β 衰减 (复现队列内部), 斜率 = Σβ_m·β_full / Σβ_full² ; 发现队列: +group 0.38, SAA+ 0.83, PD 0.35 (中位比值)")
F=res['Full cohort, age+sex']; bf=F.beta_z.values[i55]
def slope(a,bb): return (a*bb).sum()/(bb*bb).sum()
def boot_slope(a,bb,seed):
    i=np.random.default_rng(seed).integers(0,len(a),len(a)); return slope(a[i],bb[i])
att_rows=[]
for lab in ['Full cohort + group (5 groups, as discovery)','Full cohort + group (4 groups, PD combined)','Full cohort + SAA status + NSD stage','Within SAA+','Within PD']:
    bm=res[lab].beta_z.values[i55]; s=slope(bm,bf); bs=Parallel(n_jobs=NJ)(delayed(boot_slope)(bm,bf,k) for k in range(2000)); lo,hi=np.percentile(bs,[2.5,97.5])
    # 发现队列同口径斜率
    da={'Full cohort + group (5 groups, as discovery)':'beta_full_adjgroup','Within SAA+':'beta_saa','Within PD':'beta_pd'}.get(lab)
    sd=slope(att[da].values,att.beta_full.values) if da else np.nan
    att_rows.append(dict(model=lab,slope_replication=s,ci_lo=lo,ci_hi=hi,slope_discovery=sd))
    P(f"  {lab:46s} 复现斜率 {s:.2f} [{lo:.2f},{hi:.2f}]  | 发现队列同口径斜率 {sd:.2f}")
# 复现队列全队列 β 相对发现 β (赢家诅咒 + 真实差异)
s=slope(bf,b_disc[i55]); P(f"  复现全队列 β / 发现全队列 β (55 蛋白斜率) = {s:.2f}")
att_rows.append(dict(model='Replication full vs discovery full',slope_replication=s))
pd.DataFrame(att_rows).to_csv(D+"/p314_rbd_attenuation.csv",index=False)

# ---------------------------------------------------------------- C) 复合 signature 分数
H("C) 55 蛋白 signature 复合分数 (每蛋白 z 化 × 发现方向, 取均值) ~ RBDSQ: 单一检验, 不受多重比较拖累")
import statsmodels.formula.api as smf
Z=(X-X[full].mean(0))/X[full].std(0); score=(Z[:,i55]*np.sign(b_disc[i55])).mean(1)
cd=clin.copy(); cd['score']=score; cd['Plate']=plate; cd['SAA']=cd.CSFSAA.where(cd.CSFSAA.isin([0,1]))
cd['zrem']=(cd.rem-cd.rem[full].mean())/cd.rem[full].std(); cd['zscore']=(cd.score-cd.score[full].mean())/cd.score[full].std()
sc=[]
for lab,mask,f in [('Full cohort',full,'zscore ~ zrem + age + SEX'),('Full cohort + plate',full,'zscore ~ zrem + age + SEX + C(Plate)'),
                   ('Full + group (4)',full,'zscore ~ zrem + age + SEX + C(group0)'),('Full + group (5)',full,'zscore ~ zrem + age + SEX + C(group)'),
                   ('Full + SAA + NSD',full,'zscore ~ zrem + age + SEX + SAA + NSD_num'),('Within SAA+',saa,'zscore ~ zrem + age + SEX'),('Within PD',pdm,'zscore ~ zrem + age + SEX'),
                   ('Within HC',full&(clin.group0=='HC').values,'zscore ~ zrem + age + SEX'),('Within Hyposmia',full&(clin.group0=='Hyposmia').values,'zscore ~ zrem + age + SEX')]:
    m=smf.ols(f,cd[mask]).fit(); ci=m.conf_int().loc['zrem'].values
    sc.append(dict(model=lab,n=int(m.nobs),beta=m.params.zrem,ci_lo=ci[0],ci_hi=ci[1],p=m.pvalues.zrem))
    P(f"  {lab:22s} n={int(m.nobs):4d} | 标准化 β={m.params.zrem:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] P={m.pvalues.zrem:.1e}")
pd.DataFrame(sc).to_csv(D+"/p314_rbd_signature_score.csv",index=False)

# ---------------------------------------------------------------- D) 功效
H("D) 功效: 以发现队列 β 为真值 (含赢家诅咒, 为上限), 复现队列各分层的期望检出")
def power_r(r,n,alpha):
    z=np.arctanh(r)*np.sqrt(n-3); zc=stats.norm.ppf(1-alpha/2); return stats.norm.sf(zc-abs(z))+stats.norm.cdf(-zc-abs(z))
m_all=len(shared); a_bh=0.05*55/m_all; pw=[]
for lab,mask in [('Full cohort',full),('Within SAA+',saa),('Within PD',pdm)]:
    n=int(mask.sum()); pa=power_r(np.abs(b_disc[i55]),n,0.05); pb=power_r(np.abs(b_disc[i55]),n,a_bh)
    pw.append(dict(stratum=lab,n=n,median_power_nominal=np.median(pa),expected_nominal=pa.sum(),median_power_BH=np.median(pb),expected_BH=pb.sum()))
    P(f"  {lab:12s} n={n:4d} | 名义 α 中位功效 {np.median(pa):.2f} (期望 {pa.sum():.0f}/55) | BH 有效阈值 {a_bh:.1e} 中位功效 {np.median(pb):.2f} (期望 {pb.sum():.0f}/55)")
pd.DataFrame(pw).to_csv(D+"/p314_rbd_power_analytic.csv",index=False)
# 模拟: 去掉真实 RBDSQ 效应后注入发现 β (发现 SD 尺度), 数 BH 检出
def sim_once(Yres,c,Xd,beff,sdY,seed):
    r=np.random.default_rng(seed); perm=r.permutation(len(c)); cz=(c[perm]-c.mean())/c.std()
    Ys=Yres.copy(); Ys[:,i55]+=np.outer(cz,beff*sdY[i55]); Xs=Xd.copy(); Xs[:,1]=c[perm]
    b,se,p=ols_vec(Ys,Xs); fdr=multipletests(p,method='fdr_bh')[1]; return int((fdr[i55]<0.05).sum()), int((p[i55]<0.05).sum())
sims=[]
for lab,mask in [('Full cohort',full),('Within SAA+',saa),('Within PD',pdm)]:
    Xd,idx,c=design(mask); Y=X[idx]; Bh=np.linalg.pinv(Xd.T@Xd)@Xd.T@Y; Yres=Y-Xd[:,[1]]@Bh[[1]]
    out=np.array(Parallel(n_jobs=NJ)(delayed(sim_once)(Yres,c,Xd,b_disc[i55],Y.std(0),k) for k in range(NSIM)))
    obs=tab.set_index('model').loc['Full cohort, age+sex' if lab=='Full cohort' else lab]
    P(f"  模拟 {lab:12s} n={len(idx)}: 若发现效应为真, BH 期望检出 {out[:,0].mean():.1f} ({np.percentile(out[:,0],2.5):.0f}-{np.percentile(out[:,0],97.5):.0f}), "
      f"名义期望 {out[:,1].mean():.1f} | 观察: BH {obs.sig_in55}, 名义 {obs.nominal_in55}")
    sims+= [dict(stratum=lab,sim=k,bh=o[0],nominal=o[1]) for k,o in enumerate(out)]
pd.DataFrame(sims).to_csv(D+"/p314_rbd_power_simulation.csv",index=False)

# 55 蛋白逐个对照表
T=pd.DataFrame({'gene':genes[i55],'OlinkID':np.array(shared)[i55],'beta_disc':b_disc[i55]})
for lab,k in [('Full cohort, age+sex','full'),('Full cohort + group (4 groups, PD combined)','adjgroup4'),('Within SAA+','saa'),('Within PD','pd')]:
    r_=res[lab][i55]; T[f'beta_{k}']=r_.beta_z.values; T[f'p_{k}']=r_.p.values; T[f'fdr_{k}']=r_.fdr.values
T.sort_values('p_full').to_csv(D+"/p314_rbd_55_table.csv",index=False)
P("\n  55 蛋白中复现最强的 10 个 (全队列):"); P(T.sort_values('p_full').head(10)[['gene','beta_disc','beta_full','p_full','beta_saa','p_saa']].round(4).to_string(index=False))
P("\ndone"); LOG.close()
