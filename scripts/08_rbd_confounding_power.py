# -*- coding: utf-8 -*-
"""Discovery set: confounding versus power for the RBD signature (stratum sizes, covariate models, attenuation of the
55 effects, analytic and simulated power, missing-value and covariate sensitivity).
Output: derived/rev2_*.csv (Fig. 2, Tables S4 to S6).
"""
import config as cfg
import numpy as np, pandas as pd, warnings, sys
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
from scipy import stats
from statsmodels.stats.multitest import multipletests
D=cfg.D
wide=pd.read_parquet(D+"/csf_npx_wide.parquet").set_index('PATNO')
ann=pd.read_csv(D+"/protein_annotation.csv").set_index('OlinkID')
clin=pd.read_csv(D+"/analytic_clinical.csv").set_index('PATNO').reindex(wide.index)
genes=ann.reindex(wide.columns)['Assay'].astype(str).values
Xraw=wide.values.astype(float)
X=Xraw.copy(); med=np.nanmedian(X,0); ii=np.where(np.isnan(X)); X[ii]=np.take(med,ii[1])
sig55=pd.read_csv(D+"/sig_rem.csv"); sig55=sig55[sig55.fdr<0.05]; g55=set(sig55.gene.astype(str)); i55=np.array([g in g55 for g in genes])
def nsd_num(v):
    s=str(v); return 0.0 if s.startswith('Not') else (float(s[0]) if s[0].isdigit() else np.nan)
clin['NSD_num']=clin['NSD_STAGE'].map(nsd_num)

def ols_vec(Y,Xd,j=1):
    """Y n×P, Xd n×k; 返回第 j 列系数 beta, se, p"""
    XtXinv=np.linalg.pinv(Xd.T@Xd); B=XtXinv@Xd.T@Y; resid=Y-Xd@B; dof=Y.shape[0]-Xd.shape[1]
    s2=(resid**2).sum(0)/dof; se=np.sqrt(s2*XtXinv[j,j]); beta=B[j]; p=2*stats.t.sf(np.abs(beta/se),dof)
    return beta,se,p

def design(mask,extra=None):
    sub=clin[mask]; c=sub['rem'].values.astype(float); age=sub['age'].values.astype(float); sex=sub['SEX'].values.astype(float)
    cols=[np.ones(len(sub)),c,age,sex]
    if extra is not None: cols+=extra(sub)
    Xd=np.column_stack(cols); ok=~np.isnan(Xd).any(1)
    return Xd[ok], mask.nonzero()[0][ok], c[ok]

out={}
# ---------- a) 分层样本量 + 协变量模型 ----------
full=clin['rem'].notna().values
saa=(clin['CSFSAA']==1).values & full
pd_=clin['group'].isin(['PD_bodyfirst','PD_brainfirst']).values & full
print("分层样本量: 全队列",full.sum(),"| SAA+",saa.sum(),"| PD",pd_.sum())

def grp_dummies(sub):
    g=sub['group'].astype(str); cats=['Hyposmia','iRBD','PD_brainfirst','PD_bodyfirst']
    return [(g==k).astype(float).values for k in cats]
def saa_nsd(sub):
    return [sub['CSFSAA'].astype(float).values, sub['NSD_num'].astype(float).values]

rows=[]
for lab,mask,extra in [('Full cohort, age+sex',full,None),
                       ('Full cohort + group (5 groups)',full,grp_dummies),
                       ('Full cohort + SAA status + NSD stage',full,saa_nsd),
                       ('Within SAA+',saa,None),('Within PD',pd_,None)]:
    Xd,idx,c=design(mask,extra); b,se,p=ols_vec(X[idx],Xd)
    fdr=multipletests(p,method='fdr_bh')[1]
    bz=b*c.std()/X[idx].std(0)
    n_all=(fdr<0.05).sum(); n55=(fdr[i55]<0.05).sum(); nom55=(p[i55]<0.05).sum()
    rows.append(dict(model=lab,n=len(idx),sig_all=n_all,sig_in55=n55,nominal_in55=nom55,median_abs_beta55=np.median(np.abs(bz[i55]))))
    out[lab]=pd.DataFrame({'gene':genes,'beta_z':bz,'beta':b,'se':se,'p':p,'fdr':fdr})
    print(f"  {lab:40s} n={len(idx):4d} | FDR<0.05 全蛋白组 {n_all:3d} | 55 中 FDR<0.05 {n55:2d} | 55 中 P<0.05 {nom55:2d} | 55蛋白 |β_z| 中位 {np.median(np.abs(bz[i55])):.3f}")
tabA=pd.DataFrame(rows); tabA.to_csv(D+"/rev2_confounding_models.csv",index=False)

# ---------- b) β 衰减 (全队列 vs SAA+ vs PD), 带 SE ----------
F=out['Full cohort, age+sex']; S=out['Within SAA+']; P=out['Within PD']; G=out['Full cohort + group (5 groups)']
att=pd.DataFrame({'gene':genes,'beta_full':F.beta_z,'se_full':F.se*F.beta_z/F.beta,
                  'beta_saa':S.beta_z,'se_saa':S.se*S.beta_z/S.beta,'beta_pd':P.beta_z,'se_pd':P.se*P.beta_z/P.beta,
                  'beta_full_adjgroup':G.beta_z,'p_full':F.p,'p_saa':S.p,'p_pd':P.p,'fdr_full':F.fdr,'fdr_saa':S.fdr,'fdr_pd':P.fdr})[i55]
att['se_full']=att['se_full'].abs(); att['se_saa']=att['se_saa'].abs(); att['se_pd']=att['se_pd'].abs()
att['ratio_saa']=att.beta_saa/att.beta_full; att['ratio_pd']=att.beta_pd/att.beta_full; att['ratio_adjgroup']=att.beta_full_adjgroup/att.beta_full
att=att.sort_values('p_full'); att.to_csv(D+"/rev2_beta_attenuation_55.csv",index=False)
print(f"\nβ 衰减 (55 蛋白): 中位 β_SAA+/β_full = {att.ratio_saa.median():.2f}, 中位 β_PD/β_full = {att.ratio_pd.median():.2f}, 中位 β_adj-group/β_full = {att.ratio_adjgroup.median():.2f}")
print(f"  SE 变化: 中位 SE_SAA+/SE_full = {(att.se_saa/att.se_full).median():.2f}, SE_PD/SE_full = {(att.se_pd/att.se_full).median():.2f}")
print(f"  方向保持且|β|未缩小一半以上的蛋白数(SAA+): {((att.ratio_saa>0.5)).sum()} / 55")

# ---------- c) 事后功效: 解析 + 模拟 ----------
def power_t(r,n,alpha):
    """相关系数 r 在 n 下, 双侧 alpha 的功效 (Fisher z 近似)"""
    z=np.arctanh(r)*np.sqrt(n-3); zc=stats.norm.ppf(1-alpha/2); return stats.norm.sf(zc-abs(z))+stats.norm.cdf(-zc-abs(z))
# 标准化β≈偏相关; 用全队列 β_z 作为"真实效应"
alpha_bh=0.05*55/5414  # 若55个真阳性, BH 有效阈值≈ 0.05*k/m
pw=[]
for lab,n in [('SAA+',saa.sum()),('PD',pd_.sum())]:
    pa=[power_t(abs(b),n,0.05) for b in att.beta_full]; pb=[power_t(abs(b),n,alpha_bh) for b in att.beta_full]
    pw.append(dict(stratum=lab,n=n,median_power_alpha05=np.median(pa),min_power_alpha05=np.min(pa),median_power_BH=np.median(pb),min_power_BH=np.min(pb),expected_detected_BH=np.sum(pb)))
    print(f"\n解析功效 {lab} (n={n}): 55蛋白按全队列效应量, α=0.05 中位功效 {np.median(pa):.2f} (最小 {np.min(pa):.2f}); BH 有效阈值 α≈{alpha_bh:.1e} 中位功效 {np.median(pb):.2f}, 期望检出 {np.sum(pb):.0f}/55")
pd.DataFrame(pw).to_csv(D+"/rev2_power_analytic.csv",index=False)

# 模拟: 在分层内, 把全队列效应 (β_z) 注入到"去除真实RBDSQ关联后的"蛋白残差上, 看 BH 能检出多少
rng=np.random.default_rng(7); nsim=200
def simulate(mask,label):
    Xd,idx,c=design(mask); Y=X[idx]; n=len(idx)
    # 残差化: 去掉真实 RBDSQ 效应, 保留 age/sex 结构 + 噪声
    XtXinv=np.linalg.pinv(Xd.T@Xd); B=XtXinv@Xd.T@Y; R=Y-Xd[:,[1]]@B[[1]]   # 只去掉 RBDSQ 项
    cz=(c-c.mean())/c.std(); sdY=Y.std(0)
    det=[]
    for s in range(nsim):
        perm=rng.permutation(n); czp=cz[perm]                     # 打乱后与残差无关
        Ysim=R.copy(); Ysim[:,i55]+=np.outer(czp,(att.set_index('gene').loc[genes[i55],'beta_full'].values)*sdY[i55])
        Xs=Xd.copy(); Xs[:,1]=c[perm]
        b,se,p=ols_vec(Ysim,Xs); fdr=multipletests(p,method='fdr_bh')[1]
        det.append((fdr[i55]<0.05).sum())
    det=np.array(det); print(f"模拟 {label} (n={n}, {nsim} 次): 若55个全队列效应为真, BH FDR<0.05 期望检出 {det.mean():.1f} (95% 区间 {np.percentile(det,2.5):.0f}–{np.percentile(det,97.5):.0f}) / 55; 观察到 {'4' if label=='SAA+' else '0'}")
    return det
detS=simulate(saa,'SAA+'); detP=simulate(pd_,'PD')
pd.DataFrame({'stratum':['SAA+']*nsim+['PD']*nsim,'detected':np.r_[detS,detP]}).to_csv(D+"/rev2_power_simulation.csv",index=False)

# ---------- d) 缺失值敏感性: 逐蛋白完整病例 ----------
Xd,idx,c=design(full); Yr=Xraw[idx]
bcc=np.full(len(genes),np.nan); pcc=np.full(len(genes),np.nan)
for j in np.where(i55)[0]:
    ok=~np.isnan(Yr[:,j]); b,se,p=ols_vec(Yr[ok][:,[j]],Xd[ok]); bcc[j]=b[0]*c[ok].std()/Yr[ok,j].std(); pcc[j]=p[0]
miss=np.isnan(Xraw).mean(0)
cc=pd.DataFrame({'gene':genes,'missing_frac':miss,'beta_imputed':F.beta_z,'p_imputed':F.p,'beta_completecase':bcc,'p_completecase':pcc})[i55]
cc.to_csv(D+"/rev2_missing_sensitivity_55.csv",index=False)
print(f"\n缺失值: 55蛋白缺失比例中位 {cc.missing_frac.median()*100:.1f}% (最大 {cc.missing_frac.max()*100:.1f}%); 完整病例 vs 填补 β 相关 r={np.corrcoef(cc.beta_imputed,cc.beta_completecase)[0,1]:.3f}; 完整病例 P<0.05 的 {int((cc.p_completecase<0.05).sum())}/55; 全蛋白组缺失中位 {np.median(miss)*100:.1f}%")

# ---------- e) 协变量敏感性 (DDC 与 55 蛋白): + EDUCYRS + BMI + hemohi ----------
cur=pd.read_excel(cfg.CURATED,sheet_name='20260511',usecols=['PATNO','EVENT_ID','BMI','hemohi','EDUCYRS'])
bl=cur[cur.EVENT_ID=='BL'].set_index('PATNO'); clin['BMI']=bl['BMI'].reindex(clin.index); clin['hemohi']=bl['hemohi'].reindex(clin.index)
print("hemohi 分布:",clin['hemohi'].value_counts(dropna=False).to_dict())
def extra_cov(sub): return [sub['EDUCYRS'].astype(float).values, sub['BMI'].astype(float).values, sub['hemohi'].astype(float).fillna(0).values]
rows=[]
for lab,mask in [('Full cohort',full),('Within SAA+',saa)]:
    Xd,idx,c=design(mask,extra_cov); b,se,p=ols_vec(X[idx],Xd); fdr=multipletests(p,method='fdr_bh')[1]
    j=np.where(genes=='DDC')[0][0]
    rows.append(dict(model=lab+' + education + BMI + CSF haemoglobin flag',n=len(idx),sig_all=(fdr<0.05).sum(),sig_in55=(fdr[i55]<0.05).sum(),DDC_beta_z=b[j]*c.std()/X[idx,j].std(),DDC_p=p[j]))
    print(f"  {rows[-1]}")
pd.DataFrame(rows).to_csv(D+"/rev2_covariate_sensitivity.csv",index=False)
print("done")
