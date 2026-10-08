# -*- coding: utf-8 -*-
"""Sensitivity analyses: exclusion of participants in earlier PPMI CSF proteomic releases; bootstrap CIs of unique R2
in the five-component model; minimum detectable effect, Tobit and rank models for DDC-SBR among SAA-negative
participants; rank AUCs; participants untreated at sampling. Output: derived/r6_*.csv/json (Fig. 4c, Tables S11, S17).
"""
import config as cfg
import sys, glob, json, warnings
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc
import statsmodels.formula.api as smf
from statsmodels.tools.numdiff import approx_hess
from statsmodels.stats.multitest import multipletests
from scipy import stats, optimize
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
B=cfg.P; D=cfg.D; OUT=cfg.RS; NJ=30
LOG=open(OUT+"/r6_log.txt","w",encoding="utf-8")
def P(*a):
    s=" ".join(str(x) for x in a); print(s); LOG.write(s+"\n"); LOG.flush()
def H(t): P("\n"+"="*78+"\n"+t+"\n"+"="*78)
GEN=['LRRK2','GBA','LRRK2 + GBA','SNCA','PRKN','PINK1','PRKN + RBD']; SBRc=['MIA_CAUDATE_L','MIA_CAUDATE_R','MIA_PUTAMEN_L','MIA_PUTAMEN_R']
# ---------- 数据 (与 r4_D 相同口径) ----------
clin=pd.read_csv(D+"/analytic_clinical.csv").set_index('PATNO')
cur=pd.read_excel(cfg.CURATED,sheet_name='20260511'); bl=cur[cur.EVENT_ID=='BL'].drop_duplicates('PATNO').set_index('PATNO')
for c in SBRc+['updrs2_score','moca']: clin[c]=bl[c].reindex(clin.index)
clin['SBR']=clin[SBRc].mean(1); clin['SAA']=clin.CSFSAA.where(clin.CSFSAA.isin([0,1]))
wide=pd.read_parquet(D+"/csf_npx_wide.parquet").set_index('PATNO'); wide.index=wide.index.astype(int)
ann=pd.read_csv(D+"/protein_annotation.csv").set_index('OlinkID'); jD=[c for c in wide.columns if ann.Assay.get(c)=='DDC'][0]
disc=clin.reindex(wide.index).copy(); disc['DDC']=wide[jD].fillna(wide[jD].median()).values
rep=pd.read_csv(D+"/p314_rep_baseline_analytic.csv").set_index('PATNO')
for c in ['updrs2_score','moca']: rep[c]=bl[c].reindex(rep.index)
spor=rep[~rep.subgroup.isin(GEN)&rep.group0.isin(['HC','Hyposmia','iRBD','PD'])].copy()
wr=pd.read_parquet(D+"/p314_wide_BL_new.parquet").set_index('PATNO'); wr.index=wr.index.astype(int)
res={}
def tobit(y,X,lod):
    cens=y<lod; yy=np.where(cens,lod,y)
    def nll(th):
        b=th[:-1]; s=np.exp(th[-1]); mu=X@b
        return -(stats.norm.logpdf(yy[~cens],mu[~cens],s).sum()+stats.norm.logcdf((lod[cens]-mu[cens])/s).sum())
    b0=np.linalg.lstsq(X,yy,rcond=None)[0]; r=optimize.minimize(nll,np.r_[b0,np.log(yy.std())],method='BFGS')
    se=np.sqrt(np.diag(np.linalg.inv(approx_hess(r.x,nll)))); return r.x,se,cens.mean()
def lodtab(f,ids):
    t=pq.read_table(f,columns=['PATNO','EVENT_ID','SampleType','SampleQC','Assay','NPX','LOD'])
    t=t.filter(pc.and_(pc.equal(t['Assay'],'DDC'),pc.equal(t['SampleType'],'SAMPLE'))).to_pandas()
    t=t[(t.SampleQC=='PASS')&(t.EVENT_ID=='BL')&t.PATNO.notna()]; t['PATNO']=t.PATNO.astype(int); return t.groupby('PATNO').agg(NPX=('NPX','mean'),LOD=('LOD','mean')).reindex(ids)
F277=B+"/PPMI_Project_277_CSF_Screened_NPX_20260130.parquet"; F314=B+"/ppmi_proj314_csf_screened_extended_npx_20260116.parquet"

# ---------- 共用分析函数 ----------
def pd_vs_hc(df):
    s=df[df.group0.isin(['HC','PD'])].dropna(subset=['DDC','age','SEX']).copy(); s['PDx']=(s.group0=='PD').astype(int)
    m=smf.ols('DDC ~ PDx + age + SEX',s).fit(); sdhc=s.loc[s.PDx==0,'DDC'].std()
    return dict(n_pd=int(s.PDx.sum()),n_hc=int((s.PDx==0).sum()),beta=m.params.PDx,p=m.pvalues.PDx,std_effect=m.params.PDx/sdhc,auc=roc_auc_score(s.PDx,s.DDC))
def proteome_scan(W,meta):
    """全蛋白组 PD vs HC (age, sex), 向量化 OLS; 返回 FDR<0.05 的蛋白列表"""
    s=meta[meta.group0.isin(['HC','PD'])].dropna(subset=['age','SEX']); W=W.reindex(s.index)
    W=W.loc[:,W.isna().mean()<=0.2]; Y=W.values.astype(float); med=np.nanmedian(Y,0); ii=np.where(np.isnan(Y)); Y[ii]=np.take(med,ii[1])
    X=np.column_stack([np.ones(len(s)),(s.group0=='PD').astype(float),s.age.values,s.SEX.values.astype(float)])
    XtXi=np.linalg.inv(X.T@X); Bt=XtXi@X.T@Y; E=Y-X@Bt; dof=len(s)-X.shape[1]; s2=(E*E).sum(0)/dof
    t=Bt[1]/np.sqrt(s2*XtXi[1,1]); p=2*stats.t.sf(np.abs(t),dof); q=multipletests(p,method='fdr_bh')[1]
    hits=[ann.Assay.get(c,c) for c in W.columns[q<0.05]]; return len(s),W.shape[1],hits,float(q[list(W.columns).index(jD)]) if jD in W.columns else np.nan
def saa_cells(df):
    s=df.dropna(subset=['DDC','age','SEX']).copy()
    s['cell']=np.where(s.group0=='HC','HC',s.group0+np.where(s.SAA==1,'_SAA+',np.where(s.SAA==0,'_SAA-','_na'))); s=s[~s.cell.str.endswith('_na')]
    out={}
    for c in ['Hyposmia_SAA+','Hyposmia_SAA-','iRBD_SAA+','iRBD_SAA-','PD_SAA+','PD_SAA-']:
        x=s[s.cell.isin([c,'HC'])].copy(); x['g']=(x.cell==c).astype(int)
        if x.g.sum()<5: continue
        m=smf.ols('DDC ~ g + age + SEX',x).fit(); ci=m.conf_int().loc['g']; out[c]=dict(n=int(x.g.sum()),beta=m.params.g,ci_lo=ci[0],ci_hi=ci[1],p=m.pvalues.g)
    return out
def saaneg_dat(df,hcref):
    sn=df[(df.SAA==0)&(df.group0!='HC')].dropna(subset=['DDC','SBR','age','SEX']).copy()
    sn['zSBR']=(sn.SBR-hcref.SBR.mean())/hcref.SBR.std()
    m=smf.ols('DDC ~ zSBR + C(group0) + age + SEX',sn).fit(); ci=m.conf_int().loc['zSBR']
    return sn,m,dict(n=int(m.nobs),beta=m.params.zSBR,ci_lo=ci[0],ci_hi=ci[1],p=m.pvalues.zSBR,se=m.bse.zSBR)

# =====================================================================
H("A (M1) 排除既往 PPMI CSF 蛋白组发布的参与者")
prior={}
s151=pd.read_parquet(D+"/somascan151_csf.parquet"); s151=s151.reset_index() if 'PATNO' not in s151.columns else s151
prior['151 SomaScan']=set(pd.to_numeric(s151.PATNO,errors='coerce').dropna().astype(int))
for pj in ['9000','222','196']:
    ids=set()
    for f in glob.glob(B+f"/PPMI_Project_{pj}_CSF_*NPX*.csv"):
        ids|=set(pd.to_numeric(pd.read_csv(f,usecols=['PATNO']).PATNO.astype(str).str.replace('PPMI-','',regex=False),errors='coerce').dropna().astype(int))   # PATNO 形如 'PPMI-3970'
    prior[f'{pj} Olink']=ids
allprior=set().union(*prior.values())
for k,v in prior.items(): P(f"  {k}: {len(v)} 人; 与发现集重叠 {len(v&set(disc.index))}, 与复现散发集重叠 {len(v&set(spor.index))}")
d_new=disc[~disc.index.isin(allprior)]; r_new=spor[~spor.index.isin(allprior)]
P(f"  排除后: 发现集 {len(disc)}→{len(d_new)} ({d_new.group0.value_counts().to_dict()}); 复现集 {len(spor)}→{len(r_new)} ({r_new.group0.value_counts().to_dict()})")
A={}
for nm,full,new,W in [('Discovery',disc,d_new,wide),('Replication',spor,r_new,wr)]:
    for lab,df in [('all',full),('excluding prior releases',new)]:
        e=pd_vs_hc(df); n,npro,hits,qd=proteome_scan(W.drop(columns=['PATNO'],errors='ignore'),df)
        e.update(n_scan=n,n_proteins=npro,fdr_hits=hits,ddc_fdr=qd); A[f'{nm} | {lab}']=e
        P(f"  {nm:11s} {lab:26s} PD {e['n_pd']} / HC {e['n_hc']}: DDC β={e['beta']:.3f} NPX ({e['std_effect']:.2f} control SD), P={e['p']:.1e}, AUC={e['auc']:.3f}; 全蛋白组 FDR<0.05: {hits}")
cells_new=saa_cells(r_new)
for c,v in cells_new.items(): P(f"  复现集 (排除既往发布) {c:14s} n={v['n']:3d} β={v['beta']:+.3f} [{v['ci_lo']:+.3f},{v['ci_hi']:+.3f}] P={v['p']:.1e}")
_,_,dn=saaneg_dat(r_new,spor[spor.group0=='HC']); P(f"  复现集 (排除既往发布) SAA− 非对照 DDC–SBR: n={dn['n']} β={dn['beta']:+.3f} [{dn['ci_lo']:+.3f},{dn['ci_hi']:+.3f}] P={dn['p']:.3f}")
res['A']=dict(prior_sizes={k:len(v) for k,v in prior.items()},n_disc_new=len(d_new),n_rep_new=len(r_new),pd_vs_hc=A,saa_cells_rep_new=cells_new,saaneg_dat_rep_new=dn)

# =====================================================================
H("B (M3) 五成分模型独立 R² 的 bootstrap 95% CI (2,000 次)")
COMP=['SAA','zSBR','zupdrs3_score','zupdrs2_score','zmoca']
def prep5(df):
    d=df[df.NSD_STAGE.notna()].dropna(subset=['DDC','SAA','SBR','updrs3_score','updrs2_score','moca','age','SEX']).copy()   # 与 r3A 分解同一样本 (需有 NSD 分期)
    for v in ['SBR','updrs3_score','updrs2_score','moca']: d['z'+v]=(d[v]-d[v].mean())/d[v].std()
    return d
def uniq(d):
    y=d.DDC.values; base=np.column_stack([np.ones(len(d))]+[d[c].values for c in COMP]+[d.age.values,d.SEX.values.astype(float)])
    def r2(X): b=np.linalg.lstsq(X,y,rcond=None)[0]; e=y-X@b; return 1-e@e/((y-y.mean())@(y-y.mean()))
    full=r2(base); return {c:full-r2(np.delete(base,1+i,axis=1)) for i,c in enumerate(COMP)}
Bres=[]
for nm,df in [('Discovery',disc),('Replication',spor[spor.LEDD.fillna(0)==0])]:
    d=prep5(df); obs=uniq(d)
    def bs(k):
        i=np.random.default_rng(k).integers(0,len(d),len(d)); return uniq(d.iloc[i])
    bb=pd.DataFrame(Parallel(n_jobs=NJ)(delayed(bs)(k) for k in range(2000)))
    diff=bb.SAA-bb.zSBR; ratio=bb.SAA/bb.zSBR.clip(lower=1e-6)
    for c in COMP: Bres.append(dict(cohort=nm,n=len(d),term=c,unique_R2=obs[c],ci_lo=bb[c].quantile(.025),ci_hi=bb[c].quantile(.975)))
    Bres.append(dict(cohort=nm,n=len(d),term='SAA minus SBR',unique_R2=obs['SAA']-obs['zSBR'],ci_lo=diff.quantile(.025),ci_hi=diff.quantile(.975),p_boot=float((diff<=0).mean())))
    Bres.append(dict(cohort=nm,n=len(d),term='SAA / SBR ratio',unique_R2=obs['SAA']/obs['zSBR'],ci_lo=ratio.quantile(.025),ci_hi=ratio.quantile(.975)))
    P(f"  {nm:11s} n={len(d)}: "+"; ".join(f"{c} {obs[c]:.4f} [{bb[c].quantile(.025):.4f},{bb[c].quantile(.975):.4f}]" for c in COMP))
    P(f"     SAA − SBR = {obs['SAA']-obs['zSBR']:.4f} [{diff.quantile(.025):.4f},{diff.quantile(.975):.4f}], bootstrap P(差≤0)={float((diff<=0).mean()):.4f}; 比值 {obs['SAA']/obs['zSBR']:.1f} [{ratio.quantile(.025):.1f},{ratio.quantile(.975):.1f}]")
pd.DataFrame(Bres).to_csv(D+"/r6_unique_R2_bootstrap.csv",index=False)

# =====================================================================
H("C (M3, m3) SAA− 参与者 DDC–SBR: 最小可检出效应、Tobit、秩回归; 各组 SAA+ vs SAA− 秩 AUC")
sn,m,dd=saaneg_dat(spor,spor[spor.group0=='HC'])
mde=(stats.norm.ppf(0.975)+stats.norm.ppf(0.8))*dd['se']
P(f"  OLS (同 r4D): n={dd['n']} β={dd['beta']:+.3f} [{dd['ci_lo']:+.3f},{dd['ci_hi']:+.3f}]; SE={dd['se']:.3f} → 80% 功效下可检出的最小斜率 |β| = {mde:.3f} NPX/对照 SD (SAA− 患者升高 0.42 NPX 的 {mde/0.42*100:.0f}%)")
sl=sn.join(lodtab(F314,sn.index),rsuffix='_r').dropna(subset=['NPX','LOD'])
G=pd.get_dummies(sl.group0,drop_first=True).astype(float)
X=np.column_stack([np.ones(len(sl)),sl.zSBR,G.values,sl.age,sl.SEX.astype(float)]); th,se,cf=tobit(sl.NPX.values,X,sl.LOD.values)
tob=dict(n=len(sl),beta=th[1],ci_lo=th[1]-1.96*se[1],ci_hi=th[1]+1.96*se[1],p=2*stats.norm.sf(abs(th[1]/se[1])),censored=cf)
P(f"  Tobit: n={len(sl)} (删失 {cf*100:.1f}%) β={th[1]:+.3f} [{tob['ci_lo']:+.3f},{tob['ci_hi']:+.3f}] P={tob['p']:.3f}")
sn['rDDC']=stats.norm.ppf((sn.DDC.rank()-0.5)/len(sn)); mr=smf.ols('rDDC ~ zSBR + C(group0) + age + SEX',sn).fit(); cr=mr.conf_int().loc['zSBR']
rk=dict(n=int(mr.nobs),beta=mr.params.zSBR,ci_lo=cr[0],ci_hi=cr[1],p=mr.pvalues.zSBR)
P(f"  秩 (正态分数) 回归: β={rk['beta']:+.3f} SD [{rk['ci_lo']:+.3f},{rk['ci_hi']:+.3f}] P={rk['p']:.3f}")
auc={}
for g in ['Hyposmia','iRBD','PD']:
    x=spor[(spor.group0==g)&spor.SAA.notna()].dropna(subset=['DDC'])
    a=roc_auc_score(x.SAA,x.DDC); u=stats.mannwhitneyu(x.DDC[x.SAA==1],x.DDC[x.SAA==0])
    auc[g]=dict(n_pos=int(x.SAA.sum()),n_neg=int((x.SAA==0).sum()),auc=a,p=u.pvalue)
    P(f"  {g:9s} SAA+ vs SAA− 秩 AUC={a:.2f} (n {int(x.SAA.sum())} vs {int((x.SAA==0).sum())}), Mann–Whitney P={u.pvalue:.1e}")
res['C']=dict(ols=dd,mde80=mde,tobit=tob,rank=rk,auc_saa_within_group=auc)

# =====================================================================
H("D (m2) 仅采样时未用药者 (LEDD = 0), 并排除 4 名其他类别")
d_u=disc[disc.group0.isin(['HC','Hyposmia','iRBD','PD'])&(disc.LEDD.fillna(0)==0)]; r_u=spor[spor.LEDD.fillna(0)==0]
Dd={}
for nm,full,u in [('Discovery',disc,d_u),('Replication',spor,r_u)]:
    e=pd_vs_hc(u); Dd[nm]=e; P(f"  {nm:11s} n {len(full)}→{len(u)}: PD vs HC β={e['beta']:.3f}, AUC={e['auc']:.3f}, P={e['p']:.1e}")
cu=saa_cells(r_u); Dd['rep_cells']=cu
for c,v in cu.items(): P(f"  复现集未用药 {c:14s} n={v['n']:3d} β={v['beta']:+.3f} [{v['ci_lo']:+.3f},{v['ci_hi']:+.3f}] P={v['p']:.1e}")
_,_,du=saaneg_dat(r_u,spor[spor.group0=='HC']); Dd['saaneg_dat']=du
P(f"  复现集未用药 SAA− 非对照 DDC–SBR: n={du['n']} β={du['beta']:+.3f} [{du['ci_lo']:+.3f},{du['ci_hi']:+.3f}] P={du['p']:.3f}")
res['D']=Dd
json.dump(res,open(D+"/r6_reviewer.json","w"),indent=1,default=lambda o: float(o) if isinstance(o,(np.floating,np.integer)) else str(o))
P("\ndone"); LOG.close()
