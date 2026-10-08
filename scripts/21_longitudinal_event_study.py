# -*- coding: utf-8 -*-
"""Longitudinal change and treatment: participant fixed-effects event study, drift benchmark, whole-proteome
treatment scan. Output: derived/r4B_*.csv (Supplementary Fig. S7, Table S22).
"""
import config as cfg
import sys, warnings, json
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc
from patsy import dmatrix
from scipy import stats
from statsmodels.stats.multitest import multipletests
from joblib import Parallel, delayed
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
B=cfg.P; D=cfg.D; OUT=cfg.RS; NJ=30
LOG=open(OUT+"/r4B_log.txt","w",encoding="utf-8")
def P(*a):
    s=" ".join(str(x) for x in a); print(s); LOG.write(s+"\n"); LOG.flush()
def H(t): P("\n"+"="*78+"\n"+t+"\n"+"="*78)
clin=pd.read_csv(D+"/analytic_clinical.csv").set_index('PATNO')
cur=pd.read_excel(cfg.CURATED,sheet_name='20260511',usecols=['PATNO','EVENT_ID','YEAR','LEDD'])
cur=cur.drop_duplicates(['PATNO','EVENT_ID']); tx0=cur[cur.LEDD>0].groupby('PATNO').YEAR.min()
ann=pd.read_csv(D+"/protein_annotation.csv").set_index('OlinkID'); oid={v:k for k,v in ann.Assay.items()}
QC=list(pd.read_parquet(D+"/csf_npx_wide.parquet").columns.drop('PATNO',errors='ignore')); QC=[c for c in QC if c in ann.index]
FILES={'277 CSF':B+"/PPMI_Project_277_CSF_Screened_NPX_20260130.parquet",'314 CSF':B+"/ppmi_proj314_csf_screened_extended_npx_20260116.parquet",'293 plasma':B+"/ppmi_proj293_plasma_screened_extended_npx_20251121.parquet"}
def load(f):
    t=pq.read_table(f,columns=['PATNO','EVENT_ID','SampleType','SampleQC','AssayType','OlinkID','Assay','NPX','LOD','LAB_QC_FLAG'])
    t=t.filter(pc.and_(pc.equal(t['SampleType'],'SAMPLE'),pc.equal(t['AssayType'],'assay'))).to_pandas()
    t=t[(t.SampleQC=='PASS')&t.PATNO.notna()]; t=t[~t.LAB_QC_FLAG.fillna('').astype(str).str.contains('6|7')]; t['PATNO']=t.PATNO.astype(int)
    lodf={g:float((t[t.Assay==g].NPX<t[t.Assay==g].LOD).mean()) for g in ['PRL','DDC']}
    w=t.pivot_table(index=['PATNO','EVENT_ID'],columns='OlinkID',values='NPX',aggfunc='mean'); return w,lodf
def meta(w):
    m=w.index.to_frame(index=False).merge(cur,on=['PATNO','EVENT_ID'],how='left')
    m['COHORT']=m.PATNO.map(clin.COHORT)
    m.loc[(m.EVENT_ID=='BL')&m.LEDD.isna(),'LEDD']=0
    m['treated']=np.where(m.LEDD.isna(),np.nan,(m.LEDD>0).astype(float)); m['tx0']=m.PATNO.map(tx0)
    m['yrs_tx']=np.where(m.treated==1,np.maximum(0,m.YEAR-m.tx0),0.0); return m
def demean(A,g): A=pd.DataFrame(A); return (A-A.groupby(g).transform('mean')).values
def cluster_ols(y,X,g):
    """个体内已去均值的 y, X; 返回 β, 聚类协方差"""
    XtXi=np.linalg.pinv(X.T@X); b=XtXi@X.T@y; e=y-X@b; S=np.zeros((X.shape[1],X.shape[1]))
    for gg in np.unique(g):
        ix=g==gg; s=X[ix].T@e[ix]; S+=np.outer(s,s)
    G_=len(np.unique(g)); n,k=X.shape; c=G_/(G_-1)*(n-1)/(n-k); return b,XtXi@S@XtXi*c
summ=[]; scans=[]; drift=[]
for proj,f in FILES.items():
    H(proj); w,lodf=load(f); m=meta(w)
    P(f"  低于 LOD: PRL {100*lodf['PRL']:.1f}%, DDC {100*lodf['DDC']:.1f}%")
    blz=set(m[(m.EVENT_ID=='BL')&(m.LEDD==0)].PATNO)
    pdm=m[m.PATNO.isin(blz)&(m.COHORT==1)&m.treated.notna()&m.YEAR.notna()].copy()
    cnt=pdm.groupby('PATNO').size(); pdm=pdm[pdm.PATNO.isin(cnt[cnt>=2].index)]
    last=pdm.groupby('PATNO').YEAR.max(); sw=pdm.PATNO.map(lambda p: (p in tx0.index) and (tx0[p]<=last[p]))
    pdm['switcher']=sw.values; npre=pdm[pdm.treated==0].groupby('PATNO').size()
    n_sw=pdm[pdm.switcher].PATNO.nunique(); n_nv=pdm[~pdm.switcher].PATNO.nunique(); n_pre2=int((npre>=2).sum())
    P(f"  PD 分析集 {pdm.PATNO.nunique()} 人 / {len(pdm)} 样本; 转换者 {n_sw}; 从未治疗 {n_nv}; 治疗前 ≥2 样本 {n_pre2}")
    ns=dmatrix("cr(YEAR, df=3) - 1",pdm,return_type='dataframe').values
    Wp=w.loc[list(zip(pdm.PATNO,pdm.EVENT_ID))]
    for gname in ['DDC','PRL','AOC3','LPO']:
        o=oid.get(gname)
        if o not in Wp.columns: continue
        y=Wp[o].values; ok=~np.isnan(y); g=pdm.PATNO.values[ok]
        Xr=np.column_stack([ns[ok],pdm.treated.values[ok],pdm.yrs_tx.values[ok]]); yd=demean(y[ok],g).ravel(); Xd=demean(Xr,g)
        keepc=Xd.std(0)>1e-9; Xd=Xd[:,keepc]; names=([f'ns{i}' for i in range(ns.shape[1])]+['step','post_slope']); names=[n for n,kk in zip(names,keepc) if kk]
        b,V=cluster_ols(yd,Xd,g); i1=names.index('step'); i2=names.index('post_slope')
        se1=np.sqrt(V[i1,i1]); se2=np.sqrt(V[i2,i2]); bb=b[[i1,i2]]; VV=V[np.ix_([i1,i2],[i1,i2])]
        wald=float(bb@np.linalg.pinv(VV)@bb); pj=1-stats.chi2.cdf(wald,2)
        med=np.median(pdm.yrs_tx.values[ok][pdm.treated.values[ok]==1]); cvec=np.array([1,med]); con=float(cvec@bb); sec=float(np.sqrt(cvec@VV@cvec))
        # 简单 FE: treated + 线性 YEAR
        Xs=demean(np.column_stack([pdm.treated.values[ok],pdm.YEAR.values[ok]]),g); bs,Vs=cluster_ols(yd,Xs,g)
        # 从未治疗 PD 的年斜率
        nv=~pdm.switcher.values[ok]; nvs=np.nan; nvp=np.nan
        if len(np.unique(g[nv]))>=5:
            xn=demean(pdm.YEAR.values[ok][nv],g[nv]); bn,Vn=cluster_ols(demean(y[ok][nv],g[nv]).ravel(),xn,g[nv]); nvs=bn[0]; nvp=2*stats.norm.sf(abs(bn[0]/np.sqrt(Vn[0,0])))
        r=dict(project=proj,protein=gname,n=len(np.unique(g)),samples=int(ok.sum()),n_switch=n_sw,n_never=n_nv,n_pre2=n_pre2,
               step=b[i1],step_lo=b[i1]-1.96*se1,step_hi=b[i1]+1.96*se1,step_p=2*stats.norm.sf(abs(b[i1]/se1)),
               post_slope=b[i2],post_lo=b[i2]-1.96*se2,post_hi=b[i2]+1.96*se2,post_p=2*stats.norm.sf(abs(b[i2]/se2)),joint_p=pj,
               median_yrs_tx=med,contrast_at_median=con,con_lo=con-1.96*sec,con_hi=con+1.96*sec,con_p=2*stats.norm.sf(abs(con/sec)),
               fe_treated=bs[0],fe_lo=bs[0]-1.96*np.sqrt(Vs[0,0]),fe_hi=bs[0]+1.96*np.sqrt(Vs[0,0]),fe_p=2*stats.norm.sf(abs(bs[0]/np.sqrt(Vs[0,0]))),
               fe_year=bs[1],fe_year_p=2*stats.norm.sf(abs(bs[1]/np.sqrt(Vs[1,1]))),never_slope=nvs,never_p=nvp,lod_frac=lodf.get(gname,np.nan))
        summ.append(r)
        P(f"  {gname:5s} n={r['n']}: 跳变 {r['step']:+.3f} [{r['step_lo']:+.2f},{r['step_hi']:+.2f}] P={r['step_p']:.3f} | 治疗后斜率 {r['post_slope']:+.3f}/年 P={r['post_p']:.3f} | 联合 P={pj:.3f} | 中位 {med:.1f} 年处对比 {con:+.3f} [{con-1.96*sec:+.2f},{con+1.96*sec:+.2f}] P={r['con_p']:.3f} | 简单FE {bs[0]:+.3f} P={r['fe_p']:.1e} | 从未治疗斜率 {nvs:+.3f} P={nvp:.3f}")
    # 漂移基准: 从未治疗 PD 与未治疗非 PD 的全蛋白组年斜率
    cols=[c for c in QC if c in w.columns]
    for lab,sel in [('never-treated PD',pdm[~pdm.switcher]),('untreated non-PD',m[(m.COHORT!=1)&(m.treated==0)&m.YEAR.notna()])]:
        cnt=sel.groupby('PATNO').size(); sel=sel[sel.PATNO.isin(cnt[cnt>=2].index)]
        if sel.PATNO.nunique()<5: continue
        Wn=w.loc[list(zip(sel.PATNO,sel.EVENT_ID)),cols]; g=sel.PATNO.values; yr=sel.YEAR.values
        def slope(j):
            y=Wn.iloc[:,j].values; ok=~np.isnan(y)
            if ok.sum()<10: return (np.nan,np.nan)
            xd=demean(yr[ok],g[ok]); yd=demean(y[ok],g[ok]).ravel(); b,V=cluster_ols(yd,xd,g[ok]); sd=np.nanstd(y)
            return (b[0]/sd if sd>0 else np.nan, 2*stats.norm.sf(abs(b[0]/np.sqrt(V[0,0]))) if V[0,0]>0 else np.nan)
        out=np.array(Parallel(n_jobs=NJ)(delayed(slope)(j) for j in range(len(cols))))
        sdf=pd.DataFrame({'OlinkID':cols,'slope_sd_per_year':out[:,0],'p':out[:,1]}); sdf['gene']=ann.Assay.reindex(cols).values
        med=np.nanmedian(sdf.slope_sd_per_year); pos=(sdf.slope_sd_per_year>0).mean()
        for gname in ['DDC','PRL']:
            x=sdf[sdf.gene==gname].iloc[0]; pct=100*(sdf.slope_sd_per_year<x.slope_sd_per_year).mean()
            drift.append(dict(project=proj,set=lab,n=sel.PATNO.nunique(),protein=gname,slope_sd=x.slope_sd_per_year,p=x.p,percentile=pct,proteome_median=med,proteome_frac_positive=pos))
            P(f"  漂移基准 {lab:18s} n={sel.PATNO.nunique()}: {gname} 斜率 {x.slope_sd_per_year:+.3f} SD/年 (P={x.p:.3f}), 位于全蛋白组第 {pct:.0f} 百分位; 全蛋白组中位 {med:+.3f}, 正斜率比例 {100*pos:.0f}%")
    # 全蛋白组治疗扫描 (聚类 SE, 不插补)
    g_all=pdm.PATNO.values; Wq=Wp[[c for c in cols]]
    def scan(j):
        y=Wq.iloc[:,j].values; ok=~np.isnan(y)
        if ok.sum()<20: return (np.nan,np.nan)
        X=demean(np.column_stack([pdm.treated.values[ok],pdm.YEAR.values[ok]]),g_all[ok]); yd=demean(y[ok],g_all[ok]).ravel()
        if X[:,0].std()<1e-9: return (np.nan,np.nan)
        b,V=cluster_ols(yd,X,g_all[ok]); sd=np.nanstd(y); return (b[0]/sd, 2*stats.norm.sf(abs(b[0]/np.sqrt(V[0,0]))))
    out=np.array(Parallel(n_jobs=NJ)(delayed(scan)(j) for j in range(len(cols))))
    sc=pd.DataFrame({'OlinkID':cols,'gene':ann.Assay.reindex(cols).values,'beta_sd':out[:,0],'p':out[:,1]}).dropna()
    sc['fdr']=multipletests(sc.p,method='fdr_bh')[1]; sc=sc.sort_values('p'); sc['rank']=np.arange(1,len(sc)+1); sc['project']=proj; scans.append(sc)
    P(f"  全蛋白组扫描 (聚类 SE, {len(sc)} 蛋白): FDR<0.05 {int((sc.fdr<0.05).sum())}")
    for gname in ['DDC','PRL']:
        x=sc[sc.gene==gname].iloc[0]; P(f"     {gname}: rank {int(x['rank'])}/{len(sc)}, {x.beta_sd:+.2f} SD, P={x.p:.1e}, FDR={x.fdr:.2f}")
pd.DataFrame(summ).to_csv(D+"/r4B_event_study.csv",index=False); pd.DataFrame(drift).to_csv(D+"/r4B_drift_benchmark.csv",index=False)
pd.concat(scans).to_csv(D+"/r4B_proteome_scan_clustered.csv",index=False)
P("done"); LOG.close()
