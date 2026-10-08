# -*- coding: utf-8 -*-
"""Within-person effect of dopaminergic treatment on CSF DDC (mixed, fixed-effect, event-study and dose models;
whole-proteome scan; CSF haemoglobin; plasma Project 293 as positive control). Output: derived/r3B_*.csv (Table S22).
"""
import config as cfg
import sys, warnings, json
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
B=cfg.P; D=cfg.D; OUT=cfg.RS
LOG=open(OUT+"/r3B_log.txt","w",encoding="utf-8")
def P(*a):
    s=" ".join(str(x) for x in a); print(s); LOG.write(s+"\n"); LOG.flush()
def H(t): P("\n"+"="*78+"\n"+t+"\n"+"="*78)
clin=pd.read_csv(D+"/analytic_clinical.csv").set_index('PATNO')
cur=pd.read_excel(cfg.CURATED,sheet_name='20260511',usecols=['PATNO','EVENT_ID','YEAR','LEDD','hemohi','updrs3_score'])
cur=cur.drop_duplicates(['PATNO','EVENT_ID'])
txstart=cur[cur.LEDD>0].groupby('PATNO').YEAR.min()          # 首次 LEDD>0 的临床访视年份
ann=pd.read_csv(D+"/protein_annotation.csv").set_index('OlinkID')
FILES={'277':B+"/PPMI_Project_277_CSF_Screened_NPX_20260130.parquet",'314':B+"/ppmi_proj314_csf_screened_extended_npx_20260116.parquet",
       '293 plasma':B+"/ppmi_proj293_plasma_screened_extended_npx_20251121.parquet"}

def load(f):
    t=pq.read_table(f,columns=['PATNO','EVENT_ID','SampleType','SampleQC','AssayType','OlinkID','NPX','LAB_QC_FLAG'])
    t=t.filter(pc.and_(pc.equal(t['SampleType'],'SAMPLE'),pc.equal(t['AssayType'],'assay'))).to_pandas()
    t=t[(t.SampleQC=='PASS')&t.PATNO.notna()]; t=t[~t.LAB_QC_FLAG.fillna('').astype(str).str.contains('6|7')]
    t['PATNO']=t.PATNO.astype(int)
    w=t.pivot_table(index=['PATNO','EVENT_ID'],columns='OlinkID',values='NPX',aggfunc='mean'); return w
def meta(w):
    m=pd.DataFrame(index=w.index).reset_index()
    m=m.merge(cur,on=['PATNO','EVENT_ID'],how='left')
    m['COHORT']=m.PATNO.map(clin.COHORT); m['group0']=m.PATNO.map(clin.group0)
    nmiss_fu=int(((m.EVENT_ID!='BL')&m.LEDD.isna()).sum())
    m.loc[(m.EVENT_ID=='BL')&m.LEDD.isna(),'LEDD']=0                      # 基线 de novo
    m['treated']=(m.LEDD>0).astype(float); m.loc[m.LEDD.isna(),'treated']=np.nan   # 随访缺失 → NaN
    m['logLEDD']=np.log1p(m.LEDD); m['tx0']=m.PATNO.map(txstart)
    m['yrs_tx']=np.where(m.treated==1,np.maximum(0,m.YEAR-m.tx0),0.0); m.index=pd.RangeIndex(len(m))
    return m,nmiss_fu
DDCid={v:k for k,v in ann.Assay.items()}
def oid(g): return DDCid.get(g)
summary=[]; scan_all=[]
for proj,f in FILES.items():
    H(f"Project {proj}")
    w=load(f); m,nmiss=meta(w)
    blz=m[(m.EVENT_ID=='BL')&(m.LEDD==0)].PATNO
    sel=m.PATNO.isin(blz)&(m.COHORT==1)&m.treated.notna()&m.YEAR.notna()
    cnt=m[sel].groupby('PATNO').size(); keep=cnt[cnt>=2].index; sel&=m.PATNO.isin(keep)
    M=m[sel].copy(); W=w.iloc[M.index.values]; W.index=M.index
    n_sw=int(M.groupby('PATNO').treated.max().sum()); n_never=int((M.groupby('PATNO').treated.max()==0).sum())
    pre=M[M.treated==0].groupby('PATNO').size(); n_pre2=int((pre>=2).sum())
    P(f"  PD 分析集 {M.PATNO.nunique()} 人 / {len(M)} 样本; 转为治疗 {n_sw}; 从未治疗 {n_never}; 治疗前 ≥2 样本 {n_pre2}; 随访 LEDD 缺失已剔除 {nmiss} 个样本")
    P(f"  个体内 treated 与 YEAR 的相关 (去均值后) r={np.corrcoef(M.treated-M.groupby('PATNO').treated.transform('mean'),M.YEAR-M.groupby('PATNO').YEAR.transform('mean'))[0,1]:.2f}")
    for gene in ['DDC','PRL','AOC3','LPO']:
        o=oid(gene)
        if o not in W.columns: continue
        d=M.copy(); d['y']=W[o].values; d=d.dropna(subset=['y'])
        res=dict(project=proj,protein=gene,n=d.PATNO.nunique(),n_samples=len(d),n_switch=n_sw,n_never=n_never,n_pre2=n_pre2)
        try:
            mm=smf.mixedlm("y ~ treated + YEAR",d,groups=d.PATNO).fit(method='lbfgs'); res.update(mixed_b=mm.params.treated,mixed_lo=mm.conf_int().loc['treated',0],mixed_hi=mm.conf_int().loc['treated',1],mixed_p=mm.pvalues.treated)
            es=smf.mixedlm("y ~ YEAR + treated + yrs_tx",d,groups=d.PATNO).fit(method='lbfgs')
            res.update(es_jump=es.params.treated,es_lo=es.conf_int().loc['treated',0],es_hi=es.conf_int().loc['treated',1],es_p=es.pvalues.treated,es_preslope=es.params.YEAR,es_preslope_p=es.pvalues.YEAR,es_post=es.params.yrs_tx,es_post_p=es.pvalues.yrs_tx)
        except Exception as e: P('   mixed fail',gene,repr(e)[:80])
        dm=d[['y','treated','YEAR','logLEDD','hemohi','PATNO']].copy()
        for c in ['y','treated','YEAR','logLEDD']: dm[c+'_d']=dm[c]-dm.groupby('PATNO')[c].transform('mean')
        fe=smf.ols("y_d ~ treated_d + YEAR_d - 1",dm).fit(cov_type='cluster',cov_kwds={'groups':dm.PATNO})
        fd=smf.ols("y_d ~ logLEDD_d + YEAR_d - 1",dm).fit(cov_type='cluster',cov_kwds={'groups':dm.PATNO})
        res.update(fe_b=fe.params.treated_d,fe_lo=fe.conf_int().loc['treated_d',0],fe_hi=fe.conf_int().loc['treated_d',1],fe_p=fe.pvalues.treated_d,fe_year=fe.params.YEAR_d,
                   fe_dose_b=fd.params.logLEDD_d,fe_dose_p=fd.pvalues.logLEDD_d)
        # 血红蛋白: 校正 / 剔除
        hh=dm.dropna(subset=['hemohi']).copy()
        if hh.hemohi.nunique()>1:
            hh['h_d']=hh.hemohi-hh.groupby('PATNO').hemohi.transform('mean')
            fh=smf.ols("y_d ~ treated_d + YEAR_d + h_d - 1",hh).fit(cov_type='cluster',cov_kwds={'groups':hh.PATNO}); res.update(fe_hemo_adj_b=fh.params.treated_d,fe_hemo_adj_p=fh.pvalues.treated_d,hemo_pos_frac=(d.hemohi>0).mean())
        cl=d[(d.hemohi.fillna(0)==0)].copy()
        if cl.PATNO.nunique()>10:
            for c in ['y','treated','YEAR']: cl[c+'_d']=cl[c]-cl.groupby('PATNO')[c].transform('mean')
            fc=smf.ols("y_d ~ treated_d + YEAR_d - 1",cl).fit(cov_type='cluster',cov_kwds={'groups':cl.PATNO}); res.update(fe_hemo_excl_b=fc.params.treated_d,fe_hemo_excl_p=fc.pvalues.treated_d)
        # 配对 (转换者)
        sw=d.groupby('PATNO').treated.max(); ids=sw[sw==1].index; pp=d[d.PATNO.isin(ids)]
        a0=pp[pp.treated==0].groupby('PATNO').y.mean(); a1=pp[pp.treated==1].groupby('PATNO').y.mean(); ix=a0.index.intersection(a1.index); dd=a1[ix]-a0[ix]
        res.update(paired_n=len(ix),paired_median=dd.median(),paired_mean=dd.mean(),paired_frac_up=(dd>0).mean(),paired_p=stats.wilcoxon(dd).pvalue if len(ix)>5 else np.nan)
        # 从未治疗的 PD: 每年变化
        nv=d[d.PATNO.isin(sw[sw==0].index)]
        if nv.PATNO.nunique()>=5:
            nv=nv.copy(); nv['y_d']=nv.y-nv.groupby('PATNO').y.transform('mean'); nv['Y_d']=nv.YEAR-nv.groupby('PATNO').YEAR.transform('mean')
            mn=smf.ols("y_d ~ Y_d - 1",nv).fit(cov_type='cluster',cov_kwds={'groups':nv.PATNO}); res.update(never_n=nv.PATNO.nunique(),never_slope=mn.params.Y_d,never_p=mn.pvalues.Y_d)
        summary.append(res)
        P(f"  {gene:5s} | 混合 treated β={res.get('mixed_b',np.nan):+.3f} P={res.get('mixed_p',np.nan):.1e} | event-study 跳变 β={res.get('es_jump',np.nan):+.3f} [{res.get('es_lo',np.nan):+.2f},{res.get('es_hi',np.nan):+.2f}] P={res.get('es_p',np.nan):.1e}, 病程斜率 {res.get('es_preslope',np.nan):+.3f}/年 (P={res.get('es_preslope_p',np.nan):.1e}), 治疗后附加斜率 {res.get('es_post',np.nan):+.3f}/年 (P={res.get('es_post_p',np.nan):.1e})")
        P(f"        | FE treated β={res['fe_b']:+.3f} [{res['fe_lo']:+.2f},{res['fe_hi']:+.2f}] P={res['fe_p']:.1e}; FE 剂量 β={res['fe_dose_b']:+.3f} P={res['fe_dose_p']:.1e}; 血红蛋白校正 β={res.get('fe_hemo_adj_b',np.nan):+.3f} P={res.get('fe_hemo_adj_p',np.nan):.1e}; 剔除血红蛋白阳性 β={res.get('fe_hemo_excl_b',np.nan):+.3f} P={res.get('fe_hemo_excl_p',np.nan):.1e}")
        P(f"        | 配对 n={res['paired_n']} 中位 {res['paired_median']:+.3f} 均值 {res['paired_mean']:+.3f} 上升 {100*res['paired_frac_up']:.0f}% P={res['paired_p']:.1e}; 从未治疗 PD n={res.get('never_n','-')} 斜率 {res.get('never_slope',np.nan):+.3f}/年 P={res.get('never_p',np.nan):.2f}")
    # 全蛋白组 FE 扫描
    Y=W.values.astype(float); ok=(~np.isnan(Y)).mean(0)>0.8; Y=Y[:,ok]; colsY=W.columns[ok]
    med=np.nanmedian(Y,0); ii=np.where(np.isnan(Y)); Y[ii]=np.take(med,ii[1])
    g=M.PATNO.values; Xm=M[['treated','YEAR']].values.astype(float)
    def demean(A):
        A=pd.DataFrame(A); return (A-A.groupby(g).transform('mean')).values
    Yd=demean(Y); Xd=demean(Xm); XtXi=np.linalg.inv(Xd.T@Xd); Bh=XtXi@Xd.T@Yd; res_=Yd-Xd@Bh
    dof=len(M)-M.PATNO.nunique()-2; s2=(res_**2).sum(0)/dof; se=np.sqrt(s2*XtXi[0,0]); bz=Bh[0]/Y.std(0)
    pv=2*stats.t.sf(np.abs(Bh[0]/se),dof); fdr=multipletests(pv,method='fdr_bh')[1]
    sc=pd.DataFrame({'OlinkID':colsY,'gene':ann.Assay.reindex(colsY).values,'beta':Bh[0],'beta_sd':bz,'p':pv,'fdr':fdr}).sort_values('p'); sc['project']=proj
    sc['rank']=np.arange(1,len(sc)+1); scan_all.append(sc)
    r=sc.set_index('gene')
    P(f"  全蛋白组个体内 FE 扫描 ({len(sc)} 蛋白): FDR<0.05 {int((sc.fdr<0.05).sum())} 个 (上升 {int(((sc.fdr<0.05)&(sc.beta>0)).sum())}, 下降 {int(((sc.fdr<0.05)&(sc.beta<0)).sum())})")
    for gname in ['DDC','PRL','AOC3','LPO']:
        if gname in r.index: x=r.loc[gname]; x=x.iloc[0] if isinstance(x,pd.DataFrame) else x; P(f"     {gname:5s} rank {int(x['rank'])}/{len(sc)} β={x.beta:+.3f} ({x.beta_sd:+.2f} SD) P={x.p:.1e} FDR={x.fdr:.1e}")
    P("     前 10: "+", ".join(f"{a}({b:+.2f})" for a,b in zip(sc.gene.head(10),sc.beta.head(10))))
pd.DataFrame(summary).to_csv(D+"/r3B_treatment_estimators.csv",index=False)
pd.concat(scan_all).to_csv(D+"/r3B_proteome_FE_scan.csv",index=False)

# CSF 与血浆变化的配对相关 (同一人同一访视)
H("CSF (277/314) 与血浆 (293) 同人同访视 DDC 变化")
P("  (见上: 血浆 DDC 的个体内治疗效应作为已知阳性对照)")
P("\ndone"); LOG.close()
