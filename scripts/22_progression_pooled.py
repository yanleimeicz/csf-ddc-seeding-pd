# -*- coding: utf-8 -*-
"""Progression models with DDC standardised within set, pooled by inverse-variance weighting, BH across the family.
Output: derived/r4C_progression.csv, r4C_pooled.csv (Table 2, Supplementary Fig. S6, Tables S1, S21).
"""
import config as cfg
import sys, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
D=cfg.D; OUT=cfg.RS
LOG=open(OUT+"/r4C_log.txt","w",encoding="utf-8")
def P(*a):
    s=" ".join(str(x) for x in a); print(s); LOG.write(s+"\n"); LOG.flush()
GEN=['LRRK2','GBA','LRRK2 + GBA','SNCA','PRKN','PINK1','PRKN + RBD']; SBRc=['MIA_CAUDATE_L','MIA_CAUDATE_R','MIA_PUTAMEN_L','MIA_PUTAMEN_R']
clin=pd.read_csv(D+"/analytic_clinical.csv").set_index('PATNO')
cur=pd.read_excel(cfg.CURATED,sheet_name='20260511',usecols=['PATNO','EVENT_ID','YEAR','LEDD','updrs3_score','moca']+SBRc)
cur=cur.drop_duplicates(['PATNO','EVENT_ID']); cur['SBR_t']=cur[SBRc].mean(1); bl=cur[cur.EVENT_ID=='BL'].set_index('PATNO')
wide=pd.read_parquet(D+"/csf_npx_wide.parquet").set_index('PATNO'); wide.index=wide.index.astype(int)
ann=pd.read_csv(D+"/protein_annotation.csv").set_index('OlinkID'); jD=[c for c in wide.columns if ann.Assay.get(c)=='DDC'][0]
disc=clin.reindex(wide.index).copy(); disc['DDC']=wide[jD].fillna(wide[jD].median()).values
rep=pd.read_csv(D+"/p314_rep_baseline_analytic.csv").set_index('PATNO'); spor=rep[~rep.subgroup.isin(GEN)&rep.group0.isin(['HC','Hyposmia','iRBD','PD'])].copy()
for df in (disc,spor):
    df['SAA']=df.CSFSAA.where(df.CSFSAA.isin([0,1])); df['bl_U3']=bl.updrs3_score.reindex(df.index); df['bl_SBR']=bl.SBR_t.reindex(df.index)
    df['zDDC']=(df.DDC-df.DDC.mean())/df.DDC.std()       # 队列内全体基线 SD
specs={'M0':'{y} ~ zDDC*YEAR + age + SEX','M1':'{y} ~ zDDC*YEAR + C(group0)*YEAR + SAA*YEAR + age*YEAR + SEX*YEAR',
       'M2':'{y} ~ zDDC*YEAR + C(group0)*YEAR + SAA*YEAR + age*YEAR + SEX*YEAR + bl_U3*YEAR + bl_SBR*YEAR'}
rows=[]
for coh,base in [('Discovery',disc),('Replication',spor)]:
    b=base[base.group0.isin(['iRBD','Hyposmia','PD'])][['zDDC','age','SEX','group0','SAA','bl_U3','bl_SBR']]
    L=cur[cur.PATNO.isin(b.index)][['PATNO','EVENT_ID','YEAR','LEDD','updrs3_score','SBR_t','moca']].merge(b,left_on='PATNO',right_index=True)
    L.loc[(L.EVENT_ID=='BL')&L.LEDD.isna(),'LEDD']=0; L=L.dropna(subset=['YEAR','zDDC','age','SEX'])
    pro=L[L.group0.isin(['iRBD','Hyposmia'])]
    subsets={'Untreated visits':L[L.LEDD==0],'SAA-positive prodromal':pro[pro.SAA==1],'PD only':L[L.group0=='PD']}
    if coh=='Replication': subsets['All prodromal']=pro
    for sn,df in subsets.items():
        for y in ['updrs3_score','SBR_t','moca']:
            for mn,f in specs.items():
                ff=f.format(y=y)
                if df.group0.nunique()<2: ff=ff.replace(' + C(group0)*YEAR','')
                if df.SAA.dropna().nunique()<2: ff=ff.replace(' + SAA*YEAR','')
                if y=='SBR_t': ff=ff.replace(' + bl_SBR*YEAR','')
                need=['SAA'] if 'SAA*' in ff else []
                need+=['bl_U3','bl_SBR'] if mn=='M2' else []
                dd=df.dropna(subset=[y]+need); cnt=dd.groupby('PATNO').size(); dd=dd[dd.PATNO.isin(cnt[cnt>=2].index)]
                try:
                    m=smf.mixedlm(ff,dd,groups=dd.PATNO,re_formula='~YEAR').fit(method=['lbfgs','powell'])
                    k='zDDC:YEAR'; ci=m.conf_int().loc[k]
                    rows.append(dict(cohort=coh,subset=sn,outcome=y,model=mn,n_participants=dd.PATNO.nunique(),n_obs=len(dd),beta_per_SD=m.params[k],ci_lo=ci[0],ci_hi=ci[1],p=m.pvalues[k],converged=bool(m.converged)))
                except Exception as e: P('fail',coh,sn,y,mn,repr(e)[:80])
pm=pd.DataFrame(rows); pm['fdr_family']=multipletests(pm.p,method='fdr_bh')[1]
P(f"检验总数 {len(pm)} (2 集 × 子集 × 3 结局 × 3 模型); 未收敛 {int((~pm.converged).sum())}")
for _,r in pm[pm.subset.isin(['SAA-positive prodromal','Untreated visits'])].iterrows():
    P(f"  {r.cohort:11s} {r.subset:22s} {r.outcome:12s} {r.model} n={r.n_participants:3d}/{r.n_obs:4d} β/SD={r.beta_per_SD:+.4f} [{r.ci_lo:+.3f},{r.ci_hi:+.3f}] P={r.p:.3f} FDR={r.fdr_family:.3f} conv={r.converged}")
pm.to_csv(D+"/r4C_progression.csv",index=False)
pr=[]
for sub in ['SAA-positive prodromal','Untreated visits','PD only']:
    for y in ['updrs3_score','SBR_t','moca']:
        for mn in ['M0','M1','M2']:
            x=pm[(pm.subset==sub)&(pm.outcome==y)&(pm.model==mn)]
            if len(x)!=2: continue
            se=(x.ci_hi-x.ci_lo)/3.92; w=1/se**2; b=(w*x.beta_per_SD).sum()/w.sum(); s=np.sqrt(1/w.sum()); Q=(w*(x.beta_per_SD-b)**2).sum()
            pr.append(dict(subset=sub,outcome=y,model=mn,pooled_per_SD=b,ci_lo=b-1.96*s,ci_hi=b+1.96*s,p=2*stats.norm.sf(abs(b/s)),Q_p=1-stats.chi2.cdf(Q,1),
                           disc_p=x[x.cohort=='Discovery'].p.iloc[0],rep_p=x[x.cohort=='Replication'].p.iloc[0]))
prd=pd.DataFrame(pr); prd.to_csv(D+"/r4C_pooled.csv",index=False)
for _,r in prd[prd.subset=='SAA-positive prodromal'].iterrows():
    P(f"  合并 {r.subset} {r.outcome:12s} {r.model}: β/SD={r.pooled_per_SD:+.4f} [{r.ci_lo:+.3f},{r.ci_hi:+.3f}] P={r.p:.3f} | 发现 P={r.disc_p:.3f}, 复现 P={r.rep_p:.3f}, 异质性 P={r.Q_p:.2f}")
P("done"); LOG.close()
