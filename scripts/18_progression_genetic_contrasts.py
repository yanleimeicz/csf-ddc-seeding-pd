# -*- coding: utf-8 -*-
"""Progression models (confounding structure, treatment start), direct genetic contrasts and Tobit standard errors.
Output: derived/r3D_*.csv (Fig. 4, Tables S11, S16, S21).
"""
import config as cfg
import sys, warnings, json
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc
import statsmodels.formula.api as smf
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.tools.numdiff import approx_hess
from scipy import stats, optimize
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
B=cfg.P; D=cfg.D; OUT=cfg.RS
LOG=open(OUT+"/r3D_log.txt","w",encoding="utf-8")
def P(*a):
    s=" ".join(str(x) for x in a); print(s); LOG.write(s+"\n"); LOG.flush()
def H(t): P("\n"+"="*78+"\n"+t+"\n"+"="*78)
GEN=['LRRK2','GBA','LRRK2 + GBA','SNCA','PRKN','PINK1','PRKN + RBD']
SBRc=['MIA_CAUDATE_L','MIA_CAUDATE_R','MIA_PUTAMEN_L','MIA_PUTAMEN_R']
clin=pd.read_csv(D+"/analytic_clinical.csv").set_index('PATNO')
cur=pd.read_excel(cfg.CURATED,sheet_name='20260511',usecols=['PATNO','EVENT_ID','YEAR','LEDD','updrs3_score']+SBRc)
cur=cur.drop_duplicates(['PATNO','EVENT_ID']); cur['SBR_t']=cur[SBRc].mean(1)
bl=cur[cur.EVENT_ID=='BL'].set_index('PATNO')
wide=pd.read_parquet(D+"/csf_npx_wide.parquet").set_index('PATNO'); wide.index=wide.index.astype(int)
ann=pd.read_csv(D+"/protein_annotation.csv").set_index('OlinkID'); jD=[c for c in wide.columns if ann.Assay.get(c)=='DDC'][0]
disc=clin.reindex(wide.index).copy(); disc['DDC']=wide[jD].fillna(wide[jD].median()).values
rep=pd.read_csv(D+"/p314_rep_baseline_analytic.csv").set_index('PATNO'); spor=rep[~rep.subgroup.isin(GEN)&rep.group0.isin(['HC','Hyposmia','iRBD','PD'])].copy()
for df in (disc,spor):
    df['SAA']=df.CSFSAA.where(df.CSFSAA.isin([0,1])); df['bl_U3']=bl.updrs3_score.reindex(df.index); df['bl_SBR']=bl.SBR_t.reindex(df.index)

H("1. DDC × time 模型的混杂结构 (两队列, 同一组模型)")
rows=[]
def lg_data(base):
    b=base[base.group0.isin(['iRBD','Hyposmia','PD'])][['DDC','age','SEX','group0','SAA','bl_U3','bl_SBR']]
    L=cur[cur.PATNO.isin(b.index)][['PATNO','EVENT_ID','YEAR','LEDD','updrs3_score','SBR_t']].merge(b,left_on='PATNO',right_index=True)
    L.loc[(L.EVENT_ID=='BL')&L.LEDD.isna(),'LEDD']=0; L=L.dropna(subset=['YEAR','DDC','age','SEX'])
    L['zDDC']=(L.DDC-b.DDC.mean())/b.DDC.std(); return L
specs={'M0 primary (DDC×time + age + sex)':'{y} ~ DDC*YEAR + age + SEX',
       'M1 + group×time, SAA×time, age×time, sex×time':'{y} ~ DDC*YEAR + C(group0)*YEAR + SAA*YEAR + age*YEAR + SEX*YEAR',
       'M2 = M1 + baseline UPDRS-III×time + baseline SBR×time':'{y} ~ DDC*YEAR + C(group0)*YEAR + SAA*YEAR + age*YEAR + SEX*YEAR + bl_U3*YEAR + bl_SBR*YEAR'}
for coh,base in [('Discovery',disc),('Replication',spor)]:
    L=lg_data(base)
    subsets={'All prodromal + PD (all visits)':L,'Untreated visits':L[L.LEDD==0],'Prodromal only':L[L.group0.isin(['iRBD','Hyposmia'])],
             'Prodromal SAA+':L[L.group0.isin(['iRBD','Hyposmia'])&(L.SAA==1)],'PD only':L[L.group0=='PD']}
    for sn,df in subsets.items():
        for y in ['updrs3_score','SBR_t']:
            dd=df.dropna(subset=[y]); cnt=dd.groupby('PATNO').size(); dd=dd[dd.PATNO.isin(cnt[cnt>=2].index)]
            for mn,f in specs.items():
                ff=f.format(y=y)
                if dd.group0.nunique()<2: ff=ff.replace(' + C(group0)*YEAR','')
                if dd.SAA.nunique()<2: ff=ff.replace(' + SAA*YEAR','')
                if y=='SBR_t': ff=ff.replace(' + bl_SBR*YEAR','')
                try:
                    m=smf.mixedlm(ff,dd.dropna(subset=['SAA','bl_U3','bl_SBR']) if 'M2' in mn else dd.dropna(subset=['SAA']) if 'M1' in mn else dd,groups=(dd.dropna(subset=['SAA','bl_U3','bl_SBR']) if 'M2' in mn else dd.dropna(subset=['SAA']) if 'M1' in mn else dd).PATNO,re_formula='~YEAR').fit(method='lbfgs')
                    b=m.params['DDC:YEAR']; ci=m.conf_int().loc['DDC:YEAR']; p=m.pvalues['DDC:YEAR']
                    rows.append(dict(cohort=coh,subset=sn,outcome=y,model=mn,n=dd.PATNO.nunique(),beta=b,ci_lo=ci[0],ci_hi=ci[1],p=p,converged=m.converged))
                    if mn.startswith('M0') or y=='updrs3_score' or sn.startswith('Prod'):
                        P(f"  {coh:11s} {sn:32s} {y:12s} {mn[:3]} n={dd.PATNO.nunique():4d} DDC×time β={b:+.4f} [{ci[0]:+.3f},{ci[1]:+.3f}] P={p:.3f} conv={m.converged}")
                except Exception as e: P('  fail',coh,sn,y,mn[:3],repr(e)[:80])
pd.DataFrame(rows).to_csv(D+"/r3D_progression_models.csv",index=False)

H("2. DDC 是否预测治疗启动时间 (基线未治疗的 PD; Cox)")
crow=[]
for coh,base in [('Discovery',disc),('Replication',spor)]:
    pdb=base[(base.group0=='PD')&(base.LEDD.fillna(0)==0)].dropna(subset=['DDC','age','SEX','bl_U3'])
    v=cur[cur.PATNO.isin(pdb.index)].dropna(subset=['YEAR']); t1=v[v.LEDD>0].groupby('PATNO').YEAR.min(); tl=v.groupby('PATNO').YEAR.max()
    pdb=pdb.assign(event=pdb.index.isin(t1.index).astype(int),time=np.where(pdb.index.isin(t1.index),t1.reindex(pdb.index),tl.reindex(pdb.index)))
    pdb=pdb[pdb.time>0]; pdb['zDDC']=(pdb.DDC-pdb.DDC.mean())/pdb.DDC.std()
    m=PHReg.from_formula("time ~ zDDC + age + SEX + bl_U3",pdb,status=pdb.event.values).fit()
    hr=np.exp(m.params[0]); ci=np.exp(m.conf_int()[0]); crow.append(dict(cohort=coh,n=len(pdb),events=int(pdb.event.sum()),HR_per_SD=hr,ci_lo=ci[0],ci_hi=ci[1],p=m.pvalues[0],median_time=np.median(pdb.time)))
    P(f"  {coh:11s} n={len(pdb)} 事件(开始治疗)={int(pdb.event.sum())}: DDC 每 SD HR={hr:.2f} [{ci[0]:.2f},{ci[1]:.2f}] P={m.pvalues[0]:.3f} (校正年龄、性别、基线 UPDRS-III)")
pd.DataFrame(crow).to_csv(D+"/r3D_time_to_treatment.csv",index=False)

H("3. 遗传: LRRK2-PD SAA+ vs SAA− 直接对比 (R1-W6, R2-W4, DA-C5)")
g=rep[rep.subgroup.isin(['LRRK2','GBA'])&(rep.COHORT==1)].copy(); g['SAA']=g.CSFSAA.where(g.CSFSAA.isin([0,1])); g['logLEDD']=np.log1p(g.LEDD.fillna(0)); g['treated']=(g.LEDD.fillna(0)>0).astype(int)
grow=[]
for gene in ['LRRK2','GBA']:
    x=g[(g.subgroup==gene)].dropna(subset=['SAA','DDC','age','SEX'])
    P(f"  {gene}-PD: SAA+ {int((x.SAA==1).sum())} (未治疗 {int(((x.SAA==1)&(x.treated==0)).sum())}), SAA− {int((x.SAA==0).sum())} (未治疗 {int(((x.SAA==0)&(x.treated==0)).sum())})")
    for lab,f,dd in [('SAA+ vs SAA−, age+sex','DDC ~ SAA + age + SEX',x),('SAA+ vs SAA−, + log(1+LEDD) + duration','DDC ~ SAA + logLEDD + duration + age + SEX',x.assign(duration=x.duration.fillna(0))),
                     ('SAA × log(1+LEDD) interaction','DDC ~ SAA*logLEDD + age + SEX',x),('untreated only: SAA+ vs SAA−','DDC ~ SAA + age + SEX',x[x.treated==0])]:
        if dd.SAA.nunique()<2 or len(dd)<8: P(f"   {lab}: 样本不足"); continue
        m=smf.ols(f,dd).fit(); k='SAA:logLEDD' if 'interaction' in lab else 'SAA'; ci=m.conf_int().loc[k]
        grow.append(dict(gene=gene,analysis=lab,n=int(m.nobs),term=k,beta=m.params[k],ci_lo=ci[0],ci_hi=ci[1],p=m.pvalues[k]))
        P(f"   {lab:42s} n={int(m.nobs):3d} {k} β={m.params[k]:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] P={m.pvalues[k]:.3f}")
# SAA− LRRK2-PD vs HC 等效界 (校正 LEDD 与否)
hc=spor[spor.group0=='HC'].assign(cat='HC'); l0=g[(g.subgroup=='LRRK2')&(g.SAA==0)].assign(cat='LRRK2_SAAneg')
dd=pd.concat([hc,l0]); dd['logLEDD']=np.log1p(dd.LEDD.fillna(0)); dd['isL']=(dd.cat!='HC').astype(int)
for lab,f in [('age+sex','DDC ~ isL + age + SEX'),('+ log(1+LEDD)','DDC ~ isL + logLEDD + age + SEX')]:
    m=smf.ols(f,dd).fit(); ci=m.conf_int().loc['isL']; grow.append(dict(gene='LRRK2',analysis=f'SAA− LRRK2-PD vs HC, {lab}',n=int(m.nobs),term='isL',beta=m.params['isL'],ci_lo=ci[0],ci_hi=ci[1],p=m.pvalues['isL']))
    P(f"   SAA− LRRK2-PD vs HC ({lab}): β={m.params['isL']:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] P={m.pvalues['isL']:.3f}  (散发 PD vs HC = 0.78)")
pd.DataFrame(grow).to_csv(D+"/r3D_genetic_contrasts.csv",index=False)

H("4. Tobit (数值 Hessian 标准误), 两队列 H1")
def tobit(y,X,lod):
    cens=y<lod; yy=np.where(cens,lod,y)
    def nll(th):
        b=th[:-1]; s=np.exp(th[-1]); mu=X@b
        return -(stats.norm.logpdf(yy[~cens],mu[~cens],s).sum()+stats.norm.logcdf((lod[cens]-mu[cens])/s).sum())
    b0=np.linalg.lstsq(X,yy,rcond=None)[0]; r=optimize.minimize(nll,np.r_[b0,np.log(yy.std())],method='BFGS')
    Hs=approx_hess(r.x,nll); se=np.sqrt(np.diag(np.linalg.inv(Hs))); return r.x,se,cens.mean()
trow=[]
for coh,f,base in [('Discovery',B+"/PPMI_Project_277_CSF_Screened_NPX_20260130.parquet",disc),('Replication',B+"/ppmi_proj314_csf_screened_extended_npx_20260116.parquet",spor)]:
    t=pq.read_table(f,columns=['PATNO','EVENT_ID','SampleType','SampleQC','Assay','NPX','LOD'])
    t=t.filter(pc.and_(pc.equal(t['Assay'],'DDC'),pc.equal(t['SampleType'],'SAMPLE'))).to_pandas()
    t=t[(t.SampleQC=='PASS')&(t.EVENT_ID=='BL')&t.PATNO.notna()]; t['PATNO']=t.PATNO.astype(int); t=t.groupby('PATNO').agg(NPX=('NPX','mean'),LOD=('LOD','mean'))
    s=base.join(t,how='inner',rsuffix='_r').dropna(subset=['NPX','LOD','age','SEX']); s=s[s.group0.isin(['HC','Hyposmia','iRBD','PD'])]
    G=['Hyposmia','iRBD','PD']; X=np.column_stack([np.ones(len(s))]+[(s.group0==gg).astype(float) for gg in G]+[s.age,s.SEX])
    th,se,cf=tobit(s.NPX.values,X,s.LOD.values); ols=smf.ols("NPX ~ C(group0, Treatment('HC')) + age + SEX",s).fit()
    for i,gg in enumerate(G):
        p=2*stats.norm.sf(abs(th[1+i]/se[1+i])); ob=ols.params[f"C(group0, Treatment('HC'))[T.{gg}]"]
        trow.append(dict(cohort=coh,contrast=f'{gg} vs HC',tobit_beta=th[1+i],ci_lo=th[1+i]-1.96*se[1+i],ci_hi=th[1+i]+1.96*se[1+i],p=p,ols_beta=ob,censored_frac=cf))
        P(f"  {coh:11s} {gg:8s} vs HC: Tobit β={th[1+i]:+.3f} [{th[1+i]-1.96*se[1+i]:+.3f},{th[1+i]+1.96*se[1+i]:+.3f}] P={p:.1e} | OLS β={ob:+.3f} | 删失 {100*cf:.1f}%")
pd.DataFrame(trow).to_csv(D+"/r3D_tobit.csv",index=False)
H("5. 两队列合并 (固定效应, 逆方差): DDC × time")
pm=pd.DataFrame(rows); prow=[]
for sub in ['Prodromal SAA+','Untreated visits','Prodromal only']:
    for y in ['updrs3_score','SBR_t']:
        for mn in pm.model.unique():
            x=pm[(pm.subset==sub)&(pm.outcome==y)&(pm.model==mn)]
            if len(x)!=2: continue
            se=(x.ci_hi-x.ci_lo)/3.92; w=1/se**2; b=(w*x.beta).sum()/w.sum(); s_=np.sqrt(1/w.sum()); Q=(w*(x.beta-b)**2).sum()
            prow.append(dict(subset=sub,outcome=y,model=mn,pooled=b,ci_lo=b-1.96*s_,ci_hi=b+1.96*s_,p=2*stats.norm.sf(abs(b/s_)),Q_p=1-stats.chi2.cdf(Q,1)))
            P(f"  {sub:18s} {y:12s} {mn[:3]}: 合并 β={b:+.4f} [{b-1.96*s_:+.4f},{b+1.96*s_:+.4f}] P={2*stats.norm.sf(abs(b/s_)):.1e} | 异质性 P={1-stats.chi2.cdf(Q,1):.2f}")
pd.DataFrame(prow).to_csv(D+"/r3D_progression_pooled.csv",index=False)
P("\ndone"); LOG.close()
