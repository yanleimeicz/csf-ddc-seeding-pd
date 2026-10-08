# -*- coding: utf-8 -*-
"""Replication set (PPMI Project 314, CSF Olink Explore HT, participants not in the discovery set): hypotheses H1 to H5
of the written replication plan (Supplementary Note 1) with sensitivity analyses.
Pre-processing: SampleQC PASS; measurements failing or flagged in PPMI sample-level cross-validation removed
(LAB_QC_FLAG 6 and 7); values > 5 MAD from the protein median removed.
Output: derived/p314_rep_*.csv (Table S11).
"""
import config as cfg
import os, sys, json, warnings
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
B=cfg.P; D=cfg.D; OUT=cfg.RS
os.makedirs(OUT,exist_ok=True)
NJ=30; NBOOT=2000; rng0=20261003
LOG=open(OUT+"/p314_replication_log.txt","w",encoding="utf-8")
def P(*a):
    s=" ".join(str(x) for x in a); print(s); LOG.write(s+"\n"); LOG.flush()
def H(t): P("\n"+"="*78+"\n"+t+"\n"+"="*78)

# ---------------------------------------------------------------- 数据
t=pq.read_table(B+"/ppmi_proj314_csf_screened_extended_npx_20260116.parquet",
    columns=['PATNO','EVENT_ID','SampleType','SampleQC','Assay','AssayType','NPX','PCNormalizedNPX','LOD','PCNormalizedLOD','LAB_QC_FLAG','PlateID'])
t=t.filter(pc.and_(pc.equal(t['Assay'],'DDC'),pc.equal(t['SampleType'],'SAMPLE'))).to_pandas()
t=t[(t.SampleQC=='PASS')&t.PATNO.notna()].copy(); t['PATNO']=t.PATNO.astype(int)
n0=len(t); flag=t.LAB_QC_FLAG.fillna('').astype(str).str.contains('6|7'); t=t[~flag]
med=t.NPX.median(); mad=1.4826*np.median(np.abs(t.NPX-med)); out5=(t.NPX-med).abs()>5*mad; t=t[~out5]
H("预处理 (DDC, Project 314 全部样本)")
P(f"SampleQC PASS 样本 {n0}; 剔除 LAB_QC_FLAG 6/7: {int(flag.sum())}; 剔除 >5 MAD 离群: {int(out5.sum())}")
t['belowLOD']=t.NPX<t.LOD
t['NPX_LODsub']=np.maximum(t.NPX,t.LOD)

cohort733=set(pd.read_parquet(D+"/csf_npx_wide.parquet",columns=['PATNO'])['PATNO'].astype(int))
clin=pd.read_csv(D+"/analytic_clinical.csv")
cur=pd.read_excel(cfg.CURATED,sheet_name='20260511')
bl=cur[cur.EVENT_ID=='BL'].drop_duplicates('PATNO').set_index('PATNO')
SBRc=['MIA_CAUDATE_L','MIA_CAUDATE_R','MIA_PUTAMEN_L','MIA_PUTAMEN_R']
for c in SBRc+['hemohi']: clin[c]=clin.PATNO.map(bl[c]) if c in bl.columns else np.nan
clin['SBR']=clin[SBRc].mean(1)
clin['SAA']=clin.CSFSAA.where(clin.CSFSAA.isin([0,1]))
clin['NSD_num']=clin.NSD_STAGE.map(lambda v: 0.0 if str(v).startswith('Not') else (float(str(v)[0]) if str(v)[:1].isdigit() else np.nan))
GEN=['LRRK2','GBA','LRRK2 + GBA','SNCA','PRKN','PINK1','PRKN + RBD']
clin['gen']=clin.subgroup.isin(GEN)

b=t[t.EVENT_ID=='BL'].groupby('PATNO').agg(DDC=('NPX','mean'),DDC_pc=('PCNormalizedNPX','mean'),DDC_lodsub=('NPX_LODsub','mean'),
        belowLOD=('belowLOD','max'),Plate=('PlateID','first')).reset_index()
b=b[~b.PATNO.isin(cohort733)].merge(clin,on='PATNO',how='left')
b['LEDD']=b.LEDD.fillna(0)
spor=b[~b.gen & b.group0.isin(['HC','Hyposmia','iRBD','PD'])].copy()
spor['grp']=pd.Categorical(spor.group0,['HC','Hyposmia','iRBD','PD'])
P(f"733 之外基线参与者 {len(b)}; 散发分析集 {len(spor)}: "+str(spor.group0.value_counts().to_dict()))
P(f"散发 PD 中 LEDD>0: {(spor[spor.group0=='PD'].LEDD>0).sum()}; 遗传 {int(b.gen.sum())}")

# 板构成
ct=pd.crosstab(spor.Plate,spor.group0); chi=stats.chi2_contingency(ct)
P(f"板 × 组别 列联表 χ²={chi[0]:.1f}, df={chi[2]}, P={chi[1]:.1e} (板数 {ct.shape[0]}; 每板 PD 比例 {ct.PD.div(ct.sum(1)).min():.2f}-{ct.PD.div(ct.sum(1)).max():.2f})")
P("低于 LOD 比例(%) 按组: "+str((100*spor.groupby('group0').belowLOD.mean()).round(1).to_dict()))

# ---------------------------------------------------------------- 工具
def boot_auc(y,x,seed):
    r=np.random.default_rng(seed); i1=np.where(y==1)[0]; i0=np.where(y==0)[0]
    return roc_auc_score(np.r_[np.ones(len(i1)),np.zeros(len(i0))],np.r_[x[r.choice(i1,len(i1))],x[r.choice(i0,len(i0))]])
def auc_ci(y,x):
    y=np.asarray(y); x=np.asarray(x); a=roc_auc_score(y,x)
    bs=Parallel(n_jobs=NJ)(delayed(boot_auc)(y,x,rng0+k) for k in range(NBOOT))
    return a,*np.percentile(bs,[2.5,97.5])
VARS={'DDC':'(a) NPX, 年龄+性别','DDC_plate':'(b) NPX, +板固定效应','DDC_pc':'(c) PCNormalizedNPX','DDC_above':'(d) 仅高于 LOD','DDC_lodsub':'(e) <LOD 置为 LOD'}
def variants(df):
    """返回 [(标签, 数据, 结局列, 额外协变量)]"""
    return [(VARS['DDC'],df,'DDC',''),(VARS['DDC_plate'],df,'DDC',' + C(Plate)'),(VARS['DDC_pc'],df,'DDC_pc',''),
            (VARS['DDC_above'],df[~df.belowLOD],'DDC',''),(VARS['DDC_lodsub'],df,'DDC_lodsub','')]
rows=[]
def rec(h,label,model_lab,m,term):
    ci=m.conf_int().loc[term].values
    rows.append(dict(hypothesis=h,analysis=label,model=model_lab,n=int(m.nobs),term=term,beta=m.params[term],ci_lo=ci[0],ci_hi=ci[1],p=m.pvalues[term]))
    return f"{term} β={m.params[term]:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] P={m.pvalues[term]:.1e}"

# ---------------------------------------------------------------- H1
H("H1  DDC 在各组高于 HC; PD vs HC AUC (发现队列: 0.80, 0.75-0.84)")
for lab,df,y,ex in variants(spor):
    m=smf.ols(f"{y} ~ C(grp) + age + SEX{ex}",df).fit()
    P(f"  {lab:24s} n={int(m.nobs)} | "+" | ".join(rec('H1',lab,g,m,f'C(grp)[T.{g}]') for g in ['Hyposmia','iRBD','PD']))
aucs=[]
for lab,sub in [('PD vs HC',spor[spor.group0.isin(['PD','HC'])]),
                ('PD (LEDD=0) vs HC',spor[spor.group0.isin(['PD','HC'])&~((spor.group0=='PD')&(spor.LEDD>0))]),
                ('SAA+ (any group) vs HC',spor[(spor.SAA==1)|(spor.group0=='HC')&(spor.SAA!=1)]),
                ('SAA+ vs SAA- (all sporadic)',spor.dropna(subset=['SAA'])),
                ('iRBD vs HC',spor[spor.group0.isin(['iRBD','HC'])]),('Hyposmia vs HC',spor[spor.group0.isin(['Hyposmia','HC'])])]:
    if lab.startswith('SAA+ (any'): y=(sub.SAA==1).astype(int)
    elif lab.startswith('SAA+ vs'): y=sub.SAA.astype(int)
    else: y=(sub.group0!='HC').astype(int)
    for col,clab in [('DDC','NPX'),('DDC_pc','PCNormalizedNPX')]:
        a,lo,hi=auc_ci(y.values,sub[col].values)
        aucs.append(dict(contrast=lab,measure=clab,n_pos=int(y.sum()),n_neg=int((1-y).sum()),auc=a,ci_lo=lo,ci_hi=hi))
        P(f"  AUC {lab:30s} [{clab:15s}] {a:.3f} ({lo:.3f}-{hi:.3f})  n+={int(y.sum())} n-={int((1-y).sum())}")
pd.DataFrame(aucs).to_csv(D+"/p314_rep_auc.csv",index=False)
# 可检测性本身也区分组别?
lg=smf.logit("above ~ C(grp) + age + SEX",spor.assign(above=(~spor.belowLOD).astype(int))).fit(disp=0)
P("  高于 LOD 的 logistic (ref HC): "+" | ".join(f"{g} OR={np.exp(lg.params[f'C(grp)[T.{g}]']):.2f} P={lg.pvalues[f'C(grp)[T.{g}]']:.1e}" for g in ['Hyposmia','iRBD','PD']))

# ---------------------------------------------------------------- H2-H4
H("H2-H4  分期成分分解 (散发, 基线未用药); 发现队列: SAA 独立, 运动独立, SBR 不独立 (P=0.28), NSD 无残余 (ΔR² 0.003)")
z=lambda s:(s-s.mean())/s.std()
def prep(df):
    e=df.dropna(subset=['SAA','SBR','updrs3_score','age','SEX','NSD_num']).copy()
    e['zSBR']=z(e.SBR); e['zU3']=z(e.updrs3_score); return e
e=prep(spor[spor.LEDD==0])
for lab,df,y,ex in variants(e):
    full=smf.ols(f"{y} ~ SAA + zSBR + zU3 + age + SEX{ex}",df).fit()
    nsd=smf.ols(f"{y} ~ NSD_num + SAA + zSBR + zU3 + age + SEX{ex}",df).fit()
    ft=nsd.compare_f_test(full)
    rec('H4',lab,'NSD beyond components',nsd,'NSD_num')
    P(f"  {lab:24s} n={int(full.nobs)} | {rec('H2',lab,'all three',full,'SAA')} | {rec('H3',lab,'all three',full,'zSBR')} | {rec('H3',lab,'all three',full,'zU3')} | NSD ΔR²={nsd.rsquared-full.rsquared:.4f} P={ft[1]:.2f}")
es=e[e.SAA==1]
for lab,df,y,ex in variants(es):
    m=smf.ols(f"{y} ~ zSBR + zU3 + age + SEX{ex}",df).fit()
    P(f"  SAA+ 内 {lab:20s} n={int(m.nobs)} | {rec('H3',lab,'within SAA+',m,'zSBR')} | {rec('H3',lab,'within SAA+',m,'zU3')}")
pd.DataFrame(rows).to_csv(D+"/p314_rep_models.csv",index=False)

# ---------------------------------------------------------------- H3 两队列并排
H("H3  运动严重度 vs DAT SBR: 两队列同单位并排 (DDC 与预测变量均在队列内标准化) + 异质性 + bootstrap 独立 R²")
wide=pd.read_parquet(D+"/csf_npx_wide.parquet").set_index('PATNO')
ann=pd.read_csv(D+"/protein_annotation.csv").set_index('OlinkID')
jD=[c for c in wide.columns if ann.Assay.get(c)=='DDC'][0]
disc=clin.set_index('PATNO').reindex(wide.index.astype(int)).reset_index()
disc['DDC']=wide[jD].fillna(wide[jD].median()).values
cohorts={'Discovery (Project 277)':prep(disc),'Replication (Project 314)':e}
def design(df):
    X=np.column_stack([np.ones(len(df)),df.SAA,df.zSBR,df.zU3,df.age,df.SEX]); return X,z(df.DDC).values
def r2(X,y): bb=np.linalg.lstsq(X,y,rcond=None)[0]; r=y-X@bb; return 1-r@r/((y-y.mean())@(y-y.mean()))
def uniq(X,y):
    f=r2(X,y); return f-r2(np.delete(X,1,1),y), f-r2(np.delete(X,2,1),y), f-r2(np.delete(X,3,1),y)   # SAA, SBR, motor
def boot_uniq(X,y,seed):
    i=np.random.default_rng(seed).integers(0,len(y),len(y)); return uniq(X[i],y[i])
h3=[]; coefs={}
for k,df in cohorts.items():
    df=df.copy(); df['zDDC']=z(df.DDC)
    m=smf.ols("zDDC ~ SAA + zSBR + zU3 + age + SEX",df).fit(); coefs[k]=m
    X,y=design(df); u=uniq(X,y)
    bs=np.array(Parallel(n_jobs=NJ)(delayed(boot_uniq)(X,y,rng0+i) for i in range(NBOOT)))
    d=bs[:,2]-bs[:,1]; lo,hi=np.percentile(d,[2.5,97.5]); pmot=(d>0).mean()
    h3.append(dict(cohort=k,n=len(df),b_SAA=m.params.SAA,p_SAA=m.pvalues.SAA,b_zSBR=m.params.zSBR,se_zSBR=m.bse.zSBR,p_zSBR=m.pvalues.zSBR,
                   b_zU3=m.params.zU3,se_zU3=m.bse.zU3,p_zU3=m.pvalues.zU3,uR2_SAA=u[0],uR2_SBR=u[1],uR2_motor=u[2],
                   diff_motor_minus_SBR=u[2]-u[1],diff_ci_lo=lo,diff_ci_hi=hi,boot_prob_motor_gt_SBR=pmot,r_SBR_U3=np.corrcoef(df.SBR,df.updrs3_score)[0,1]))
    P(f"  {k:27s} n={len(df)} | SAA β={m.params.SAA:+.2f} | zSBR β={m.params.zSBR:+.3f} (P={m.pvalues.zSBR:.3f}) | zU3 β={m.params.zU3:+.3f} (P={m.pvalues.zU3:.3f})"
      f" | 独立R² SAA {u[0]:.3f} SBR {u[1]:.4f} 运动 {u[2]:.4f} | 运动-SBR 差 {u[2]-u[1]:+.4f} [{lo:+.4f},{hi:+.4f}], P(运动>SBR)={pmot:.2f} | r(SBR,UPDRS-III)={h3[-1]['r_SBR_U3']:+.2f}")
h3=pd.DataFrame(h3)
for term in ['zSBR','zU3']:
    bb=h3[f'b_{term}'].values; se=h3[f'se_{term}'].values; w=1/se**2
    pooled=(w*bb).sum()/w.sum(); sep=np.sqrt(1/w.sum()); Q=(w*(bb-pooled)**2).sum(); pq_=1-stats.chi2.cdf(Q,1)
    zz=pooled/sep; pp=2*(1-stats.norm.cdf(abs(zz)))
    P(f"  合并(固定效应) {term}: β={pooled:+.3f} [{pooled-1.96*sep:+.3f},{pooled+1.96*sep:+.3f}] P={pp:.1e} | 异质性 Q={Q:.2f} P={pq_:.3f} I²={max(0,(Q-1)/Q)*100 if Q>0 else 0:.0f}%")
    h3[f'pooled_{term}']=pooled; h3[f'pooled_se_{term}']=sep; h3[f'pooled_p_{term}']=pp; h3[f'Q_p_{term}']=pq_
# 合并个体数据 (队列内标准化) + 队列固定效应, 严重度复合 (SBR 反向 + UPDRS)
pool=pd.concat([df.assign(cohort=k,zDDC=z(df.DDC)) for k,df in cohorts.items()])
pool['severity']=(pool.zU3-pool.zSBR)/2
mp=smf.ols("zDDC ~ SAA + zSBR + zU3 + age + SEX + C(cohort)",pool).fit()
mc=smf.ols("zDDC ~ SAA + severity + age + SEX + C(cohort)",pool).fit()
P(f"  个体合并 (n={int(mp.nobs)}): zSBR β={mp.params.zSBR:+.3f} P={mp.pvalues.zSBR:.1e}; zU3 β={mp.params.zU3:+.3f} P={mp.pvalues.zU3:.1e}; "
  f"复合严重度 β={mc.params.severity:+.3f} P={mc.pvalues.severity:.1e}; SAA β={mc.params.SAA:+.3f} P={mc.pvalues.SAA:.1e}")
h3['pooled_ipd_zSBR_p']=mp.pvalues.zSBR; h3['pooled_ipd_zU3_p']=mp.pvalues.zU3; h3['pooled_ipd_severity_b']=mc.params.severity; h3['pooled_ipd_severity_p']=mc.pvalues.severity
h3.to_csv(D+"/p314_rep_H3_two_cohorts.csv",index=False)

# ---------------------------------------------------------------- H5
H("H5  DDC × time 线性混合模型 (发现队列: 仅未用药访视 β=+0.85 P=0.001; 仅前驱 +0.72 P=0.003; 纵向 SBR -0.024 P<0.001; PD 内 null)")
syn_ids=spor.PATNO[spor.group0.isin(['iRBD','Hyposmia','PD'])]
base=spor.set_index('PATNO')[['DDC','age','SEX','group0','SAA']]
lgd=cur[cur.PATNO.isin(syn_ids)][['PATNO','EVENT_ID','YEAR','updrs3_score','updrs3_score_on','LEDD']+SBRc].copy()
lgd['SBR_t']=lgd[SBRc].mean(1); lgd=lgd.merge(base,left_on='PATNO',right_index=True).dropna(subset=['DDC','age','SEX','YEAR']); lgd['LEDD']=lgd.LEDD.fillna(0)
h5=[]
def lme(df,y,label):
    df=df.dropna(subset=[y]); cnt=df.groupby('PATNO').size(); df=df[df.PATNO.isin(cnt[cnt>=2].index)]
    if df.PATNO.nunique()<20: P(f"  {label:58s} 样本不足 (n={df.PATNO.nunique()})"); return
    try:
        md=smf.mixedlm(f'{y} ~ DDC*YEAR + age + SEX',df,groups=df['PATNO'],re_formula='~YEAR').fit(method='lbfgs')
        bb=md.params['DDC:YEAR']; p=md.pvalues['DDC:YEAR']; ci=md.conf_int().loc['DDC:YEAR'].values; fu=df.groupby('PATNO').YEAR.max().median()
        h5.append(dict(analysis=label,outcome=y,n=df.PATNO.nunique(),obs=len(df),median_fu=fu,beta_DDCxtime=bb,ci_lo=ci[0],ci_hi=ci[1],p=p))
        P(f"  {label:58s} n={df.PATNO.nunique():4d} obs={len(df):5d} 中位随访={fu:.1f}y | DDC×time β={bb:+.4f} [{ci[0]:+.3f},{ci[1]:+.3f}] P={p:.3f}")
    except Exception as ex: P(label,'失败',repr(ex)[:120])
lme(lgd,'updrs3_score','(a) MDS-UPDRS-III, 全部前驱+PD')
lme(lgd[lgd.LEDD==0],'updrs3_score','(c) 仅未用药访视')
lme(lgd[lgd.group0.isin(['iRBD','Hyposmia'])],'updrs3_score','(d) 仅前驱')
lme(lgd[lgd.group0.isin(['iRBD','Hyposmia'])&(lgd.SAA==1)],'updrs3_score','(d2) 仅前驱且 SAA+')
lme(lgd[lgd.group0=='PD'],'updrs3_score','(e) 仅 PD')
lme(lgd,'SBR_t','(g) 纵向纹状体 SBR, 全部')
lme(lgd[lgd.group0.isin(['iRBD','Hyposmia'])],'SBR_t','(i) 纵向 SBR, 仅前驱')
pd.DataFrame(h5).to_csv(D+"/p314_rep_H5_progression.csv",index=False)

# ---------------------------------------------------------------- 遗传
H("遗传队列: DDC 跟 α-syn 种子还是多巴胺能缺损? (SAA 阴性 LRRK2-PD 为关键对照)")
g=b[b.subgroup.isin(['LRRK2','GBA'])].copy()
g['status']=np.where(g.COHORT==1,'PD','NMC')   # NMC = 未发病携带者
g['cat']=g.subgroup+'-'+g.status+np.where(g.SAA==1,'-SAA+',np.where(g.SAA==0,'-SAA-','-SAA?'))
hc=spor[spor.group0=='HC'].assign(cat='HC (sporadic)')
sp=spor[spor.group0=='PD'].assign(cat='Sporadic PD')
gg=pd.concat([hc,sp,g])
tab=gg.groupby('cat').agg(n=('DDC','size'),DDC_mean=('DDC','mean'),DDC_sd=('DDC','std'),belowLOD=('belowLOD','mean'),
    SBR=('SBR','mean'),UPDRS3=('updrs3_score','mean'),LEDDpos=('LEDD',lambda s:(s>0).mean()),age=('age','mean')).round(3)
P(tab.to_string())
tab.to_csv(D+"/p314_rep_genetic_groups.csv")
gm=gg[gg.cat.map(gg.cat.value_counts())>=10].copy(); gm['cat']=pd.Categorical(gm.cat,['HC (sporadic)']+sorted(c for c in gm.cat.unique() if c!='HC (sporadic)'))
m=smf.ols("DDC ~ C(cat) + age + SEX",gm).fit(); gen_rows=[]
for c in gm.cat.cat.categories[1:]:
    k=f'C(cat)[T.{c}]'; gen_rows.append(dict(contrast=f'{c} vs HC',beta=m.params[k],p=m.pvalues[k]))
    P(f"  {c:22s} vs HC: β={m.params[k]:+.3f} P={m.pvalues[k]:.1e}")
for gene in ['LRRK2','GBA']:
    x=g[(g.subgroup==gene)&(g.status=='PD')].dropna(subset=['SAA','SBR','updrs3_score']).copy()
    if len(x)>=30:
        x['zSBR']=z(x.SBR); x['zU3']=z(x.updrs3_score); x['treated']=(x.LEDD>0).astype(int)
        mm=smf.ols("DDC ~ SAA + zSBR + zU3 + treated + age + SEX",x).fit()
        P(f"  {gene}-PD 内 (n={int(mm.nobs)}, 用药 {x.treated.mean():.0%}): SAA β={mm.params.SAA:+.3f} P={mm.pvalues.SAA:.1e} | zSBR β={mm.params.zSBR:+.3f} P={mm.pvalues.zSBR:.1e} | zU3 β={mm.params.zU3:+.3f} P={mm.pvalues.zU3:.1e} | treated β={mm.params.treated:+.3f} P={mm.pvalues.treated:.1e}")
        for term in ['SAA','zSBR','zU3','treated']: gen_rows.append(dict(contrast=f'within {gene}-PD: {term}',beta=mm.params[term],p=mm.pvalues[term],n=int(mm.nobs)))
    x=g[g.subgroup==gene].dropna(subset=['SAA'])
    mm=smf.ols("DDC ~ SAA + C(status) + age + SEX",x).fit()
    P(f"  {gene} 全部携带者 (n={int(mm.nobs)}): SAA β={mm.params.SAA:+.3f} P={mm.pvalues.SAA:.1e} | PD vs NMC β={mm.params['C(status)[T.PD]']:+.3f} P={mm.pvalues['C(status)[T.PD]']:.1e}")
    gen_rows.append(dict(contrast=f'{gene} carriers: SAA (adj status)',beta=mm.params.SAA,p=mm.pvalues.SAA,n=int(mm.nobs)))
    gen_rows.append(dict(contrast=f'{gene} carriers: PD vs NMC (adj SAA)',beta=mm.params['C(status)[T.PD]'],p=mm.pvalues['C(status)[T.PD]'],n=int(mm.nobs)))
pd.DataFrame(gen_rows).to_csv(D+"/p314_rep_genetic_models.csv",index=False)
b.to_csv(D+"/p314_rep_baseline_analytic.csv",index=False)
P("\ndone"); LOG.close()
