# -*- coding: utf-8 -*-
"""Replication set: genetic participants (LRRK2, GBA1) and within-person effect of dopaminergic treatment on CSF DDC.
Output: derived/p314_gen_*.csv (Fig. 4e, Table S16).
"""
import config as cfg
import sys, warnings
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc
import statsmodels.formula.api as smf
from scipy import stats
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
B=cfg.P; D=cfg.D; OUT=cfg.RS
LOG=open(OUT+"/p314_genetic_log.txt","w",encoding="utf-8")
def P(*a):
    s=" ".join(str(x) for x in a); print(s); LOG.write(s+"\n"); LOG.flush()
def H(t): P("\n"+"="*78+"\n"+t+"\n"+"="*78)

GEN=['LRRK2','GBA','LRRK2 + GBA','SNCA','PRKN','PINK1','PRKN + RBD']
clin=pd.read_csv(D+"/analytic_clinical.csv").set_index('PATNO')
cur=pd.read_excel(cfg.CURATED,sheet_name='20260511')
cur=cur.drop_duplicates(['PATNO','EVENT_ID']).set_index(['PATNO','EVENT_ID'])
SBRc=['MIA_CAUDATE_L','MIA_CAUDATE_R','MIA_PUTAMEN_L','MIA_PUTAMEN_R']

def ddc_long(f,proj):
    cols=['PATNO','EVENT_ID','SampleType','SampleQC','Assay','NPX','PCNormalizedNPX','LAB_QC_FLAG','PlateID']
    t=pq.read_table(f,columns=cols); t=t.filter(pc.and_(pc.equal(t['Assay'],'DDC'),pc.equal(t['SampleType'],'SAMPLE'))).to_pandas()
    t=t[(t.SampleQC=='PASS')&t.PATNO.notna()]; t=t[~t.LAB_QC_FLAG.fillna('').astype(str).str.contains('6|7')]
    t['PATNO']=t.PATNO.astype(int)
    t=t.groupby(['PATNO','EVENT_ID']).agg(DDC=('NPX','mean'),DDC_pc=('PCNormalizedNPX','mean'),Plate=('PlateID','first')).reset_index()
    for c in ['YEAR','LEDD','updrs3_score']+SBRc:
        t[c]=[cur[c].get((p,e),np.nan) if c in cur.columns else np.nan for p,e in zip(t.PATNO,t.EVENT_ID)]
    t['LEDD']=t.LEDD.fillna(0); t['treated']=(t.LEDD>0).astype(int); t['logLEDD']=np.log1p(t.LEDD)
    for c in ['group0','subgroup','COHORT','age','SEX','CSFSAA']: t[c]=t.PATNO.map(clin[c])
    t['gen']=t.subgroup.isin(GEN); t['project']=proj
    return t
L314=ddc_long(B+"/ppmi_proj314_csf_screened_extended_npx_20260116.parquet",'314')
L277=ddc_long(B+"/PPMI_Project_277_CSF_Screened_NPX_20260130.parquet",'277')

# ---------------------------------------------------------------- A) 个体内治疗效应
H("A) 个体内: 基线未用药, 随访开始多巴胺能治疗后 CSF DDC 的变化 (两项目分别估计)")
arows=[]
for lab,Lx in [('Project 277',L277),('Project 314',L314)]:
    Lx=Lx.dropna(subset=['YEAR']).copy()
    bl=Lx[Lx.EVENT_ID=='BL'].set_index('PATNO')
    ok=bl.index[(bl.LEDD==0)]; Lx=Lx[Lx.PATNO.isin(ok)]
    cnt=Lx.groupby('PATNO').size(); Lx=Lx[Lx.PATNO.isin(cnt[cnt>=2].index)]
    pdset=Lx[Lx.COHORT==1]                                        # 已诊断 PD (散发+遗传)
    ctrl=Lx[(Lx.COHORT!=1)&(Lx.treated==0)]
    sw=pdset.groupby('PATNO').treated.max(); nsw=int(sw.sum())
    samep=(pdset.groupby('PATNO').Plate.nunique()==1).mean()
    P(f"  {lab}: 有基线(未用药)+≥1 随访的 PD {pdset.PATNO.nunique()} 人, 其中随访时已用药 {nsw} 人; 样本 {len(pdset)}; 同一人全部样本同板比例 {samep:.0%}")
    for out_,dd in [('NPX','DDC'),('PCNormalizedNPX','DDC_pc')]:
        for expo in ['treated','logLEDD']:
            try:
                m=smf.mixedlm(f"{dd} ~ {expo} + YEAR",pdset,groups=pdset.PATNO).fit(method='lbfgs')
                ci=m.conf_int().loc[expo].values
                # 个体内去均值 (固定效应) 估计
                dm=pdset[[dd,expo,'YEAR','PATNO']].dropna().copy()
                for c in [dd,expo,'YEAR']: dm[c]=dm[c]-dm.groupby('PATNO')[c].transform('mean')
                fe=smf.ols(f"{dd} ~ {expo} + YEAR - 1",dm).fit(cov_type='cluster',cov_kwds={'groups':pdset.loc[dm.index,'PATNO']})
                arows.append(dict(project=lab,measure=out_,exposure=expo,n_pd=pdset.PATNO.nunique(),n_switch=nsw,beta_mixed=m.params[expo],ci_lo=ci[0],ci_hi=ci[1],p_mixed=m.pvalues[expo],
                                  beta_fe=fe.params[expo],p_fe=fe.pvalues[expo],year_beta=m.params['YEAR'],year_p=m.pvalues['YEAR']))
                P(f"    [{out_:15s}] {expo:8s}: 混合模型 β={m.params[expo]:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] P={m.pvalues[expo]:.1e} | 个体内FE β={fe.params[expo]:+.3f} P={fe.pvalues[expo]:.1e} | 时间 β={m.params['YEAR']:+.3f}/年 P={m.pvalues['YEAR']:.2f}")
            except Exception as ex: P('    失败',expo,repr(ex)[:100])
    # 未用药对照: 前驱/HC 的随访时间趋势 (检验时间/储存漂移)
    ctrl=Lx[(Lx.COHORT!=1)&(Lx.treated==0)]
    if ctrl.PATNO.nunique()>=20:
        m=smf.mixedlm("DDC ~ YEAR",ctrl,groups=ctrl.PATNO).fit(method='lbfgs')
        P(f"    对照 (未用药非 PD, n={ctrl.PATNO.nunique()} 人, {len(ctrl)} 样本): 时间 β={m.params.YEAR:+.3f}/年 P={m.pvalues.YEAR:.2f}")
        arows.append(dict(project=lab,measure='NPX',exposure='YEAR (untreated non-PD controls)',n_pd=ctrl.PATNO.nunique(),beta_mixed=m.params.YEAR,p_mixed=m.pvalues.YEAR))
    # 以未用药非 PD 对照的时间漂移为基准 (PD + 对照合并, 共用时间斜率; treated 效应 = 用药后超出对照漂移的部分, 仍含疾病进展)
    both_=pd.concat([pdset,ctrl]) if ctrl.PATNO.nunique()>=20 else pdset
    m=smf.mixedlm("DDC ~ treated + YEAR",both_,groups=both_.PATNO).fit(method='lbfgs'); ci=m.conf_int().loc['treated'].values
    fu=pdset[pdset.treated==1].YEAR.median()
    P(f"    PD+未用药对照共用时间斜率: treated β={m.params.treated:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] P={m.pvalues.treated:.1e} (时间 β={m.params.YEAR:+.3f}/年); 用药后样本中位时间 {fu:.1f} 年")
    arows.append(dict(project=lab,measure='NPX',exposure='treated (common time slope with untreated controls)',n_pd=pdset.PATNO.nunique(),beta_mixed=m.params.treated,ci_lo=ci[0],ci_hi=ci[1],p_mixed=m.pvalues.treated,year_beta=m.params.YEAR,median_fu_treated=fu))
    # 仅转换者: 用药前 vs 用药后配对
    sw_ids=sw[sw==1].index; pp=pdset[pdset.PATNO.isin(sw_ids)]
    pre=pp[pp.treated==0].groupby('PATNO').DDC.mean(); post=pp[pp.treated==1].groupby('PATNO').DDC.mean(); both=pre.index.intersection(post.index)
    if len(both)>=5:
        d=post[both]-pre[both]; w=stats.wilcoxon(d)
        P(f"    转换者配对 (n={len(both)}): 用药后-用药前 DDC 中位差 {d.median():+.3f} NPX (均值 {d.mean():+.3f}), 上升比例 {(d>0).mean():.0%}, Wilcoxon P={w.pvalue:.1e}")
        arows.append(dict(project=lab,measure='NPX',exposure='paired post-pre (switchers)',n_switch=len(both),beta_mixed=d.mean(),p_mixed=w.pvalue))
pd.DataFrame(arows).to_csv(D+"/p314_gen_treatment_within.csv",index=False)

# ---------------------------------------------------------------- B) 遗传携带者横断面
H("B) 遗传携带者 (314 基线, 733 之外): 校正治疗后")
base=pd.read_csv(D+"/p314_rep_baseline_analytic.csv")
base['SBR']=base[SBRc].mean(1) if set(SBRc)<=set(base.columns) else base.PATNO.map(lambda p: np.nanmean([cur[c].get((p,'BL'),np.nan) for c in SBRc]))
base['SAA']=base.CSFSAA.where(base.CSFSAA.isin([0,1])); base['logLEDD']=np.log1p(base.LEDD.fillna(0)); base['treated']=(base.LEDD.fillna(0)>0).astype(int)
pcm=L314[L314.EVENT_ID=='BL'].set_index('PATNO').DDC_pc; base['DDC_pc']=base.PATNO.map(pcm)
g=base[base.subgroup.isin(['LRRK2','GBA'])].copy(); g['status']=np.where(g.COHORT==1,'PD','NMC')
g['cat']=g.subgroup+'-'+g.status+np.where(g.SAA==1,'-SAA+',np.where(g.SAA==0,'-SAA-','-SAA?'))
spor=base[~base.subgroup.isin(GEN)]
hc=spor[spor.group0=='HC'].assign(cat='HC'); sp=spor[spor.group0=='PD'].assign(cat='Sporadic PD')
gg=pd.concat([hc,sp,g]); gg=gg[gg.cat.map(gg.cat.value_counts())>=10].copy()
gg['cat']=pd.Categorical(gg.cat,['HC']+sorted(c for c in gg.cat.unique() if c!='HC'))
gg['duration']=gg.duration.fillna(0)
P("  用药比例 / 病程(月) 中位: "+"; ".join(f"{c}: {gg[gg.cat==c].treated.mean():.0%} / {gg[gg.cat==c].duration.median():.0f}" for c in gg.cat.cat.categories))
brows=[]
specs=[('(a) 年龄+性别','DDC ~ C(cat) + age + SEX',gg),('(b) + log(1+LEDD)','DDC ~ C(cat) + age + SEX + logLEDD',gg),
       ('(c) + log(1+LEDD) + 病程 + 板','DDC ~ C(cat) + age + SEX + logLEDD + duration + C(Plate)',gg),
       ('(d) PCNormalizedNPX + LEDD','DDC_pc ~ C(cat) + age + SEX + logLEDD',gg),('(e) 仅未用药者','DDC ~ C(cat) + age + SEX',gg[gg.treated==0])]
for lab,f,df in specs:
    cs=df.cat.astype(str); df=df[(cs.map(cs.value_counts())>=8).values]; df=df.assign(cat=df.cat.cat.remove_unused_categories())
    m=smf.ols(f,df).fit(); terms=[k for k in m.params.index if k.startswith('C(cat)')]
    P(f"  {lab} (n={int(m.nobs)})"+(f", LEDD β={m.params.logLEDD:+.3f}/log单位 P={m.pvalues.logLEDD:.1e}" if 'logLEDD' in m.params else ''))
    for k in terms:
        c=k.split('T.')[1][:-1]; n=int((df.cat==c).sum())
        brows.append(dict(model=lab,contrast=f'{c} vs HC',n=n,beta=m.params[k],ci_lo=m.conf_int().loc[k,0],ci_hi=m.conf_int().loc[k,1],p=m.pvalues[k]))
        P(f"     {c:16s} n={n:3d} β={m.params[k]:+.3f} [{m.conf_int().loc[k,0]:+.3f},{m.conf_int().loc[k,1]:+.3f}] P={m.pvalues[k]:.1e}")
pd.DataFrame(brows).to_csv(D+"/p314_gen_crosssectional.csv",index=False)
# LRRK2-PD 内: SAA 与 SBR 谁解释 DDC (校正治疗)
for gene in ['LRRK2','GBA']:
    x=g[(g.subgroup==gene)&(g.status=='PD')].dropna(subset=['SAA','SBR','updrs3_score']).copy()
    x['zSBR']=(x.SBR-x.SBR.mean())/x.SBR.std(); x['zU3']=(x.updrs3_score-x.updrs3_score.mean())/x.updrs3_score.std()
    m=smf.ols("DDC ~ SAA + zSBR + zU3 + logLEDD + duration + age + SEX",x).fit()
    P(f"  {gene}-PD 内 (n={int(m.nobs)}): "+" | ".join(f"{k} β={m.params[k]:+.3f} P={m.pvalues[k]:.1e}" for k in ['SAA','zSBR','zU3','logLEDD','duration']))

# ---------------------------------------------------------------- C) NMC
H("C) 未发病携带者 (NMC) DDC 偏高排查")
nm=pd.concat([hc,g[g.status=='NMC']]); nm['grp']=np.where(nm.cat=='HC','HC',nm.subgroup+'-NMC'); nm['grp']=pd.Categorical(nm.grp,['HC','GBA-NMC','LRRK2-NMC'])
ct=pd.crosstab(nm.Plate,nm.grp); P(f"  板分布: HC 分布于 {int((ct.HC>0).sum())} 块板, NMC 分布于 {int(((ct['GBA-NMC']+ct['LRRK2-NMC'])>0).sum())} 块板; 两者共板 {int(((ct.HC>0)&((ct['GBA-NMC']+ct['LRRK2-NMC'])>0)).sum())} 块; χ² P={stats.chi2_contingency(ct)[1]:.1e}")
shared_pl=ct.index[(ct.HC>0)&((ct['GBA-NMC']+ct['LRRK2-NMC'])>0)]
crows=[]
for lab,f,df in [('(a) 年龄+性别','DDC ~ C(grp) + age + SEX',nm),('(b) + 板固定效应','DDC ~ C(grp) + age + SEX + C(Plate)',nm),
                 ('(c) PCNormalizedNPX','DDC_pc ~ C(grp) + age + SEX',nm),('(d) 仅 HC 与 NMC 共板','DDC ~ C(grp) + age + SEX + C(Plate)',nm[nm.Plate.isin(shared_pl)]),
                 ('(e) 仅 SAA 阴性 NMC','DDC ~ C(grp) + age + SEX',nm[(nm.grp=='HC')|(nm.SAA==0)]),
                 ('(f) SAA- 且 SBR 正常(≥HC 第 10 百分位)','DDC ~ C(grp) + age + SEX',nm[(nm.grp=='HC')|((nm.SAA==0)&(nm.SBR>=hc.SBR.quantile(0.10)))])]:
    m=smf.ols(f,df).fit()
    s=" | ".join(f"{gr} β={m.params[f'C(grp)[T.{gr}]']:+.3f} P={m.pvalues[f'C(grp)[T.{gr}]']:.1e}" for gr in ['GBA-NMC','LRRK2-NMC'] if f'C(grp)[T.{gr}]' in m.params)
    P(f"  {lab:38s} n={int(m.nobs):4d} | {s}")
    for gr in ['GBA-NMC','LRRK2-NMC']:
        k=f'C(grp)[T.{gr}]'
        if k in m.params: crows.append(dict(model=lab,contrast=f'{gr} vs HC',beta=m.params[k],p=m.pvalues[k],n=int(m.nobs)))
nmc=g[g.status=='NMC'].copy()
for c in ['SBR','upsit','rem','SAA','age']:
    x=nmc.dropna(subset=[c]);
    if len(x)>30:
        m=smf.ols(f"DDC ~ {c} + age + SEX + C(subgroup)" if c!='age' else "DDC ~ age + SEX + C(subgroup)",x).fit()
        P(f"  NMC 内 DDC ~ {c:6s} (n={int(m.nobs)}): β={m.params[c]:+.4f} P={m.pvalues[c]:.1e}")
        crows.append(dict(model='within NMC',contrast=c,beta=m.params[c],p=m.pvalues[c],n=int(m.nobs)))
pd.DataFrame(crows).to_csv(D+"/p314_gen_nmc.csv",index=False)
P("\ndone"); LOG.close()
