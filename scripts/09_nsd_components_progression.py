# -*- coding: utf-8 -*-
"""Discovery set: NSD stage decomposed into SAA, dopamine-transporter binding and clinical severity; alternative
body-first definitions; progression sensitivity analyses.
Output: derived/rev2_nsd_*.csv, subtype and progression tables (Tables S8, S14, S18, S20).
"""
import config as cfg
import numpy as np, pandas as pd, warnings, sys, json
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
from scipy import stats
import statsmodels.formula.api as smf, statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
D=cfg.D
wide=pd.read_parquet(D+"/csf_npx_wide.parquet").set_index('PATNO')
ann=pd.read_csv(D+"/protein_annotation.csv").set_index('OlinkID')
clin=pd.read_csv(D+"/analytic_clinical.csv").set_index('PATNO').reindex(wide.index)
genes=ann.reindex(wide.columns)['Assay'].astype(str).values
X=wide.values.astype(float); med=np.nanmedian(X,0); ii=np.where(np.isnan(X)); X[ii]=np.take(med,ii[1])
def nsd_num(v):
    s=str(v); return 0.0 if s.startswith('Not') else (float(s[0]) if s[0].isdigit() else np.nan)
clin['NSD_num']=clin['NSD_STAGE'].map(nsd_num)
jD=int(np.where(genes=='DDC')[0][0]); clin['DDC']=X[:,jD]
clin['SBR']=clin[['PUTAMEN_ASYM']].shape[0] and np.nan
cur=pd.read_excel(cfg.CURATED,sheet_name='20260511')
bl=cur[cur.EVENT_ID=='BL'].set_index('PATNO')
for c in ['MIA_CAUDATE_L','MIA_CAUDATE_R','MIA_PUTAMEN_L','MIA_PUTAMEN_R','updrs3_score','updrs3_score_on','updrs2_score','updrs1_score','LEDD','hemohi']:
    clin[c]=bl[c].reindex(clin.index)
clin['SBR']=clin[['MIA_CAUDATE_L','MIA_CAUDATE_R','MIA_PUTAMEN_L','MIA_PUTAMEN_R']].mean(1)
clin['SAA']=clin['CSFSAA'].astype(float)

print("="*70+"\nR2  NSD 分期分解: DDC 与 SAA / DAT / 临床严重度 的独立关联\n"+"="*70)
d=clin.dropna(subset=['DDC','SAA','SBR','updrs3_score','age','SEX','NSD_num']).copy()
d['zSBR']=(d.SBR-d.SBR.mean())/d.SBR.std(); d['zU3']=(d.updrs3_score-d.updrs3_score.mean())/d.updrs3_score.std()
rows=[]
def fit(f,dd,label):
    m=smf.ols(f,dd).fit(); rows.append(dict(model=label,n=int(m.nobs),r2=m.rsquared,**{k:m.params[k] for k in m.params.index if k!='Intercept'},**{'p_'+k:m.pvalues[k] for k in m.pvalues.index if k!='Intercept'}))
    print(f"  [{label}] n={int(m.nobs)} R²={m.rsquared:.3f} | "+"; ".join(f"{k}: β={m.params[k]:+.3f} P={m.pvalues[k]:.1e}" for k in m.params.index if k not in('Intercept','age','SEX'))); return m
m0=fit('DDC ~ age + SEX',d,'covariates only')
m1=fit('DDC ~ SAA + age + SEX',d,'SAA only')
m2=fit('DDC ~ zSBR + age + SEX',d,'DAT SBR only')
m3=fit('DDC ~ zU3 + age + SEX',d,'motor severity only')
m4=fit('DDC ~ SAA + zSBR + zU3 + age + SEX',d,'all three components')
m5=fit('DDC ~ NSD_num + age + SEX',d,'NSD stage only')
m6=fit('DDC ~ NSD_num + SAA + zSBR + zU3 + age + SEX',d,'NSD stage + all components (residual stage information)')
# 增量 R² (半偏 R²)
def dR2(full,drop,dd):
    mf=smf.ols(full,dd).fit(); md=smf.ols(drop,dd).fit(); return mf.rsquared-md.rsquared, sm.stats.anova_lm(md,mf).iloc[1]['Pr(>F)']
inc={}
for comp,drop in [('SAA','DDC ~ zSBR + zU3 + age + SEX'),('DAT SBR','DDC ~ SAA + zU3 + age + SEX'),('motor severity','DDC ~ SAA + zSBR + age + SEX')]:
    inc[comp]=dR2('DDC ~ SAA + zSBR + zU3 + age + SEX',drop,d); print(f"  独立增量 R² ({comp}): {inc[comp][0]:.3f}, F 检验 P={inc[comp][1]:.1e}")
inc['NSD stage beyond components']=dR2('DDC ~ NSD_num + SAA + zSBR + zU3 + age + SEX','DDC ~ SAA + zSBR + zU3 + age + SEX',d); print(f"  NSD 分期在三成分之外的增量 R²: {inc['NSD stage beyond components'][0]:.3f}, P={inc['NSD stage beyond components'][1]:.1e}")
# 在 SAA+ 内部 (去掉 SAA 成分), 且 DAT 与临床同时进模型
ds=d[d.SAA==1]
m7=fit('DDC ~ zSBR + zU3 + age + SEX',ds,'within SAA+: DAT + motor')
m8=fit('DDC ~ NSD_num + zSBR + zU3 + age + SEX',ds,'within SAA+: NSD stage + DAT + motor')
# 在 DAT 缺损定义相同的层内: NSD stage 2 vs 3 仅由临床严重度区分 -> DDC 是否与临床成分相关 (SAA+ 且 DAT 缺损者)
d23=ds[ds.NSD_num.isin([2,3])]
m9=fit('DDC ~ C(NSD_num) + zSBR + age + SEX',d23,'SAA+ stage 2 vs 3 (clinical component) adjusted for SBR')
pd.DataFrame(rows).to_csv(D+"/rev2_nsd_decomposition.csv",index=False)
json.dump({k:{'dR2':v[0],'p':v[1]} for k,v in inc.items()},open(D+"/rev2_nsd_incrementalR2.json","w"),indent=1)

print("\n"+"="*70+"\nR3  body-first 定义敏感性 (PD 内 body vs brain 差异蛋白数, FDR<0.05)\n"+"="*70)
pdm=clin[clin.group.isin(['PD_bodyfirst','PD_brainfirst'])].copy()
pdm['asym']=pdm['PUTAMEN_ASYM']; med_asym=pdm.asym.median(); med_scopa=pdm.scopa.median()
defs={'RBDSQ ≥ 5 (primary)':pdm.rem>=5,
      'RBDSQ ≥ 6 (stricter)':pdm.rem>=6,
      'RBDSQ ≥ 5 AND symmetric putaminal DAT (asymmetry ≤ median)':(pdm.rem>=5)&(pdm.asym<=med_asym),
      'RBDSQ ≥ 5 AND autonomic burden ≥ median (SCOPA-AUT)':(pdm.rem>=5)&(pdm.scopa>=med_scopa),
      'RBDSQ ≥ 5 AND symmetric DAT AND autonomic ≥ median':(pdm.rem>=5)&(pdm.asym<=med_asym)&(pdm.scopa>=med_scopa),
      'Composite score ≥ 2 of 3 (RBDSQ ≥ 5, symmetric DAT, autonomic ≥ median)':((pdm.rem>=5).astype(int)+(pdm.asym<=med_asym).astype(int)+(pdm.scopa>=med_scopa).astype(int))>=2}
prim=(pdm.rem>=5)
rows=[]
for lab,bf in defs.items():
    ok=bf.notna()&pdm.rem.notna()&pdm.age.notna()
    sub=pdm[ok]; y=bf[ok].astype(float).values
    Xd=np.column_stack([np.ones(len(sub)),y,sub.age.values,sub.SEX.values]); Y=X[[wide.index.get_loc(i) for i in sub.index]]
    XtXinv=np.linalg.pinv(Xd.T@Xd); B=XtXinv@Xd.T@Y; r=Y-Xd@B; dof=len(sub)-4; se=np.sqrt((r**2).sum(0)/dof*XtXinv[1,1]); p=2*stats.t.sf(np.abs(B[1]/se),dof)
    fdr=multipletests(p,method='fdr_bh')[1]; agree=(bf[ok]==prim[ok]).mean()
    ddc_p=p[jD]; ddc_b=B[1][jD]/Y[:,jD].std()
    rows.append(dict(definition=lab,n_body=int(y.sum()),n_brain=int(len(y)-y.sum()),agreement_with_primary=agree,n_sig_FDR05=int((fdr<0.05).sum()),min_FDR=fdr.min(),DDC_beta_z=ddc_b,DDC_p=ddc_p))
    print(f"  {lab:75s} body={int(y.sum()):3d} brain={int(len(y)-y.sum()):3d} 与主定义一致率={agree:.2f} | FDR<0.05 差异蛋白={int((fdr<0.05).sum())} (最小FDR {fdr.min():.2f}) | DDC β_z={ddc_b:+.2f} P={ddc_p:.2f}")
pd.DataFrame(rows).to_csv(D+"/rev2_subtype_definitions.csv",index=False)

print("\n"+"="*70+"\nR6  进展阴性结论的敏感性 (DDC × time 混合模型)\n"+"="*70)
syn=clin.index[clin.group.isin(['iRBD','Hyposmia','PD_bodyfirst','PD_brainfirst'])]
base=clin[['DDC','SBR','age','SEX','group']].copy(); base['updrs3_bl']=clin['updrs3_score']
lg=cur[cur.PATNO.isin(syn)][['PATNO','EVENT_ID','YEAR','updrs3_score','updrs3_score_on','LEDD','MIA_CAUDATE_L','MIA_CAUDATE_R','MIA_PUTAMEN_L','MIA_PUTAMEN_R']].copy()
lg['SBR_t']=lg[['MIA_CAUDATE_L','MIA_CAUDATE_R','MIA_PUTAMEN_L','MIA_PUTAMEN_R']].mean(1)
lg=lg.merge(base,left_on='PATNO',right_index=True).dropna(subset=['DDC','age','SEX','YEAR']); lg['LEDD']=lg.LEDD.fillna(0)
rows=[]
def lme(df,y,label,extra=''):
    df=df.dropna(subset=[y]); cnt=df.groupby('PATNO').size(); df=df[df.PATNO.isin(cnt[cnt>=2].index)]
    try:
        md=smf.mixedlm(f'{y} ~ DDC*YEAR + age + SEX{extra}',df,groups=df['PATNO'],re_formula='~YEAR').fit(method='lbfgs')
        b=md.params['DDC:YEAR']; p=md.pvalues['DDC:YEAR']; ci=md.conf_int().loc['DDC:YEAR'].values
        rows.append(dict(analysis=label,outcome=y,n=df.PATNO.nunique(),obs=len(df),beta_DDCxtime=b,ci_lo=ci[0],ci_hi=ci[1],p=p,median_fu=df.groupby('PATNO').YEAR.max().median()))
        print(f"  {label:70s} n={df.PATNO.nunique():4d} obs={len(df):5d} 中位随访={df.groupby('PATNO').YEAR.max().median():.0f}y | DDC×time β={b:+.4f} [{ci[0]:+.3f},{ci[1]:+.3f}] P={p:.3f}")
    except Exception as e: print(label,'失败',repr(e)[:100])
lme(lg,'updrs3_score','(a) Primary: MDS-UPDRS-III, all biologically positive (as in main text)')
lme(lg,'updrs3_score','(b) + baseline SBR + time-varying LEDD',' + SBR + LEDD')
lme(lg[lg.LEDD==0],'updrs3_score','(c) Untreated visits only (LEDD = 0 at the visit)')
lme(lg[lg.group.isin(['iRBD','Hyposmia'])],'updrs3_score','(d) Prodromal participants only (largely untreated)')
lme(lg[lg.group.str.startswith('PD')],'updrs3_score','(e) PD only')
lme(lg,'updrs3_score_on','(f) ON-state MDS-UPDRS-III (treated visits)')
lme(lg,'SBR_t','(g) Treatment-insensitive outcome: longitudinal striatal DAT SBR')
lme(lg[lg.group.str.startswith('PD')],'SBR_t','(h) Longitudinal striatal SBR, PD only')
lme(lg[lg.group.isin(['iRBD','Hyposmia'])],'SBR_t','(i) Longitudinal striatal SBR, prodromal only')
print("  注: PPMI 'updrs3_score' 为未用药/OFF 状态评分, 'updrs3_score_on' 为 ON 状态评分")
pd.DataFrame(rows).to_csv(D+"/rev2_progression_sensitivity.csv",index=False)
# 检验力: 混合模型能检出多大的 DDC×time 效应? 用 (b) 的 SE 估计最小可检出效应 (MDE, 80% 功效)
r=pd.DataFrame(rows); r0=r.iloc[0]; se=(r0.ci_hi-r0.ci_lo)/3.92; mde=2.8*se
print(f"  主分析 DDC×time 的 SE={se:.3f} 分/年/NPX → 80% 功效可检出的最小效应 ≈ {mde:.2f} UPDRS-III 分/年 per NPX 单位 (DDC SD≈{clin.DDC.std():.2f} NPX → 每 SD {mde*clin.DDC.std():.2f} 分/年)")
json.dump({'se':se,'mde_per_NPX':mde,'mde_per_SD':mde*clin.DDC.std(),'DDC_sd':clin.DDC.std()},open(D+"/rev2_progression_mde.json","w"))

print('done')
