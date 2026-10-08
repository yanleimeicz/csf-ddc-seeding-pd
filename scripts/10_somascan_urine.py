# -*- coding: utf-8 -*-
"""Other PPMI platforms: SomaScan CSF (Project 151) and urine mass spectrometry (Project 190): DDC diagnostic
performance, Olink-SomaScan concordance, genetic carriers, RBD signature within PD.
Output: derived/rev2_soma_*.csv, rev2_urine_*.csv/json (Supplementary Fig. S2, Table S13).
"""
import config as cfg
import numpy as np, pandas as pd, warnings, sys, json, glob
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score
D=cfg.D
clin_all=pd.read_csv(D+"/analytic_clinical.csv").set_index('PATNO')
cur=pd.read_excel(cfg.CURATED,sheet_name='20260511')
bl=cur[cur.EVENT_ID=='BL'].set_index('PATNO')
def nsd_num(v):
    s=str(v); return 0.0 if s.startswith('Not') else (float(s[0]) if s[0].isdigit() else np.nan)
def auc_ci(y,x,n=1000,seed=42):
    a=roc_auc_score(y,x); rng=np.random.RandomState(seed); bs=[]
    y=np.asarray(y); x=np.asarray(x)
    for _ in range(n):
        s=rng.randint(0,len(y),len(y))
        if len(set(y[s]))<2: continue
        bs.append(roc_auc_score(y[s],x[s]))
    return a,np.percentile(bs,2.5),np.percentile(bs,97.5)

# ---------------- 1) SomaScan CSF ----------------
soma=pd.read_parquet(D+"/somascan151_csf.parquet"); soma['PATNO']=soma.PATNO.astype(int); soma=soma[soma.EVENT_ID=='BL'].set_index('PATNO')
lk=pd.read_csv(D+"/somascan151_lookup.csv")
S=soma.drop(columns=['EVENT_ID']).astype(float)
S.columns=[c.replace('P151_CSF_','') for c in S.columns]
c=clin_all.reindex(S.index).copy()
for k in ['MIA_CAUDATE_L','MIA_CAUDATE_R','MIA_PUTAMEN_L','MIA_PUTAMEN_R','updrs3_score','LEDD','NSD_STAGE','CSFSAA','COHORT','PRIMDIAG']:
    c[k]=bl[k].reindex(c.index)
c['SBR']=c[['MIA_CAUDATE_L','MIA_CAUDATE_R','MIA_PUTAMEN_L','MIA_PUTAMEN_R']].mean(1); c['NSD_num']=c['NSD_STAGE'].map(nsd_num)
c['DDC_soma']=S['DDC']
olink=pd.read_parquet(D+"/csf_npx_wide.parquet").set_index('PATNO'); ann=pd.read_csv(D+"/protein_annotation.csv").set_index('OlinkID')
og=ann.reindex(olink.columns)['Assay'].astype(str).values; jD=int(np.where(og=='DDC')[0][0])
c['DDC_olink']=olink.iloc[:,jD].reindex(c.index)
c['in_olinkHT']=c.index.isin(olink.index)
# Olink Explore HT vs SomaScan concordance for every protein measured on both platforms (overlapping participants)
_ov=olink.index.intersection(S.index); _cc=[]
for _g in sorted(set(og)&set(S.columns)):
    _x=olink.loc[_ov,olink.columns[np.where(og==_g)[0][0]]]; _y=S.loc[_ov,_g]; _m=_x.notna()&_y.notna()
    _cc.append(dict(gene=_g,r=stats.pearsonr(_x[_m],_y[_m])[0],sd_soma=_y.std()))
pd.DataFrame(_cc).to_csv(D+"/rev2_soma_olink_concordance.csv",index=False)
print("SomaScan 基线 n=",len(c)," 分组:",c.group.value_counts().to_dict())
print("  subgroup 列:",c['subgroup'].value_counts().head(12).to_dict())
res={}
# 平台相关
ov=c.dropna(subset=['DDC_soma','DDC_olink']); r,p=stats.pearsonr(ov.DDC_soma,ov.DDC_olink); rs,ps=stats.spearmanr(ov.DDC_soma,ov.DDC_olink)
print(f"  Olink HT vs SomaScan DDC (n={len(ov)}): Pearson r={r:.2f} P={p:.1e}; Spearman ρ={rs:.2f}")
res['platform_corr']=dict(n=len(ov),pearson_r=r,p=p,spearman=rs)
# PD vs HC (全部 / 非重叠)
pdhc=c[c.group.isin(['PD_bodyfirst','PD_brainfirst','HC'])].dropna(subset=['DDC_soma']); y=(pdhc.group!='HC').astype(int)
a=auc_ci(y,pdhc.DDC_soma); print(f"  SomaScan DDC PD vs HC: n={len(pdhc)} (PD {y.sum()}, HC {(1-y).sum()}) AUC={a[0]:.3f} ({a[1]:.3f}-{a[2]:.3f})")
res['auc_all']=dict(n=len(pdhc),n_pd=int(y.sum()),n_hc=int((1-y).sum()),auc=a[0],lo=a[1],hi=a[2])
ind=pdhc[~pdhc.in_olinkHT]; y2=(ind.group!='HC').astype(int); a2=auc_ci(y2,ind.DDC_soma)
print(f"  SomaScan DDC PD vs HC, 与 Olink HT 无重叠参与者: n={len(ind)} (PD {y2.sum()}, HC {(1-y2).sum()}) AUC={a2[0]:.3f} ({a2[1]:.3f}-{a2[2]:.3f})")
res['auc_nonoverlap']=dict(n=len(ind),n_pd=int(y2.sum()),n_hc=int((1-y2).sum()),auc=a2[0],lo=a2[1],hi=a2[2])
# 组间均值
print("  SomaScan DDC (log2 RFU) 组均值:",pdhc.groupby('group').DDC_soma.agg(['count','mean','std']).round(3).to_dict('index'))
kw=stats.kruskal(*[g.DDC_soma.values for _,g in pdhc.groupby('group')]); res['kw_p']=kw.pvalue
# 关联: SBR / UPDRS3 / SAA / NSD / LEDD
for f,lab,sub in [('DDC_soma ~ SBR + age + SEX','SBR (all)',c),('DDC_soma ~ updrs3_score + age + SEX','UPDRS-III (all)',c),
                  ('DDC_soma ~ CSFSAA + age + SEX','SAA (all)',c),('DDC_soma ~ NSD_num + age + SEX','NSD stage (all)',c),
                  ('DDC_soma ~ NSD_num + age + SEX','NSD stage (SAA+)',c[c.CSFSAA==1]),('DDC_soma ~ SBR + age + SEX','SBR (SAA+)',c[c.CSFSAA==1]),
                  ('DDC_soma ~ SBR + updrs3_score + age + SEX','SBR + UPDRS-III (SAA+)',c[c.CSFSAA==1]),
                  ('DDC_soma ~ LEDD + age + SEX','LEDD (PD)',c[c.group.str.startswith('PD')])]:
    d=sub.dropna(subset=[v for v in ['DDC_soma','SBR','updrs3_score','CSFSAA','NSD_num','LEDD','age','SEX'] if v in f]);
    if len(d)<30: print('  跳过',lab); continue
    m=smf.ols(f,d).fit(); k=[t for t in m.params.index if t not in('Intercept','age','SEX')]
    print(f"  {lab:28s} n={int(m.nobs):4d} | "+"; ".join(f"{t}: β={m.params[t]:+.3f} P={m.pvalues[t]:.1e}" for t in k))
    res['assoc_'+lab]={t:dict(beta=m.params[t],p=m.pvalues[t],n=int(m.nobs)) for t in k}
# 遗传队列
gen=c[c.group=='GV/other'].copy(); print("  GV/other subgroup:",gen.subgroup.value_counts().to_dict(), "| COHORT:",gen.COHORT.value_counts().to_dict())
hc=c[c.group=='HC'].dropna(subset=['DDC_soma'])
rowsG=[]
for sg,g in gen.dropna(subset=['DDC_soma']).groupby('subgroup'):
    if len(g)<10: continue
    yy=np.r_[np.ones(len(g)),np.zeros(len(hc))]; xx=np.r_[g.DDC_soma.values,hc.DDC_soma.values]
    a=auc_ci(yy,xx); t=stats.mannwhitneyu(g.DDC_soma,hc.DDC_soma)
    rowsG.append(dict(subgroup=sg,n=len(g),n_hc=len(hc),mean_diff=g.DDC_soma.mean()-hc.DDC_soma.mean(),auc=a[0],lo=a[1],hi=a[2],p_mwu=t.pvalue,saa_pos=g.CSFSAA.mean()))
    print(f"  遗传亚组 {sg:35s} n={len(g):3d} vs HC {len(hc)}: Δmean={g.DDC_soma.mean()-hc.DDC_soma.mean():+.3f} AUC={a[0]:.2f} ({a[1]:.2f}-{a[2]:.2f}) MWU P={t.pvalue:.1e} SAA+ {g.CSFSAA.mean()*100:.0f}%")
pd.DataFrame(rowsG).to_csv(D+"/rev2_soma_genetic.csv",index=False)
# 55 个 RBD 蛋白在 SomaScan: 全体 vs PD 内 RBDSQ 关联 (SomaScan 无前驱组, 全体=HC+PD+GV)
sig55=pd.read_csv(D+"/sig_rem.csv"); sig55=sig55[sig55.fdr<0.05]; g55=[g for g in sig55.gene.astype(str) if g in S.columns]
print(f"  55 个 RBD 蛋白中 SomaScan 有 {len(g55)} 个")
def rbd_soma(sub,label):
    d=sub.dropna(subset=['rem','age','SEX']); Y=S.loc[d.index,g55].values; Xd=np.column_stack([np.ones(len(d)),d.rem.values,d.age.values,d.SEX.values])
    XtXinv=np.linalg.pinv(Xd.T@Xd); B=XtXinv@Xd.T@Y; r=Y-Xd@B; dof=len(d)-4; se=np.sqrt((r**2).sum(0)/dof*XtXinv[1,1]); p=2*stats.t.sf(np.abs(B[1]/se),dof)
    bz=B[1]*d.rem.std()/Y.std(0); fdr=multipletests(p,method='fdr_bh')[1]
    # 方向一致性 与 Olink 全队列 β
    ob=sig55.set_index('gene').loc[g55,'beta_z'].values; agree=(np.sign(bz)==np.sign(ob)).mean(); rr=stats.spearmanr(bz,ob)
    print(f"  {label:45s} n={len(d)} | P<0.05: {(p<0.05).sum()}/{len(g55)} | FDR<0.05 (within 55): {(fdr<0.05).sum()} | 与 Olink β 方向一致 {agree*100:.0f}%, ρ={rr[0]:+.2f} (P={rr[1]:.2f})")
    return pd.DataFrame({'gene':g55,'beta_z_soma':bz,'p':p,'fdr_within55':fdr,'beta_z_olink_full':ob})
r1=rbd_soma(c[c.group.isin(['HC','PD_bodyfirst','PD_brainfirst'])],'SomaScan HC+PD (mixed, disease status varies)')
r2=rbd_soma(c[c.group.str.startswith('PD')],'SomaScan within PD')
r1.to_csv(D+"/rev2_soma_rbd55_mixed.csv",index=False); r2.to_csv(D+"/rev2_soma_rbd55_withinPD.csv",index=False)
json.dump(res,open(D+"/rev2_soma_summary.json","w"),indent=1,default=float)
c[['group','subgroup','DDC_soma','DDC_olink','in_olinkHT','SBR','updrs3_score','CSFSAA','NSD_num','age','SEX']].to_csv(D+"/rev2_soma_ddc.csv")

# ---------------- 2) 尿液质谱 DDC ----------------
print("\n尿液质谱 (Project 190):")
fr=[]
for f in sorted(glob.glob(cfg.P+"/Targeted___untargeted_MS-based_proteomics_of_urine_in_PD_*_02Jul2026.csv")):
    fr.append(pd.read_csv(f,usecols=['PATNO','COHORT','CLINICAL_EVENT','TESTNAME','TESTVALUE']).query("TESTNAME.str.startswith('DDC_')",engine='python'))
u=pd.concat(fr); u['PATNO']=u.PATNO.astype(int); u['val']=np.log2(pd.to_numeric(u.TESTVALUE,errors='coerce'))
print("  DDC 行数",len(u),"| 访视:",u.CLINICAL_EVENT.value_counts().to_dict())
ub=u[u.CLINICAL_EVENT=='BL'].groupby('PATNO').val.mean().to_frame('DDC_urine')
ub=ub.join(clin_all[['group','age','SEX','CSFSAA','rem']]);
for k in ['updrs3_score']: ub[k]=bl[k].reindex(ub.index)
ub['SBR']=bl[['MIA_CAUDATE_L','MIA_CAUDATE_R','MIA_PUTAMEN_L','MIA_PUTAMEN_R']].mean(1).reindex(ub.index)
print("  基线可用:",ub.group.value_counts(dropna=False).to_dict())
print("  组均值 log2 LFQ:",ub.groupby('group').DDC_urine.agg(['count','mean','std']).round(2).to_dict('index'))
resU={}
for lab,gs in [('PD vs HC',['PD_bodyfirst','PD_brainfirst']),('iRBD vs HC',['iRBD']),('Hyposmia vs HC',['Hyposmia'])]:
    d=ub[ub.group.isin(gs+['HC'])].dropna(subset=['DDC_urine'])
    if (d.group=='HC').sum()<5 or d.group.isin(gs).sum()<5: print('  跳过',lab); continue
    y=d.group.isin(gs).astype(int); a=auc_ci(y,d.DDC_urine); t=stats.mannwhitneyu(d.DDC_urine[y==1],d.DDC_urine[y==0])
    m=smf.ols('DDC_urine ~ y + age + SEX',d.assign(y=y)).fit()
    print(f"  尿 DDC {lab:15s}: n={len(d)} ({y.sum()} vs {(1-y).sum()}) AUC={a[0]:.2f} ({a[1]:.2f}-{a[2]:.2f}) MWU P={t.pvalue:.1e} | 校正年龄性别 β={m.params['y']:+.2f} P={m.pvalues['y']:.1e}")
    resU[lab]=dict(n=len(d),n_case=int(y.sum()),n_hc=int((1-y).sum()),auc=a[0],lo=a[1],hi=a[2],p_mwu=t.pvalue,beta_adj=m.params['y'],p_adj=m.pvalues['y'])
for f,lab in [('DDC_urine ~ SBR + age + SEX','urine DDC ~ SBR'),('DDC_urine ~ updrs3_score + age + SEX','urine DDC ~ UPDRS-III'),('DDC_urine ~ CSFSAA + age + SEX','urine DDC ~ SAA')]:
    d=ub.dropna(subset=[v for v in ['DDC_urine','SBR','updrs3_score','CSFSAA','age','SEX'] if v in f])
    m=smf.ols(f,d).fit(); k=[t for t in m.params.index if t not in('Intercept','age','SEX')][0]
    print(f"  {lab:28s} n={int(m.nobs)} β={m.params[k]:+.3f} P={m.pvalues[k]:.1e}"); resU[lab]=dict(n=int(m.nobs),beta=m.params[k],p=m.pvalues[k])
ub.to_csv(D+"/rev2_urine_ddc.csv"); json.dump(resU,open(D+"/rev2_urine_summary.json","w"),indent=1,default=float)
print('done')
