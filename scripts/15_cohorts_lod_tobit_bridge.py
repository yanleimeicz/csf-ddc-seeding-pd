# -*- coding: utf-8 -*-
"""Cohort composition (Table 1), SAA-negative hyposmia, left-censored (Tobit) models, same-sample bridge between
projects, NSD components and SAA AUC. Output: derived/r3A_*.csv (Table 1, Fig. 4, Tables S2, S11, S15, S17).
"""
import config as cfg
import sys, warnings, json
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc
import statsmodels.formula.api as smf
from scipy import stats, optimize
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
B=cfg.P; D=cfg.D; OUT=cfg.RS
LOG=open(OUT+"/r3A_log.txt","w",encoding="utf-8")
def P(*a):
    s=" ".join(str(x) for x in a); print(s); LOG.write(s+"\n"); LOG.flush()
def H(t): P("\n"+"="*78+"\n"+t+"\n"+"="*78)
GEN=['LRRK2','GBA','LRRK2 + GBA','SNCA','PRKN','PINK1','PRKN + RBD']
SBRc=['MIA_CAUDATE_L','MIA_CAUDATE_R','MIA_PUTAMEN_L','MIA_PUTAMEN_R']
clin=pd.read_csv(D+"/analytic_clinical.csv").set_index('PATNO')
cur=pd.read_excel(cfg.CURATED,sheet_name='20260511')
bl=cur[cur.EVENT_ID=='BL'].drop_duplicates('PATNO').set_index('PATNO')
for c in SBRc+['updrs2_score','hemohi','race','moca']: clin[c]=bl[c].reindex(clin.index)
clin['SBR']=clin[SBRc].mean(1); clin['SAA']=clin.CSFSAA.where(clin.CSFSAA.isin([0,1]))
clin['NSD_cat']=clin.NSD_STAGE.astype(str).where(clin.NSD_STAGE.notna())
clin['NSD_num']=clin.NSD_STAGE.map(lambda v: 0.0 if str(v).startswith('Not') else (float(str(v)[0]) if str(v)[:1].isdigit() else np.nan))
wide=pd.read_parquet(D+"/csf_npx_wide.parquet").set_index('PATNO'); wide.index=wide.index.astype(int)
ann=pd.read_csv(D+"/protein_annotation.csv").set_index('OlinkID'); jD=[c for c in wide.columns if ann.Assay.get(c)=='DDC'][0]
disc=clin.reindex(wide.index).copy(); disc['DDC']=wide[jD].fillna(wide[jD].median()).values; disc['cohort']='Discovery'
rep=pd.read_csv(D+"/p314_rep_baseline_analytic.csv").set_index('PATNO')
for c in ['updrs2_score','race','moca']: rep[c]=bl[c].reindex(rep.index)
rep['NSD_cat']=rep.NSD_STAGE.astype(str).where(rep.NSD_STAGE.notna())
spor=rep[~rep.subgroup.isin(GEN)&rep.group0.isin(['HC','Hyposmia','iRBD','PD'])].copy(); spor['cohort']='Replication'
disc['group0']=disc.group0.where(disc.group0.isin(['HC','Hyposmia','iRBD','PD']))

# ---------------------------------------------------------------- 1 组别 × SAA
H("1. 组别 × SAA 交叉表与 SAA 阴性嗅觉减退者的 DDC (R2-W2)")
rows=[]
for nm,df in [('Discovery',disc),('Replication',spor)]:
    ct=pd.crosstab(df.group0,df.SAA.map({1.0:'SAA+',0.0:'SAA-'}).fillna('SAA n/a'),margins=True); P(f"-- {nm}\n{ct.to_string()}")
    ct.to_csv(D+f"/r3A_crosstab_{nm}.csv")
    d=df.dropna(subset=['DDC','age','SEX']).copy()
    d['cell']=np.where(d.group0=='HC','HC',d.group0+np.where(d.SAA==1,'_SAA+',np.where(d.SAA==0,'_SAA-','_SAAna')))
    d=d[d.cell!='HC']._append(d[d.cell=='HC']) if False else d
    keep=d.cell.map(d.cell.value_counts())>=5; d=d[keep]
    d['cell']=pd.Categorical(d.cell,['HC']+sorted(c for c in d.cell.unique() if c!='HC'))
    m=smf.ols("DDC ~ C(cell) + age + SEX",d).fit()
    for c in d.cell.cat.categories[1:]:
        k=f'C(cell)[T.{c}]'; ci=m.conf_int().loc[k]
        sub=d[d.cell.isin([c,'HC'])]; y=(sub.cell==c).astype(int); auc=roc_auc_score(y,sub.DDC)
        sb=d.loc[d.cell==c,'SBR']
        rows.append(dict(cohort=nm,cell=c,n=int((d.cell==c).sum()),beta=m.params[k],ci_lo=ci[0],ci_hi=ci[1],p=m.pvalues[k],auc_vs_HC=auc,mean_SBR=sb.mean()))
        P(f"  {nm:11s} {c:16s} n={int((d.cell==c).sum()):4d} | DDC vs HC β={m.params[k]:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] P={m.pvalues[k]:.1e} | AUC={auc:.2f} | SBR={sb.mean():.2f}")
    P(f"  HC SBR mean {d.loc[d.cell=='HC','SBR'].mean():.2f}")
pd.DataFrame(rows).to_csv(D+"/r3A_group_by_SAA_DDC.csv",index=False)
# SAA- 散发者中 DAT 缺损者 (EIC Q4): SBR < HC 第 5 百分位
H("1b. SAA 阴性散发参与者中 DAT 缺损者 vs 正常者 (EIC-Q4)")
hc_p5=spor.loc[spor.group0=='HC','SBR'].quantile(0.05)
sn=spor[(spor.SAA==0)].dropna(subset=['SBR','DDC']).copy(); sn['DATdef']=(sn.SBR<hc_p5).astype(int)
m=smf.ols("DDC ~ DATdef + age + SEX",sn).fit()
P(f"  SAA- 散发 n={len(sn)} (DAT 缺损 {int(sn.DATdef.sum())}; 阈值 SBR<{hc_p5:.2f}) | 缺损 vs 正常 β={m.params.DATdef:+.3f} P={m.pvalues.DATdef:.2e} | 组成: {sn[sn.DATdef==1].group0.value_counts().to_dict()}")
json.dump(dict(n=len(sn),n_def=int(sn.DATdef.sum()),beta=m.params.DATdef,p=m.pvalues.DATdef,thr=hc_p5),open(D+"/r3A_SAAneg_DATdef.json","w"))

# ---------------------------------------------------------------- 2 SAA AUC 订正
H("2. SAA 在同一 733 人中的 PD vs HC AUC (EIC/R1 不一致 #5)")
pdhc=disc[disc.group0.isin(['PD','HC'])].dropna(subset=['SAA'])
a_saa=roc_auc_score((pdhc.group0=='PD').astype(int),pdhc.SAA); a_ddc=roc_auc_score((pdhc.group0=='PD').astype(int),pdhc.DDC)
P(f"  733 人内 PD vs HC (n={len(pdhc)}): SAA AUC={a_saa:.3f}; DDC AUC={a_ddc:.3f}")
pdhc2=spor[spor.group0.isin(['PD','HC'])].dropna(subset=['SAA'])
a2=roc_auc_score((pdhc2.group0=='PD').astype(int),pdhc2.SAA); P(f"  复现队列 PD vs HC: SAA AUC={a2:.3f}")
# 增量: DDC 加入 SAA 后 (逻辑回归, 5 折 CV)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
for nm,df in [('Discovery',pdhc),('Replication',pdhc2)]:
    y=(df.group0=='PD').astype(int).values; cv=StratifiedKFold(5,shuffle=True,random_state=1)
    p1=cross_val_predict(LogisticRegression(),df[['SAA']].values,y,cv=cv,method='predict_proba')[:,1]
    p2=cross_val_predict(LogisticRegression(),df[['SAA','DDC']].values,y,cv=cv,method='predict_proba')[:,1]
    P(f"  {nm}: CV AUC SAA {roc_auc_score(y,p1):.3f} → SAA+DDC {roc_auc_score(y,p2):.3f}")
json.dump(dict(saa_auc_disc=a_saa,ddc_auc_disc=a_ddc,saa_auc_rep=a2),open(D+"/r3A_saa_auc.json","w"))

# ---------------------------------------------------------------- 3 删失 (Tobit) 模型
H("3. 低于 LOD 的删失 (Tobit) 模型: H1 组间差 (DA-C4, R1-W5)")
def tobit(y,X,lod):
    cens=y<lod; yy=np.where(cens,lod,y)
    def nll(th):
        b=th[:-1]; s=np.exp(th[-1]); mu=X@b
        return -(stats.norm.logpdf(yy[~cens],mu[~cens],s).sum()+stats.norm.logcdf((lod[cens]-mu[cens])/s).sum())
    b0=np.linalg.lstsq(X,yy,rcond=None)[0]; th0=np.r_[b0,np.log(yy.std())]
    r=optimize.minimize(nll,th0,method='BFGS'); se=np.sqrt(np.diag(r.hess_inv)); return r.x,se,cens.mean()
t=pq.read_table(B+"/ppmi_proj314_csf_screened_extended_npx_20260116.parquet",columns=['PATNO','EVENT_ID','SampleType','SampleQC','Assay','NPX','LOD','LAB_QC_FLAG'])
t=t.filter(pc.and_(pc.equal(t['Assay'],'DDC'),pc.equal(t['SampleType'],'SAMPLE'))).to_pandas()
t=t[(t.SampleQC=='PASS')&(t.EVENT_ID=='BL')&t.PATNO.notna()]; t=t[~t.LAB_QC_FLAG.fillna('').astype(str).str.contains('6|7')]
t['PATNO']=t.PATNO.astype(int); t=t.groupby('PATNO').agg(NPX=('NPX','mean'),LOD=('LOD','mean'))
s=spor.join(t,how='inner').dropna(subset=['NPX','LOD','age','SEX'])
P("  各组低于 LOD (%): "+str((100*(s.NPX<s.LOD).groupby(s.group0).mean()).round(1).to_dict())+f"; 散发总体 {(100*(s.NPX<s.LOD).mean()):.1f}%; 含遗传的全部新基线 {100*(rep.belowLOD.mean()):.1f}%")
G=['Hyposmia','iRBD','PD']; X=np.column_stack([np.ones(len(s))]+[(s.group0==g).astype(float) for g in G]+[s.age,s.SEX]); y=s.NPX.values
th,se,cf=tobit(y,X,s.LOD.values); ols=np.linalg.lstsq(X,np.maximum(y,-99),rcond=None)[0]
trows=[]
for i,g in enumerate(G):
    z=th[1+i]/se[1+i]; p=2*stats.norm.sf(abs(z))
    trows.append(dict(model='Tobit (left-censored at LOD)',contrast=f'{g} vs HC',beta=th[1+i],ci_lo=th[1+i]-1.96*se[1+i],ci_hi=th[1+i]+1.96*se[1+i],p=p,censored_frac=cf))
    P(f"  Tobit {g:8s} vs HC: β={th[1+i]:+.3f} [{th[1+i]-1.96*se[1+i]:+.3f},{th[1+i]+1.96*se[1+i]:+.3f}] P={p:.1e} | OLS(全部NPX) β={ols[1+i]:+.3f}")
# 发现队列同样处理 (277 BL)
t2=pq.read_table(B+"/PPMI_Project_277_CSF_Screened_NPX_20260130.parquet",columns=['PATNO','EVENT_ID','SampleType','SampleQC','Assay','NPX','LOD'])
t2=t2.filter(pc.and_(pc.equal(t2['Assay'],'DDC'),pc.equal(t2['SampleType'],'SAMPLE'))).to_pandas()
t2=t2[(t2.SampleQC=='PASS')&(t2.EVENT_ID=='BL')&t2.PATNO.notna()]; t2['PATNO']=t2.PATNO.astype(int); t2=t2.groupby('PATNO').agg(NPX=('NPX','mean'),LOD=('LOD','mean'))
s2=disc.join(t2,how='inner',rsuffix='_r').dropna(subset=['NPX','LOD','age','SEX','group0'])
X2=np.column_stack([np.ones(len(s2))]+[(s2.group0==g).astype(float) for g in G]+[s2.age,s2.SEX])
th2,se2,cf2=tobit(s2.NPX.values,X2,s2.LOD.values)
for i,g in enumerate(G):
    p=2*stats.norm.sf(abs(th2[1+i]/se2[1+i])); trows.append(dict(model='Tobit, discovery',contrast=f'{g} vs HC',beta=th2[1+i],ci_lo=th2[1+i]-1.96*se2[1+i],ci_hi=th2[1+i]+1.96*se2[1+i],p=p,censored_frac=cf2))
    P(f"  发现队列 Tobit {g:8s} vs HC: β={th2[1+i]:+.3f} P={p:.1e} (删失 {100*cf2:.1f}%)")
# Tobit 下 SAA 独立关联 (H2)
s3=s.dropna(subset=['SAA','SBR','updrs3_score']).copy(); s3=s3[s3.LEDD.fillna(0)==0]
z_=lambda v:(v-v.mean())/v.std()
X3=np.column_stack([np.ones(len(s3)),s3.SAA,z_(s3.SBR),z_(s3.updrs3_score),s3.age,s3.SEX]); th3,se3,_=tobit(s3.NPX.values,X3,s3.LOD.values)
for i,nm in enumerate(['SAA','zSBR','zUPDRS-III']):
    p=2*stats.norm.sf(abs(th3[1+i]/se3[1+i])); trows.append(dict(model='Tobit H2/H3, replication untreated',contrast=nm,beta=th3[1+i],ci_lo=th3[1+i]-1.96*se3[1+i],ci_hi=th3[1+i]+1.96*se3[1+i],p=p))
    P(f"  Tobit H2/H3 {nm}: β={th3[1+i]:+.3f} P={p:.1e}")
pd.DataFrame(trows).to_csv(D+"/r3A_tobit.csv",index=False)

# ---------------------------------------------------------------- 4 同样本桥接
H("4. 两项目中同一样本 (PATNO+访视) 的桥接 (R1-W5)")
def long(f,cols=['PATNO','EVENT_ID','SampleType','SampleQC','OlinkID','Assay','AssayType','NPX']):
    tt=pq.read_table(f,columns=cols); tt=tt.filter(pc.and_(pc.equal(tt['SampleType'],'SAMPLE'),pc.equal(tt['AssayType'],'assay'))).to_pandas()
    tt=tt[(tt.SampleQC=='PASS')&tt.PATNO.notna()]; tt['PATNO']=tt.PATNO.astype(int); return tt
a=long(B+"/PPMI_Project_277_CSF_Screened_NPX_20260130.parquet"); b=long(B+"/ppmi_proj314_csf_screened_extended_npx_20260116.parquet")
ka=set(zip(a.PATNO,a.EVENT_ID)); kb=set(zip(b.PATNO,b.EVENT_ID)); both=ka&kb
a=a[[k in both for k in zip(a.PATNO,a.EVENT_ID)]]; b=b[[k in both for k in zip(b.PATNO,b.EVENT_ID)]]
wa=a.pivot_table(index=['PATNO','EVENT_ID'],columns='OlinkID',values='NPX'); wb=b.pivot_table(index=['PATNO','EVENT_ID'],columns='OlinkID',values='NPX')
ix=wa.index.intersection(wb.index); cols=wa.columns.intersection(wb.columns); wa=wa.loc[ix,cols]; wb=wb.loc[ix,cols]
P(f"  同一样本 n={len(ix)} (访视: {pd.Series([k[1] for k in ix]).value_counts().to_dict()}), 共有检测 {len(cols)}")
r=[stats.spearmanr(wa[c],wb[c],nan_policy='omit')[0] for c in cols]; rr=pd.Series(r,index=cols)
def icc(x,y):  # ICC(2,1) 绝对一致
    M=np.c_[x,y]; M=M[~np.isnan(M).any(1)]; n,k=M.shape; gm=M.mean()
    msr=k*((M.mean(1)-gm)**2).sum()/(n-1); msc=n*((M.mean(0)-gm)**2).sum()/(k-1)
    mse=((M-M.mean(1,keepdims=True)-M.mean(0,keepdims=True)+gm)**2).sum()/((n-1)*(k-1))
    return (msr-mse)/(msr+(k-1)*mse+k*(msc-mse)/n), (msr-mse)/(msr+(k-1)*mse)
dc=[c for c in cols if ann.Assay.get(c)=='DDC'][0]
ia,ic=icc(wa[dc].values,wb[dc].values); diff=(wb[dc]-wa[dc])
P(f"  DDC: Spearman ρ={rr[dc]:.2f}; ICC(绝对一致)={ia:.2f}, ICC(一致性)={ic:.2f}; 均值差(314−277)={diff.mean():+.2f} NPX, Bland–Altman 95% LoA [{diff.mean()-1.96*diff.std():+.2f},{diff.mean()+1.96*diff.std():+.2f}]")
P(f"  全部检测 Spearman ρ 中位 {rr.median():.2f} (IQR {rr.quantile(.25):.2f}–{rr.quantile(.75):.2f}); DDC 位于第 {100*(rr<rr[dc]).mean():.0f} 百分位")
# 55 RBD 蛋白与年龄/性别对照蛋白的技术一致性
sig=pd.read_csv(D+"/sig_rem.csv"); s55=set(sig[sig.fdr<0.05].OlinkID)
pcb=pd.read_csv(D+"/p314_posctrl_betas.csv"); agep=set(pcb[(pcb.exposure=='age')&(pcb.fdr_277<0.05)].OlinkID); saap=set(pcb[(pcb.exposure=='SAA')&(pcb.fdr_277<0.05)].OlinkID)
for nm,ss in [('55 RBD proteins',s55),('age-associated (positive control)',agep),('SAA-associated',saap)]:
    v=rr[[c for c in ss if c in rr.index]]; P(f"  {nm:34s} n={len(v)}: 跨项目同样本 ρ 中位 {v.median():.2f}")
pd.DataFrame({'OlinkID':rr.index,'rho_same_sample':rr.values}).to_csv(D+"/r3A_bridge_rho.csv",index=False)
json.dump(dict(n=len(ix),ddc_rho=rr[dc],ddc_icc_abs=ia,ddc_icc_cons=ic,ddc_meandiff=diff.mean(),ddc_loa=[diff.mean()-1.96*diff.std(),diff.mean()+1.96*diff.std()],median_rho=rr.median()),open(D+"/r3A_bridge.json","w"))

# ---------------------------------------------------------------- 5 Table 1 两队列
H("5. 两队列 Table 1 (R1/EIC)")
def summ(df,nm):
    out=[]
    for g in ['HC','Hyposmia','iRBD','PD']:
        x=df[df.group0==g]
        out.append(dict(cohort=nm,group=g,n=len(x),age=f"{x.age.mean():.1f} ({x.age.std():.1f})",female=f"{100*(x.SEX==0).mean():.0f}%",
            SAA_pos=f"{100*(x.SAA==1).sum()/max(1,x.SAA.notna().sum()):.0f}%",RBDSQ=f"{x.rem.mean():.1f} ({x.rem.std():.1f})",UPSIT=f"{x.upsit.mean():.1f} ({x.upsit.std():.1f})",
            UPDRS3=f"{x.updrs3_score.mean():.1f} ({x.updrs3_score.std():.1f})",SBR=f"{x.SBR.mean():.2f} ({x.SBR.std():.2f})",MoCA=f"{x.moca.mean():.1f} ({x.moca.std():.1f})",
            LEDD_pos=f"{100*(x.LEDD.fillna(0)>0).mean():.0f}%",DDC=f"{x.DDC.mean():+.2f} ({x.DDC.std():.2f})"))
    return out
T1=pd.DataFrame(summ(disc,'Discovery (Project 277)')+summ(spor,'Replication (Project 314)')); T1.to_csv(D+"/r3A_table1_two_cohorts.csv",index=False); P(T1.to_string())

# ---------------------------------------------------------------- 6 NSD 操作化成分 + 分类编码 + 各期 n
H("6. NSD 分期: 操作化成分 (SAA, SBR, UPDRS-III, UPDRS-II, MoCA) 与分类编码 (R2-W5, R1)")
nrows=[]; stage_tab=[]
for nm,df in [('Discovery',disc),('Replication',spor[spor.LEDD.fillna(0)==0])]:
    st=df.groupby('NSD_cat').DDC.agg(['count','mean','sem']).reset_index(); st['cohort']=nm; stage_tab.append(st); P(f"-- {nm} NSD 各期 DDC:\n{st.to_string(index=False)}")
    d=df.dropna(subset=['DDC','SAA','SBR','updrs3_score','updrs2_score','moca','age','SEX','NSD_cat']).copy()
    for v in ['SBR','updrs3_score','updrs2_score','moca']: d['z'+v]=(d[v]-d[v].mean())/d[v].std()
    base="DDC ~ SAA + zSBR + zupdrs3_score + zupdrs2_score + zmoca + age + SEX"
    full=smf.ols(base,d).fit(); P(f"   {nm} 五成分模型 n={int(full.nobs)}: "+" | ".join(f"{k} β={full.params[k]:+.3f} P={full.pvalues[k]:.1e}" for k in ['SAA','zSBR','zupdrs3_score','zupdrs2_score','zmoca']))
    for comp in ['SAA','zSBR','zupdrs3_score','zupdrs2_score','zmoca']:
        red=smf.ols(base.replace(' + '+comp,'').replace(comp+' + ',''),d).fit(); ft=full.compare_f_test(red)
        nrows.append(dict(cohort=nm,n=int(full.nobs),term=comp,beta=full.params[comp],p=full.pvalues[comp],unique_R2=full.rsquared-red.rsquared,F_p=ft[1]))
    cat=smf.ols(base+" + C(NSD_cat)",d).fit(); ft=cat.compare_f_test(full); dr2=cat.rsquared-full.rsquared
    def bs(seed):
        i=np.random.default_rng(seed).integers(0,len(d),len(d)); dd=d.iloc[i]
        try: return smf.ols(base+" + C(NSD_cat)",dd).fit().rsquared-smf.ols(base,dd).fit().rsquared
        except Exception: return np.nan
    b=np.array(Parallel(n_jobs=30)(delayed(bs)(k) for k in range(1000))); lo,hi=np.nanpercentile(b,[2.5,97.5])
    nrows.append(dict(cohort=nm,n=int(full.nobs),term='NSD stage (categorical) beyond 5 components',unique_R2=dr2,F_p=ft[1],ci_lo=lo,ci_hi=hi,df=int(ft[2])))
    P(f"   {nm}: 分类 NSD 在五成分之外 ΔR²={dr2:.4f} (bootstrap 95% CI {lo:.4f}–{hi:.4f}), F={ft[0]:.2f}, df={int(ft[2])}, P={ft[1]:.2f}")
    d23=d[d.NSD_cat.isin(['2b','3'])].copy(); d23['st3']=(d23.NSD_cat=='3').astype(int)
    m=smf.ols("DDC ~ st3 + zSBR + age + SEX",d23).fit(); ci=m.conf_int().loc['st3']
    nrows.append(dict(cohort=nm,n=int(m.nobs),term='stage 3 vs 2B (adj SBR)',beta=m.params.st3,p=m.pvalues.st3,ci_lo=ci[0],ci_hi=ci[1]))
    P(f"   {nm}: stage 3 vs 2B (校正 SBR) n={int(m.nobs)} β={m.params.st3:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] P={m.pvalues.st3:.1e}")
pd.DataFrame(nrows).to_csv(D+"/r3A_nsd_components.csv",index=False); pd.concat(stage_tab).to_csv(D+"/r3A_nsd_stage_table.csv",index=False)
P("\ndone"); LOG.close()
