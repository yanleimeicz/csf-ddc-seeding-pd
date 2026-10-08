# -*- coding: utf-8 -*-
"""Group x SAA Tobit models; DDC and striatal binding among SAA-negative participants; NSD stage beyond its
components (Freedman-Lane permutation); PD vs control among technically reproducible proteins.
Output: derived/r4D_*.csv/json (Table 2, Fig. 4, Tables S2, S11, S17).
"""
import config as cfg
import sys, warnings, json
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc
import statsmodels.formula.api as smf
from statsmodels.tools.numdiff import approx_hess
from statsmodels.stats.multitest import multipletests
from scipy import stats, optimize
from joblib import Parallel, delayed
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
B=cfg.P; D=cfg.D; OUT=cfg.RS; NJ=30
LOG=open(OUT+"/r4D_log.txt","w",encoding="utf-8")
def P(*a):
    s=" ".join(str(x) for x in a); print(s); LOG.write(s+"\n"); LOG.flush()
def H(t): P("\n"+"="*78+"\n"+t+"\n"+"="*78)
GEN=['LRRK2','GBA','LRRK2 + GBA','SNCA','PRKN','PINK1','PRKN + RBD']; SBRc=['MIA_CAUDATE_L','MIA_CAUDATE_R','MIA_PUTAMEN_L','MIA_PUTAMEN_R']
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
H("A-R4a 组别 × SAA 单元: Tobit (删失于 LOD)")
trow=[]
for coh,f,base in [('Discovery',B+"/PPMI_Project_277_CSF_Screened_NPX_20260130.parquet",disc),('Replication',B+"/ppmi_proj314_csf_screened_extended_npx_20260116.parquet",spor)]:
    s=base.join(lodtab(f,base.index),rsuffix='_r').dropna(subset=['NPX','LOD','age','SEX','group0']); s=s[s.group0.isin(['HC','Hyposmia','iRBD','PD'])]
    s['cell']=np.where(s.group0=='HC','HC',s.group0+np.where(s.SAA==1,'_SAA+',np.where(s.SAA==0,'_SAA-','_na'))); s=s[~s.cell.str.endswith('_na')]
    cells=[c for c in sorted(s.cell.unique()) if c!='HC' and (s.cell==c).sum()>=10]; s=s[s.cell.isin(cells+['HC'])]
    X=np.column_stack([np.ones(len(s))]+[(s.cell==c).astype(float) for c in cells]+[s.age,s.SEX]); th,se,cf=tobit(s.NPX.values,X,s.LOD.values)
    for i,c in enumerate(cells):
        p=2*stats.norm.sf(abs(th[1+i]/se[1+i])); trow.append(dict(cohort=coh,cell=c,n=int((s.cell==c).sum()),tobit_beta=th[1+i],ci_lo=th[1+i]-1.96*se[1+i],ci_hi=th[1+i]+1.96*se[1+i],p=p,censored=cf))
        P(f"  {coh:11s} {c:16s} n={int((s.cell==c).sum()):3d} Tobit β={th[1+i]:+.3f} [{th[1+i]-1.96*se[1+i]:+.3f},{th[1+i]+1.96*se[1+i]:+.3f}] P={p:.1e}")
pd.DataFrame(trow).to_csv(D+"/r4D_tobit_cells.csv",index=False)
H("A-R4b SAA 阴性参与者 (排除 HC): DDC 与多巴胺转运体结合 (组内)")
sn=spor[(spor.SAA==0)&(spor.group0!='HC')].dropna(subset=['DDC','SBR','age','SEX']).copy()
hc5=spor.loc[spor.group0=='HC','SBR'].quantile(0.05); sn['deficit']=(sn.SBR<hc5).astype(int); sn['zSBR']=(sn.SBR-spor.loc[spor.group0=='HC','SBR'].mean())/spor.loc[spor.group0=='HC','SBR'].std()
out={}
for lab,f in [('continuous SBR (per control SD), + group','DDC ~ zSBR + C(group0) + age + SEX'),('deficit (SBR < 5th percentile of controls), + group','DDC ~ deficit + C(group0) + age + SEX')]:
    m=smf.ols(f,sn).fit(); k='zSBR' if 'zSBR' in f else 'deficit'; ci=m.conf_int().loc[k]
    out[lab]=dict(n=int(m.nobs),n_def=int(sn['deficit'].sum()),beta=m.params[k],ci_lo=ci[0],ci_hi=ci[1],p=m.pvalues[k])
    P(f"  {lab:52s} n={int(m.nobs)} (缺损 {int(sn['deficit'].sum())}): β={m.params[k]:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] P={m.pvalues[k]:.3f}")
P(f"  组成: {sn.group0.value_counts().to_dict()}; 参照: SAA− PD vs HC = 0.42 NPX")
json.dump(out,open(D+"/r4D_SAAneg_DAT.json","w"))
H("A-R5 NSD: 合并稀疏分期, adjusted ΔR², Freedman–Lane 零分布")
def stage(v):
    s=str(v)
    if s.startswith('Not'): return '0'
    if s[:1]=='1': return '1'
    if s in ('2a','2b','3'): return s
    if s[:1] in ('4','5'): return '4+'
    return np.nan
nrow=[]
for coh,df in [('Discovery',disc),('Replication',spor[spor.LEDD.fillna(0)==0])]:
    d=df.dropna(subset=['DDC','SAA','SBR','updrs3_score','updrs2_score','moca','age','SEX']).copy(); d['st']=d.NSD_STAGE.map(stage); d=d.dropna(subset=['st'])
    cnt=d.st.value_counts(); d=d[d.st.isin(cnt[cnt>=5].index)]
    for v in ['SBR','updrs3_score','updrs2_score','moca']: d['z'+v]=(d[v]-d[v].mean())/d[v].std()
    Z=np.column_stack([np.ones(len(d)),d.SAA,d.zSBR,d.zupdrs3_score,d.zupdrs2_score,d.zmoca,d.age,d.SEX]); S=pd.get_dummies(d.st,drop_first=True).values.astype(float); y=d.DDC.values
    def r2(X,y): b=np.linalg.lstsq(X,y,rcond=None)[0]; e=y-X@b; return 1-e@e/((y-y.mean())@(y-y.mean())),e
    r0,e0=r2(Z,y); r1,_=r2(np.column_stack([Z,S]),y); n=len(y); k0=Z.shape[1]; k1=k0+S.shape[1]
    adj=lambda r,k:1-(1-r)*(n-1)/(n-k); dadj=adj(r1,k1)-adj(r0,k0); obs=r1-r0; fit0=y-e0
    def fl(seed): yy=fit0+e0[np.random.default_rng(seed).permutation(n)]; a,_=r2(Z,yy); b,_=r2(np.column_stack([Z,S]),yy); return b-a
    nul=np.array(Parallel(n_jobs=NJ)(delayed(fl)(k) for k in range(2000))); p=(1+(nul>=obs).sum())/2001
    nrow.append(dict(cohort=coh,n=n,stages=dict(cnt[cnt>=5]),dR2=obs,adj_dR2=dadj,null_mean=nul.mean(),perm_p=p))
    P(f"  {coh:11s} n={n} 分期 {dict(cnt[cnt>=5])}: ΔR²={obs:.4f} (零分布均值 {nul.mean():.4f}), adjusted ΔR²={dadj:+.4f}, 置换 P={p:.3f}")
pd.DataFrame(nrow).to_csv(D+"/r4D_nsd_perm.csv",index=False)
H("A-R6 技术可复现蛋白子集 (同样本跨项目 ρ ≥ 0.5) 内的 PD vs HC; 桥接 ρ 的 CI")
br=pd.read_csv(D+"/r3A_bridge_rho.csv").set_index('OlinkID').rho_same_sample
pcb=pd.read_csv(D+"/p314_posctrl_betas.csv"); q=pcb[pcb.exposure=='PD_vs_HC'].set_index('OlinkID')
good=[o for o in q.index if br.get(o,np.nan)>=0.5]; P(f"  ρ ≥ 0.5 的蛋白 {len(good)} / {len(q)}")
srow={}
for coh,pc_ in [('Discovery','p_277'),('Replication','p_314')]:
    pv=q.loc[good,pc_].values; f=multipletests(pv,method='fdr_bh')[1]; hits=[ann.Assay.get(o) for o,ff in zip(good,f) if ff<0.05]
    srow[coh]=hits; P(f"  {coh}: 子集内 BH FDR<0.05 的蛋白: {hits}")
def fisher_ci(r,n): z=np.arctanh(r); s=1/np.sqrt(n-3); return np.tanh(z-1.96*s),np.tanh(z+1.96*s)
dd=br[[c for c in br.index if ann.Assay.get(c)=='DDC'][0]]; lo,hi=fisher_ci(dd,30)
sig=pd.read_csv(D+"/sig_rem.csv"); s55=set(sig[sig.fdr<0.05].OlinkID); age=set(pcb[(pcb.exposure=='age')&(pcb.fdr_277<0.05)].OlinkID)
def bootmed(ids,seed=1):
    # sorted(ids): resampling order does not depend on Python set hashing
    v=br.reindex(sorted(ids)).dropna().values; r=np.random.default_rng(seed); bs=[np.median(r.choice(v,len(v))) for _ in range(2000)]; return np.median(v),*np.percentile(bs,[2.5,97.5])
m55=bootmed(s55); mage=bootmed(age)
P(f"  DDC 同样本 ρ={dd:.2f} (95% CI {lo:.2f}–{hi:.2f}); 55 RBD 蛋白中位 ρ {m55[0]:.2f} ({m55[1]:.2f}–{m55[2]:.2f}); 年龄蛋白 {mage[0]:.2f} ({mage[1]:.2f}–{mage[2]:.2f}); 全部检测中 ρ ≥ 0.5 的比例 {100*(br>=0.5).mean():.0f}%")
json.dump(dict(n_good=len(good),hits=srow,ddc_rho=dd,ddc_ci=[lo,hi],rbd55=m55,age=mage,frac_ge05=float((br>=0.5).mean())),open(D+"/r4D_detect_bridge.json","w"))
P("done"); LOG.close()
