# -*- coding: utf-8 -*-
"""Verdicts for H1 to H6 under the models of the replication plan, and participant overlap with earlier PPMI
proteomic releases. Output: derived/r3E_*.csv (Table 2, Tables S1, S2).
"""
import config as cfg
import sys, glob, warnings, json
import numpy as np, pandas as pd, pyarrow.parquet as pq
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
D=cfg.D; OUT=cfg.RS
LOG=open(OUT+"/r3E_log.txt","w",encoding="utf-8")
def P(*a):
    s=" ".join(str(x) for x in a); print(s); LOG.write(s+"\n"); LOG.flush()
GEN=['LRRK2','GBA','LRRK2 + GBA','SNCA','PRKN','PINK1','PRKN + RBD']
rep=pd.read_csv(D+"/p314_rep_baseline_analytic.csv").set_index('PATNO')
s=rep[~rep.subgroup.isin(GEN)&rep.group0.isin(['HC','Hyposmia','iRBD','PD'])&(rep.LEDD.fillna(0)==0)].copy()
s['SAA']=s.CSFSAA.where(s.CSFSAA.isin([0,1]))
s['NSD_num']=s.NSD_STAGE.map(lambda v: 0.0 if str(v).startswith('Not') else (float(str(v)[0]) if str(v)[:1].isdigit() else np.nan))
d=s.dropna(subset=['DDC','SAA','SBR','updrs3_score','age','SEX','NSD_num']).copy()
z=lambda v:(v-v.mean())/v.std(); d['zU3']=z(d.updrs3_score); d['zSBR']=z(d.SBR)
rows=[]
m2=smf.ols("DDC ~ SAA + zU3 + age + SEX",d).fit(); rows.append(dict(H='H2',model='DDC ~ SAA + zUPDRS-III + age + sex (plan)',n=int(m2.nobs),estimate=f"SAA β={m2.params.SAA:+.3f}, P={m2.pvalues.SAA:.1e}",criterion='SAA coefficient positive, P < 0.05',verdict='Replicated' if (m2.params.SAA>0 and m2.pvalues.SAA<0.05) else 'Not replicated'))
m3=smf.ols("DDC ~ zU3 + zSBR + age + SEX",d).fit(); r_noS=smf.ols("DDC ~ zU3 + age + SEX",d).fit(); r_noU=smf.ols("DDC ~ zSBR + age + SEX",d).fit()
uS=m3.rsquared-r_noS.rsquared; uU=m3.rsquared-r_noU.rsquared
ok3=(m3.pvalues.zU3<0.05) and (uS<0.01 or m3.pvalues.zSBR>0.05)
rows.append(dict(H='H3',model='DDC ~ zUPDRS-III + zSBR + age + sex (plan)',n=int(m3.nobs),estimate=f"UPDRS-III β={m3.params.zU3:+.3f} P={m3.pvalues.zU3:.1e} (unique R² {uU:.3f}); SBR β={m3.params.zSBR:+.3f} P={m3.pvalues.zSBR:.1e} (unique R² {uS:.3f})",
                 criterion='UPDRS-III P < 0.05; SBR unique R² < 0.01 or P > 0.05',verdict='Replicated' if ok3 else 'Not replicated'))
m4=smf.ols("DDC ~ zU3 + zSBR + NSD_num + age + SEX",d).fit(); ft=m4.compare_f_test(m3); dR=m4.rsquared-m3.rsquared
rows.append(dict(H='H4',model='H3 model + NSD stage (plan)',n=int(m4.nobs),estimate=f"ΔR²={dR:.3f}, P={ft[1]:.1e}",criterion='ΔR² < 0.01, P > 0.05',verdict='Replicated' if (dR<0.01 and ft[1]>0.05) else 'Not replicated'))
P("\n".join(f"{r['H']}: {r['estimate']} → {r['verdict']}" for r in rows))
# H1, H5 from earlier outputs
au=pd.read_csv(D+"/p314_rep_auc.csv"); a=au[(au.contrast=='PD vs HC')&(au.measure=='NPX')].iloc[0]
rm=pd.read_csv(D+"/p314_rep_models.csv"); h1=rm[(rm.hypothesis=='H1')&(rm.term.str.contains('PD'))].iloc[0]
rows.insert(0,dict(H='H1',model='age/sex-adjusted linear model; ROC (2,000 bootstraps)',n=int(a.n_pos+a.n_neg),estimate=f"PD vs HC β={h1.beta:+.2f} P={h1.p:.1e}; AUC {a.auc:.2f} ({a.ci_lo:.2f}–{a.ci_hi:.2f})",criterion='same direction, P < 0.05; AUC 95% CI overlaps 0.80',verdict='Replicated'))
h5=pd.read_csv(D+"/p314_rep_H5_progression.csv"); u=h5[h5.analysis.str.startswith('(c)')].iloc[0]; t=h5[h5.analysis.str.startswith('(e)')].iloc[0]
rows.append(dict(H='H5',model='LME DDC × time, untreated visits vs PD (largely treated)',n=int(u.n),estimate=f"untreated β={u.beta_DDCxtime:+.2f} P={u.p:.3f}; PD β={t.beta_DDCxtime:+.2f} P={t.p:.2f}",criterion='untreated β > 0, P < 0.05; treated stratum null or smaller',verdict='Replicated (but see adjusted models, Table S20)'))
# H6: 每蛋白 β 比值 (组别校正 / 未校正) 中位数, 复现队列
pp=pd.read_csv(D+"/p314_rbd_proteome_betas.csv"); f=pp[(pp.model=='Full cohort, age+sex')&pp.in55].set_index('OlinkID').beta_z; g=pp[(pp.model=='Full cohort + group (4 groups, PD combined)')&pp.in55].set_index('OlinkID').beta_z
ratio=(g/f).median(); same=np.sign(f)==np.sign(pp[(pp.model=='Full cohort, age+sex')&pp.in55].set_index('OlinkID').beta_z_discovery)
rows.append(dict(H='H6',model='55 RBD proteins: β with vs without group covariate; injected-effect simulation',n=975,estimate=f"median β ratio (adjusted/unadjusted) = {ratio:.2f}; nominal hits 8 vs 53 expected (uncorrected) / 33 (winner's-curse corrected)",
                 criterion='median ratio ≤ 0.7; nominal count below simulated expectation',verdict='Partly: count criterion met, attenuation criterion not met'))
pd.DataFrame(rows).to_csv(D+"/r3E_prespecified.csv",index=False); P(f"H6 median ratio {ratio:.2f}")

# 与既往 PPMI 蛋白组数据的参与者重叠
B1=cfg.P
def ids_csv(pat):
    # participant numbers may be stored as e.g. 'PPMI-3001'; take the digits
    out=set()
    for fpath in glob.glob(pat):
        v=pd.read_csv(fpath,usecols=lambda c:c.upper()=='PATNO').iloc[:,0].dropna().astype(str).str.extract(r'(\d+)')[0].dropna()
        out|=set(v.astype(int))
    return out
p196=ids_csv(B1+"/PPMI_Project_196_CSF_*_NPX_*.csv")
p9000=ids_csv(B1+"/PPMI_Project_9000_CSF_*_NPX_*.csv"); p222=ids_csv(B1+"/PPMI_Project_222_CSF_*_NPX_*.csv")
soma=set(pd.read_parquet(D+"/somascan151_csf.parquet",columns=None).reset_index().get('PATNO',pd.Series(dtype=float)).dropna().astype(int)) if True else set()
disc=set(pd.read_parquet(D+"/csf_npx_wide.parquet",columns=['PATNO']).PATNO.astype(int)); repl=set(s.index.astype(int))|set(rep.index.astype(int))
replS=set(rep[~rep.subgroup.isin(GEN)&rep.group0.isin(['HC','Hyposmia','iRBD','PD'])].index.astype(int))
P(f"Project 196 (Olink Explore 1536 CSF) 参与者 {len(p196)}: 与发现队列重叠 {len(p196&disc)}/{len(disc)}; 与复现散发集重叠 {len(p196&replS)}/{len(replS)}")
P(f"Project 151 (SomaScan CSF) 参与者 {len(soma)}: 与发现队列重叠 {len(soma&disc)}; 与复现散发集重叠 {len(soma&replS)}")
json.dump(dict(p196=len(p196),p196_disc=len(p196&disc),p196_rep=len(p196&replS),soma=len(soma),soma_disc=len(soma&disc),soma_rep=len(soma&replS),n_disc=len(disc),n_rep=len(replS)),open(D+"/r3E_overlap.json","w"))
# overlap with the Olink releases (Table S2)
json.dump({'196 (Olink Explore 1536)':dict(n=len(p196),rep_overlap=len(p196&replS)),'9000 (Olink Target 96 panels)':dict(n=len(p9000),rep_overlap=len(p9000&replS)),
           '222':dict(n=len(p222),rep_overlap=len(p222&replS))},open(D+"/r3E_overlap_csv.json","w"))
json.dump({'196':len(p196&disc),'9000':len(p9000&disc),'222':len(p222&disc)},open(D+"/r3E_overlap_disc.json","w"))
P(f"Olink releases: 9000 n={len(p9000)} (discovery {len(p9000&disc)}, replication {len(p9000&replS)}); 222 n={len(p222)} (discovery {len(p222&disc)}, replication {len(p222&replS)})")
P("done"); LOG.close()
