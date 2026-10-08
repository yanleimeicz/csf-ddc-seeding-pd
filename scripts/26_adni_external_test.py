# -*- coding: utf-8 -*-
"""External test in ADNI: CSF DDC (NULISAseq CNS Disease Panel 120) and alpha-synuclein SAA, following the written
analysis plan (Supplementary Note 3; E1 to E4, pre-specified sensitivity and post hoc analyses).
ADNI participant-level data are processed locally and only summary statistics are written.
Output: results/ADNI_NULISA/ (T1 to T4, adni_results.json; Table S19) and figures/Figure_R7_* (Fig. 5, Supplementary Fig. S5).
"""
import config as cfg
import sys, os, re, json, zipfile, warnings
import numpy as np, pandas as pd, pyreadr
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy import stats
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')
X=cfg.A; T=cfg.A+"/raw/ADNIMERGE2_tables"; D=cfg.D
OUT=cfg.RS+"/ADNI_NULISA"; os.makedirs(OUT,exist_ok=True); NJ=30
LOG=open(OUT+"/adni_log.txt","w",encoding="utf-8")
def P(*a):
    s=" ".join(str(x) for x in a); print(s); LOG.write(s+"\n"); LOG.flush()
def H(t): P("\n"+"="*78+"\n"+t+"\n"+"="*78)
rd=lambda n: list(pyreadr.read_r(T+f"/{n}.rda").values())[0]
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})

# ---------------- 数据 ----------------
nul=pd.read_csv(X+"/nulisa/BSHRI_PLA_CSF_NULISA_CNS_04Oct2026.csv",low_memory=False).drop_duplicates()
nul=nul[(nul.SampleType=='Sample')&nul.RID.notna()].copy(); nul['RID']=nul.RID.astype(int)
nul['plate']=nul.PlateID.str.extract(r'(Plate\d+)')[0]; nul['d']=pd.to_datetime(nul.EXAMDATE,errors='coerce')
saa=pd.read_csv(X+"/AMPRION_ASYN_SAA_04Oct2026.csv",low_memory=False)
saa['SAA']=saa.Result.map({'Detected-1':1,'Detected-2':1,'Not_Detected':0})
dx=rd('DXSUM'); dx['RID']=dx.RID.astype(int); dx=dx[['RID','VISCODE2','DIAGNOSIS']].dropna(subset=['VISCODE2']).drop_duplicates(['RID','VISCODE2'])
dm=rd('PTDEMOG'); dm['RID']=dm.RID.astype(int); dm=dm.sort_values('RID').groupby('RID').agg(sex=('PTGENDER','first'),dob=('PTDOB','first'),yob=('PTDOBYY','first')).reset_index()
ap=rd('APOERES'); ap['RID']=ap.RID.astype(int); ap=ap.dropna(subset=['GENOTYPE']).drop_duplicates('RID')
ap['e4']=ap.GENOTYPE.astype(str).str.count('4').clip(upper=2)
el=pd.read_csv(zipfile.ZipFile(X+"/download2.zip").open('UPENNBIOMK_ROCHE_ELECSYS_04Oct2026.csv'),low_memory=False)
num=lambda s: pd.to_numeric(s.astype(str).str.replace(r'[<>]','',regex=True),errors='coerce')
P(f"Elecsys 截断值: Aβ42 含 '>' {el.ABETA42.astype(str).str.contains('>').sum()}, '<' {el.ABETA42.astype(str).str.contains('<').sum()}; p-tau '<' {el.PTAU.astype(str).str.contains('<').sum()}, '>' {el.PTAU.astype(str).str.contains('>').sum()} (按截断值本身计)")
el['ab42']=num(el.ABETA42); el['ptau']=num(el.PTAU); el=el.dropna(subset=['ab42','ptau']).drop_duplicates(['RID','VISCODE2'])
med=pd.read_csv(zipfile.ZipFile(X+"/download3.zip").open('RECCMEDS_04Oct2026.csv'),low_memory=False,usecols=['RID','CMMED'])
DOPA=r'levodopa|carbidopa|sinemet|rytary|stalevo|madopar|pramipexole|mirapex|ropinirole|requip|rotigotine|neupro|rasagiline|azilect|selegiline|eldepryl|zelapar|safinamide|xadago|amantadine|entacapone|comtan|opicapone|apomorphine|benserazide'
dopa_ids=set(med[med.CMMED.astype(str).str.contains(DOPA,case=False)].RID.astype(int))

def build(matrix,target,qc_pass=True):
    """同次访视 NULISA–SAA 配对; 每人取最早一次; 同访视多样本取均值"""
    x=nul[(nul.SampleMatrixType==matrix)&(nul.Target==target)]
    if qc_pass: x=x[x.SampleQC=='passed']
    x=x.groupby(['RID','VISCODE2']).agg(y=('NPQ','mean'),plate=('plate','first'),d=('d','first'),PHASE=('PHASE','first'),qc=('SampleQC','first')).reset_index()
    p=x.merge(saa[['RID','VISCODE2','SAA','Result']].dropna(subset=['SAA']),on=['RID','VISCODE2']).sort_values('d').drop_duplicates('RID')
    p=p.merge(dx,on=['RID','VISCODE2'],how='left').merge(dm,on='RID',how='left').merge(ap[['RID','e4']],on='RID',how='left')
    p=p.merge(el[['RID','VISCODE2','ab42','ptau']],on=['RID','VISCODE2'],how='left')
    dob=pd.to_datetime(p.dob,errors='coerce'); p['age']=((p.d-dob).dt.days/365.25).where(dob.notna(),p.d.dt.year-pd.to_numeric(p.yob,errors='coerce'))
    p['male']=(p.sex=='Male').astype(int); p['lab42']=np.log(p.ab42); p['lptau']=np.log(p.ptau)
    p['dopa']=p.RID.isin(dopa_ids).astype(int); p['SAA']=p.SAA.astype(int)
    p['yc']=p.y-p.groupby('plate').y.transform('mean')+p.y.mean()          # 板中心化 (仅用于标准化、AUC 与作图)
    return p

def fit(f,d,term='SAA'):
    m=smf.ols(f,d).fit(); ci=m.conf_int().loc[term]
    return dict(n=int(m.nobs),n_pos=int(d.loc[m.model.data.row_labels,'SAA'].sum()),beta=m.params[term],ci_lo=ci[0],ci_hi=ci[1],p=m.pvalues[term]),m
def fmt(r,sd=None):
    s=f"n={r['n']} (SAA+ {r['n_pos']}) β={r['beta']:+.3f} NPQ [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] P={r['p']:.2g}"
    return s+(f"; {r['beta']/sd:+.2f} SD [{r['ci_lo']/sd:+.2f}, {r['ci_hi']/sd:+.2f}]" if sd else '')

res={}; rows=[]
def add(analysis,label,r,sd,kind):
    rows.append(dict(analysis=analysis,label=label,kind=kind,**r,beta_SD=r['beta']/sd,ci_lo_SD=r['ci_lo']/sd,ci_hi_SD=r['ci_hi']/sd))

# =====================================================================
H("样本")
a=build('CSF','DDC'); aw=build('CSF','DDC',qc_pass=False)
P(f"主分析样本 (QC passed, 同次访视): {len(a)} 人; SAA+ {a.SAA.sum()} (Detected-2 {int((a.Result=='Detected-2').sum())}), SAA− {(a.SAA==0).sum()}; 板 {a.plate.nunique()}")
P("诊断 × SAA:\n"+pd.crosstab(a.DIAGNOSIS.fillna('missing'),a.SAA,margins=True).to_string())
desc=a.groupby('SAA').agg(n=('RID','size'),age_mean=('age','mean'),age_sd=('age','std'),male_pct=('male','mean'),e4_carrier_pct=('e4',lambda s:(s>0).mean()),
    ab42_median=('ab42','median'),ptau_median=('ptau','median'),dopa_n=('dopa','sum')).T
desc.loc['male_pct']*=100; desc.loc['e4_carrier_pct']*=100
for g in ['CN','MCI','Dementia']: desc.loc[f'dx_{g}_n']=a.groupby('SAA').DIAGNOSIS.apply(lambda s:(s==g).sum())
desc.to_csv(OUT+"/T1_sample_characteristics.csv"); P(desc.round(1).to_string())
ref=a[(a.SAA==0)&(a.DIAGNOSIS=='CN')]; SD=ref.yc.std(); SDraw=ref.y.std()
P(f"标准化分母: SAA− CN 的 DDC SD (板中心化) = {SD:.3f} NPQ (n={len(ref)}); 未中心化 SD = {SDraw:.3f}")
res['sample']=dict(n=len(a),n_pos=int(a.SAA.sum()),n_neg=int((a.SAA==0).sum()),SD_ref=SD,SD_ref_raw=SDraw,n_ref=len(ref))

# =====================================================================
H("E1 (主要): DDC ~ SAA + age + sex + plate")
e1,m1=fit('y ~ SAA + age + male + C(plate)',a); P(fmt(e1,SD)); add('E1','E1: age, sex, plate',e1,SD,'primary')
res['E1']=dict(**e1,beta_SD=e1['beta']/SD,met=bool(e1['beta']>0 and e1['p']<0.05))
P(f"  判定 (β > 0 且 P < 0.05): {'达到' if res['E1']['met'] else '未达到'}")

H("E2: + 诊断 + log Aβ42 + log p-tau181")
a2=a.dropna(subset=['DIAGNOSIS','lab42','lptau'])
e2,m2=fit('y ~ SAA + age + male + C(plate) + C(DIAGNOSIS) + lab42 + lptau',a2); P(fmt(e2,SD)); add('E2','E2: + diagnosis, Aβ42, p-tau181',e2,SD,'primary')
for t in ['lab42','lptau','C(DIAGNOSIS)[T.Dementia]','C(DIAGNOSIS)[T.MCI]','age','male']:
    ci=m2.conf_int().loc[t]; P(f"   协变量 {t}: β={m2.params[t]:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] P={m2.pvalues[t]:.2g}")
res['E2']=dict(**e2,beta_SD=e2['beta']/SD,met=bool(e2['beta']>0 and e2['p']<0.05),
    covariates={t:dict(beta=m2.params[t],p=m2.pvalues[t]) for t in ['lab42','lptau','age','male']})
P(f"  判定: {'达到' if res['E2']['met'] else '未达到'}")

H("E3: 各诊断组内 SAA+ − SAA− (age, sex, plate); 判定: SAA+ ≥ 10 的各组方向一致为正")
res['E3']={}
for g in ['CN','MCI','Dementia']:
    s=a[a.DIAGNOSIS==g]; r,_=fit('y ~ SAA + age + male + C(plate)',s)
    au=roc_auc_score(s.SAA,s.yc); u=stats.mannwhitneyu(s.yc[s.SAA==1],s.yc[s.SAA==0])
    P(f"  {g:8s} {fmt(r,SD)}; 秩 AUC (板中心化) {au:.2f}, Mann–Whitney P={u.pvalue:.2g}")
    add('E3',f'E3: {g}',r,SD,'primary'); res['E3'][g]=dict(**r,beta_SD=r['beta']/SD,auc=au,p_mw=u.pvalue)
res['E3_met']=bool(all(v['beta']>0 for v in res['E3'].values() if v['n_pos']>=10)); P(f"  判定: {'达到 (三组均为正)' if res['E3_met'] else '未达到'}")
mi=smf.ols('y ~ SAA*C(DIAGNOSIS) + age + male + C(plate)',a.dropna(subset=['DIAGNOSIS'])).fit()
mr=smf.ols('y ~ SAA + C(DIAGNOSIS) + age + male + C(plate)',a.dropna(subset=['DIAGNOSIS'])).fit()
fi=mi.compare_f_test(mr); P(f"  [事后] SAA × 诊断交互 F 检验: F={fi[0]:.2f}, P={fi[1]:.3f}"); res['E3_interaction_p_posthoc']=fi[1]

# =====================================================================
H("敏感性分析 (预设)")
S={}
s=a[a.dopa==0]; r,_=fit('y ~ SAA + age + male + C(plate)',s); P("  排除多巴胺能药物 (n 去除 %d): "%a.dopa.sum()+fmt(r,SD)); add('Sens','Excluding dopaminergic drugs',r,SD,'sensitivity'); S['no_dopa']=r
r,_=fit('y ~ SAA + age + male + C(plate)',aw); P("  纳入 QC warning: "+fmt(r,SD)); add('Sens','Including QC-warning samples',r,SD,'sensitivity'); S['incl_qc_warning']=r
r,_=fit('y ~ SAA + age + male',a); P("  不调整板号: "+fmt(r,SD)); add('Sens','Without plate adjustment',r,SD,'sensitivity'); S['no_plate']=r
r,_=fit('y ~ SAA + age + male + C(plate) + C(PHASE)',a); P("  加 ADNI phase: "+fmt(r,SD)); add('Sens','Adding ADNI phase',r,SD,'sensitivity'); S['plus_phase']=r
a['rz']=stats.norm.ppf((a.y.rank()-0.5)/len(a)); r,_=fit('rz ~ SAA + age + male + C(plate)',a)
P(f"  秩 (正态分数) 回归: β={r['beta']:+.3f} SD [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] P={r['p']:.2g}"); S['rank_regression_SD']=r
au=roc_auc_score(a.SAA,a.yc); u=stats.mannwhitneyu(a.yc[a.SAA==1],a.yc[a.SAA==0]); P(f"  Mann–Whitney (板中心化): AUC={au:.3f}, P={u.pvalue:.2g}")
bs=Parallel(n_jobs=NJ)(delayed(lambda i: (lambda ix: roc_auc_score(a.SAA.values[ix],a.yc.values[ix]) if a.SAA.values[ix].std()>0 else np.nan)(np.random.default_rng(i).integers(0,len(a),len(a))))(i) for i in range(2000))
S['auc_overall']=dict(auc=au,ci_lo=np.nanpercentile(bs,2.5),ci_hi=np.nanpercentile(bs,97.5),p=u.pvalue); P(f"     AUC 95% CI (bootstrap 2,000) {S['auc_overall']['ci_lo']:.3f}–{S['auc_overall']['ci_hi']:.3f}")
H("敏感性/补充分析 (事后, post hoc)")
s=a[a.Result!='Detected-2']; r,_=fit('y ~ SAA + age + male + C(plate)',s); P("  [事后] 排除 Detected-2: "+fmt(r,SD)); add('PostHoc','Excluding SAA type 2 (post hoc)',r,SD,'posthoc'); S['excl_type2']=r
s=a2.dropna(subset=['e4']); r,_=fit('y ~ SAA + age + male + C(plate) + C(DIAGNOSIS) + lab42 + lptau + e4',s); P("  [事后] E2 + APOE ε4: "+fmt(r,SD)); add('PostHoc','E2 + APOE ε4 (post hoc)',r,SD,'posthoc'); S['E2_plus_apoe']=r
res['sensitivity']=S

# =====================================================================
H("E4 (次要): 与 PPMI 复现集比较 (效应以参照组 SD 为单位: ADNI = SAA− CN; PPMI = SAA− HC)")
GEN=['LRRK2','GBA','LRRK2 + GBA','SNCA','PRKN','PINK1','PRKN + RBD']
rep=pd.read_csv(D+"/p314_rep_baseline_analytic.csv")
spor=rep[~rep.subgroup.isin(GEN)&rep.group0.isin(['HC','Hyposmia','iRBD','PD'])].copy(); spor['SAA']=spor.CSFSAA.where(spor.CSFSAA.isin([0,1]))
hcref=spor[(spor.group0=='HC')&(spor.SAA==0)].DDC; SDp=hcref.std(); P(f"  PPMI 参照 SD (SAA− HC, n={hcref.notna().sum()}) = {SDp:.3f} NPX")
E4=[]
nh=spor[(spor.group0!='HC')&spor.SAA.notna()].dropna(subset=['DDC','age','SEX']).copy(); nh['SAA']=nh.SAA.astype(int)
r,_=fit('DDC ~ SAA + C(group0) + age + SEX',nh); E4.append(dict(cohort='PPMI',group='Pooled (hyposmia, iRBD, PD)',**r,beta_SD=r['beta']/SDp,ci_lo_SD=r['ci_lo']/SDp,ci_hi_SD=r['ci_hi']/SDp,auc=np.nan))
for g in ['Hyposmia','iRBD','PD']:
    s=nh[nh.group0==g]; r,_=fit('DDC ~ SAA + age + SEX',s)
    E4.append(dict(cohort='PPMI',group=g,**r,beta_SD=r['beta']/SDp,ci_lo_SD=r['ci_lo']/SDp,ci_hi_SD=r['ci_hi']/SDp,auc=roc_auc_score(s.SAA,s.DDC)))
E4.append(dict(cohort='ADNI',group='All (E1)',**e1,beta_SD=e1['beta']/SD,ci_lo_SD=e1['ci_lo']/SD,ci_hi_SD=e1['ci_hi']/SD,auc=S['auc_overall']['auc']))
for g,v in res['E3'].items(): E4.append(dict(cohort='ADNI',group=g,**{k:v[k] for k in ['n','n_pos','beta','ci_lo','ci_hi','p']},beta_SD=v['beta_SD'],ci_lo_SD=v['ci_lo']/SD,ci_hi_SD=v['ci_hi']/SD,auc=v['auc']))
E4=pd.DataFrame(E4); E4.to_csv(OUT+"/T3_E4_PPMI_comparison.csv",index=False)
for _,r in E4.iterrows(): P(f"  {r.cohort:4s} {r.group:28s} n={r.n} (SAA+ {r.n_pos}) {r.beta_SD:+.2f} SD [{r.ci_lo_SD:+.2f}, {r.ci_hi_SD:+.2f}] P={r.p:.2g}  AUC={r.auc:.2f}")
pp=E4[(E4.cohort=='PPMI')&(E4.group.str.startswith('Pooled'))].iloc[0]; ad=E4[(E4.cohort=='ADNI')&(E4.group=='All (E1)')].iloc[0]
se=lambda r:(r.ci_hi_SD-r.ci_lo_SD)/(2*1.96); z=(ad.beta_SD-pp.beta_SD)/np.hypot(se(ad),se(pp))
P(f"  ADNI − PPMI (合并) 差值 {ad.beta_SD-pp.beta_SD:+.2f} SD, z={z:.2f}, P={2*stats.norm.sf(abs(z)):.2g}; 比值 {ad.beta_SD/pp.beta_SD:.2f}")
res['E4']=dict(table=E4.to_dict('records'),diff_SD=ad.beta_SD-pp.beta_SD,p_diff=2*stats.norm.sf(abs(z)),ratio=ad.beta_SD/pp.beta_SD,SD_ppmi=SDp)

# =====================================================================
H("[事后] 全 panel CSF 扫描: 各靶点 ~ SAA + age + sex + plate (FDR); DDC 的排名")
cov=a[['RID','VISCODE2','SAA','age','male']]
tg=nul[(nul.SampleMatrixType=='CSF')&(nul.SampleQC=='passed')].groupby(['RID','VISCODE2','Target']).agg(y=('NPQ','mean'),plate=('plate','first'),det=('TargetDetectability',lambda s: pd.to_numeric(s.astype(str).str.rstrip('%'),errors='coerce').mean())).reset_index()
tg=tg.merge(cov,on=['RID','VISCODE2'])
def scan(t,g):
    g=g.dropna(subset=['y'])
    if len(g)<100 or g.y.std()==0: return None
    m=smf.ols('y ~ SAA + age + male + C(plate)',g).fit(); sdr=g[g.SAA==0].assign(yc=lambda d:d.y-d.groupby('plate').y.transform('mean')).yc.std()
    return dict(target=t,n=int(m.nobs),beta=m.params.SAA,t=m.tvalues.SAA,p=m.pvalues.SAA,beta_SD=m.params.SAA/sdr,detect_pct=g.det.mean())
sc=[r for r in Parallel(n_jobs=NJ)(delayed(scan)(t,g) for t,g in tg.groupby('Target')) if r and r['target']!='APOE4']
sc=pd.DataFrame(sc); sc['q']=multipletests(sc.p,method='fdr_bh')[1]; sc=sc.sort_values('p').reset_index(drop=True); sc['rank']=sc.index+1
sc.to_csv(OUT+"/T4_posthoc_panel_scan_CSF.csv",index=False)
det50=sc[sc.detect_pct>=50]
P(f"  靶点 {len(sc)} 个 (检出率 ≥ 50%: {len(det50)}); FDR < 0.05: {(sc.q<0.05).sum()} 个")
P(sc.head(15)[['rank','target','n','beta','beta_SD','p','q','detect_pct']].round(4).to_string(index=False))
dr=sc[sc.target=='DDC'].iloc[0]; P(f"  DDC 排名 {int(dr['rank'])}/{len(sc)}, q={dr.q:.2g}")
res['panel_scan']=dict(n_targets=len(sc),n_fdr=int((sc.q<0.05).sum()),ddc_rank=int(dr['rank']),ddc_q=dr.q,top10=sc.head(10)[['target','beta_SD','p','q']].to_dict('records'))

H("[事后] 血浆 DDC 与 SAA (同次访视)")
pl=build('PLASMA','DDC'); r,_=fit('y ~ SAA + age + male + C(plate)',pl); sdp=pl[(pl.SAA==0)&(pl.DIAGNOSIS=='CN')].yc.std()
P("  血浆: "+fmt(r,sdp)+f"; AUC (板中心化) {roc_auc_score(pl.SAA,pl.yc):.3f}")
res['plasma']=dict(**r,beta_SD=r['beta']/sdp,auc=roc_auc_score(pl.SAA,pl.yc))
both=a[['RID','VISCODE2','yc']].merge(pl[['RID','VISCODE2','yc']],on=['RID','VISCODE2'],suffixes=('_csf','_pla'))
rho=stats.spearmanr(both.yc_csf,both.yc_pla); P(f"  同次访视 CSF–血浆 DDC Spearman ρ={rho.statistic:.2f} (n={len(both)}, P={rho.pvalue:.2g})")
res['plasma']['rho_csf_plasma']=rho.statistic; res['plasma']['n_both']=len(both)

pd.DataFrame(rows).to_csv(OUT+"/T2_E1_E3_sensitivity.csv",index=False)
json.dump(res,open(OUT+"/adni_results.json","w",encoding="utf-8"),indent=1,default=float,ensure_ascii=False)

# =====================================================================

FIGD=cfg.F
def _save(fig,stem_res,stem_pub):
    fig.savefig(OUT+f"/{stem_res}.png",dpi=300,bbox_inches='tight'); fig.savefig(OUT+f"/{stem_res}.pdf",bbox_inches='tight')
    fig.savefig(FIGD+f"/{stem_pub}.pdf",bbox_inches='tight'); fig.savefig(FIGD+f"/{stem_pub}.tiff",dpi=500,bbox_inches='tight',pil_kwargs={'compression':'tiff_lzw'})
# 图 1: (a) 各诊断组 × SAA 的板中心化 DDC 分布 (箱线图, 只显示分位数, 不画个体点); (b) 森林图 (SD 单位)
fig=plt.figure(figsize=(7.4,4.0)); gs=fig.add_gridspec(1,2,width_ratios=[0.95,1.25],wspace=1.05)
ax=fig.add_subplot(gs[0]); pos=[]; data=[]; cols=[]
for i,g in enumerate(['CN','MCI','Dementia']):
    for j,sv in enumerate([0,1]):
        data.append(a[(a.DIAGNOSIS==g)&(a.SAA==sv)].yc.values); pos.append(i*2.4+j*0.9); cols.append('#9e9e9e' if sv==0 else '#c0392b')
bp=ax.boxplot(data,positions=pos,widths=0.7,showfliers=False,whis=(5,95),patch_artist=True,medianprops=dict(color='k',lw=1))
for b,c in zip(bp['boxes'],cols): b.set_facecolor(c); b.set_alpha(0.75)
ax.set_xticks([i*2.4+0.45 for i in range(3)]); ax.set_xticklabels([f"{g}\n(n {(a[(a.DIAGNOSIS==g)].SAA==0).sum()}/{(a[(a.DIAGNOSIS==g)].SAA==1).sum()})" for g in ['CN','MCI','Dementia']])
ax.set_ylabel('CSF DDC, NPQ (plate-centred)'); ax.set_title('a  ADNI: CSF DDC by SAA status',loc='left',fontweight='bold')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc='#9e9e9e',alpha=.75,label='SAA−'),Patch(fc='#c0392b',alpha=.75,label='SAA+')],frameon=False,loc='upper left',fontsize=7,ncol=2)
ax.set_ylim(top=ax.get_ylim()[1]+0.25)
ax.text(0,-0.2,'n shown as SAA−/SAA+. Boxes: median and IQR;\nwhiskers: 5th–95th percentiles.',transform=ax.transAxes,fontsize=6.5,va='top')
ax=fig.add_subplot(gs[1]); fr=pd.DataFrame(rows); pe=E4[E4.cohort=='PPMI']
blocks=[('ADNI, pre-specified',fr[fr.kind=='primary']),('ADNI, pre-specified sensitivity',fr[fr.kind=='sensitivity']),('ADNI, post hoc',fr[fr.kind=='posthoc']),
        ('PPMI replication set (Olink)',pd.DataFrame(dict(label=['Pooled (group-adjusted)']+list(pe.group[1:]),beta_SD=pe.beta_SD.values,ci_lo_SD=pe.ci_lo_SD.values,ci_hi_SD=pe.ci_hi_SD.values,kind='ppmi')))]
cmap={'primary':'#c0392b','sensitivity':'#e08e7f','posthoc':'#aaaaaa','ppmi':'#2e6da4'}
yy=0; yt=[]; yl=[]
for title,b in blocks:
    ax.text(-0.02,yy,title,transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=6.7,fontweight='bold'); yy-=1
    for _,r in b.iterrows():
        ax.errorbar(r.beta_SD,yy,xerr=[[r.beta_SD-r.ci_lo_SD],[r.ci_hi_SD-r.beta_SD]],fmt='o',color=cmap[r.kind],ms=4,capsize=2,lw=1)
        yt.append(yy); yl.append(r.label.replace('E1: age, sex, plate','E1: all (age, sex, plate)').replace(' (post hoc)','')); yy-=1
    yy-=0.4
ax.set_yticks(yt); ax.set_yticklabels(yl,fontsize=6.5); ax.tick_params(axis='y',length=0); ax.set_ylim(yy+0.4,0.8)
ax.axvline(0,color='k',lw=0.6,ls=':')
ax.set_xlabel('SAA+ minus SAA−, SD units\n(ADNI: SAA− CN; PPMI: SAA− controls)'); ax.set_title('b  Effect estimates with 95% CI',loc='left',fontweight='bold')
_save(fig,'Fig_ADNI_1_DDC_SAA','Figure_R7_Fig5_ADNI'); plt.close(fig)
# 图 2: [事后] 全 panel 火山图 (标签用引线排在右侧, 避免重叠)
fig,ax=plt.subplots(figsize=(3.9,3.2))
c=np.where(sc.target=='DDC','#c0392b',np.where(sc.q<0.05,'#555555','#c8c8c8'))
ax.scatter(sc.beta_SD,-np.log10(sc.p),c=c,s=12,lw=0,zorder=3)
ax.annotate(f"DDC (q = {dr.q:.3f})",(dr.beta_SD,-np.log10(dr.p)),fontsize=6.5,xytext=(-4,4),textcoords='offset points',ha='right',color='#c0392b')
oth=sc[sc.target!='DDC'].head(8).sort_values('p'); ys=np.linspace(3.25,1.75,len(oth))
for (_,r),yv in zip(oth.iterrows(),ys):
    ax.annotate(r.target,(r.beta_SD,-np.log10(r.p)),xytext=(0.43,yv),textcoords='data',fontsize=6,va='center',
                arrowprops=dict(arrowstyle='-',lw=0.4,color='#888888',shrinkA=0,shrinkB=2))
ax.set_xlim(right=0.56)
ax.set_xlabel('SAA+ minus SAA− (SD of SAA− participants)'); ax.set_ylabel('−log$_{10}$ P')
ax.set_title(f'CSF NULISA panel ({len(sc)} targets), post hoc\nonly DDC passes FDR 5%',fontsize=8)
_save(fig,'Fig_ADNI_2_panel_scan_posthoc','Figure_R7_FigS5_ADNI_panel_scan'); plt.close(fig)
P("\n输出目录: "+OUT)
