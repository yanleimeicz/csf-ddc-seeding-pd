# -*- coding: utf-8 -*-
"""Discovery-set figures. Output: figures/Figure1_cohort, Figure2_RBD_channel, Figure3_axis_validation,
Figure4_DDC, FigureS1_GSEA_discovery (Fig. 1, Fig. 2, Supplementary Figs. S1 to S3).
"""
import config as cfg
import os, re, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats
from adjustText import adjust_text
LOG10=r'$-\log_{10}\,P$'
D=cfg.D; FIG=cfg.F; os.makedirs(FIG,exist_ok=True)

# ---- 全局样式 (Cell Press) ----
plt.rcParams.update({
 'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans','Liberation Sans'],
 'svg.fonttype':'none','pdf.fonttype':42,'font.size':7,
 'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':0.8,
 'legend.frameon':False,'xtick.major.width':0.8,'ytick.major.width':0.8,
 'axes.labelsize':8,'axes.titlesize':8,
 'text.color':'black','axes.labelcolor':'black','axes.titlecolor':'black',
 'xtick.color':'black','ytick.color':'black','axes.edgecolor':'black',
 'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold','mathtext.default':'regular'})
M_=lambda v,f: format(v,f).replace('-','−')   # QA: 数字中的减号统一为 U+2212
def sci(p):
    e=int(np.floor(np.log10(p))); m=p/10**e; return f'{m:.1f} × 10$^{{{e}}}$'
C={'HC':'#767676','Hyposmia':'#3775BA','PD_brainfirst':'#0F4D92',
   'iRBD':'#B64342','PD_bodyfirst':'#E9A6A1'}
LAB={'HC':'HC','Hyposmia':'Hyposmia','PD_brainfirst':'PD brain-first',
     'iRBD':'iRBD','PD_bodyfirst':'PD body-first'}
ORDER=['HC','Hyposmia','iRBD','PD_brainfirst','PD_bodyfirst']
RED='#B64342'; BLUE='#2c6fbb'; GREY='#BFBFBF'
def panel(ax,l,x=-0.14,y=1.04): ax.text(x,y,l.lower(),transform=ax.transAxes,fontsize=10,fontweight='bold',va='bottom',ha='left')
def save(fig,name):
    for ext,dpi in [('svg',None),('pdf',None),('png',600),('tiff',500)]:
        kw={'pil_kwargs':{'compression':'tiff_lzw'}} if ext=='tiff' else {}
        fig.savefig(os.path.join(FIG,f'{name}.{ext}'),dpi=dpi,bbox_inches='tight',**kw)
    plt.close(fig); print('saved',name)

clin=pd.read_csv(D+"/analytic_clinical.csv").set_index('PATNO')
def nsd_num(v):
    s=str(v); return 0.0 if s.startswith('Not') else (float(s[0]) if s[0].isdigit() else np.nan)
clin['NSD_num']=clin['NSD_STAGE'].map(nsd_num); clin['SAA_pos']=(clin['CSFSAA']==1)

# ================= FIGURE 1 =================
fig=plt.figure(figsize=(7.2,4.6)); gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.55,wspace=0.45)
wide_idx=pd.read_parquet(D+"/csf_npx_wide.parquet",columns=["PATNO"])["PATNO"].astype(int).values  # 限定733人CSF蛋白组子集,与Table 1一致
sub=clin[clin.group.isin(ORDER)&clin.index.isin(wide_idx)]
# A 队列schematic (连续谱条)
axA=fig.add_subplot(gs[0,0]); axA.axis('off'); panel(axA,'A',x=-0.05,y=1.04)
ns={g:(sub.group==g).sum() for g in ORDER}
axA.text(0.1,1.04,'Prodromal-to-manifest continuum',transform=axA.transAxes,fontsize=8,va='bottom')
ypos=np.arange(len(ORDER))[::-1]
for i,g in enumerate(ORDER):
    axA.barh(ypos[i],1,color=C[g],height=0.7)
    axA.text(1.05,ypos[i],f'{LAB[g]} (n={ns[g]})',va='center',fontsize=6.5)
axA.set_xlim(0,3.2); axA.set_ylim(-0.6,len(ORDER)-0.2)
# B 通道量表
axB=fig.add_subplot(gs[0,1]); panel(axB,'B')
chans=[('rem','RBDSQ'),('scopa','SCOPA-AUT'),('upsit','UPSIT')]
xx=np.arange(len(ORDER)); w=0.26
for k,(col,nm) in enumerate(chans):
    m=[sub[sub.group==g][col].mean() for g in ORDER]
    se=[sub[sub.group==g][col].sem() for g in ORDER]
    axB.bar(xx+(k-1)*w,m,w,yerr=se,label=nm,error_kw={'elinewidth':0.6,'capsize':1.5},
            color=['#4C72B0','#DD8452','#55A868'][k],edgecolor='k',linewidth=0.4)
axB.set_xticks(xx); axB.set_xticklabels([LAB[g] for g in ORDER],rotation=40,ha='right',fontsize=6)
axB.set_ylabel('Score'); axB.legend(fontsize=6,ncol=1,loc='upper right'); axB.set_title('Prodromal channels',fontsize=8,loc='left')
# C SAA/NSD阳性比例
axC=fig.add_subplot(gs[0,2]); panel(axC,'C')
saa=[100*(sub[(sub.group==g)&sub.CSFSAA.isin([0,1])].CSFSAA==1).mean() for g in ORDER]   # QA: 分母只计有结果者
nsd=[100*(sub[(sub.group==g)&sub.NSD_Status.notna()].NSD_Status==1).mean() for g in ORDER]
axC.bar(xx-0.2,saa,0.4,label='SAA+',color='#8C6BB1',edgecolor='k',linewidth=0.4)
axC.bar(xx+0.2,nsd,0.4,label='NSD+',color='#C994C7',edgecolor='k',linewidth=0.4)
axC.set_xticks(xx); axC.set_xticklabels([LAB[g] for g in ORDER],rotation=40,ha='right',fontsize=6)
axC.set_ylabel('% positive'); axC.set_ylim(0,128); axC.set_yticks([0,25,50,75,100]); axC.legend(fontsize=6,loc='upper left',ncol=2,handlelength=1.2,columnspacing=0.8,bbox_to_anchor=(0.0,1.02)); axC.set_title('Biological anchoring',fontsize=8,loc='left')
# D DDC预览 by group (箱线)
axD=fig.add_subplot(gs[1,:]); panel(axD,'D',x=-0.05)
dd=pd.read_csv(D+"/ddc_deepdive.csv")
data=[dd[dd.group==g]['DDC'].dropna() for g in ORDER]
bp=axD.boxplot(data,positions=xx,widths=0.6,patch_artist=True,showfliers=False,
               medianprops={'color':'k','linewidth':1})
for patch,g in zip(bp['boxes'],ORDER): patch.set_facecolor(C[g]); patch.set_edgecolor('k'); patch.set_linewidth(0.5)
for i,g in enumerate(ORDER):
    yv=dd[dd.group==g]['DDC'].dropna(); axD.scatter(np.random.normal(xx[i],0.06,len(yv)),yv,s=1.5,c='k',alpha=0.25,zorder=3)
H,pk=stats.kruskal(*[d.values for d in data])
axD.set_xticks(xx); axD.set_xticklabels([LAB[g] for g in ORDER],fontsize=7)
axD.set_ylabel('CSF DDC (NPX)'); axD.set_title(f'CSF DOPA decarboxylase across the continuum (Kruskal–Wallis P = {sci(pk)})',fontsize=8,loc='left')
save(fig,'Figure1_cohort')

# ================= FIGURE 2 (RBD通道: 混杂 vs 效能, 修回版) =================
sig=pd.read_csv(D+"/sig_rem.csv")
fig=plt.figure(figsize=(7.2,7.4)); gs=gridspec.GridSpec(3,2,figure=fig,width_ratios=[1.02,1.0],height_ratios=[1,1,1.05],hspace=0.75,wspace=0.62)
from matplotlib.lines import Line2D
axA=fig.add_subplot(gs[:2,0]); panel(axA,'A',x=-0.34,y=1.01)
sig['nl']=-np.log10(sig['p']); s=sig.fdr<0.05; up=s&(sig.beta_z>0); dn=s&(sig.beta_z<0)
top=sig[s].reindex(sig[s]['nl'].sort_values(ascending=False).index).head(18).sort_values('beta_z')
yy=np.arange(len(top)); cols=[RED if b>0 else BLUE for b in top.beta_z]
axA.hlines(yy,0,top.beta_z,color=cols,lw=1.5,alpha=0.55,zorder=1)
axA.scatter(top.beta_z,yy,c=cols,s=28+(top.nl/top.nl.max())*60,edgecolors='white',linewidths=0.4,zorder=3)
axA.axvline(0,c='k',lw=0.7)
axA.set_yticks(yy); axA.set_yticklabels(top.gene.astype(str).str.replace('_','/'),fontsize=6.4); axA.tick_params(axis='y',length=0)
axA.set_ylim(-0.7,len(top)-0.3); axA.set_xlabel('Standardised β (association with RBDSQ)')
_n2a=int(clin.loc[clin.index.isin(wide_idx),'rem'].notna().sum())   # QA: n 由数据计算
axA.set_title(f'CSF proteins associated with RBD severity (n = {_n2a})',fontsize=8,loc='left')
axA.legend(handles=[Line2D([0],[0],marker='o',ls='',mfc=RED,mec='white',ms=6,label='higher with RBD'),
                    Line2D([0],[0],marker='o',ls='',mfc=BLUE,mec='white',ms=6,label='lower with RBD')],
           title='dot size: −log$_{10}$ P',fontsize=6.3,title_fontsize=6,loc='lower right',handletextpad=0.3,labelspacing=0.4)
# B channel bar
axB=fig.add_subplot(gs[0,1]); panel(axB,'B')
chn=['RBDSQ','SCOPA-AUT','SCOPA-GI','UPSIT']; nsig=[55,1,1,1]
axB.bar(chn,nsig,color=['#B64342','#999','#999','#999'],edgecolor='k',linewidth=0.5)
for i,v in enumerate(nsig): axB.text(i,v+1,str(v),ha='center',fontsize=7)
axB.set_ylabel('# proteins (FDR<0.05)'); axB.set_title('Signal by channel',fontsize=8,loc='left')
axB.set_xticklabels(chn,rotation=25,ha='right',fontsize=6.5)
# C 五种模型下 55 蛋白的检出: FDR<0.05 / 名义 P<0.05 / 若效应为真的模拟期望
cm=pd.read_csv(D+"/rev2_confounding_models.csv"); ps=pd.read_csv(D+"/rev2_power_simulation.csv")
sim=ps.groupby('stratum').detected.agg(['mean',lambda x:np.percentile(x,2.5),lambda x:np.percentile(x,97.5)]); sim.columns=['m','lo','hi']
axC=fig.add_subplot(gs[1,1]); panel(axC,'C')
labs=['Full\ncohort','+ group','+ SAA\n+ NSD','Within\nSAA+','Within\nPD']; xx=np.arange(5); w=0.36
axC.bar(xx-w/2,cm.sig_in55,w,color='#B64342',edgecolor='k',linewidth=0.5,label='FDR < 0.05')
axC.bar(xx+w/2,cm.nominal_in55,w,color='#E9A6A1',edgecolor='k',linewidth=0.5,label='nominal P < 0.05')
for i,(a1,a2) in enumerate(zip(cm.sig_in55,cm.nominal_in55)): axC.text(i-w/2,a1+1.2,str(int(a1)),ha='center',fontsize=6); axC.text(i+w/2,a2+1.2,str(int(a2)),ha='center',fontsize=6)
for i,k in [(3,'SAA+'),(4,'PD')]:
    axC.errorbar(i-w/2-0.27,sim.loc[k,'m'],yerr=[[sim.loc[k,'m']-sim.loc[k,'lo']],[sim.loc[k,'hi']-sim.loc[k,'m']]],fmt='D',ms=3.2,c='k',capsize=2,lw=0.8,zorder=4)
axC.plot([],[],'kD',ms=3.2,label='expected at FDR < 0.05 if\neffects true (95% range)')
axC.set_xticks(xx); axC.set_xticklabels(labs,fontsize=6.2); axC.set_ylabel('# of the 55 proteins'); axC.set_ylim(0,100)
axC.legend(fontsize=6,loc='upper left',handlelength=1.2,ncol=2,columnspacing=0.8,bbox_to_anchor=(-0.02,1.02)); axC.set_title('Confounding versus power',fontsize=7.5,loc='left')
# D β 衰减: 全队列 β vs (+group) 与 (within PD)
at=pd.read_csv(D+"/rev2_beta_attenuation_55.csv")
axD=fig.add_subplot(gs[2,:]); panel(axD,'D',x=-0.09)
lim=max(abs(at.beta_full).max(),abs(at.beta_pd).max())*1.15
axD.plot([-lim,lim],[-lim,lim],c='#bbb',lw=0.8,ls='--',zorder=1); axD.axhline(0,c='k',lw=0.5); axD.axvline(0,c='k',lw=0.5)
axD.scatter(at.beta_full,at.beta_saa,s=16,c='#E9A6A1',edgecolors='k',linewidths=0.3,label=f'within SAA+ (standardised median ratio {at.ratio_saa.median():.2f})',zorder=3)
axD.scatter(at.beta_full,at.beta_full_adjgroup,s=16,c='#8C6BB1',edgecolors='k',linewidths=0.3,label=f'full cohort + group (standardised median ratio {at.ratio_adjgroup.median():.2f})',zorder=3)
axD.scatter(at.beta_full,at.beta_pd,s=16,c='#CFCECE',edgecolors='k',linewidths=0.3,label=f'within PD (standardised median ratio {at.ratio_pd.median():.2f})',zorder=3)
axD.set_xlabel('Standardised β, full cohort (age, sex adjusted)'); axD.set_ylabel('Standardised β,\nadjusted or stratified')
axD.set_xlim(-lim,lim); axD.set_ylim(-lim,lim); axD.legend(fontsize=6,loc='upper left',handletextpad=0.3)
axD.set_title('Effect-size attenuation of the 55 RBD-associated proteins (dashed line = no attenuation)',fontsize=7.5,loc='left')
save(fig,'Figure2_RBD_channel')

# ================= FIGURE 3 (replication & orthogonal validation, 修回版) =================
g3=pd.read_csv(D+"/gsea_iRBD_vs_Hyposmia.csv"); g3['NES']=g3['NES'].astype(float); g3['FDR q-val']=g3['FDR q-val'].astype(float)
def clean_term(t):
    t=re.sub(r'.*__','',str(t)); t=re.sub(r'\s*\(GO:\d+\)','',t); t=re.sub(r'\s*R-HSA-\d+','',t)
    return (t[:38]+'…') if len(t)>39 else t
g3['T']=g3['Term'].map(clean_term)
# ---- Supplementary Fig. S1: GSEA (discovery only) ----
figS=plt.figure(figsize=(4.4,3.0)); axS=figS.add_subplot(111)
top=g3[g3['FDR q-val']<0.25].reindex(g3[g3['FDR q-val']<0.25]['NES'].abs().sort_values(ascending=False).index)   # QA: 与 Table S4 同一规则 (全部 q<0.25 术语)
axS.barh(range(len(top))[::-1],top['NES'],color=[RED if v>0 else BLUE for v in top['NES']],edgecolor='k',linewidth=0.5)
axS.set_yticks(range(len(top))[::-1]); axS.set_yticklabels(top['T'],fontsize=6.3); axS.axvline(0,c='k',lw=0.6)
axS.set_xlabel('NES (>0: higher in iRBD)'); axS.set_title('Discovery GSEA, iRBD vs Hyposmia CSF (1,000 permutations; not replicated)',fontsize=7,loc='left')
save(figS,'FigureS1_GSEA_discovery')
# ---- main Fig. 3 ----
fig=plt.figure(figsize=(7.2,6.0)); gs=gridspec.GridSpec(2,2,figure=fig,hspace=0.62,wspace=0.5)
# A Olink Target 独立面板复现
axA=fig.add_subplot(gs[0,0]); panel(axA,'A',x=-0.30)
labs=['DDC ~ NSD stage','DDC ~ RBDSQ']; logp=[14.0,4.1]
axA.barh([1,0],logp,color='#2E9E44',edgecolor='k',linewidth=0.5,height=0.5)
axA.set_yticks([1,0]); axA.set_yticklabels(labs,fontsize=6.5); axA.axvline(-np.log10(0.05),ls='--',c='#888',lw=0.6)
axA.set_xlabel(r'$-\log_{10}\,P$'); axA.set_title('Separate Olink Target run\n(same cohort, overlapping participants)',fontsize=7.5,loc='left')
# B SomaScan vs Olink 跨平台一致性分布, DDC 标记
cc=pd.read_csv(D+"/rev2_soma_olink_concordance.csv"); rD=cc.loc[cc.gene=='DDC','r'].values[0]
axB=fig.add_subplot(gs[0,1]); panel(axB,'B',x=-0.22)
axB.hist(cc.r,bins=40,color='#BFBFBF',edgecolor='k',linewidth=0.3)
axB.axvline(rD,c=RED,lw=1.4); axB.text(rD-0.03,0.92*len(cc)/8,f"DDC r = {M_(rD,'+.2f')}",color=RED,fontsize=6.5,ha='right')
axB.axvline(cc.r.median(),c='k',lw=0.8,ls='--'); axB.text(cc.r.median()+0.03,0.75*len(cc)/8,f'median r = {cc.r.median():.2f}',fontsize=6)
axB.set_xlabel('Olink HT vs SomaScan r (n = 369)'); axB.set_ylabel('# proteins (1,555 shared)')
axB.set_title('Orthogonal aptamer platform (SomaScan CSF)',fontsize=7.5,loc='left')
# C 尿液质谱 DDC
ub=pd.read_csv(D+"/rev2_urine_ddc.csv"); ub=ub[ub.group.isin(['HC','PD_brainfirst','PD_bodyfirst'])].dropna(subset=['DDC_urine'])
ub['g']=np.where(ub.group=='HC','HC','PD')
import json as _j; ur=_j.load(open(D+"/rev2_urine_summary.json"))
axC=fig.add_subplot(gs[1,0]); panel(axC,'C',x=-0.30)
dat=[ub[ub.g=='HC'].DDC_urine.values,ub[ub.g=='PD'].DDC_urine.values]
bp=axC.boxplot(dat,positions=[0,1],widths=0.55,patch_artist=True,showfliers=False,medianprops={'color':'k'})
for patch,cc_ in zip(bp['boxes'],[C['HC'],C['PD_brainfirst']]): patch.set_facecolor(cc_); patch.set_edgecolor('k'); patch.set_linewidth(0.5)
for i,v in enumerate(dat): axC.scatter(np.random.normal(i,0.06,len(v)),v,s=2,c='k',alpha=0.25,zorder=3)
axC.set_xticks([0,1]); axC.set_xticklabels([f'HC (n={len(dat[0])})',f'PD (n={len(dat[1])})'],fontsize=6.5)
axC.set_ylabel('Urine DDC (log2 LFQ intensity)'); axC.set_title(f"Urine mass spectrometry: AUC {ur['PD vs HC']['auc']:.2f}, P = {ur['PD vs HC']['p_adj']:.2f}",fontsize=7.5,loc='left')
# D 独立炎症面板: 趋化因子 iRBD vs hyposmia forest
ch=pd.read_csv(D+"/validation_chemokines_INF.csv").sort_values('beta_iRBD')
z=pd.Series(stats.norm.ppf(1-ch.p.values/2),index=ch.index).replace(0,np.nan); se=(ch.beta_iRBD.abs()/z).fillna(ch.beta_iRBD.abs())
axD=fig.add_subplot(gs[1,1]); panel(axD,'D',x=-0.22)
yy=np.arange(len(ch)); axD.errorbar(ch.beta_iRBD,yy,xerr=1.96*se,fmt='o',ms=2.6,c='#555',ecolor='#999',elinewidth=0.6,capsize=0)
axD.axvline(0,c='k',lw=0.7); axD.set_yticks(yy); axD.set_yticklabels(ch.chemokine,fontsize=6); axD.tick_params(axis='y',length=0)
axD.set_xlabel('β, iRBD vs hyposmia (95% CI)'); axD.set_title('Independent inflammation panel: chemokines\n(23 iRBD vs 16 hyposmia; all FDR > 0.8)',fontsize=7.5,loc='left')
save(fig,'Figure3_axis_validation')

# ================= FIGURE 4 (DDC dual biomarker, money) =================
from sklearn.metrics import roc_curve, roc_auc_score
dd=pd.read_csv(D+"/ddc_deepdive.csv"); pc=pd.read_csv(D+"/ddc_plasma_vs_csf.csv")
mo=pd.read_csv(D+"/module5_score_outcomes.csv")
fig=plt.figure(figsize=(7.35,5.0)); gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.6,wspace=0.8,width_ratios=[1,1,1.3])
# A group box (hero-ish)
axA=fig.add_subplot(gs[0,0]); panel(axA,'A')
data=[dd[dd.group==g]['DDC'].dropna() for g in ORDER]
bp=axA.boxplot(data,widths=0.6,patch_artist=True,showfliers=False,medianprops={'color':'k','linewidth':1})
for p,g in zip(bp['boxes'],ORDER): p.set_facecolor(C[g]); p.set_edgecolor('k'); p.set_linewidth(0.5)
axA.set_xticks(range(1,6)); axA.set_xticklabels([LAB[g] for g in ORDER],rotation=40,ha='right',fontsize=6)
axA.set_ylabel('CSF DDC (NPX)'); axA.set_title('DDC by group',fontsize=8,loc='left')
# B NSD trajectory
axB=fig.add_subplot(gs[0,1]); panel(axB,'B')
st=sorted(dd['NSD_num'].dropna().unique())
mn=[dd[dd.NSD_num==s]['DDC'].mean() for s in st]; se=[dd[dd.NSD_num==s]['DDC'].sem() for s in st]
axB.errorbar(st,mn,yerr=se,marker='o',ms=3,c=RED,capsize=2,lw=1.2)
axB.set_xlabel('NSD stage'); axB.set_ylabel('CSF DDC (NPX)'); axB.set_title('DDC vs biological stage',fontsize=8,loc='left')
# D DDC vs SBR (原C)
axC=fig.add_subplot(gs[1,0]); panel(axC,'D')
# 与正文一致: 年龄、性别校正的线性模型 DDC ~ SBR + age + sex (同 ddc_deepdive.py)
import statsmodels.formula.api as smf
d3=dd.merge(clin[['age','SEX']],left_on='PATNO',right_index=True).dropna(subset=['DDC','striatum_mean','age','SEX'])
axC.scatter(d3['striatum_mean'],d3['DDC'],s=3,c='#555',alpha=0.4,rasterized=True)
fC=smf.ols('DDC ~ striatum_mean + age + SEX',data=d3).fit(); bC=fC.params['striatum_mean']; pC=fC.pvalues['striatum_mean']
xr=np.linspace(d3['striatum_mean'].min(),d3['striatum_mean'].max(),50)
yr=fC.params['Intercept']+bC*xr+fC.params['age']*d3['age'].mean()+fC.params['SEX']*d3['SEX'].mean()  # 校正后在均值年龄/性别处的拟合线
axC.plot(xr,yr,c=RED,lw=1.2)
pexp=int(np.floor(np.log10(pC))); pman=pC/10**pexp
_nd=pd.read_csv(D+r'\rev2_nsd_decomposition.csv').set_index('model'); _b10=_nd.loc['within SAA+: DAT + motor','zSBR']; _p10=_nd.loc['within SAA+: DAT + motor','p_zSBR']
axC.text(0.03,0.99,f"Age/sex-adjusted: β = {M_(bC,'.2f')} per SBR unit,\nP = {pman:.1f} × 10$^{{{pexp}}}$; within SAA+, adjusted for\nmotor severity: β = {M_(_b10,'+.3f')} per SD, P = {_p10:.2f}",transform=axC.transAxes,ha='left',va='top',fontsize=6)
print(f'Fig4C: n={len(d3)}, beta={bC:.3f}, P={pC:.2e}')
axC.set_xlabel('Striatal SBR (mean)'); axC.set_ylabel('CSF DDC (NPX)'); axC.set_title('DDC vs dopaminergic imaging',fontsize=8,loc='left'); axC.set_ylim(-2.8,5.6)
# C NSD 分期分解: 各成分独立增量 R²
import json as _j; inc=_j.load(open(D+"/rev2_nsd_incrementalR2.json")); nd=pd.read_csv(D+"/rev2_nsd_decomposition.csv")
axN=fig.add_subplot(gs[0,2]); panel(axN,'C')
comp=[('SAA status','SAA'),('Motor severity\n(MDS-UPDRS-III)','motor severity'),('DAT binding\n(SBR)','DAT SBR'),('NSD stage beyond\ncomponents','NSD stage beyond components')]
vals=[inc[k]['dR2'] for _,k in comp]; pv=[inc[k]['p'] for _,k in comp]
axN.bar(range(4),vals,color=['#8C6BB1',RED,'#3775BA','#BFBFBF'],edgecolor='k',linewidth=0.5)
for i,(v,pp) in enumerate(zip(vals,pv)): axN.text(i,v+0.001,('P < 0.001' if pp<0.001 else f'P = {pp:.2f}'),ha='center',fontsize=6)
axN.set_xticks(range(4)); axN.set_xticklabels(['SAA','Motor\nseverity','DAT\nSBR','NSD\nresidual'],fontsize=6); axN.set_ylabel('Unique R² for CSF DDC'); axN.set_ylim(0,max(vals)*1.35)
axN.set_title('Decomposition of stage',fontsize=8,loc='left')
# E ROC CSF vs plasma
axD=fig.add_subplot(gs[1,1]); panel(axD,'E')
both=pc[pc.group.isin(['PD_bodyfirst','PD_brainfirst','HC'])].dropna(subset=['DDC_csf','DDC_plasma']).copy()
both['y']=(both.group!='HC').astype(int)
for x,lab,c in [('DDC_csf','CSF',RED),('DDC_plasma','Plasma',BLUE)]:
    fpr,tpr,_=roc_curve(both['y'],both[x]); au=roc_auc_score(both['y'],both[x])
    axD.plot(fpr,tpr,c=c,lw=1.4,label=f'{lab} (AUC={au:.2f})')
axD.plot([0,1],[0,1],'k--',lw=0.6); axD.set_xlabel('1 − Specificity'); axD.set_ylabel('Sensitivity')
axD.legend(fontsize=6,loc='lower right'); axD.set_title('Diagnostic: PD vs HC',fontsize=8,loc='left')
# E ROC CSF vs plasma (原 D) —— 已在上方绘制为 D; 此处 D 面板改为分解, 重新安排: C=分解, D=SBR散点, E=ROC, F=进展敏感性
# F 进展敏感性 forest (DDC × time, MDS-UPDRS-III 结局)
pr=pd.read_csv(D+"/rev2_progression_sensitivity.csv"); prU=pr[pr.outcome.isin(['updrs3_score','updrs3_score_on'])].copy()
labF={'(a) Primary: MDS-UPDRS-III, all biologically positive (as in main text)':'All participants',
      '(b) + baseline SBR + time-varying LEDD':'+ baseline SBR + LEDD',
      '(c) Untreated visits only (LEDD = 0 at the visit)':'Untreated visits only',
      '(d) Prodromal participants only (largely untreated)':'Prodromal (all SAA+)',
      '(e) PD only':'PD only',
      '(f) ON-state MDS-UPDRS-III (treated visits)':'ON-state score'}
prU['lab']=prU.analysis.map(labF); prU=prU.dropna(subset=['lab'])
axF=fig.add_subplot(gs[1,2]); panel(axF,'F')
yF=np.arange(len(prU))[::-1]
for y,(_,r) in zip(yF,prU.iterrows()):
    col='#444444'   # QA: 不再用红色强调 P<0.05
    axF.plot([r.ci_lo,r.ci_hi],[y,y],c=col,lw=1.3,solid_capstyle='round'); axF.scatter(r.beta_DDCxtime,y,s=22,c=col,edgecolors='k',linewidths=0.4,zorder=3)
    axF.text(1.45,y,f"n = {int(r.n)}, P = {r.p:.3f}" if r.p>=0.001 else f"n = {int(r.n)}, P < 0.001",fontsize=6,va='center')
axF.axvline(0,c='k',lw=0.7); axF.set_yticks(yF); axF.set_yticklabels(prU.lab,fontsize=6); axF.tick_params(axis='y',length=0)
axF.set_xlim(-0.6,2.5); axF.set_xlabel('DDC × time (MDS-UPDRS-III points/year per NPX)',fontsize=6.8)
axF.set_title('DDC × time sensitivity (age, sex)',fontsize=8,loc='left')
save(fig,'Figure4_DDC')
print('\nAll figures saved to',FIG)
