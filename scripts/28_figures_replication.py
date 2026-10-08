# -*- coding: utf-8 -*-
"""Replication figures. Output: figures/Figure_R3_Fig3_RBD_replication (Fig. 3), Figure_R3_Fig4_DDC_two_sets (Fig. 4),
Figure_R3_Fig5_progression (Supplementary Fig. S6), Figure_R3_Fig6_longitudinal_treatment (Supplementary Fig. S7).
"""
import config as cfg
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score
import pyarrow.parquet as pq, pyarrow.compute as pc
D=cfg.D; FIG=cfg.F; B=cfg.P
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'svg.fonttype':'none','pdf.fonttype':42,'font.size':7,
 'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':0.8,'legend.frameon':False,'axes.labelsize':7.5,'axes.titlesize':7.5,
 'text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','axes.edgecolor':'black',
 'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold','mathtext.default':'regular'})
MODEL={'Full cohort':'#333333','+ clinical group':'#7B3294','Within SAA+':'#1B9E77','Within PD':'#A6761D'}   # QA: 模型配色不与 Discovery/Replication 撞色
RED='#B64342'; BLUE='#2c6fbb'; GREY='#BFBFBF'; DISC='#0F4D92'; REP='#E08A3C'; POOL='#333333'
def panel(ax,l,x=-0.16,y=1.04): ax.text(x,y,l,transform=ax.transAxes,fontsize=10,fontweight='bold',va='bottom')
def save(fig,name):
    for ext,dpi in [('svg',None),('pdf',None),('png',600),('tiff',500)]:
        fig.savefig(os.path.join(FIG,f'{name}.{ext}'),dpi=dpi,bbox_inches='tight',**({'pil_kwargs':{'compression':'tiff_lzw'}} if ext=='tiff' else {}))
    plt.close(fig); print('saved',name)
def pf(p,pre='P = '):
    if p<0.001:
        m_,e_=f'{p:.0e}'.split('e'); return f'{pre}{m_} × 10$^{{{int(e_)}}}$'
    return f'{pre}{p:.3f}'
def forest(ax,labels,b,lo,hi,cols,p=None,xlabel='',zero=0):
    y=np.arange(len(labels))[::-1]; b=np.asarray(b,float); lo=np.asarray(lo,float); hi=np.asarray(hi,float)
    for i in range(len(labels)):
        ax.plot([lo[i],hi[i]],[y[i],y[i]],color=cols[i],lw=1.2); ax.plot(b[i],y[i],'o',color=cols[i],ms=4,mec='white',mew=0.4,zorder=3)
    l_=np.nanmin(np.r_[lo,zero]); h_=np.nanmax(np.r_[hi,zero]); rg=h_-l_
    if p is not None:
        for i in range(len(labels)):
            if np.isfinite(p[i]): ax.text(h_+0.12*rg,y[i],pf(p[i]),va='center',fontsize=6)
        ax.set_xlim(l_-0.05*rg,h_+0.85*rg)
    ax.axvline(zero,color='black',lw=0.6,ls=(0,(3,2))); ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=6); ax.set_xlabel(xlabel); ax.set_ylim(-0.7,len(labels)-0.3)
GEN=['LRRK2','GBA','LRRK2 + GBA','SNCA','PRKN','PINK1','PRKN + RBD']
clin=pd.read_csv(D+"/analytic_clinical.csv").set_index('PATNO'); ann=pd.read_csv(D+"/protein_annotation.csv").set_index('OlinkID')

# ============================== FIG 3: RBD replication ==============================
sig=pd.read_csv(D+"/sig_rem.csv").set_index('OlinkID'); s55=set(sig[sig.fdr<0.05].index)
def prep(wf,filt):
    w=pd.read_parquet(wf).set_index('PATNO'); w.index=w.index.astype(int); w=w.loc[:,w.isna().mean()<=0.2]; w=w.loc[w.isna().mean(1)<=0.3]
    c=clin.reindex(w.index); m=filt(c)&c.rem.notna()&c.age.notna()&c.SEX.notna(); return w[m.values],c[m.values]
wd,cd=prep(D+"/csf_npx_wide.parquet",lambda c:c.group0.isin(['HC','Hyposmia','iRBD','PD']))
wr,cr=prep(D+"/p314_wide_BL_new.parquet",lambda c:(~c.subgroup.isin(GEN))&c.group0.isin(['HC','Hyposmia','iRBD','PD']))
sh=[c for c in wd.columns if c in wr.columns]; i55=np.array([c in s55 for c in sh])
def imp(w):
    X=w[sh].values.astype(float); med=np.nanmedian(X,0); ii=np.where(np.isnan(X)); X[ii]=np.take(med,ii[1]); return X
Xd=imp(wd); Xr=imp(wr); Yd=Xd/Xd[(cd.group0=='HC').values].std(0); Yr=Xr/Xr[(cr.group0=='HC').values].std(0)
def ppb(Y,c,m,g4=False):
    cols=[np.ones(m.sum()),c.rem.values[m],c.age.values[m],c.SEX.values[m].astype(float)]
    if g4: cols+=[(c.group0.values[m]==g).astype(float) for g in ['Hyposmia','iRBD','PD']]
    X=np.column_stack(cols); return (np.linalg.pinv(X.T@X)@X.T@Y[m])[1]
bd=ppb(Yd,cd,np.ones(len(cd),bool))[i55]
reps={'Full cohort':ppb(Yr,cr,np.ones(len(cr),bool))[i55],'+ clinical group':ppb(Yr,cr,np.ones(len(cr),bool),True)[i55],
      'Within SAA+':ppb(Yr,cr,(cr.CSFSAA==1).values)[i55],'Within PD':ppb(Yr,cr,(cr.group0=='PD').values)[i55]}
fig=plt.figure(figsize=(7.0,5.4)); gs=gridspec.GridSpec(2,2,figure=fig,hspace=0.55,wspace=0.45)
ax=fig.add_subplot(gs[0,0]); panel(ax,'a')
cl=dict(MODEL)
for k,v in reps.items():
    s=(v*bd).sum()/(bd*bd).sum(); ax.scatter(bd,v,s=9,color=cl[k],alpha=0.75,lw=0,label=f'{k} (slope {s:.2f})')
lim=[min(bd.min(),-0.08)*1.1,max(bd.max(),0.08)*1.1]; ax.plot(lim,lim,'k--',lw=0.6); ax.axhline(0,color=GREY,lw=0.5); ax.axvline(0,color=GREY,lw=0.5)
ax.set_xticks([-0.05,0,0.05]); ax.set_xlabel('Discovery effect (control-SD per RBDSQ point)'); ax.set_ylabel('Replication effect (control-SD per point)'); ax.legend(fontsize=6,loc='upper left'); ax.set_title('55 RBD-associated proteins',fontsize=7.5)
ax=fig.add_subplot(gs[0,1]); panel(ax,'b')
eh=pd.read_csv(D+"/r3C_expected_hits.csv"); strata=['Full','Within SAA+','Within PD']; x=np.arange(3); w=0.26
obs={'Full':8,'Within SAA+':0,'Within PD':17}
for k,(lab,col,face) in enumerate([('uncorrected per-point','black','white'),('FIQT-corrected per-point',GREY,GREY)]):
    e=eh[eh.effect==lab].set_index('stratum').reindex(strata)
    ax.bar(x+(k-1)*w,e.exp_nominal,w,color=face,edgecolor=col,lw=0.7,label='Expected, '+('uncorrected' if k==0 else "winner's-curse corrected"))
    ax.errorbar(x+(k-1)*w,e.exp_nominal,yerr=[e.exp_nominal-e.exp_nom_lo,e.exp_nom_hi-e.exp_nominal],fmt='none',ecolor='black',lw=0.6,capsize=1.5)
ax.bar(x+w,[obs[s] for s in strata],w,color=RED,label='Observed')
for i,s in enumerate(strata): ax.text(x[i]+w,obs[s]+1,str(obs[s]),ha='center',fontsize=6)
ax.set_xticks(x); ax.set_xticklabels(['Full\ncohort','Within\nSAA+','Within\nPD']); ax.set_ylabel('Proteins (of 55) at nominal P < 0.05'); ax.set_ylim(0,80); ax.legend(fontsize=6,loc='upper center',ncol=1,bbox_to_anchor=(0.5,1.0))
ax.set_title('Expected versus observed replication',fontsize=7.5)
ax=fig.add_subplot(gs[1,0]); panel(ax,'c')
pm=pd.read_csv(D+"/r3C_permutation.csv"); lab={('Full','age,sex'):'Full cohort',('Full','g4'):'+ clinical group',('Within SAA+','age,sex'):'Within SAA+',('Within PD','age,sex'):'Within PD'}
pm['lab']=[lab[(a,b)] for a,b in zip(pm.stratum,pm.adjust)]; _fl=pd.read_csv(D+r'\r4A_group_adjusted_permutation.csv').set_index('method')
pm.loc[pm.lab=='+ clinical group','perm_p_score']=_fl.loc['Freedman–Lane','p_score']; pm.loc[pm.lab=='+ clinical group','lab']='+ clinical group (Freedman–Lane)'; cl['+ clinical group (Freedman–Lane)']=MODEL['+ clinical group']; yy=np.arange(len(pm))[::-1]
ax.barh(yy,pm.score_beta,color=[cl[l] for l in pm.lab],height=0.6)
for i,(b,p) in enumerate(zip(pm.score_beta,pm.perm_p_score)): ax.text(max(b,0)+0.0008,yy[i],f'permutation {pf(p)}',va='center',fontsize=6)
ax.set_yticks(yy); ax.set_yticklabels(pm.lab,fontsize=6.3); ax.axvline(0,color='black',lw=0.6); ax.set_xlabel('Composite score per RBDSQ point (SD)'); ax.set_xlim(-0.005,0.034); ax.set_xticks([0,0.01,0.02,0.03])
ax.set_title('Signature tested as a whole (replication)',fontsize=7.5)
ax=fig.add_subplot(gs[1,1]); panel(ax,'d')
pcs=pd.read_csv(D+"/p314_posctrl_summary.csv").set_index('exposure'); br=pd.read_csv(D+"/r3A_bridge_rho.csv").set_index('OlinkID').rho_same_sample
pcb=pd.read_csv(D+"/p314_posctrl_betas.csv")
sets={'age':set(pcb[(pcb.exposure=='age')&(pcb.fdr_277<0.05)].OlinkID),'sex':set(pcb[(pcb.exposure=='sex')&(pcb.fdr_277<0.05)].OlinkID),'SAA':set(pcb[(pcb.exposure=='SAA')&(pcb.fdr_277<0.05)].OlinkID),'RBDSQ':set(s55)}   # QA: 跨项目 ρ 用 55 个 RBD 蛋白 (与正文、Table S23 一致); 柱 = 四组模型 FDR<0.05 的 33 个
order=['age','sex','SAA','RBDSQ']; xx=np.arange(4)
ax.bar(xx,pcs.loc[order,'slope'],color=[GREY,GREY,'#808080','#404040'],width=0.6)
for i,o in enumerate(order): ax.text(i,pcs.loc[o,'slope']+0.04,f"{pcs.loc[o,'rep_rate']*100:.0f}%",ha='center',fontsize=6)
ax2=ax.twinx(); ax2.plot(xx,[br.reindex(list(sets[o])).median() for o in order],'D',color='black',ms=4); ax2.set_ylim(0,1.05); ax2.set_ylabel('Same-sample cross-project ρ (diamonds)',fontsize=7.5)
ax2.spines['right'].set_visible(True)
ax.axhline(1,color='black',lw=0.6,ls=(0,(3,2))); ax.set_xticks(xx); ax.set_xticklabels(['Age','Sex','SAA','RBDSQ']); ax.set_ylabel('Replication / discovery slope (bars)'); ax.set_ylim(0,1.3)
ax.set_title('Assay comparability',fontsize=7.5)
save(fig,'Figure_R3_Fig3_RBD_replication')

# ============================== FIG 4: DDC two sets ==============================
fig=plt.figure(figsize=(7.4,5.8)); gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.6,wspace=1.25,width_ratios=[1,1.15,1])
ax=fig.add_subplot(gs[0,0]); panel(ax,'a')
q=pcb[pcb.exposure=='PD_vs_HC'].copy(); q['gene']=ann.reindex(q.OlinkID).Assay.values
ax.scatter(q.beta_277,q.beta_314,s=2,color=GREY,alpha=0.5,lw=0,rasterized=True); d_=q[q.gene=='DDC'].iloc[0]
ax.scatter(d_.beta_277,d_.beta_314,s=22,color=RED,zorder=3); ax.annotate('DDC',(d_.beta_277,d_.beta_314),xytext=(-8,0),textcoords='offset points',ha='right',va='center',color=RED,fontweight='bold',fontsize=6.5)
lim=[-0.8,1.2]; ax.plot(lim,lim,'k--',lw=0.6); ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xticks([-0.5,0,0.5,1.0]); ax.set_yticks([-0.5,0,0.5,1.0]); ax.set_xlabel('PD − HC, discovery (SD)'); ax.set_ylabel('PD − HC, replication (SD)'); ax.set_title('5,414 proteins',fontsize=7.5)
ax=fig.add_subplot(gs[0,1]); panel(ax,'b',x=-0.55)
gx=pd.read_csv(D+"/r3A_group_by_SAA_DDC.csv"); gx=gx[~gx.cell.str.contains('SAAna')]
nice=lambda c: c.replace('_SAA+',', SAA+').replace('_SAA-',', SAA−')
labs=[f"{'Disc.' if co=='Discovery' else 'Rep.'}: {nice(c)} ({n})" for c,co,n in zip(gx.cell,gx.cohort,gx.n)]
forest(ax,labs,gx.beta,gx.ci_lo,gx.ci_hi,[DISC if c=='Discovery' else REP for c in gx.cohort],gx.p.values,xlabel='DDC vs HC (NPX)')
ax.set_title('By clinical group and seed status',fontsize=7.5)
ax=fig.add_subplot(gs[0,2]); panel(ax,'c',x=-0.3)
nc=pd.read_csv(D+"/r3A_nsd_components.csv"); npm_=pd.read_csv(D+"/r4D_nsd_perm.csv"); comps=['SAA','zSBR','zupdrs3_score','zupdrs2_score','zmoca','NSD stage (categorical) beyond 5 components']
nm={'SAA':'SAA','zSBR':'SBR','zupdrs3_score':'MDS-UPDRS-III','zupdrs2_score':'MDS-UPDRS-II','zmoca':'MoCA','NSD stage (categorical) beyond 5 components':'NSD stage (adj. ΔR²)'}
xx=np.arange(len(comps)); w=0.38
for k,(co,col) in enumerate([('Discovery',DISC),('Replication',REP)]):
    v=nc[(nc.cohort==co)].set_index('term').reindex(comps).unique_R2.copy()
    v.iloc[-1]=max(npm_.set_index('cohort').adj_dR2[co],0)   # 终审 E1: NSD 用合并稀疏分期后的 adjusted ΔR² (r4D, 与 Table S18 一致)
    _bs=pd.read_csv(D+"/r6_unique_R2_bootstrap.csv"); _bs=_bs[_bs.cohort==co].set_index('term')   # 第五轮 M3: bootstrap 95% CI (2,000 次)
    _lo=[v.iloc[i]-_bs.loc[c,'ci_lo'] if c in _bs.index else 0 for i,c in enumerate(comps)]; _hi=[_bs.loc[c,'ci_hi']-v.iloc[i] if c in _bs.index else 0 for i,c in enumerate(comps)]
    ax.bar(xx+(k-0.5)*w,v,w,color=col,label=co,yerr=[np.clip(_lo,0,None),np.clip(_hi,0,None)],error_kw={'elinewidth':0.6,'capsize':1.5,'ecolor':'black'})
    # QA: NSD 置换 P (0.089/0.087) 写在图注, 不再竖排小字
ax.set_xticks(xx); ax.set_xticklabels([nm[c] for c in comps],rotation=45,ha='right',fontsize=6); ax.set_ylabel('Unique R² for DDC'); ax.legend(fontsize=6.5); ax.set_title('Components of stage',fontsize=7.5)
gsb=gs[1,:].subgridspec(1,2,width_ratios=[1,1.55],wspace=1.05)   # QA: 底行单独分栏, 给 e 的行标签留空间
ax=fig.add_subplot(gsb[0,0]); panel(ax,'d')
wide=pd.read_parquet(D+"/csf_npx_wide.parquet").set_index('PATNO'); jD=[c for c in wide.columns if ann.Assay.get(c)=='DDC'][0]
dc=clin.reindex(wide.index.astype(int)); dc['DDC']=wide[jD].values; rp=pd.read_csv(D+"/p314_rep_baseline_analytic.csv"); rp=rp[~rp.subgroup.isin(GEN)]
for df,col,lab in [(dc,DISC,'Discovery'),(rp,REP,'Replication')]:
    s=df[df.group0.isin(['PD','HC'])].dropna(subset=['DDC']); y=(s.group0=='PD').astype(int); f,t,_=roc_curve(y,s.DDC); ax.plot(f,t,color=col,lw=1.3,label=f'{lab}: {roc_auc_score(y,s.DDC):.2f}')
ax.plot([0,1],[0,1],color=GREY,ls='--',lw=0.6); ax.set_xlabel('1 − specificity'); ax.set_ylabel('Sensitivity'); ax.legend(fontsize=6,loc='lower right',title='PD vs HC, AUC',title_fontsize=6,handlelength=1.0); ax.set_title('CSF DDC',fontsize=7.5)
ax=fig.add_subplot(gsb[0,1]); panel(ax,'e',x=-0.62)
gc=pd.read_csv(D+"/r3D_genetic_contrasts.csv"); gc=gc[gc.gene=='LRRK2']
lab2={'SAA+ vs SAA−, age+sex':'LRRK2: SAA+ vs SAA−','SAA+ vs SAA−, + log(1+LEDD) + duration':'LRRK2: SAA+ vs SAA−, adj.*','untreated only: SAA+ vs SAA−':'LRRK2: SAA+ vs SAA−, untreated',
      'SAA− LRRK2-PD vs HC, age+sex':'LRRK2 SAA− vs HC','SAA− LRRK2-PD vs HC, + log(1+LEDD)':'LRRK2 SAA− vs HC, adj.†'}
gc=gc[gc.analysis.isin(lab2)]
# QA: 补未用药者对照 (正文引用 Fig. 4e 的 −0.08 与 1.20), 散发 PD 作参照
cu=pd.read_csv(D+"/p314_gen_crosssectional.csv"); cu=cu[cu.model=='(e) 仅未用药者'].set_index('contrast')
add=[('LRRK2-PD-SAA+ vs HC','Untreated LRRK2 SAA+ vs HC'),('GBA-PD-SAA+ vs HC','Untreated GBA1 SAA+ vs HC'),('Sporadic PD vs HC','Untreated sporadic vs HC')]
L4=[lab2[a] for a in gc.analysis]+[b for _,b in add]; B4=list(gc.beta)+[cu.loc[a,'beta'] for a,_ in add]
Lo4=list(gc.ci_lo)+[cu.loc[a,'ci_lo'] for a,_ in add]; Hi4=list(gc.ci_hi)+[cu.loc[a,'ci_hi'] for a,_ in add]; P4=list(gc.p)+[cu.loc[a,'p'] for a,_ in add]
C4=['#E08A3C' if 'SAA+ vs SAA−' in l else '#8C4A12' for l in L4[:len(gc)]]+['#8C4A12','#8C4A12','#808080']
forest(ax,L4,B4,Lo4,Hi4,C4,np.array(P4),xlabel='Difference in CSF DDC (NPX)')
ax.set_title("Genetic Parkinson's disease (replication set)",fontsize=7.5)
save(fig,'Figure_R3_Fig4_DDC_two_sets')

# ============================== FIG 5: progression (per SD) ==============================
pm=pd.read_csv(D+"/r4C_progression.csv"); plz=pd.read_csv(D+"/r4C_pooled.csv")
fig=plt.figure(figsize=(7.4,4.6)); gs=gridspec.GridSpec(1,3,figure=fig,wspace=1.25)
for j,(y,ttl,xl) in enumerate([('updrs3_score','MDS-UPDRS-III','points/year per SD of DDC'),('SBR_t','Striatal SBR','SBR/year per SD of DDC'),('moca','MoCA','points/year per SD of DDC')]):
    ax=fig.add_subplot(gs[0,j]); panel(ax,'abc'[j],x=-0.9)
    labs=[];b=[];lo=[];hi=[];cols=[];pp=[]
    for sub,sl in [('Untreated visits','Untreated'),('SAA-positive prodromal','SAA+ prodromal')]:
        for mm in ['M0','M1','M2']:
            for co,col in [('Discovery',DISC),('Replication',REP)]:
                r=pm[(pm.subset==sub)&(pm.outcome==y)&(pm.model==mm)&(pm.cohort==co)]
                if len(r): r=r.iloc[0]; labs.append(f"{sl} {mm} {'Disc.' if co=='Discovery' else 'Rep.'}"); b.append(r.beta_per_SD); lo.append(r.ci_lo); hi.append(r.ci_hi); cols.append(col); pp.append(r.p)
            r=plz[(plz.subset==sub)&(plz.outcome==y)&(plz.model==mm)]
            if len(r): r=r.iloc[0]; labs.append(f'{sl} {mm} pooled'); b.append(r.pooled_per_SD); lo.append(r.ci_lo); hi.append(r.ci_hi); cols.append(POOL); pp.append(r.p)
    forest(ax,labs,b,lo,hi,cols,np.array(pp),xlabel=xl); ax.set_title(ttl,fontsize=7.5); ax.tick_params(axis='y',labelsize=6)
save(fig,'Figure_R3_Fig5_progression')

# ============================== FIG 6: within-person change ==============================
es=pd.read_csv(D+"/r4B_event_study.csv"); dr=pd.read_csv(D+"/r4B_proteome_scan_clustered.csv")
fig=plt.figure(figsize=(6.3,5.8)); gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.62,wspace=0.8)
ax=fig.add_subplot(gs[0,0:2]); panel(ax,'a',x=-0.3)
rows=[]
for pj,lab_,col in [('277 CSF','CSF discovery',DISC),('314 CSF','CSF replication',REP),('293 plasma','Plasma',POOL)]:
    for g in ['DDC','PRL']:
        r=es[(es.project==pj)&(es.protein==g)].iloc[0]; c=col if g=='DDC' else GREY
        rows.append((f'{g}, {lab_}: step',r.step,r.step_lo,r.step_hi,r.step_p,c))
        rows.append((f"{g}, {lab_}: at {r.median_yrs_tx:.0f} y (joint {pf(r.joint_p,'P = ')})",r.contrast_at_median,r.con_lo,r.con_hi,r.con_p,c))
forest(ax,[x[0] for x in rows],[x[1] for x in rows],[x[2] for x in rows],[x[3] for x in rows],[x[5] for x in rows],np.array([x[4] for x in rows]),xlabel='Change with treatment (NPX), fixed effects + flexible time')
ax.set_title('Treatment step and cumulative change; prolactin as pharmacodynamic control',fontsize=7.2); ax.tick_params(axis='y',labelsize=6)
ax=fig.add_subplot(gs[0,2]); panel(ax,'b',x=-0.3)
import pyarrow  # noqa
bm=pd.read_csv(D+"/r4B_drift_benchmark.csv"); x=bm[(bm.project=='277 CSF')&(bm.set=='untreated non-PD')]
ax.bar([0,1],x.slope_sd.values,color=[RED,'#666666'],width=0.55); ax.axhline(x.proteome_median.iloc[0],color='black',ls=(0,(3,2)),lw=0.7)
ax.text(1.75,x.proteome_median.iloc[0]+0.003,'proteome median',fontsize=6,va='bottom',ha='right')
for i_,(v,pc_) in enumerate(zip(x.slope_sd,x.percentile)): ax.text(i_,v+0.004,f'{pc_:.0f}th pct',ha='center',fontsize=6)
ax.set_xticks([0,1]); ax.set_xticklabels(x.protein.values); ax.set_xlim(-0.6,1.9); ax.set_ylim(0,0.16); ax.set_ylabel('Yearly within-person slope (SD)')
ax.set_title('Drift, untreated non-PD (discovery)',fontsize=7)
ax.text(0.02,0.98,f'n = {int(x.n.iloc[0])}; {100*x.proteome_frac_positive.iloc[0]:.0f}% of proteins rise',transform=ax.transAxes,fontsize=6,va='top')
for k,(pj,lab_) in enumerate([('277 CSF','CSF, discovery'),('314 CSF','CSF, replication'),('293 plasma','Plasma')]):
    ax=fig.add_subplot(gs[1,k]); panel(ax,'cde'[k]); s_=dr[dr.project==pj]
    ax.scatter(s_.beta_sd,-np.log10(s_.p),s=2,color=GREY,lw=0,rasterized=True)
    for g,col in [('DDC',RED),('PRL','#555555')]:
        r=s_[s_.gene==g]
        if len(r): r=r.iloc[0]; ax.scatter(r.beta_sd,-np.log10(r.p),s=20,color=col,zorder=3); ax.annotate(f"{g} (rank {int(r['rank'])})",(r.beta_sd,-np.log10(r.p)),xytext=(4,3),textcoords='offset points',fontsize=6,color=col,bbox=dict(boxstyle='round,pad=0.1',fc='white',ec='none',alpha=0.85))
    ax.set_xlabel('Treatment effect, linear time (SD)'); ax.set_ylabel('−log$_{10}$ P (clustered)'); ax.set_title(lab_,fontsize=7.2)
save(fig,'Figure_R3_Fig6_longitudinal_treatment')
