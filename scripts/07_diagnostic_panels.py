# -*- coding: utf-8 -*-
"""Discovery set: DDC clinical correlates, multi-marker diagnostic panels, whole-proteome classifier and comparison
with established markers. Output: derived/diagnostic_panel_AUC.csv and related tables (Supplementary Fig. S4, Table S15).
"""
import config as cfg
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve
D=cfg.D; FIG=cfg.F
wide=pd.read_parquet(D+"/csf_npx_wide.parquet").set_index('PATNO')
ann=pd.read_csv(D+"/protein_annotation.csv").set_index('OlinkID')
clin=pd.read_csv(D+"/analytic_clinical.csv").set_index('PATNO')
oid=[o for o in wide.columns if str(ann.reindex([o])['Assay'].values[0])=='DDC'][0]
clin['DDC']=wide[oid].reindex(clin.index)
cur=pd.read_excel(cfg.CURATED,sheet_name='20260511')
bl=cur[cur.EVENT_ID=='BL'].set_index('PATNO')
for m in ['NFL_CSF','IU_pTau181_CSF','IU_GFAP_plasma_mean','IU_NFL_plasma_mean','updrs3_score','moca','scopa']:
    clin[m]=bl[m].reindex(clin.index)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
 'svg.fonttype':'none','pdf.fonttype':42,'font.size':7,'axes.spines.right':False,
 'axes.spines.top':False,'axes.linewidth':0.8,'legend.frameon':False,'axes.titlesize':8,'axes.labelsize':8,
 'text.color':'black','axes.labelcolor':'black','axes.titlecolor':'black',
 'xtick.color':'black','ytick.color':'black','axes.edgecolor':'black',
 'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold'})
RED='#B64342'; BLUE='#2c6fbb'; GREY='#9aa0a6'
def panel(ax,l,x=-0.16,y=1.03): ax.text(x,y,l.lower(),transform=ax.transAxes,fontsize=10,fontweight='bold',va='bottom')   # QA: 面板字母统一小写
def auc_ci(y,x,n=1000):
    a=roc_auc_score(y,x); rng=np.random.RandomState(1); bs=[]; idx=np.arange(len(y))
    for _ in range(n):
        s=rng.choice(idx,len(idx),True)
        if len(np.unique(y[s]))>1: bs.append(roc_auc_score(y[s],x[s]))
    return a,np.percentile(bs,2.5),np.percentile(bs,97.5)

print("="*60+"\n① DDC 与基线临床严重度 (偏相关, 校正年龄性别)\n"+"="*60)
res1=[]
for v,nm in [('updrs3_score','MDS-UPDRS-III (motor)'),('moca','MoCA (cognition)'),('scopa','SCOPA-AUT (autonomic)')]:
    d=clin.dropna(subset=['DDC',v,'age','SEX'])
    f=smf.ols(f'{v} ~ DDC + age + SEX',data=d).fit()
    r_par=np.sign(f.params['DDC'])*np.sqrt(f.tvalues['DDC']**2/(f.tvalues['DDC']**2+f.df_resid))
    print(f"  DDC vs {nm} (n={len(d)}): partial r={r_par:+.3f}, P={f.pvalues['DDC']:.2e}")
    res1.append((nm,r_par,f.pvalues['DDC']))

print("\n"+"="*60+"\n② 诊断组合 (PD vs HC): 单标志物 vs 组合 vs 全蛋白组ML\n"+"="*60)
pdhc=clin[clin.group.isin(['PD_bodyfirst','PD_brainfirst','HC'])].copy(); pdhc['y']=(pdhc.group!='HC').astype(int)
markers={'DDC':'DDC (CSF)','IU_pTau181_CSF':'pTau181 (CSF)','IU_GFAP_plasma_mean':'GFAP (plasma)','IU_NFL_plasma_mean':'NfL (plasma)'}
aucs={}
for m,nm in markers.items():
    d=pdhc.dropna(subset=[m,'y']); a,lo,hi=auc_ci(d['y'].values,d[m].values*np.sign(1))
    a=max(a,1-a)  # 方向无关的判别力
    aucs[nm]=(a,lo if a==roc_auc_score(d['y'],d[m]) else 1-hi, hi if a==roc_auc_score(d['y'],d[m]) else 1-lo)
    print(f"  {nm}: AUC={a:.3f} (n={len(d)})")
# α-syn SAA 参照 (现行金标准)
ds=pdhc[pdhc.index.isin(wide.index)].dropna(subset=['CSFSAA','y']); aS=roc_auc_score(ds['y'].values,(ds['CSFSAA']==1).astype(int).values)
aucs['α-syn SAA (CSF)']=(aS,None,None); print(f"  α-syn SAA (CSF): AUC={aS:.3f}")
# 组合(逻辑回归, 交叉验证) DDC+pTau+GFAP+NfL
combo=pdhc.dropna(subset=list(markers)+['y'])
Xc=combo[list(markers)].values; yc=combo['y'].values
cv=StratifiedKFold(5,shuffle=True,random_state=1)
prob=cross_val_predict(LogisticRegression(max_iter=1000),Xc,yc,cv=cv,method='predict_proba')[:,1]
aC=roc_auc_score(yc,prob); aucs['Combined panel']=(aC,None,None)
print(f"  Combined (DDC+pTau+GFAP+NfL, 5-fold CV): AUC={aC:.3f} (n={len(combo)})")
# 全蛋白组ML —— 仅限有CSF蛋白组数据的样本(否则大量无蛋白组样本被中位数插补,稀释信号)
pdhc_p=pdhc[pdhc.index.isin(wide.index)]
X=wide.reindex(pdhc_p.index).values.astype(float); med=np.nanmedian(X,0); ii=np.where(np.isnan(X)); X[ii]=np.take(med,ii[1])
y=pdhc_p['y'].values
probP=cross_val_predict(HistGradientBoostingClassifier(max_iter=200,random_state=1),X,y,cv=cv,method='predict_proba')[:,1]
aP=roc_auc_score(y,probP); aucs['Whole CSF proteome (ML)']=(aP,None,None)
print(f"  Whole CSF proteome (HistGB, 5-fold CV): AUC={aP:.3f}")

print("\n"+"="*60+"\n③ 全蛋白组ML 检出 iRBD(身体优先前驱) + top特征\n"+"="*60)
ih=clin[clin.group.isin(['iRBD','HC'])].copy(); ih['y']=(ih.group=='iRBD').astype(int)
ih=ih[ih.index.isin(wide.index)]  # 仅限有蛋白组数据的样本
Xi=wide.reindex(ih.index).values.astype(float); med=np.nanmedian(Xi,0); ii=np.where(np.isnan(Xi)); Xi[ii]=np.take(med,ii[1])
yi=ih['y'].values
probI=cross_val_predict(HistGradientBoostingClassifier(max_iter=200,random_state=1),Xi,yi,cv=cv,method='predict_proba')[:,1]
aI=roc_auc_score(yi,probI); print(f"  iRBD vs HC (全蛋白组ML): AUC={aI:.3f} (n={len(ih)})")
# 快速: 单变量判别力(|AUC-0.5|)取top特征
uni=np.array([abs(roc_auc_score(yi,Xi[:,j])-0.5) for j in range(Xi.shape[1])])
topidx=np.argsort(uni)[::-1][:12]
impvals=uni[topidx]*2  # 0-1标度
topg=[str(ann.reindex([wide.columns[i]])['Assay'].values[0]) for i in topidx]
print("  Top12 判别蛋白:", ', '.join(topg))

print("\n"+"="*60+"\n④ DDC vs 既往标志物 预测运动/认知进展\n"+"="*60)
prog=pd.read_csv(D+"/module5_score_outcomes.csv")[['PATNO','slope_moca','slope_updrs3_score']].set_index('PATNO')
for m,nm in [('DDC','DDC'),('IU_NFL_plasma_mean','NfL(plasma)'),('IU_GFAP_plasma_mean','GFAP(plasma)')]:
    cc=clin.join(prog)
    for sl,sn in [('slope_updrs3_score','UPDRS-III'),('slope_moca','MoCA')]:
        d=cc.dropna(subset=[m,sl,'age','SEX'])
        f=smf.ols(f'{sl} ~ {m} + age + SEX',data=d).fit()
        print(f"  {nm} -> {sn}进展 (n={len(d)}): beta={f.params[m]:+.4f}, P={f.pvalues[m]:.4f}")

# ---------- 预测因子重要性排序 + 累积AUC (诊断 PD vs HC, 仿参考文献Fig6A) ----------
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
uni=np.array([abs(roc_auc_score(y,X[:,j])-0.5) for j in range(X.shape[1])])
rank=np.argsort(uni)[::-1]; K=40; topK=rank[:K]
imp_disp=uni[topK]*2  # 全数据排序(仅用于柱状"重要性"展示)
gene_disp=[str(ann.reindex([wide.columns[i]])['Assay'].values[0]) for i in topK]
# 累积AUC: 嵌套交叉验证(每折内重新做单变量排序), 避免特征选择泄漏 → 与全蛋白组ML(0.81)口径一致
cvA=StratifiedKFold(5,shuffle=True,random_state=1); foldA=np.zeros((5,K))
for fi,(tr,te) in enumerate(cvA.split(X,y)):
    ur=np.array([abs(roc_auc_score(y[tr],X[tr,j])-0.5) for j in range(X.shape[1])]); rk=np.argsort(ur)[::-1]
    for k in range(1,K+1):
        ck=rk[:k]; scl=StandardScaler().fit(X[tr][:,ck])
        lr=LogisticRegression(max_iter=1000).fit(scl.transform(X[tr][:,ck]),y[tr])
        foldA[fi,k-1]=roc_auc_score(y[te],lr.predict_proba(scl.transform(X[te][:,ck]))[:,1])
cumm=foldA.mean(0); cums=foldA.std(0)
nsel=int(np.argmax(cumm>=cumm.max()-0.005)+1)   # QA: 去掉强制下限 6
print(f'PD-vs-HC 累积AUC(嵌套CV): 1蛋白={cumm[0]:.3f}, {K}蛋白={cumm[-1]:.3f}, 平台@{nsel}蛋白')

# ---------- FIGURE 5 ----------
fig=plt.figure(figsize=(7.2,7.6)); gs=gridspec.GridSpec(3,2,figure=fig,height_ratios=[1.25,1,0.85],hspace=0.9,wspace=0.62)
# A 预测因子重要性排序 + 累积AUC (全宽)
axA=fig.add_subplot(gs[0,:]); panel(axA,'A',x=-0.05,y=1.04)
xk=np.arange(K)
axA.bar(xk,imp_disp,color=plt.cm.Blues(np.linspace(0.92,0.32,K)),width=0.82,zorder=2)
axA.set_ylabel('Predictor importance\n(2×|AUC−0.5|)',fontsize=7)
axA.set_xticks(xk); axA.set_xticklabels(gene_disp,rotation=90,fontsize=6)
axA.set_xlim(-0.7,K-0.3); axA.tick_params(axis='x',length=0)
axR=axA.twinx(); axR.spines['top'].set_visible(False)
axR.fill_between(xk,cumm-cums,cumm+cums,color=RED,alpha=0.15,lw=0,zorder=1)
axR.plot(xk,cumm,'-o',c=RED,ms=2.6,lw=1.3,zorder=4)
axR.set_ylabel('Cumulative AUC (PD vs HC)',fontsize=7); axR.set_ylim(min(cumm)-0.03,1.0)
axA.set_title('CSF protein ranking and cumulative diagnostic accuracy (PD vs HC)',fontsize=8,loc='left')
# QA: 红线说明移入图注
# B ROC组合
axB=fig.add_subplot(gs[1,0]); panel(axB,'B')
for m,nm,c in [('DDC','DDC',RED),('IU_pTau181_CSF','pTau181',BLUE),('IU_GFAP_plasma_mean','GFAP',GREY)]:
    d=pdhc.dropna(subset=[m,'y']); fpr,tpr,_=roc_curve(d['y'],d[m]); a=roc_auc_score(d['y'],d[m]); a=max(a,1-a)
    if roc_auc_score(d['y'],d[m])<0.5: fpr,tpr,_=roc_curve(d['y'],-d[m])
    axB.plot(fpr,tpr,c=c,lw=1.3,label=f'{nm} ({a:.2f})')
axB.plot(roc_curve(yc,prob)[0],roc_curve(yc,prob)[1],c='#2E9E44',lw=1.6,label=f'Combined ({aC:.2f})')
axB.plot(roc_curve(y,probP)[0],roc_curve(y,probP)[1],c='#111',lw=1.6,ls='-',label=f'Whole CSF proteome (ML) ({aP:.2f})')
axB.plot([0,1],[0,1],'k--',lw=0.5); axB.set_xlabel('1 − Specificity'); axB.set_ylabel('Sensitivity')
axB.legend(fontsize=6,loc='lower right',bbox_to_anchor=(1.0,0.0),handlelength=1.3,borderaxespad=0.2,frameon=True,facecolor='white',edgecolor='none',framealpha=0.92); axB.set_title('Diagnostic panels: PD vs HC',fontsize=8,loc='left')
# C AUC bar
axC=fig.add_subplot(gs[1,1]); panel(axC,'C')
order=['α-syn SAA (CSF)','DDC (CSF)','pTau181 (CSF)','GFAP (plasma)','NfL (plasma)','Combined panel','Whole CSF proteome (ML)']
av=[aucs[k][0] for k in order]; cols=['#8C6BB1',RED,BLUE,GREY,GREY,'#2E9E44','#111']
axC.barh(range(len(order))[::-1],av,color=cols,edgecolor='k',linewidth=0.4)
axC.set_yticks(range(len(order))[::-1]); axC.set_yticklabels(order,fontsize=6)
for i,v in enumerate(av[::-1]): axC.text(v+0.01,i,(f'{v:.3f}' if v>0.99 else f'{v:.2f}'),va='center',fontsize=6)
axC.set_xlim(0.4,1.0); axC.axvline(0.5,ls='--',c='#999',lw=0.5); axC.set_xlabel('AUC (PD vs HC)')
axC.set_title('Marker comparison',fontsize=8,loc='left')
# D DDC clinical corr
axD=fig.add_subplot(gs[2,0]); panel(axD,'D')
nms=[r[0] for r in res1]; rr=[r[1] for r in res1]
axD.bar(range(3),rr,color=[RED if x>0 else BLUE for x in rr],edgecolor='k',linewidth=0.4)
axD.set_xticks(range(3)); axD.set_xticklabels(['Motor\n(MDS-UPDRS-III)','Cognition\n(MoCA)','Autonomic\n(SCOPA-AUT)'],fontsize=6)
axD.axhline(0,c='k',lw=0.6); axD.set_ylabel('Partial r (DDC)'); axD.set_title('DDC vs baseline severity',fontsize=8,loc='left',pad=10)
lo_,hi_=min(min(rr),0),max(max(rr),0); axD.set_ylim(lo_-0.08,hi_+0.08)
for i,(n_,r_,p_) in enumerate(res1):
    axD.text(i,r_+(0.01 if r_>0 else -0.01),'*' if p_<0.05 else '',ha='center',va='bottom' if r_>0 else 'top',fontsize=9)
# (E 面板 iRBD 分类器已按审稿意见移除; 结果仍存于 derived/ 与正文一句说明)
for ext,dpi in [('svg',None),('pdf',None),('png',600),('tiff',500)]:   # QA: 与其他图统一 500 dpi
    kw={'pil_kwargs':{'compression':'tiff_lzw'}} if ext=='tiff' else {}
    fig.savefig(os.path.join(FIG,f'Figure5_diagnostic_ML.{ext}'),dpi=dpi,bbox_inches='tight',**kw)
plt.close(fig); print('\nFigure5 已保存')
pd.DataFrame([(k,)+tuple(str(x) for x in v) for k,v in aucs.items()],columns=['marker','AUC','lo','hi']).to_csv(D+"/diagnostic_panel_AUC.csv",index=False)
