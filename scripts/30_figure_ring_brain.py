# -*- coding: utf-8 -*-
"""Protein x striatal binding ring heatmap and striatal map (Supplementary Fig. S9, Table S24).
Uses the Harvard-Oxford atlas from nilearn (downloaded to the cache folder).
"""
import config as cfg
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from scipy import stats
from statsmodels.stats.multitest import multipletests
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.cm import ScalarMappable
import nibabel as nib
from nilearn import datasets, plotting
D=cfg.D; FIG=cfg.F
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
 'svg.fonttype':'none','pdf.fonttype':42,'font.size':7,
 'text.color':'black','axes.labelcolor':'black','axes.titlecolor':'black',
 'xtick.color':'black','ytick.color':'black','axes.edgecolor':'black',
 'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold','mathtext.default':'regular'})
wide=pd.read_parquet(D+"/csf_npx_wide.parquet").set_index('PATNO')
ann=pd.read_csv(D+"/protein_annotation.csv").set_index('OlinkID')
sig=pd.read_csv(D+"/sig_rem.csv"); sig=sig[sig.fdr<0.05].copy()
cur=pd.read_excel(cfg.CURATED,sheet_name='20260511')
bl=cur[cur.EVENT_ID=='BL'].set_index('PATNO'); clin=pd.read_csv(D+"/analytic_clinical.csv").set_index('PATNO')
dat=pd.DataFrame(index=wide.index)
for c in ['MIA_CAUDATE_L','MIA_CAUDATE_R','MIA_PUTAMEN_L','MIA_PUTAMEN_R','MIA_STRIATUM_L','MIA_STRIATUM_R']:
    dat[c]=bl[c].reindex(wide.index)
dat['Striatum_mean']=dat[['MIA_STRIATUM_L','MIA_STRIATUM_R']].mean(1)
cols=[('MIA_CAUDATE_L','Caudate L'),('MIA_CAUDATE_R','Caudate R'),('MIA_PUTAMEN_L','Putamen L'),
      ('MIA_PUTAMEN_R','Putamen R'),('Striatum_mean','Striatum mean')]
age=clin['age'].reindex(wide.index).values.astype(float); sex=clin['SEX'].reindex(wide.index).values.astype(float)
pid=sig['OlinkID'].tolist(); genes=sig.set_index('OlinkID')['gene']
B=np.full((len(pid),len(cols)),np.nan); Pv=np.full_like(B,np.nan)
for i,oid in enumerate(pid):
    x=wide[oid].values.astype(float)
    for j,(col,_) in enumerate(cols):
        y=dat[col].values.astype(float); m=~(np.isnan(x)|np.isnan(y)|np.isnan(age)|np.isnan(sex))
        if m.sum()<50: continue
        xz=(x[m]-x[m].mean())/x[m].std(); yz=(y[m]-y[m].mean())/y[m].std()
        Xd=np.column_stack([np.ones(m.sum()),xz,age[m],sex[m]]); bb=np.linalg.lstsq(Xd,yz,rcond=None)[0]
        resid=yz-Xd@bb; dof=m.sum()-4; se=np.sqrt((resid@resid)/dof*np.linalg.inv(Xd.T@Xd)[1,1])
        B[i,j]=bb[1]; Pv[i,j]=2*stats.t.sf(abs(bb[1]/se),dof)
flat=Pv.flatten(); ok=~np.isnan(flat); fdr=np.full_like(flat,np.nan); fdr[ok]=multipletests(flat[ok],method='fdr_bh')[1]
FDR=fdr.reshape(Pv.shape); bonf=0.05/ok.sum()
keep=np.where(np.nansum(FDR<0.05,axis=1)>0)[0]
order=keep[np.argsort(np.nanmin(Pv[keep],axis=1))]
Bk=B[order]; Fk=FDR[order]; Pk=Pv[order]; glab=[str(genes[pid[k]]) for k in order]
# 每个解剖脑区显著蛋白数(4个ROI, 用全部55蛋白)
roicols={'Caudate L':0,'Caudate R':1,'Putamen L':2,'Putamen R':3}
cnt={r:int(np.nansum(FDR[:,j]<0.05)) for r,j in roicols.items()}
print('每ROI显著蛋白数:',cnt,'| 圆环蛋白数:',len(order))
# 导出 Fig7 支撑表(补充表S7): 各蛋白 × 纹状体区域 标准化β, 最小P, FDR显著区域数
_lab=[c[1] for c in cols]
_exp=pd.DataFrame(Bk,columns=[f'beta_{l}' for l in _lab],index=glab); _exp.index.name='Protein'
_exp.insert(0,'n_sig_regions_FDR',np.nansum(Fk<0.05,axis=1).astype(int))
_exp.insert(1,'min_P',np.nanmin(Pk,axis=1))
_exp[[c for c in _exp.columns if c!='min_P']]=_exp[[c for c in _exp.columns if c!='min_P']].round(4)   # QA: min_P 不再取整 (原先舍成 0, 表中显示 <1e-300)
_exp.to_csv(D+"/protein_DAT_full.csv"); print('protein_DAT_full.csv 已导出(S7)')

fig=plt.figure(figsize=(7.3,4.3))   # QA: 按 190 mm 通栏绘制, 最小字号 6 pt
gs=gridspec.GridSpec(1,2,figure=fig,width_ratios=[1.3,1.0],wspace=0.22)

# ===== A 圆环热图 =====
axA=fig.add_subplot(gs[0],projection='polar'); axA.set_theta_zero_location('N'); axA.set_theta_direction(-1)
N=len(order); M=len(cols); r0=2.6
gap=np.radians(58)   # QA: 加宽顶部留白, 环标签不压色块  # 顶部留白扇区,用于放脑区(环)标签,避免文字压在深色格上
theta=np.linspace(gap/2,2*np.pi-gap/2,N+1); r=np.arange(M+1)+r0
C=Bk.T  # (regions, proteins)
vmax=np.nanmax(np.abs(Bk)); cmap=plt.cm.RdBu_r; cmap.set_bad('#f2f2f2')
axA.pcolormesh(theta,r,np.ma.masked_invalid(C),cmap=cmap,vmin=-vmax,vmax=vmax,edgecolors='white',linewidth=0.4)
# 显著性点
tc=(theta[:-1]+theta[1:])/2; rc=(r[:-1]+r[1:])/2
for i in range(N):
    for j in range(M):
        if np.isnan(Bk[i,j]): continue
        if Pk[i,j]<bonf: axA.scatter(tc[i],rc[j],s=9,c='k',marker='o',zorder=5)
        elif Fk[i,j]<0.05: axA.scatter(tc[i],rc[j],s=3,c='k',marker='o',zorder=5)
# 蛋白标签(外圈) —— 按屏幕实际角度(theta_zero=N,顺时针)计算旋转,保证全部正向可读
for i,g in enumerate(glab):
    ang=tc[i]; screen=(90-np.degrees(ang))%360  # 屏幕上从东逆时针的角度
    if 90<screen<270: rot=screen-180; ha='right'
    else:             rot=screen;     ha='left'
    axA.text(ang,r[-1]+0.25,str(g).replace('_','/'),rotation=rot,rotation_mode='anchor',ha=ha,
             va='center',fontsize=6)
# 环(脑区)标签 —— 放在顶部留白扇区(白底,居中,黑字),不再压在色块上
for j,(_,lab) in enumerate(cols):
    axA.text(0,rc[j],lab,fontsize=6,ha='center',va='center',color='black',fontweight='medium',zorder=6,
             bbox=dict(boxstyle='round,pad=0.12',fc='white',ec='none',alpha=0.92))
axA.set_rlim(0,r[-1]+1.3); axA.set_xticks([]); axA.set_yticks([]); axA.spines['polar'].set_visible(False)
axA.set_title('CSF proteins × dopamine-transporter binding',fontsize=8,loc='left',pad=22)
axA.text(-0.08,1.13,'a',transform=axA.transAxes,fontsize=10,fontweight='bold',va='bottom')
cb=fig.colorbar(ScalarMappable(norm=mpl.colors.Normalize(-vmax,vmax),cmap=cmap),ax=axA,fraction=0.032,pad=0.13,shrink=0.5)
cb.set_label('Standardised β',fontsize=6.5); cb.ax.tick_params(labelsize=6)
# QA: 点和环的含义移入图注

# ===== B 纹状体脑图 =====
axB=fig.add_subplot(gs[1]); axB.set_title('Associated proteins per striatal region',fontsize=8,loc='left',pad=6)
axB.text(-0.06,1.02,'b',transform=axB.transAxes,fontsize=10,fontweight='bold',va='bottom')
try:
    ho=datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm',data_dir=str(cfg.CACHE))   # 规则 2: 数据不放 C 盘
    atlas=nib.load(ho.maps) if isinstance(ho.maps,str) else ho.maps
    adata=np.asarray(atlas.dataobj); stat=np.zeros(adata.shape,dtype=float)
    for idx,rname in [(5,'Caudate L'),(16,'Caudate R'),(6,'Putamen L'),(17,'Putamen R')]:
        stat[adata==idx]=cnt[rname]
    statimg=nib.Nifti1Image(stat,atlas.affine)
    tmpl=datasets.load_mni152_template(resolution=1)  # 1mm 高清 MNI152 背景
    disp=plotting.plot_stat_map(statimg,bg_img=tmpl,display_mode='z',cut_coords=[6],dim=-0.1,cmap='YlOrRd',
         colorbar=False,axes=axB,annotate=False,threshold=0.5,black_bg=False,vmax=max(cnt.values()),
         resampling_interpolation='continuous')
    axB.text(0.5,-0.06,'Axial slice, z = 6 mm',transform=axB.transAxes,ha='center',fontsize=6.5,color='black')
    for xx_,tt_ in [(0.04,'L'),(0.96,'R')]: axB.text(xx_,0.5,tt_,transform=axB.transAxes,ha='center',va='center',fontsize=8,fontweight='bold')
    _vm=max(cnt.values()); cbB=fig.colorbar(ScalarMappable(norm=mpl.colors.Normalize(0,_vm),cmap=plt.cm.YlOrRd),ax=axB,fraction=0.04,pad=0.02,shrink=0.6,ticks=list(range(0,_vm+1,5)))
    cbB.set_label('No. of proteins (FDR < 0.05)',fontsize=6.5); cbB.ax.tick_params(labelsize=6)
except Exception as ex:
    axB.axis('off'); axB.text(0.5,0.5,'brain render failed:\n'+repr(ex)[:120],ha='center',fontsize=7); print('brain fail',repr(ex)[:200])

# QA: 删除图内脚注 (已撤回的 denervation 表述与旧图号)
for ext,dpi in [('svg',None),('pdf',None),('png',300),('tiff',500)]:
    kw={'pil_kwargs':{'compression':'tiff_lzw'}} if ext=='tiff' else {}
    fig.savefig(os.path.join(FIG,f'Figure8_ring_brain.{ext}'),dpi=dpi,bbox_inches='tight',**kw)
plt.close(fig); print('Figure8 ring+brain saved')
