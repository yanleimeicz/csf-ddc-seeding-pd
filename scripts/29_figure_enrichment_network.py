# -*- coding: utf-8 -*-
"""Enrichment and STRING network figure for the RBD signature (Supplementary Fig. S8).
Inputs: derived/enrich_multi.csv, derived/sig_rem.csv; queries the STRING API (responses cached).
"""
import config as cfg
import os, io, re, math, numpy as np, pandas as pd, warnings, requests
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import networkx as nx
from adjustText import adjust_text
def trunc(name,a,b,n=256):
    base=plt.cm.get_cmap(name); return mpl.colors.LinearSegmentedColormap.from_list(name+'_t',base(np.linspace(a,b,n)))
D=cfg.D; FIG=cfg.F
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
 'svg.fonttype':'none','pdf.fonttype':42,'font.size':7,'axes.spines.right':False,
 'axes.spines.top':False,'axes.linewidth':0.8,'axes.titlesize':9.5,'legend.frameon':False,
 'text.color':'black','axes.labelcolor':'black','axes.titlecolor':'black',
 'xtick.color':'black','ytick.color':'black','axes.edgecolor':'black',
 'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold','mathtext.default':'regular'})
OKABE={'BP':'#0072B2','CC':'#E69F00','MF':'#CC79A7','KEGG':'#009E73'}; TF_COL='#3A3A3A'
def plabel(ax,l,x=-0.02,y=1.01): ax.text(x,y,l.lower(),transform=ax.transAxes,fontsize=10,fontweight='bold',va='bottom',ha='right')   # QA: 小写, 与主图一致
def lum_text(rgba): return 'white' if (0.299*rgba[0]+0.587*rgba[1]+0.114*rgba[2])<0.55 else 'black'
def fill_axes(ax,pos,mx=0.12):
    xs=[p[0] for p in pos.values()]; ys=[p[1] for p in pos.values()]
    dx=(max(xs)-min(xs))*mx or 0.1; dy=(max(ys)-min(ys))*mx or 0.1
    ax.set_xlim(min(xs)-dx,max(xs)+dx); ax.set_ylim(min(ys)-dy,max(ys)+dy)

sig=pd.read_csv(D+"/sig_rem.csv"); sig=sig[sig.fdr<0.05]
neglogfdr={}
for _,r in sig.iterrows():
    v=-math.log10(max(float(r['fdr']),1e-300))
    for g in str(r['gene']).split('_'): neglogfdr[g]=v
genes=sorted(set(g for gg in sig.gene.astype(str) for g in gg.split('_') if g and g!='nan'))
en=pd.read_csv(D+"/enrich_multi.csv")
def clean(t):
    t=re.sub(r'\s*\(GO:\d+\)','',t); t=re.sub(r'\s+R-HSA-\d+','',t); t=t.replace(' human','').replace(' mouse','')
    t=t.replace('TGF-beta','TGF-β'); return t[:1].upper()+t[1:]

fig=plt.figure(figsize=(7.2,8.6))   # QA: 190 mm 通栏; 顶行轴右移使术语标签留在画布内, 底行 b/c 用满全宽
gsT=gridspec.GridSpec(1,1,figure=fig,left=0.50,right=0.98,top=0.97,bottom=0.60)
gsB=gridspec.GridSpec(1,2,figure=fig,left=0.02,right=0.97,top=0.50,bottom=0.02,wspace=0.12,width_ratios=[0.95,1.05])

# ================= A 富集点图 (左侧整列) =================
axA=fig.add_subplot(gsT[0,0]); fig.text(0.01,0.975,'a',fontsize=10,fontweight='bold',va='bottom')
cats=[('GO_Biological_Process_2021','GO:BP',OKABE['BP']),('GO_Cellular_Component_2021','GO:CC',OKABE['CC']),
      ('GO_Molecular_Function_2021','GO:MF',OKABE['MF']),('KEGG_2021_Human','KEGG',OKABE['KEGG'])]
rows=[]
for gs_id,lab,col in cats:
    for _,x in en[en.Gene_set==gs_id].sort_values('Adjusted P-value').head(3).iterrows():
        rows.append((clean(x['Term']),x['neglog10'],x['nGenes'],lab,col))
rows=rows[::-1]
for i,(term,nl,ng,lab,col) in enumerate(rows):
    axA.scatter(nl,i,s=48+ng*30,c=col,edgecolors='white',linewidths=0.8,zorder=3)
axA.set_yticks(range(len(rows))); axA.set_yticklabels([r[0] for r in rows],fontsize=7)
axA.set_xlabel('Enrichment significance (−log$_{10}$ adjusted P)',fontsize=7)
axA.set_title('Pathway and Gene Ontology enrichment',fontsize=8,loc='left',pad=5)
axA.grid(axis='x',ls=':',lw=0.5,alpha=0.55); axA.set_ylim(-0.7,len(rows)-0.3); axA.set_xlim(left=0); axA.tick_params(length=3)
leg=[Line2D([0],[0],marker='o',ls='',mfc=c,mec='white',mew=0.6,ms=8,label=l) for _,l,c in cats]
_ng=[r[2] for r in rows]
leg+=[Line2D([0],[0],marker='o',ls='',mfc='#9a9a9a',mec='white',ms=np.sqrt(48+min(_ng)*30)*0.85,label=f'{min(_ng)} genes'),
      Line2D([0],[0],marker='o',ls='',mfc='#9a9a9a',mec='white',ms=np.sqrt(48+max(_ng)*30)*0.85,label=f'{max(_ng)} genes')]
axA.legend(handles=leg,fontsize=6.5,loc='lower right',handletextpad=0.3,labelspacing=0.6,borderpad=0.6)

# ================= B TRRUST 转录因子网络 (右上) =================
axB=fig.add_subplot(gsB[0,0]); plabel(axB,'B',x=0.0,y=1.02); axB.axis('off')
axB.set_title('   Transcription factors (TRRUST)',fontsize=8,loc='left',pad=4)
tf=en[en.Gene_set=='TRRUST_Transcription_Factors_2019'].sort_values('Adjusted P-value').head(4)
G=nx.Graph(); tfn=[]
for _,x in tf.iterrows():
    t=x['Term'].split()[0]; tfn.append(t); G.add_node(t,kind='tf')
    for g_ in str(x['Genes']).split(';'):
        g_=g_.strip()
        if g_: G.add_node(g_,kind='tgt'); G.add_edge(t,g_)
posB=nx.spring_layout(G,seed=5,k=1.7,iterations=500)
for n in posB: posB[n]=(posB[n][0]*1.8,posB[n][1]*1.0)
nx.draw_networkx_edges(G,posB,ax=axB,edge_color='#cccccc',width=1.0)
tgt=[n for n in G if G.nodes[n]['kind']=='tgt']; tvals=np.array([neglogfdr.get(n,0) for n in tgt])
cmapB=trunc('YlOrRd',0.10,0.78); nB=mpl.colors.Normalize(vmin=tvals.min(),vmax=tvals.max())
nx.draw_networkx_nodes(G,posB,nodelist=tgt,ax=axB,node_size=120,node_color=tvals,cmap=cmapB,vmin=tvals.min(),vmax=tvals.max(),edgecolors='white',linewidths=0.8)
nx.draw_networkx_nodes(G,posB,nodelist=tfn,ax=axB,node_shape='s',node_size=1250,node_color=TF_COL,edgecolors='white',linewidths=1.1)
for n,(xx,yy) in posB.items():
    if n in tfn: axB.text(xx,yy,n,fontsize=6.2,fontweight='bold',color='white',ha='center',va='center')
    else: axB.text(xx,yy+0.14,n,fontsize=6,ha='center',va='bottom',color='black',bbox=dict(boxstyle='round,pad=0.05',fc='white',ec='none',alpha=0.8))
fill_axes(axB,posB,0.16)
cbB=fig.colorbar(ScalarMappable(norm=nB,cmap=cmapB),ax=axB,orientation='horizontal',fraction=0.035,pad=0.02,shrink=0.6)
cbB.set_label('−log$_{10}$ FDR (RBD association)',fontsize=6.5); cbB.ax.tick_params(labelsize=6)
axB.legend(handles=[Line2D([0],[0],marker='s',ls='',mfc=TF_COL,mec='white',ms=9,label='transcription factor'),
                    Line2D([0],[0],marker='o',ls='',mfc='#F0A030',mec='white',ms=8,label='target protein')],
           fontsize=6.5,loc='upper left',handletextpad=0.3,bbox_to_anchor=(0,1.02))

# ================= C STRING PPI 网络 (右下, HERO, 全节点标注) =================
axC=fig.add_subplot(gsB[0,1]); plabel(axC,'C',x=0.0,y=1.02); axC.axis('off')
axC.set_title('   Protein–protein interactions (STRING)',fontsize=8,loc='left',pad=4)
try:
    _cache=D+"/string_network_55.tsv"   # QA: 缓存 STRING 结果, 保证可复现
    if os.path.exists(_cache): net=pd.read_csv(_cache,sep='\t')
    else:
        rr=requests.post("https://string-db.org/api/tsv/network",
            data={'identifiers':'%0d'.join(genes),'species':9606,'required_score':400,'caller_identity':'ppmi'},timeout=90)
        net=pd.read_csv(io.StringIO(rr.text),sep='\t'); net.to_csv(_cache,sep='\t',index=False)
    P=nx.Graph()
    for g in genes: P.add_node(g)
    for _,e in net.iterrows():
        a,b=e['preferredName_A'],e['preferredName_B']
        if a in P and b in P: P.add_edge(a,b,w=float(e['score']))
    deg=dict(P.degree()); Pc=P.subgraph([n for n in P if deg[n]>0])
    hub=sorted(Pc.nodes(),key=lambda x:-deg[x])[:6]
    posC=nx.spring_layout(Pc,seed=42,k=2.6,iterations=1500)   # 大k更疏
    nx.draw_networkx_edges(Pc,posC,ax=axC,edge_color='#d2d2d2',width=[Pc[u][v]['w']*1.7 for u,v in Pc.edges()],alpha=0.65)
    dvals=np.array([deg[n] for n in Pc.nodes()]); cmapC=trunc('GnBu',0.10,0.66); nC=mpl.colors.Normalize(vmin=dvals.min(),vmax=dvals.max())
    nx.draw_networkx_nodes(Pc,posC,ax=axC,node_size=[25+deg[n]*25 for n in Pc.nodes()],
        node_color=dvals,cmap=cmapC,vmin=dvals.min(),vmax=dvals.max(),edgecolors='#5a5a5a',linewidths=0.5)
    # 浅色系节点, 黑字直接可读, 无描边; hub粗体居中, 其余adjustText排开
    texts=[]
    for n,(xx,yy) in posC.items():
        if n in hub:
            texts.append(axC.text(xx,yy,n,fontsize=7,fontweight='bold',color='black',ha='center',va='center',zorder=8,bbox=dict(boxstyle='round,pad=0.05',fc='white',ec='none',alpha=0.75)))
        else:
            texts.append(axC.text(xx,yy,n,fontsize=6,color='black',ha='center',va='center',zorder=7,bbox=dict(boxstyle='round,pad=0.05',fc='white',ec='none',alpha=0.75)))
    fill_axes(axC,posC,0.10)
    adjust_text(texts,ax=axC,expand=(1.3,1.5),force_text=(0.5,0.6),iter_lim=2000,
                arrowprops=dict(arrowstyle='-',color='#9a9a9a',lw=0.4))
    cbC=fig.colorbar(ScalarMappable(norm=nC,cmap=cmapC),ax=axC,orientation='horizontal',fraction=0.035,pad=0.02,shrink=0.6)
    cbC.set_label('network degree',fontsize=6.5); cbC.ax.tick_params(labelsize=6)
except Exception as ex:
    axC.text(0.5,0.5,'STRING unavailable',ha='center'); print('STRING fail',repr(ex)[:120])

# QA: 删除图内脚注 (旧框架与旧图号); 说明放图注
for ext,dpi in [('svg',None),('pdf',None),('png',300),('tiff',500)]:
    kw={'pil_kwargs':{'compression':'tiff_lzw'}} if ext=='tiff' else {}
    fig.savefig(os.path.join(FIG,f'Figure7_molecular_characterization.{ext}'),dpi=dpi,bbox_inches='tight',**kw)
plt.close(fig); print('Fig6 (A-left, B/C-right, C labelled) saved | hubs:', hub if 'hub' in dir() else 'NA')
