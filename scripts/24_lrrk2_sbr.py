# -*- coding: utf-8 -*-
"""Striatal binding in SAA-negative versus SAA-positive LRRK2-associated PD (replication set baseline).
Output: derived/r5_lrrk2_sbr.csv (Table S16 note).
"""
import config as cfg
import sys, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
D=cfg.D
r=pd.read_csv(D+"/p314_rep_baseline_analytic.csv")
hc=r[(r.group0=='HC')&(r.subgroup=='Healthy Control')].SBR.dropna(); cut=hc.quantile(.05)
L=r[(r.subgroup=='LRRK2')&r.CSFSAA.isin([0,1])]
rows=[]
for (c,s),d in L.groupby(['COHORT','CSFSAA']):
    x=d.SBR.dropna()
    rows.append(dict(group={1:'LRRK2 PD',4:'LRRK2 non-manifest carrier'}.get(c,c),SAA='positive' if s==1 else 'negative',n=len(d),n_SBR=len(x),
                     SBR_mean=x.mean(),SBR_sd=x.std(),pct_deficit=100*(x<cut).mean()))
out=pd.DataFrame(rows); out.to_csv(D+"/r5_lrrk2_sbr.csv",index=False); print(f"control 5th percentile {cut:.3f}"); print(out.round(3).to_string(index=False))
