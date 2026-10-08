# -*- coding: utf-8 -*-
"""Step 0b: SomaScan CSF data (PPMI Project 151, Proteomic Working Group curated release), baseline visit.

Outputs (derived/):
  somascan151_csf.parquet   PATNO, EVENT_ID and one column per aptamer target (log2 RFU), baseline only
  somascan151_lookup.csv    target annotation (Lookup sheet)
"""
import sys
import pandas as pd
from config import PWG_DIR, DERIVED
sys.stdout.reconfigure(encoding='utf-8')

src = PWG_DIR / 'ppmi_project_151.xlsx'
res = pd.read_excel(src, sheet_name='Results', dtype=str)
soma = res[res.EVENT_ID == 'BL'].copy()
soma['PATNO'] = soma.PATNO.astype(int)
for c in soma.columns[2:]:
    soma[c] = pd.to_numeric(soma[c], errors='coerce')
soma.reset_index(drop=True).to_parquet(DERIVED / 'somascan151_csf.parquet', index=False)
lk = pd.read_excel(src, sheet_name='Lookup').dropna(axis=1, how='all')
lk.to_csv(DERIVED / 'somascan151_lookup.csv', index=False)
print('SomaScan baseline samples', len(soma), 'targets', soma.shape[1] - 2)
