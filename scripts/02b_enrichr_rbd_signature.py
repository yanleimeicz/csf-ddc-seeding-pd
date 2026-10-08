# -*- coding: utf-8 -*-
"""Over-representation analysis (Enrichr) of the 55 CSF proteins associated with RBDSQ at FDR < 0.05.

Input:  derived/sig_rem.csv (01_channel_signatures.py)
Output: derived/enrich_multi.csv (Supplementary Table S23, Supplementary Fig. S8)
Requires internet access to the Enrichr API (gseapy). Multi-gene Olink assays (e.g. CKMT1A_CKMT1B)
are split into their gene symbols.
"""
import sys
import numpy as np, pandas as pd
import gseapy as gp
from config import DERIVED
sys.stdout.reconfigure(encoding='utf-8')

LIBS = ['GO_Biological_Process_2021', 'GO_Cellular_Component_2021', 'GO_Molecular_Function_2021',
        'KEGG_2021_Human', 'TRRUST_Transcription_Factors_2019']
sig = pd.read_csv(DERIVED / 'sig_rem.csv')
genes = sorted({g for a in sig.loc[sig.fdr < 0.05, 'gene'].dropna() for g in str(a).split('_')})
print('genes submitted:', len(genes))
enr = gp.enrichr(gene_list=genes, gene_sets=LIBS, organism='human', outdir=None)
r = enr.results.copy()
r['neglog10'] = -np.log10(r['Adjusted P-value'])
r['nGenes'] = r.Genes.str.split(';').str.len()
r = r.sort_values(['Gene_set', 'Adjusted P-value'], kind='stable')
r[['Gene_set', 'Term', 'Adjusted P-value', 'neglog10', 'nGenes', 'Genes']].to_csv(DERIVED / 'enrich_multi.csv', index=False)
print('terms:', len(r), ' FDR<0.05:', int((r['Adjusted P-value'] < 0.05).sum()))
