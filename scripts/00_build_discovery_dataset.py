# -*- coding: utf-8 -*-
"""Step 0: discovery data set (PPMI Project 277, baseline CSF, Olink Explore HT) and clinical table.

Outputs (derived/):
  csf_npx_wide.parquet      733 participants x 5,414 proteins (NPX; PATNO column + one column per OlinkID)
  protein_annotation.csv    OlinkID, UniProt, Assay
  analytic_clinical.csv     baseline clinical variables for all PPMI participants in the curated data cut
  diff_iRBD_vs_Hyposmia.csv per-protein iRBD vs hyposmia difference (NPX ~ group + age + sex)
  diff_PD_body_vs_brain.csv per-protein body-first vs brain-first PD difference (same model)

Quality control: baseline (BL) participant samples, assay rows only; values with Olink assay QC
'WARN' set to missing; proteins with > 20% and samples with > 30% missing values removed.
"""
import sys, warnings
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from config import PPMI_DIR, DERIVED, PPMI_CURATED, PPMI_CURATED_SHEET
warnings.filterwarnings('ignore'); sys.stdout.reconfigure(encoding='utf-8')

# ---------------------------------------------------------------- proteomics
t = pq.read_table(PPMI_DIR / 'PPMI_Project_277_CSF_Screened_NPX_20260130.parquet',
                  columns=['PATNO', 'EVENT_ID', 'SampleType', 'AssayType', 'AssayQC', 'OlinkID', 'UniProt', 'Assay', 'NPX'])
t = t.filter(pc.and_(pc.equal(t['SampleType'], 'SAMPLE'), pc.equal(t['EVENT_ID'], 'BL'))).to_pandas()
t = t[t.AssayType == 'assay'].copy()
t['PATNO'] = t.PATNO.astype(float).astype(int)
t.loc[t.AssayQC == 'WARN', 'NPX'] = np.nan
wide = t.pivot(index='PATNO', columns='OlinkID', values='NPX')
keep_p = wide.columns[wide.isna().mean(0) <= 0.20]
wide = wide[keep_p]
wide = wide[wide.isna().mean(1) <= 0.30]
print(f"proteins {t.OlinkID.nunique()} -> {wide.shape[1]}; participants {wide.shape[0]}; missing values {int(wide.isna().sum().sum())}")
wide.columns.name = None
wide.reset_index().to_parquet(DERIVED / 'csf_npx_wide.parquet', index=False)
ann = t[['OlinkID', 'UniProt', 'Assay']].drop_duplicates('OlinkID').set_index('OlinkID').reindex(wide.columns)
ann.index.name = 'OlinkID'
ann.reset_index().to_csv(DERIVED / 'protein_annotation.csv', index=False)

# ---------------------------------------------------------------- clinical (baseline row of the curated data cut)
COLS = ['PATNO', 'COHORT', 'subgroup', 'age', 'age_at_visit', 'age_LP', 'SEX', 'EDUCYRS', 'duration', 'agediag',
        'DOMSIDE', 'LEDD', 'rem', 'upsit', 'scopa', 'scopa_gi', 'scopa_cv', 'scopa_ur', 'CSFSAA', 'NSD_Status',
        'NSD_STAGE', 'moca', 'updrs1_score', 'updrs2_score', 'updrs3_score', 'IU_GFAP_plasma_mean', 'IU_NFL_plasma_mean']
cur = pd.read_excel(PPMI_CURATED, sheet_name=PPMI_CURATED_SHEET)
bl = cur[cur.EVENT_ID == 'BL'].drop_duplicates('PATNO').copy()

def group0(s):
    s = str(s)
    if s == 'Healthy Control' or 'Normosmic' in s: return 'HC'
    if s == 'Hyposmia': return 'Hyposmia'
    if 'RBD' in s: return 'iRBD'
    if s in ('Sporadic PD', 'SWEDD'): return 'PD'
    return 'GV/other'

clin = bl[COLS].copy()
clin.insert(1, 'group0', bl.subgroup.map(group0))
# PD split by RBD Screening Questionnaire: body-first RBDSQ >= 5, brain-first < 5
clin.insert(2, 'group', np.where(clin.group0 != 'PD', clin.group0,
                                 np.where(clin.rem >= 5, 'PD_bodyfirst', np.where(clin.rem < 5, 'PD_brainfirst', 'PD'))))
for side in ('PUTAMEN', 'CAUDATE'):   # dopamine-transporter asymmetry index |L - R| / mean(L, R)
    L, R = bl['MIA_%s_L' % side], bl['MIA_%s_R' % side]
    clin[side + '_ASYM'] = (L - R).abs() / ((L + R) / 2)
order = COLS[:1] + ['group0', 'group'] + COLS[1:COLS.index('NSD_STAGE') + 1] + ['PUTAMEN_ASYM', 'CAUDATE_ASYM'] + COLS[COLS.index('NSD_STAGE') + 1:]
clin = clin[order]
clin.to_csv(DERIVED / 'analytic_clinical.csv', index=False)
print('clinical table', clin.shape, clin.group.value_counts().to_dict())

# ---------------------------------------------------------------- group contrasts (complete cases per protein)
c = clin.set_index('PATNO').reindex(wide.index)
for fname, a, b in [('diff_iRBD_vs_Hyposmia.csv', 'iRBD', 'Hyposmia'),
                    ('diff_PD_body_vs_brain.csv', 'PD_bodyfirst', 'PD_brainfirst')]:
    s = c[c.group.isin([a, b])]
    g = (s.group == a).astype(float)
    rows = []
    for o in wide.columns:
        y = wide.loc[s.index, o]
        m = y.notna() & s.age.notna() & s.SEX.notna()
        X = sm.add_constant(pd.DataFrame({'g': g[m], 'age': s.age[m], 'sex': s.SEX[m]}))
        r = sm.OLS(y[m], X).fit()
        rows.append((o, r.params['g'], r.pvalues['g'], int(m.sum())))
    d = pd.DataFrame(rows, columns=['OlinkID', 'beta_bodyfirst', 'p', 'n'])
    d['gene'] = ann.reindex(d.OlinkID)['Assay'].values
    d['fdr'] = multipletests(d.p, method='fdr_bh')[1]
    d.to_csv(DERIVED / fname, index=False)
    print(fname, 'FDR<0.05:', int((d.fdr < 0.05).sum()))
