# -*- coding: utf-8 -*-
"""Paths used by every script.

Defaults assume this layout under the project root (the repository folder unless
CSF_DDC_ROOT is set):

    data/ppmi/      PPMI downloads (Olink, SomaScan, urine, curated data cut)
    data/adni/      ADNI downloads (NULISA, Amprion SAA, ADNIMERGE2 tables)
    derived/        intermediate tables written by the scripts
    results/        logs and summary results
    figures/        figure files

Any path can be overridden with an environment variable of the same name
(e.g. CSF_DDC_PPMI_DIR) or in an untracked config_local.py next to this file.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get('CSF_DDC_ROOT', Path(__file__).resolve().parents[1]))

PPMI_DIR = Path(os.environ.get('CSF_DDC_PPMI_DIR', ROOT / 'data' / 'ppmi'))
ADNI_DIR = Path(os.environ.get('CSF_DDC_ADNI_DIR', ROOT / 'data' / 'adni'))
DERIVED = Path(os.environ.get('CSF_DDC_DERIVED', ROOT / 'derived'))
RESULTS = Path(os.environ.get('CSF_DDC_RESULTS', ROOT / 'results'))
FIGURES = Path(os.environ.get('CSF_DDC_FIGURES', ROOT / 'figures'))
CACHE = Path(os.environ.get('CSF_DDC_CACHE', ROOT / 'cache'))   # nilearn atlas, STRING responses

# PPMI files (names as downloaded from LONI; data freeze used in the paper)
PPMI_CURATED = Path(os.environ.get('CSF_DDC_PPMI_CURATED', PPMI_DIR / 'PPMI_Curated_Data_Cut_Public_20260511.xlsx'))
PPMI_CURATED_SHEET = '20260511'
PWG_DIR = PPMI_DIR / 'Proteomic_Working_Group_Curated_Dataset(multi-project)'

try:
    from config_local import *  # noqa: F401,F403  (untracked local overrides)
except ImportError:
    pass

for _d in (DERIVED, RESULTS, FIGURES, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

# string forms for scripts that build paths by concatenation
P = str(PPMI_DIR); A = str(ADNI_DIR); D = str(DERIVED); RS = str(RESULTS); F = str(FIGURES)
CURATED = str(PPMI_CURATED)
