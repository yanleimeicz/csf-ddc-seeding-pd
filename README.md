# CSF DOPA decarboxylase, α-synuclein seeding and dopamine-transporter loss in Parkinson's disease

Analysis code for:

> Yan L, Xing Y, Yang Y, Zhao M, Xu L, for the Alzheimer's Disease Neuroimaging Initiative.
> *Cerebrospinal fluid DOPA decarboxylase relates to α-synuclein seeding more than to dopamine-transporter loss in Parkinson's disease.* (submitted)

The scripts reproduce every analysis, table value and figure in the paper from the original PPMI and ADNI downloads:
a discovery set (PPMI Project 277, CSF Olink Explore HT, n = 733), a non-overlapping replication set on the same platform
(PPMI Project 314, n = 980), other PPMI platforms (SomaScan, Olink Target 96, plasma Olink, urine mass spectrometry) and an
external test in ADNI (CSF NULISAseq CNS Disease Panel 120 and Amprion α-synuclein SAA, n = 702).

**This repository contains code only.** No participant-level data, derived tables or participant identifiers are included.

## Data access

Both data sets are available to qualified researchers on application; they cannot be redistributed.

**PPMI** (www.ppmi-info.org, RRID:SCR_006431; files downloaded 2 July 2026). Place these files in `data/ppmi/`:

| File | Content |
|---|---|
| `PPMI_Curated_Data_Cut_Public_20260511.xlsx` | Curated clinical data cut (sheet `20260511`) |
| `PPMI_Project_277_CSF_Screened_NPX_20260130.parquet` | Discovery set, CSF Olink Explore HT |
| `ppmi_proj314_csf_screened_extended_npx_20260116.parquet` | Replication set, CSF Olink Explore HT |
| `ppmi_proj293_plasma_screened_extended_npx_20251121.parquet` | Plasma Olink Explore HT |
| `PPMI_Project_9000_CSF_*_NPX_02Jul2026.csv` | CSF Olink Target 96 panels |
| `PPMI_Project_222_CSF_*_NPX_02Jul2026.csv`, `PPMI_Project_196_CSF_*_NPX_02Jul2026.csv` | CSF Olink Target 96 (Project 222) and Explore 1536 (Project 196); used only to identify participants in earlier releases |
| `Targeted___untargeted_MS-based_proteomics_of_urine_in_PD_*_02Jul2026.csv` | Urine mass spectrometry (Project 190) |
| `Proteomic_Working_Group_Curated_Dataset(multi-project)/ppmi_project_151.xlsx` | CSF SomaScan (Project 151, PWG release 2024-05-29) |

**ADNI** (adni.loni.usc.edu; tables downloaded 4 October 2026). Place these files in `data/adni/`:

| Path | Content |
|---|---|
| `nulisa/BSHRI_PLA_CSF_NULISA_CNS_04Oct2026.csv` | NULISAseq CNS Disease Panel 120 (CSF and plasma) |
| `AMPRION_ASYN_SAA_04Oct2026.csv` | Amprion α-synuclein SAA |
| `download2.zip` → `UPENNBIOMK_ROCHE_ELECSYS_04Oct2026.csv` | Elecsys CSF Aβ42 and p-tau181 |
| `download3.zip` → `RECCMEDS_04Oct2026.csv` | Concomitant medications |
| `raw/ADNIMERGE2_tables/DXSUM.rda`, `PTDEMOG.rda`, `APOERES.rda` | Diagnosis, demographics, APOE (ADNIMERGE2 R package tables) |

ADNI participant-level data were analysed on the investigators' computer only, and the ADNI script writes summary
statistics only, as required by the ADNI Data Use Agreement.

Folder locations can be changed with environment variables (`CSF_DDC_PPMI_DIR`, `CSF_DDC_ADNI_DIR`, `CSF_DDC_DERIVED`, …)
or an untracked `scripts/config_local.py`; see `scripts/config.py`.

## Software

Python 3.14.5 on Windows 11 with the package versions in `requirements.txt`:

```
pip install -r requirements.txt
```

Scripts 02, 02b and 29 query online resources (Enrichr gene-set libraries through gseapy, and the STRING API), and script 30
downloads the Harvard-Oxford atlas through nilearn. Random seeds are fixed throughout; the pre-ranked GSEA in script 02 uses multithreaded permutations, so its permutation P and q values can differ in the third decimal place between runs (enrichment scores are identical). Parallel steps use joblib
(`NJ = 30` workers in several scripts; reduce this on smaller machines). Some inline comments are in Chinese.

## Running

Run the scripts from the `scripts/` folder in numerical order:

```
cd scripts
python 00_build_discovery_dataset.py
python 00_build_somascan151.py
python 01_channel_signatures.py
...
python 30_figure_ring_brain.py
```

Each script reads earlier outputs from `derived/` and writes its own tables to `derived/`, logs to `results/` and figures
to `figures/`.

| Script | Analysis | Paper items |
|---|---|---|
| 00_build_discovery_dataset | Discovery CSF matrix (QC), clinical table, group contrasts | all discovery analyses |
| 00_build_somascan151 | SomaScan CSF baseline data | Supplementary Fig. S2, Table S13 |
| 01_channel_signatures | Proteome-wide association with RBDSQ, UPSIT, SCOPA-AUT | Fig. 2, Table S3 |
| 02_gsea_nsd_trajectory | GSEA (iRBD vs hyposmia); NSD-stage trajectories | Supplementary Fig. S1, Tables S9, S12 |
| 02b_enrichr_rbd_signature | Enrichr over-representation of the RBD signature | Supplementary Fig. S8, Table S23 |
| 03_signature_score | Composite RBD-signature score | Supplementary Fig. S3 |
| 04_ddc_discovery | DDC by group, stage, binding ratio | Fig. 1, Supplementary Fig. S3, Table S14 |
| 05_plasma_ddc | Plasma versus CSF DDC | Supplementary Fig. S3e, Table S14 |
| 06_olink_target_panels | Olink Target 96 panels (Project 9000) | Supplementary Fig. S2d, Table S10 |
| 07_diagnostic_panels | Diagnostic panels and classifiers | Supplementary Fig. S4, Table S15 |
| 08_rbd_confounding_power | Confounding versus power for the RBD signature | Fig. 2, Tables S4 to S6 |
| 09_nsd_components_progression | NSD components, subtype definitions, progression | Tables S8, S14, S18, S20 |
| 10_somascan_urine | SomaScan and urine DDC; Olink–SomaScan concordance | Supplementary Fig. S2, Table S13 |
| 11_replication_ddc_hypotheses | Replication of H1 to H5 | Table S11 |
| 12_replication_rbd_signature | Replication of the RBD signature | Fig. 3 |
| 13_replication_positive_controls | Platform positive controls | Fig. 3, Table S7 |
| 14_replication_genetic | Genetic carriers, treatment | Fig. 4e, Table S16 |
| 15_cohorts_lod_tobit_bridge | Cohort composition, Tobit, same-sample bridge | Table 1, Fig. 4, Tables S2, S11, S15, S17 |
| 16_treatment_within_person | Within-person treatment effect | Table S22 |
| 17_rbd_per_point_replication | Per-point replication, winner's-curse correction, permutation | Fig. 3, Table S7 |
| 18_progression_genetic_contrasts | Progression and genetic contrasts | Fig. 4, Tables S11, S16, S21 |
| 19_prespecified_verdicts_overlap | Pre-specified verdicts, overlap with earlier releases | Table 2, Tables S1, S2 |
| 20_rbd_group_adjusted_null | Group-adjusted permutation and simulation | Fig. 3, Table S7 |
| 21_longitudinal_event_study | Longitudinal event study | Supplementary Fig. S7, Table S22 |
| 22_progression_pooled | Pooled progression models | Table 2, Supplementary Fig. S6, Tables S1, S21 |
| 23_seed_sbr_nsd_detection | Seed status, binding ratio, NSD stage | Table 2, Fig. 4, Tables S2, S11, S17 |
| 24_lrrk2_sbr | Binding ratio in LRRK2-associated PD | Table S16 |
| 25_sensitivity_prior_releases_r2 | Exclusion of earlier releases; bootstrap unique R² | Fig. 4c, Tables S11, S17 |
| 26_adni_external_test | ADNI external test (E1 to E4) | Fig. 5, Supplementary Fig. S5, Table S19 |
| 27_figures_discovery | Discovery figures | Fig. 1, Fig. 2, Supplementary Figs. S1 to S3 |
| 28_figures_replication | Replication figures | Figs. 3, 4, Supplementary Figs. S6, S7 |
| 29_figure_enrichment_network | Enrichment and STRING network | Supplementary Fig. S8 |
| 30_figure_ring_brain | Protein × binding-ratio ring and striatal map | Supplementary Fig. S9, Table S24 |

Figure files are written under working names; their numbers in the paper are:

| Output file (figures/) | Paper |
|---|---|
| Figure1_cohort | Fig. 1 |
| Figure2_RBD_channel | Fig. 2 |
| Figure_R3_Fig3_RBD_replication | Fig. 3 |
| Figure_R3_Fig4_DDC_two_sets | Fig. 4 |
| Figure_R7_Fig5_ADNI | Fig. 5 |
| FigureS1_GSEA_discovery | Supplementary Fig. S1 |
| Figure3_axis_validation | Supplementary Fig. S2 |
| Figure4_DDC | Supplementary Fig. S3 |
| Figure5_diagnostic_ML | Supplementary Fig. S4 |
| Figure_R7_FigS5_ADNI_panel_scan | Supplementary Fig. S5 |
| Figure_R3_Fig5_progression | Supplementary Fig. S6 |
| Figure_R3_Fig6_longitudinal_treatment | Supplementary Fig. S7 |
| Figure7_molecular_characterization | Supplementary Fig. S8 |
| Figure8_ring_brain | Supplementary Fig. S9 |

Tables 1 and 2 and Supplementary Tables S1 to S24 were assembled from the `derived/` tables named above.

## Analysis plans

The replication plan (Supplementary Note 1) and the ADNI analysis plan with its two amendments (Supplementary Note 3)
are part of the paper's supplementary material.

## Participant lists

Lists of PPMI participant numbers (PATNO) defining each analysis set are not posted here, because PPMI data may be shared
only with registered users. Approved PPMI users can obtain them from the corresponding author (Ligang Xu).

## Licence and citation

Code: MIT licence (see `LICENSE`). Please cite the paper when using this code.
