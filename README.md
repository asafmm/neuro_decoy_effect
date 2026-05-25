# Neural similarity between choice options predicts group-level context effects

Code and data for the manuscript *"Neural similarity between choice options predicts group-level context effects"* by **Asaf Madar, Tom Zemer, Ido Tavor\*, and Dino J. Levy\***.

---

## Overview

The study includes **two tasks**:

| # | Task | Description |
|---|------|-------------|
| 1 | **Lottery evaluation** | Willingness-to-pay task over 31 unique lotteries. |
| 2 | **Decoy** | Multi-alternative choice task. Participants are assigned to a *binary* group, choosing between a target (A) and a competitor (B), or a *trinary* group, with an additional *decoy* option (C). The task includes 27 unique lottery sets, constructed from the 31 lotteries of Task 1. |

Both tasks were completed by **two types of samples**:

| Sample | n | Setting |
|--------|---|---------|
| Behavioral | 122 | Both tasks completed in a computer lab |
| fMRI (first) | 28 | Lottery evaluation in scanner; decoy task outside (not used in the study) |
| fMRI (replication) | 34 | Lottery evaluation in scanner; decoy task outside (not used in the study) |

The fMRI analyses derive **Representational Dissimilarity Matrices (RDMs)** from individual-subject responses to the 31 lotteries and use their average, via cross-validated regression, to predict each lottery set's decoy effect from the behavioral sample.

---

## Data

The analyzed fMRI data (~50 GB) is provided via the **Open Science Framework (OSF)**: <https://osf.io/uex4m/>

> To run the notebooks, download the `data` folder from OSF and place it at the repository root, replacing the `data` folder from this repo.

The ROIs used throughout the paper are available in the `mri_masks` folder on OSF. They are *not* required to run the code.

---

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Core dependencies: `numpy`, `pandas`, `scipy`, `statsmodels`, `scikit-learn`, `matplotlib`, `seaborn`, `pingouin`, `nibabel`, `tqdm`, `joblib`, `torch`, `torchvision`.

---

## Repository layout

```
neuro_decoy_effect/
├── code/
│   ├── behavioral_computational/    # Decoy choice task analysis & computational models
│   └── mri/                         # fMRI analyses & figure notebooks
│       └── utils/                   # Supporting modules
├── stimuli/                         # Lottery and lottery-set definitions (CSV)
├── results/                         # Pre-computed behavioral tables & model results
└── data/                            # behavioral WTP and fMRI data (download from OSF)
```

### `code/behavioral_computational/`
Analysis of the decoy choice experiment and fits of computational context-effect models.

| File | Description |
|------|-------------|
| `read_files.py` | Utilities for loading raw behavioral CSVs, screening subjects, and computing per-set choice ratios and decoy effects. |
| `choice_main.ipynb` | Main behavioral analyses: aggregates subjects, computes the decoy effect per lottery set, and produces the behavioral summary figures. |
| `computational_models.ipynb` | Fits and compares context-effect models (Adaptive Gain, Divisive Normalization, Range Normalization, etc.) to the observed decoy effects. |

### `code/mri/`
fMRI analyses producing the figures of the paper.

| Notebook | Description |
|----------|-------------|
| `fig2_fitting.ipynb` | Stepwise RDM regression that selects ROIs whose representational geometry predicts the decoy effect. |
| `fig2_CV_visualization.ipynb` | Visualizes the cross-validated predictions of the selected model. |
| `fig2_predictions.ipynb` | Leave-one-lottery-out predictions of the decoy effect from neural RDMs. |
| `fig3_attribute_representation_levels.ipynb` | Relates ROI RDMs to attribute-based (amount, probability) RDMs. |
| `fig4_rep_geometry.ipynb` | Analyses of representational geometry (effective dimensionality). |

### `code/mri/utils/`
Supporting modules used by the notebooks (see module docstrings for details).

| Module | Description |
|--------|-------------|
| `load_params.py` | Loads subject lists, lottery sets, behavioral results, and saved subject representations. |
| `lottery_sets.py` | `Lottery` and `Set` classes describing single lotteries and target/competitor/decoy triplets. |
| `mri_subject.py` | `Subject` class encapsulating an individual's choice data, motion, representations, and RDM. |
| `rdms.py` | RDM construction and helpers (distance metrics, normalization, plotting). |
| `rdm_regression.py` | `RDMRegression` class that predicts behavioral decoy effects from per-set neural dissimilarities. |
| `stepwise_rdm.py` | Forward stepwise ROI selection for the RDM regression model. |
| `read_subjects_data.py` | Builds `Subject` objects with the appropriate run exclusions and computes their RDMs in parallel. |
| `visualization.py` | Writes Schaefer-parcellation results to CIFTI surface files for cortical visualization. |

### `stimuli/` and `results/`
- **`stimuli/`** — CSV definitions of the binary and trinary lottery sets and of the individual lotteries used in the fMRI evaluation task.
- **`results/`** — pre-computed behavioral tables and model results referenced by the fMRI notebooks (e.g., `decoy_table.csv`).

---

## Running the code

The notebooks are intended to be run with the `data/` folder placed at the repository root (downloaded from OSF). Each notebook is self-contained; loading helpers in `code/mri/utils/load_params.py` resolve all data paths relative to the notebook directory.
