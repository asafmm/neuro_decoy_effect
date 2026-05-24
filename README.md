## Neural similarity between choice options predicts group-level context effects
Code and data for the manuscript "Neural similarity between choice options predicts group-level context effects" by Asaf Madar, Tom Zemer, Ido Tavor* & Dino J Levy*.

### Overview
The study includes two tasks:

(Task 1) Lottery evaluation task - willingness-to-pay task over 31 unique lotteries.

(Task 2) Decoy task - multi-alternative choice task, where participants choose which lottery they would like to participate in. Participants are assigned either to a binary group, choosing between a target (A) and competitor (B), or a trinary group, with an additional "decoy" option (C). The task includes 27 unique lottery sets, constructed by the 31 lotteries from task 1.

Both tasks were completed by two types of samples:

(1) Behavioral sample (n=122) completing both tasks in a computer lab.

(2) fMRI samples - first (n=28) and replication (n=34) samples, completing the lottery evaluation task inside the fMRI, and the decoy task outside the scanner.

The fMRI analyses derive Representational Dissimilarity Matrices (RDMs) from individual-subject responses to 31 lotteries and use their average, via cross-validated regression, to predict each lottery set's behavioral decoy effect.

### Data
The analyzed fMRI data (~50GB) is provided via the Open Science Framework (OSF) at: https://osf.io/uex4m/

To run the notebooks, please download the "data" folder from OSF, and replace it with the data folder from this repo.
To access the ROIs used throughout the paper, please refer to the "mri_masks" folder in OSF. Note that it is not required in order to run the code.

### Repository layout
- `code/behavioral_computational/` — analysis of the online multi-alternative choice experiment and fits of computational context-effect models.
  - `read_files.py` — utilities for loading raw behavioral CSVs, screening subjects, and computing per-set choice ratios and decoy effects.
  - `choice_main.ipynb` — main behavioral analyses: aggregates subjects, computes the decoy effect per lottery set, and produces the behavioral summary figures.
  - `computational_models.ipynb` — fits and compares context-effect models (Adaptive Gain, Divisive Normalization, Range Normalization, etc.) to the observed decoy effects.
- `code/mri/` — fMRI analyses producing the figures of the paper.
  - `fig2_fitting.ipynb` — stepwise RDM regression that selects ROIs whose representational geometry predicts the decoy effect.
  - `fig2_CV_visualization.ipynb` — visualizes the cross-validated predictions of the selected model.
  - `fig2_predictions.ipynb` — leave-one-lottery-out predictions of the decoy effect from neural RDMs.
  - `fig3_attribute_representation_levels.ipynb` — relates ROI RDMs to attribute-based (amount, probability) reference RDMs.
  - `fig4_rep_geometry.ipynb` — analyses of representational geometry (effective dimensionality, PCA-based geometry features).
  - `utils/` — supporting modules used by the notebooks (see module docstrings for details):
    - `load_params.py` — loads subject lists, lottery sets, behavioral results, and saved subject representations.
    - `lottery_sets.py` — `Lottery` and `Set` classes describing single lotteries and target/competitor/decoy triplets.
    - `mri_subject.py` — `Subject` class encapsulating an individual's choice data, motion, representations, and RDM.
    - `rdms.py` — RDM construction and helpers (distance metrics, normalization, plotting).
    - `rdm_regression.py` — `RDMRegression` class that predicts behavioral decoy effects from per-set neural dissimilarities.
    - `stepwise_rdm.py` — forward stepwise ROI selection for the RDM regression model.
    - `read_subjects_data.py` — builds `Subject` objects with the appropriate run exclusions and computes their RDMs in parallel.
    - `visualization.py` — writes Schaefer-parcellation results to CIFTI surface files for cortical visualization.
- `stimuli/` — CSV definitions of the binary and trinary lottery sets and of the individual lotteries used in the fMRI evaluation task.
- `results/` — pre-computed behavioral tables and model results referenced by the fMRI notebooks (e.g., `decoy_table.csv`).

### Running the code
The notebooks are intended to be run with the `data/` folder placed at the repository root (downloaded from OSF). Each notebook is self-contained; loading helpers in `code/mri/utils/load_params.py` resolve all data paths relative to the notebook directory. A standard scientific Python stack is required (pandas, numpy, scipy, statsmodels, scikit-learn, matplotlib, seaborn, pingouin, nibabel, tqdm, joblib).
