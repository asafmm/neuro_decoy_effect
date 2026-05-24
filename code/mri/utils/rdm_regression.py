"""Linear regression fitting per-set behavioral decoy effects from
neural RDMs.

For each ROI, every subject contributes three pairwise distances per
lottery set (target-competitor, target-decoy, competitor-decoy). After
normalizing each subject's full RDM and averaging across subjects,
these three columns (per included ROI) become the predictors in an OLS
regression whose target is the empirical decoy effect of each set.
"""

import numpy as np
import pandas as pd
from utils import rdms
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import MissingDataError

class RDMRegression():
    """OLS regression of decoy effects on per-set neural representational dissimilarities.

    Parameters:
        subjects (list[Subject]): subjects whose RDMs feed into the model.
        rois (list[str] | None): ROIs whose RDMs serve as predictors. If
            ``None`` the model reduces to an intercept-only baseline.
        sets (list[Set]): lottery sets to predict (target variable is
            ``set.decoy_effect``).

    Workflow:
        1. ``__init__`` z-scores each subject's RDM and averages across
           subjects to produce per-set features.
        2. `fit` runs OLS on the design matrix.
        3. `transform_set` / `predict` project a held-out
           lottery set onto the fitted model.
    """

    def __init__(self, subjects, rois, sets):
        self.subjects = subjects
        self.rois = rois
        self.sets = sets
        self.y = [s.decoy_effect for s in self.sets]
        self.set_nums = [s.set_num for s in self.sets]
        self.lottery_ids = np.unique([s.lottery_ids for s in self.sets])
        if self.rois is not None:
            self._get_avg_rdms()
        else:
            self.rdms = {}
            self.full_rdms = {}
            self.rois_mean = {}
            self.rois_std = {}
        
    def _get_avg_rdms(self):
        """Z-score each subject's RDM and average them into per-set features.

        Stores group-mean per-set RDM features in ``self.rdms[roi]`` and
        the full 31x31 group RDM in ``self.full_rdms[roi]``. The per-
        subject mean/std used for z-scoring are cached in
        ``self.rois_mean`` / ``self.rois_std`` so that held-out sets can
        be normalized with the same statistics.
        """
        avg_rdms = {}
        std_rdms = {}
        full_avg_rdms = {}
        rois_mean = {}
        rois_std = {}
        for roi in self.rois:
            subjects_full_rdms = []
            subjects_set_rdms = []
            subjects_rdm_means = {}
            subjects_rdm_stds = {}
            for subject in self.subjects:
                # if the regression is using only a subset of the lotteries, use subset of the RDM
                rdm_subset = subject.RDM[roi].loc[self.lottery_ids, self.lottery_ids]
                subject_norm_rdm, subject_rdm_mean, subject_rdm_std = rdms.normalize_RDM(rdm_subset, return_stats=True)
                subjects_rdm_means[subject.sub_num] = subject_rdm_mean
                subjects_rdm_stds[subject.sub_num] = subject_rdm_std
                subjects_full_rdms.append(subject_norm_rdm)
                set_rdms = rdms.get_set_RDMs_obj(subject_norm_rdm, self.sets, roi)
                subjects_set_rdms.append(set_rdms)
            rois_mean[roi] = subjects_rdm_means
            rois_std[roi] = subjects_rdm_stds
            subjects_full_rdms = np.array(subjects_full_rdms)
            full_avg_rdms[roi] = np.mean(subjects_full_rdms, axis=0)
            subjects_rdms = pd.concat(subjects_set_rdms, axis=0)
            avg_rdms[roi] = subjects_rdms.groupby(level=0).mean()
            std_rdms[roi] = subjects_rdms.groupby(level=0).std()
        self.rdms = avg_rdms
        self.full_rdms = full_avg_rdms
        self.rois_mean = rois_mean
        self.rois_std = rois_std
        self.std_rdms = std_rdms
    
    def fit(self, use_explicit=False):
        """Fit the OLS model and cache ``lm``, ``adj_r2`` and ``pval``.

        With ``use_explicit=True`` the design matrix is augmented with
        the explicit lottery attributes (amount and probability of each
        of the three options) to control for attribute-level effects.
        """
        X = []
        if use_explicit:
            attributes = pd.DataFrame(np.zeros((len(self.sets), 3*2)), columns=3*['amount', 'prob'], index=[s.set_num for s in self.sets])
            for i, set_obj in enumerate(self.sets):
                lotteries = [set_obj.target, set_obj.competitor, set_obj.decoy]
                for j, lottery in enumerate(lotteries):
                    attributes.iloc[i, 2*j:2*j+2] = [lottery.amount, lottery.prob]
            X = [attributes]
        if self.rois is not None:
            for roi in self.rois:
                X.append(self.rdms[roi])
            X = pd.concat(X, axis=1)
            intercept = pd.Series([1] * len(self.y), index=self.set_nums, name='intercept')
            self.X_for_lm = pd.concat([intercept, X], axis=1)
        else:
            intercept = pd.Series([1] * len(self.y), index=self.set_nums, name='intercept')
            self.X_for_lm = intercept
        try:
            lm = sm.OLS(self.y, self.X_for_lm).fit()
            self.lm = lm
            self.adj_r2 = lm.rsquared_adj
            self.pval = lm.f_pvalue
        except MissingDataError:
            print(f'{roi} has missing data')

    def transform_set(self, new_set):
        """Build a design-matrix row for one or more held-out lottery sets.

        For each new set, slices each subject's RDM to its three
        lotteries, normalizes with the subject's training-time mean/std,
        and averages across subjects. The resulting row(s) live in
        ``self.new_set_X`` and are also returned.
        """
        if type(new_set) == list:
            new_sets = new_set
        else:
            new_sets = [new_set]
        new_set_X = pd.DataFrame()
        for new_set in new_sets:
            avg_new_set = {}
            new_set_X_row = pd.DataFrame({'intercept':[1]}, index=[new_set.set_num])
            for roi in self.rois:
                new_set_rdms = []
                for subject in self.subjects:
                    new_set_rdm = subject.RDM[roi].loc[new_set.lottery_ids, new_set.lottery_ids]
                    new_set_rdm_norm = (new_set_rdm - self.rois_mean[roi][subject.sub_num]) / self.rois_std[roi][subject.sub_num]
                    new_set_rdm_flat = rdms.get_set_RDMs_obj(new_set_rdm_norm, [new_set], roi)
                    new_set_rdms.append(new_set_rdm_flat)
                avg_new_set[roi] = pd.concat(new_set_rdms, axis=0)
                avg_new_set[roi] = avg_new_set[roi].groupby(level=0).mean()
                new_set_X_row = pd.concat([new_set_X_row, avg_new_set[roi]], axis=1)
            new_set_X = pd.concat([new_set_X, new_set_X_row], axis=0)
        self.new_set_X = new_set_X
        return new_set_X

    def predict(self, new_set):
        """Predict the decoy effect of held-out set(s) using the fitted model.

        Fits the model lazily if it has not been fitted yet. Returns the
        model's predictions and the projected design matrix.
        """
        if not hasattr(self, 'lm'):
            self.fit()
        if self.rois is not None:
            self.transform_set(new_set)
            new_prediction = self.lm.predict(self.new_set_X)
        else:
            # if rois is None, the lm is constant
            set_nums = [set_i.set_num for set_i in new_set]
            self.new_set_X = pd.Series([1] * len(new_set), index=set_nums, name='intercept')
            new_prediction = self.lm.predict(self.new_set_X)
        return new_prediction, self.new_set_X