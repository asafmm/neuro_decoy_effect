import numpy as np
import pandas as pd
from utils import rdms
import copy
import matplotlib.pyplot as plt
import os
import pingouin
import pickle

class Subject():
    def __init__(self, sub_num, age, female, right_hand, runs=np.arange(1, 6)):
        # general info
        self.sub_num = sub_num
        self.age = age
        self.female = female
        self.right_hand = right_hand
        self.runs = runs
        # motion
        self.motion = self._read_motion()
        self.average_motion = np.mean(self.motion, axis=0) # average move per run 
        self.motion.loc[:, 'time'] = self.motion.index*0.75
        # choice behavior
        self.all_trials = pd.read_csv(f'../../data/choices/{sub_num}_all_runs.csv')
        # remove blank trials
        self.blank_trials = self.all_trials.lottery_id==0
        self.trials = self.all_trials[~self.blank_trials]
        # exclude runs, if necessary
        self.trials = self.trials.loc[self.trials.run.isin(runs)]
        
        self.choices = self.trials.choice
        self.rt = self.trials.rt
        self.missed_trials = self.rt.isna()
        self.num_missed = np.sum(np.isnan(self.rt))
        self.EVs = self.trials.EV
        self.ev_choice_corr = self._calc_ev_choice_corr()
        self._add_movements_to_trials()
        self._calc_self_consistency()
        # exclude?
        self.to_exclude = False
        if self.num_missed / len(self.trials) > 0.3:
            # more than 30% missed trials
            self.to_exclude = True
        if self.icc < 0.6 or self.mean_self_corr < 0.6:
            # inconsistent choices
            self.to_exclude = True
    
    def _calc_self_consistency(self):
        '''
        calculate average correlation and ICC between the choices of all runs
        '''
        subject_runs = pd.DataFrame()
        for run in self.runs:
            run_df = self.trials.loc[self.trials.run==run]
            run_df = run_df.sort_values('lottery_id')
            subject_runs.loc[:, f'run{run}'] = run_df.choice.values
        ICC_df = pingouin.intraclass_corr(data=self.trials, targets='lottery_id', raters='run', ratings='choice', nan_policy='omit')
        self.icc = ICC_df.ICC[0]
        n_runs = len(subject_runs.columns)
        tril = np.tril(np.ones((n_runs, n_runs), dtype=bool), -1)
        self.mean_self_corr = np.mean(subject_runs.corr()[tril])

    def _calc_ev_choice_corr(self):
        '''
        calculate the correlation between the EVs and the choices
        '''
        non_missed_choices = self.choices[~self.missed_trials]
        non_missed_EVs = self.EVs[~self.missed_trials]
        self.ev_corr = np.corrcoef(non_missed_choices, non_missed_EVs)[0, 1]
        return self.ev_corr
    
    def _read_motion(self, move_type='RelativeRMS'):
        motion_df = pd.read_csv(f'../../data/motion/{self.sub_num}_{move_type}.csv')
        return motion_df

    def _add_movements_to_trials(self):
        '''
        merge the movements data (per TR) to the trials data (per trial)
        find the maximal movement during each trial and add to the trials' dataframe
        '''
        for run_i in self.all_trials.run.unique():
            run_df = self.all_trials.loc[self.all_trials.run == run_i]
            run_rms_df = self.motion.loc[:, [f'run{run_i}', 'time']]
            for trial_i in run_df.index:
                trial_view_time = run_df.loc[trial_i, 'view_time']
                ev_duration = run_df.loc[trial_i, 'ev_duration']
                trial_end_time = trial_view_time + ev_duration
                # find the maximal movement during the trial
                trial_rms = run_rms_df.loc[(run_rms_df.time >= trial_view_time) & (run_rms_df.time <= trial_end_time), f'run{run_i}']
                self.all_trials.loc[trial_i, 'trial_rms'] = np.max(trial_rms)
        return self.all_trials

    def read_representations(self, mask_dict_type='roi', map_type='zstat', design_prefix='view_unified', verbose=False):
        '''
        read subject's representations from all their task fMRI maps
        map_type: 'zstat' (contrast maps of stimulus-blank) or 'pe' (parameter estimates of stimulus)
        mask_dict_ype: 'roi' for pre-defined ROIs, 'schaefer419' for Schaefer 400+19 subcortex atlas
        '''
        dir_to_load = f'../../data/subject_representations/{mask_dict_type}/{design_prefix}/{map_type}'
        path_to_load = f'{dir_to_load}/{self.sub_num}_{map_type}_{mask_dict_type}.pkl'
        if os.path.exists(path_to_load):
            if verbose:
                print(f'loading {self.sub_num} from file: {path_to_load}')
            with open(path_to_load, 'rb') as f:
                roi_rep_dict = pickle.load(f)
            self.representations = roi_rep_dict
        else:
            raise(ValueError, f'import subject representations to {dir_to_load}.\nAlternatively, just load the whole sample pkl file (e.g., data/first_subjects.pkl)')

    def _filter_representation_runs(self):
        '''
        filter the representations to include only runs with no problems in behavior/fMRI data, based on self.runs
        '''
        nan_representation = copy.deepcopy(self.representations)
        for roi in self.representations.keys():
            nan_representation[roi][:, :, :] = np.nan
            nan_representation[roi][:, :, self.runs-1] = self.representations[roi][:, :, self.runs-1]
        self.representations = nan_representation
        
    def _filter_representations_with_motion(self, threshold=1.5):
        '''
        remove representations with high movements of rms > threshold (default 1.5)
        '''
        high_motion_trials = self.all_trials.loc[self.all_trials.trial_rms > threshold]
        self.excluded_trial_num = len(high_motion_trials)
        for high_motion_run in high_motion_trials.run.unique():
            high_motion_lotteries = high_motion_trials.loc[high_motion_trials.run == high_motion_run].lottery_id.values
            # filter only lotteries and not blanks (ID of blanks is 32)
            high_motion_lotteries = high_motion_lotteries[high_motion_lotteries<=31]
            for roi in self.representations.keys():
                self.representations[roi][high_motion_lotteries-1, :, high_motion_run-1] = np.nan
                
    def calc_RDM(self, mask_dict_type='roi', distance_metric='euclidean', motion_thresh=0.75, verbose=False, map_type='zstat', design_prefix='view_unified'):
        '''
        calculate the representational dissimilarity matrix (RDM) based on the subject's average representations
        '''
        if not hasattr(self, 'representations'):
            self.read_representations(mask_dict_type=mask_dict_type, verbose=verbose, map_type=map_type, design_prefix=design_prefix)
        self._filter_representation_runs()
        if motion_thresh:
            self._filter_representations_with_motion(threshold=motion_thresh)
        self.avg_representation = rdms.get_avg_rep(self.representations)
        self.RDM = rdms.create_subject_rdm_dict(self.avg_representation, distance_metric=distance_metric)

    def plot_RDM(self, roi):
        '''
        plot the RDM of the subject
        '''
        fig, ax, cax = rdms.plot_rdm(self.RDM[roi], title=f'Subject {self.sub_num}, {roi}')
        return fig, ax, cax
            
    def plot_subject_stats(self):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
        # choices
        axs[0].hist(self.all_trials.choice, color='lightcoral')
        axs[0].set_title('Choice distribution', fontsize=18)
        axs[0].set_xlabel('Choice (NIS)', fontsize=16)
        axs[0].set_xlim(-5, 85)
        axs[0].tick_params(axis='x', labelsize=13)
        axs[0].tick_params(axis='y', labelsize=13)
        # RT and misses
        n_blank_trials = len(self.all_trials[self.all_trials.amount==0])
        missed_trials = self.all_trials.rt.isna().sum() - n_blank_trials
        axs[1].hist(self.all_trials.rt, color='darkseagreen')
        axs[1].set_title(f'RT distribution (missed: {missed_trials})', fontsize=18)
        axs[1].set_xlabel('RT (s)', fontsize=16)
        axs[1].set_xlim(0, 5.5)
        axs[1].tick_params(axis='x', labelsize=13)
        axs[1].tick_params(axis='y', labelsize=13)
        # EV sensitivity
        axs[2].scatter(self.all_trials.EV, self.all_trials.choice, color='steelblue', alpha=0.5)
        axs[2].set_title(f'EV sensitivity (r={self.ev_choice_corr:.2f})', fontsize=18)
        axs[2].set_xlabel('EV', fontsize=16)
        axs[2].set_ylabel('Choice', fontsize=16)
        axs[2].set_ylim(-5, 85)
        axs[2].tick_params(axis='x', labelsize=13)
        axs[2].tick_params(axis='y', labelsize=13)
        plt.suptitle(f'Subject {self.sub_num}', fontsize=18)
        plt.show()

    def plot_motion(self):
        mean_rms = np.mean(self.motion.iloc[:, :-1], axis=0)
        mean_rms = np.round(mean_rms, 2)
        plt.plot(self.motion.iloc[:, :-1], alpha=0.7)
        plt.title(f'Subject {self.sub_num} ({mean_rms.values})', fontsize=10)
        plt.ylabel(f'Movement RelativeRMS')
        plt.show()