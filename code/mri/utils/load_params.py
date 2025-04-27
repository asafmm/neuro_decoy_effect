import os
import re
import pickle
import pandas as pd
import numpy as np
import glob
from utils import lottery_sets

SET_NUMBERS = np.arange(1, 28)
LOTTERY_IDS = np.arange(1, 32)

def load_subject_numbers(sample_type='first'):
    if sample_type == 'first':
        subject_numbers = np.arange(1, 32)
        subject_numbers = ['{:03d}'.format(i) for i in subject_numbers]
        # remove 006 because of artifacts in all task scans
        subject_numbers.remove('006')
    elif sample_type == 'replication':
        subject_numbers = np.arange(32, 71)
        subject_numbers = ['{:03d}'.format(i) for i in subject_numbers]
    elif sample_type == 'all':
        pass
    return subject_numbers

def create_set_dicts():
    lotteries = pd.read_csv('../../stimuli/evaluation_stimuli.csv')
    lotteries.loc[:, 'EV'] = lotteries.prob * lotteries.amount / 100
    evs = lotteries.EV.values.flatten()
    amounts = lotteries.amount.values.flatten()
    probs = lotteries.prob.values.flatten()
    set_ids = lotteries[['lottery_id', 'set']]
    option_letters = ['A', 'B', 'C']
    set_dicts = {}
    set_ids_lists = [this_set_id.split('-') for this_set_id in set_ids.set]
    for set_num in SET_NUMBERS:
        set_dict = {}
        for letter in option_letters:
            option_str = str(set_num) + letter
            option_str_bool = [option_str in set_id_list for set_id_list in set_ids_lists]
            option_str_id = set_ids.loc[option_str_bool, 'lottery_id'].values[0]
            set_dict[letter] = option_str_id
        set_dicts[set_num] = set_dict
    return set_dicts

def load_set_dicts():
    with open('../../data/lottery_sets/set_dicts.pkl', 'rb') as f:
        set_dicts = pickle.load(f)
    return set_dicts

def load_lotteries():
    with open('../../data/lottery_sets/lotteries.pkl', 'rb') as f:
        lottery_objs = pickle.load(f)
    return lottery_objs

def load_behavior_results(path='../../results/decoy_table.csv'):
    behavior_results = pd.read_csv(path, index_col=0)
    return behavior_results

def load_sets(behavior_results=None):
    if behavior_results is None:
        behavior_results = load_behavior_results()
    set_dicts = load_set_dicts()
    lottery_objs = load_lotteries()
    set_objs = np.zeros(len(set_dicts), dtype=object)
    for i, set_num in enumerate(set_dicts.keys()):
        set_dict = set_dicts[set_num]
        target_id = set_dict['A'] - 1
        competitor_id = set_dict['B'] - 1
        decoy_id = set_dict['C'] - 1
        target = lottery_objs[target_id]
        competitor = lottery_objs[competitor_id]
        decoy = lottery_objs[decoy_id]
        decoy_effect = behavior_results.loc[set_num, 'decoy_effect_A']
        target_ratio_binary = behavior_results.loc[set_num, 'A_ratio_binary']
        target_ratio_ternary = behavior_results.loc[set_num, 'A_ratio_trinary']
        s = lottery_sets.Set(set_num, target, competitor, decoy, decoy_effect, target_ratio_binary, target_ratio_ternary)
        set_objs[i] = s
    return set_objs

def load_samples(roi_type='roi'):
    if roi_type=='roi':
        with open('../../data/subject_representations/first_subjects.pkl', 'rb') as f:
            first_subjects = pickle.load(f)
        with open('../../data/subject_representations/replication_subjects.pkl', 'rb') as f:
            replication_subjects = pickle.load(f)
    elif roi_type=='schaefer':
        with open('../../data/subject_representations/first_subjects_schaefer.pkl', 'rb') as f:
            first_subjects = pickle.load(f)
        with open('../../data/subject_representations/replication_subjects_schaefer.pkl', 'rb') as f:
            replication_subjects = pickle.load(f)
    else:
        raise(ValueError, 'roi_type should be "roi" or "schaefer"')
    return first_subjects, replication_subjects