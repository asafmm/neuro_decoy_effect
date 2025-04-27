import os
import re
import pickle
import tqdm
import pandas as pd
import numpy as np
import glob
import datetime
import matplotlib.pyplot as plt
import nibabel as nib
import scipy.io

SET_NUMBERS = np.arange(1, 28)
LOTTERY_IDS = np.arange(1, 32)

def calc_euclidean(a):
    # calculate the euclidean distance between all pairs of rows in a matrix
    sum_sq = np.sum(a**2, axis=1, keepdims=True)
    euc_sq = sum_sq + sum_sq.T - 2*np.dot(a, a.T)
    return euc_sq

def calc_corr_dist(a):
    # calculate the correlation distance between all pairs of rows in a matrix
    corr_dist = 1 - np.corrcoef(a)
    return corr_dist

def get_avg_rep(representations_dict):
    avg_representation = representations_dict.copy()
    for roi, reps in avg_representation.items():
        avg_representation[roi] = np.nanmean(reps, axis=2)
    return avg_representation

def get_set_RDMs_obj(rdm, sets, roi):
    set_numbers = [s.set_num for s in sets]
    set_RDMs = pd.DataFrame({f'{roi}_Target_Competitor':np.nan, f'{roi}_Target_Decoy':np.nan, f'{roi}_Competitor_Decoy':np.nan}, index=set_numbers)
    for lottery_set in sets:
        set_stimuli = lottery_set.lottery_ids
        set_RDM = rdm.loc[set_stimuli, set_stimuli]
        tril_ind = np.tril_indices(set_RDM.shape[0], k=-1)
        flat_RDM = set_RDM.values[tril_ind]
        set_RDMs.loc[lottery_set.set_num, :] = flat_RDM
    return set_RDMs
   
def get_RDM(stimuli_reps, dist='correlation'):
    if dist=='correlation':
        non_set_RDM = calc_corr_dist(stimuli_reps)
    elif dist=='euclidean':
        non_set_RDM = calc_euclidean(stimuli_reps)
    return non_set_RDM

def create_subject_rdm_dict(avg_representation, distance_metric='euclidean'):
    roi_rdm_dict = {}
    for roi in avg_representation.keys():
        if avg_representation[roi].shape[1] == 0:
            print(f'empty array, passing {roi}')
            continue
        non_set_rdm = get_RDM(avg_representation[roi], dist=distance_metric)
        if np.all(np.isnan(non_set_rdm)):
            print(f'NaN RDM, passing {roi}')
            continue
        non_set_rdm = pd.DataFrame(non_set_rdm, index=LOTTERY_IDS, columns=LOTTERY_IDS)
        roi_rdm_dict[roi] = non_set_rdm
    return roi_rdm_dict

def normalize_RDM(roi_rdm, return_stats=False):
    rdm_mean = np.mean(roi_rdm.values)
    rdm_std = np.std(roi_rdm.values)
    norm_rdm = (roi_rdm - rdm_mean) / rdm_std
    if return_stats:
        return norm_rdm, rdm_mean, rdm_std
    else:
        return norm_rdm

def plot_rdm(rdm, title=None):
    fig ,ax = plt.subplots(figsize=(10, 10), dpi=150)
    cax = ax.imshow(rdm, cmap='Purples')
    # plt.xticks(np.arange(0, rdm.shape[0], 2), np.arange(1, rdm.shape[0]+1, 2))
    ax.set_title(title, fontsize=42, pad=25)
    ax.set_xticks([])
    ax.set_yticks(np.arange(0, rdm.shape[0], 2), np.arange(1, rdm.shape[0]+1, 2), fontsize=24)
    return fig, ax, cax