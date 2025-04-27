import nibabel as nib
import numpy as np
import pandas as pd
import os
import pickle
import glob
import re
import matplotlib.pyplot as plt

subcortex_labels = ['ACCUMBENS_LEFT',
 'ACCUMBENS_RIGHT',
 'AMYGDALA_LEFT',
 'AMYGDALA_RIGHT',
 'BRAIN_STEM',
 'CAUDATE_LEFT',
 'CAUDATE_RIGHT',
 'CEREBELLUM_LEFT',
 'CEREBELLUM_RIGHT',
 'DIENCEPHALON_VENTRAL_LEFT',
 'DIENCEPHALON_VENTRAL_RIGHT',
 'HIPPOCAMPUS_LEFT',
 'HIPPOCAMPUS_RIGHT',
 'PALLIDUM_LEFT',
 'PALLIDUM_RIGHT',
 'PUTAMEN_LEFT',
 'PUTAMEN_RIGHT',
 'THALAMUS_LEFT',
 'THALAMUS_RIGHT']

def write_schaefer_results_to_surface(schaefer_df, result_column, output_file_path):
    # read schaefer atlas and labels
    schaefer = nib.load('../../mri_data/schaefer/Asaf_Schaefer400_19SubCortex.dtseries.nii')
    with open('/Volumes/HCP/Parcellation/schafer/Schaefer2018_400Parcels_7Networks_order_info.txt', 'r') as f:
        text_labels = f.read()
    text_labels = text_labels.split('\n')
    text_labels = text_labels[:-1:2]
    schaefer_data = schaefer.get_fdata().copy()
    schaefer_labels = np.unique(schaefer_data)  
    # write schaefer results to file
    if 'label' not in schaefer_df.columns:
        raise('dataframe needs "label" columns with schaefer parcel number')
    elif type(schaefer_df.label.values[0]) != float:
        try:
            schaefer_df.label = schaefer_df.label.astype(float)
        except ValueError:
            schaefer_df = schaefer_df.rename({'label':'label_str'}, axis=1)
            schaefer_df.loc[:, 'label'] = schaefer_df.label_str.str.split('_').str[1].astype(float)
    schaefer_copy = schaefer_data.copy()
    for label in schaefer_labels:
        if label==0:
            # skip the background
            continue
        result = schaefer_df.loc[schaefer_df.label==label, result_column].values[0]
        schaefer_copy[schaefer_copy==label] = result
    # r2_path = '/Volumes/homes/Asaf/python/decoy/mri_data/first_schaefer419_effective_dims.dtseries.nii'
    new_image = nib.Cifti2Image(schaefer_copy, schaefer.header)
    nib.save(new_image, output_file_path)
    print(f'Saved dataframe to surface in {output_file_path}')