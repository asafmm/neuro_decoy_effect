"""Helpers for projecting Schaefer-parcellated results onto a CIFTI surface.

The ``subcortex_labels`` list enumerates the 19 subcortical structures
appended to the 400-parcel Schaefer atlas used in this study, in the
same order they appear in the CIFTI label file.
"""

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
    """Write a per-parcel scalar to a CIFTI dtseries for surface visualization.

    Replaces each Schaefer parcel index in the reference CIFTI with the
    value taken from ``schaefer_df[result_column]`` (matched on the
    ``label`` column) and saves the modified image to
    ``output_file_path``. Useful for producing brain maps of regression
    statistics on the Schaefer 400+19 atlas.
    """
    # read schaefer atlas and labels
    schaefer = nib.load('../../mri_masks/schaefer/Schaefer400_19SubCortex.dtseries.nii')
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
    new_image = nib.Cifti2Image(schaefer_copy, schaefer.header)
    nib.save(new_image, output_file_path)
    print(f'Saved dataframe to surface in {output_file_path}')