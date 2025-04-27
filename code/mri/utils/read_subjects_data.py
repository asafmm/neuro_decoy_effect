import tqdm
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from utils import mri_subject, rdms, load_params, stepwise_rdm

def get_subjects_object(subject_numbers):
    subjects = []
    info_survey_pilot = pd.read_csv('../../results/replication_results/survey_details.csv', index_col=0)
    info_survey_replication = pd.read_csv('../../results/first_results/survey_details.csv', index_col=0)
    info_survey = pd.concat([info_survey_pilot, info_survey_replication])
    # subject_numbers = subject_numbers[30:]
    for sub_num in subject_numbers:
        if sub_num=='002':
            # technical problem in scan #2
            runs = np.array([1, 3, 4, 5])
        elif sub_num=='009':
            # tiredness, did 4 runs
            runs = np.array([1, 2, 3, 4])
        elif sub_num=='022':
            # did not understand task in run 1
            runs = np.array([2, 3, 4, 5])
        elif sub_num=='035':
            # tiredness, did 4 runs
            runs = np.array([1, 2, 3, 4])
        elif sub_num=='039':
            # subject currently removed - bad brain extraction 
            print(f'Excluding {sub_num} due to bad brain extraction')
            continue
        elif sub_num=='052':
            # did not understand task in run 1
            runs = np.array([2, 3, 4, 5])
        elif sub_num=='058':
            # almost fell asleep in run 3
            runs = np.array([1, 2, 4, 5])
        elif sub_num=='062':
            # did not understand task in run 1
            runs = np.array([2, 3, 4, 5])
        elif sub_num=='066':
            # tiredness, did 4 runs
            # fasted 24h due to religious reasons
            # comment in hidsight: removed due to inconsistency...
            runs = np.array([1, 2, 3, 4])
        else:
            runs = np.arange(1, 6)
        age = info_survey.loc[info_survey.UID==int(sub_num), 'Age']
        female = info_survey.loc[info_survey.UID==int(sub_num), 'Female']
        right_hand = info_survey.loc[info_survey.UID==int(sub_num), 'RightHanded']
        subject = mri_subject.Subject(sub_num, age=age, female=female, right_hand=right_hand, runs=runs)
        if not subject.to_exclude:    
            subjects.append(subject)
        else:
            print(f'Excluding {sub_num}')
    return subjects
            
def calc_subjects_RDM(subjects, mask_dict_type='roi', map_type='zstat', design_prefix='view_unified'):
    subjects_with_RDM = Parallel(n_jobs=-1)(delayed(_calc_RDM)(subject, mask_dict_type, map_type=map_type, design_prefix=design_prefix) for subject in tqdm.tqdm(subjects))
    return subjects_with_RDM

def _calc_RDM(subject, mask_dict_type='roi', motion_thresh=0.75, map_type='zstat', design_prefix='view_unified'):
    subject.calc_RDM(mask_dict_type=mask_dict_type, motion_thresh=motion_thresh, verbose=False, map_type=map_type, design_prefix=design_prefix)
    return subject