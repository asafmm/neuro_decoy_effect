import pandas as pd
import numpy as np
import glob
import os
import re
import platform
import json
import tqdm
import matplotlib
import matplotlib.pyplot as plt

MIN_LENGTH = 194
TRINARY_PATH = '../../stimuli/stimuli_trinary.csv'
BINARY_PATH = '../../stimuli/stimuli_binary.csv'
EVALUATION_PATH = '../../stimuli/evaluation_stimuli.csv'

def extract_choices(df):
    is_trinary = any(df.part=='trinary')
    if is_trinary:
        choices_df = df.loc[df.part=='trinary', ['subject_id', 'binary_id', 'trinary_id', 'response', 'actual_choice', 'direction', 'catch']]
        choices_df.loc[:, 'trinary_group'] = 1
        choices_df = choices_df.sort_values('binary_id')
    else:
        choices_df = df.loc[df.part=='binary', ['subject_id', 'binary_id', 'trinary_id', 'response', 'actual_choice', 'catch']]
        choices_df.loc[:, 'trinary_group'] = 0
        choices_df = choices_df.sort_values('binary_id')
    return choices_df

def passed_catch_trials(choices_df, verbose=True):
    # check catch trials
    # if the subject didn't choose A in the catch trials, drop them!
    catch_choices = choices_df[choices_df.catch==1].actual_choice
    catch_fails = sum(catch_choices!='A')
    if catch_fails > 1:
        if verbose:
            print(f'Dropping subject due to failing {catch_fails}/{len(catch_choices)} catch trials')    
            subject_group = 'trinary' if 'trinary_id' in choices_df.columns else 'binary'
            print(f'Group: {subject_group}')
            print(f'Subject ID: {choices_df.subject_id.values[0]}')
            print(f'Trials: {choices_df[choices_df.catch==1][catch_choices!="A"].binary_id.values}')
            print(f'{choices_df[choices_df.catch==1][catch_choices!="A"].actual_choice.values}')
        return False
    else:
        return True

def calculate_choice_ratios(choices_df, verbose=True):
    # print choices
    responses = choices_df.response.values
    
    passed_catch = passed_catch_trials(choices_df, verbose=verbose)
    if passed_catch == False:
        return 'dropped'
    
    # if 'trinary_id' in choices_df.columns:
    if any(choices_df.trinary_group==1):
        # read the struct of binary and trinary IDs
        ratio_df = pd.read_csv(TRINARY_PATH)[['binary_id', 'trinary_id']]
        ratio_df = ratio_df.set_index(['binary_id', 'trinary_id'])
        # choose a binary_id and save only the choices of it
        for bin_id in choices_df.binary_id.unique():
            bin_id_df = choices_df[choices_df.binary_id==bin_id]
            # for each binary_id choose a trinary_id, for lotteries which have several trinary options for a single binary
            # such as: A B DecoyA, A B DecoyB, these are two trinaries which match one binary lottery A vs. B
            for tri_id in bin_id_df.trinary_id:
                tri_id_df = bin_id_df[bin_id_df.trinary_id==tri_id]
                # count how many A+B choices (excluding decoy choices, if any)
                if tri_id == 22:
                    # for set #22, the geometry is like the compromise effect, with "B" being the target
                    A_total = sum(tri_id_df.actual_choice=='B')
                    B_total = sum(tri_id_df.actual_choice=='A')    
                else:
                    A_total = sum(tri_id_df.actual_choice=='A')
                    B_total = sum(tri_id_df.actual_choice=='B')
                C_total = sum(tri_id_df.actual_choice=='C')
                A_B_total = A_total + B_total
                A_B_C_total = A_B_total + C_total
                ratio_df.loc[(bin_id, tri_id), 'A_total'] = A_total
                ratio_df.loc[(bin_id, tri_id), 'B_total'] = B_total
                ratio_df.loc[(bin_id, tri_id), 'A_B_total'] = A_B_total
                ratio_df.loc[(bin_id, tri_id), 'C_total'] = C_total
                ratio_df.loc[(bin_id, tri_id), 'A_B_C_total'] = A_B_C_total
                # calculate relative choice share: A/(A+B), B/(A+B)
                if A_B_total!=0:
                    ratio_df.loc[(bin_id, tri_id), 'A_ratio'] = A_total / A_B_total
                    ratio_df.loc[(bin_id, tri_id), 'B_ratio'] = B_total / A_B_total
                else:
                    ratio_df.loc[(bin_id, tri_id), 'A_ratio'] = 0
                    ratio_df.loc[(bin_id, tri_id), 'B_ratio'] = 0
    else:
        ratio_df = pd.DataFrame(pd.read_csv(BINARY_PATH)['binary_id'])
        ratio_df = ratio_df.set_index('binary_id')
        for bin_id in choices_df.binary_id.unique():
            id_df = choices_df[choices_df.binary_id==bin_id]
            # count how many A+B choices (excluding decoy choices, if any)
            if bin_id == 22:
                # for set #22, the geometry is like the compromise effect, with "B" being the target
                A_total = sum(id_df.actual_choice=='B')
                B_total = sum(id_df.actual_choice=='A')
            else:
                A_total = sum(id_df.actual_choice=='A')
                B_total = sum(id_df.actual_choice=='B')
            A_B_total = A_total + B_total
            ratio_df.loc[bin_id, 'A_total'] = A_total
            ratio_df.loc[bin_id, 'B_total'] = B_total
            ratio_df.loc[bin_id, 'A_B_total'] = A_B_total
            # calculate relative choice share: A/(A+B), B/(A+B)
            if A_B_total!=0:
                ratio_df.loc[bin_id, 'A_ratio'] = A_total / A_B_total
                ratio_df.loc[bin_id, 'B_ratio'] = B_total / A_B_total
            else:
                ratio_df.loc[bin_id, 'A_ratio'] = 0
                ratio_df.loc[bin_id, 'B_ratio'] = 0
    return ratio_df

def get_ratios(files, full_length, verbose=True):
    binary_ratios = []
    trinary_ratios = []
    subjects_dropped = 0
    good_subjects = []
    subject_ids = []
    subject_ratio_dfs = []
    raw_choices = pd.DataFrame()
    for i in range(len(files)):
        f = files[i]
        choices = pd.read_csv(f)
        sub_id = choices.subject_id.values[0]
        subject_ids.append(sub_id)
        if verbose:
            print(f'working on subject {i+1}/{len(files)}')
        
        if len(choices) < full_length:
            subjects_dropped += 1
            if verbose:
                print(f'Dropping subject due to not completing all trials')
            continue
        is_binary = any(choices.part == 'binary')
        choices = extract_choices(choices)
        ratio = calculate_choice_ratios(choices, verbose=verbose)
        if type(ratio) is str:
            # subject dropped due to catch trails
            subjects_dropped += 1
            continue
        # if it's a good subjects, add choices to all raw choices
        raw_choices = pd.concat([raw_choices, choices])
        if is_binary:
            binary_ratios.append(ratio)
        else:
            trinary_ratios.append(ratio)
        good_subjects.append(sub_id)
        subject_ratio_df = ratio.copy()
        subject_ratio_df.loc[:, 'subject_id'] = sub_id
        subject_ratio_df.loc[:, 'trinary_group'] = is_binary==0
        subject_ratio_dfs.append(subject_ratio_df)
    if verbose:
        print(f'Using {len(files) - subjects_dropped} out of {len(files)} subjects')
    return binary_ratios, trinary_ratios, good_subjects, raw_choices, subject_ratio_dfs

def average_ratios(binary_ratios, trinary_ratios):
    sum_binary = pd.concat(binary_ratios).groupby('binary_id').sum()
    avg_binary = sum_binary.copy()
    avg_binary.loc[:, 'A_ratio'] = sum_binary.A_total / sum_binary.A_B_total
    avg_binary.loc[:, 'B_ratio'] = sum_binary.B_total / sum_binary.A_B_total
    sum_trinary = pd.concat(trinary_ratios).groupby(['binary_id', 'trinary_id']).sum()
    avg_trinary = sum_trinary.copy()
    avg_trinary.loc[:, 'A_ratio'] = sum_trinary.A_total / sum_trinary.A_B_total
    avg_trinary.loc[:, 'B_ratio'] = sum_trinary.B_total / sum_trinary.A_B_total
    avg_trinary.loc[:, 'C_ABC_ratio'] = sum_trinary.C_total / sum_trinary.A_B_C_total
    return avg_binary, avg_trinary

def calculate_decoy_effect(binary_ratios, trinary_ratios):
    avg_binary, avg_trinary = average_ratios(binary_ratios, trinary_ratios)
    avg_trinary = avg_trinary.reset_index()
    avg_binary.loc[:, 'n'] = len(binary_ratios)
    avg_trinary.loc[:, 'n'] = len(trinary_ratios)
    trinary_template = pd.read_csv(TRINARY_PATH)
    binary_template = pd.read_csv(BINARY_PATH)
    catch_trials = trinary_template.catch.values
    # merge data
    all_ratios = avg_trinary.merge(avg_binary, left_on='binary_id', right_on='binary_id', suffixes=['_trinary', '_binary'])
    all_ratios.loc[:, 'decoy_effect_A'] = all_ratios.A_ratio_trinary - all_ratios.A_ratio_binary
    all_ratios.loc[:, 'decoy_effect_B'] = all_ratios.B_ratio_trinary - all_ratios.B_ratio_binary
    all_ratios.loc[:, 'decoy_direction'] = ['A' if effect>0 else 'B' for effect in all_ratios.decoy_effect_A]
    all_ratios.loc[:, 'catch'] = catch_trials
    all_ratios.loc[:, 'amountA'] = trinary_template.amountA
    all_ratios.loc[:, 'probA'] = trinary_template.probA
    all_ratios.loc[:, 'amountB'] = trinary_template.amountB
    all_ratios.loc[:, 'probB'] = trinary_template.probB
    all_ratios.loc[:, 'amountC'] = trinary_template.amountC
    all_ratios.loc[:, 'probC'] = trinary_template.probC
    return all_ratios

def analyze_data_files(files, verbose=True):
    if verbose:
        print(f'Overall {len(files)} subjects')
    # analyze decoy effect
    binary_ratios, trinary_ratios, good_subjects, raw_choices, subject_ratio_dfs = get_ratios(files, full_length=MIN_LENGTH, verbose=verbose)
    all_ratios = calculate_decoy_effect(binary_ratios, trinary_ratios)
    return all_ratios, binary_ratios, trinary_ratios, raw_choices, subject_ratio_dfs

def bootstrap_std(decoy_table, binary_ratios, trinary_ratios, B=10_000):
    # calculate bootstrap standard deviation for the difference between binary and trinary means
    n_sets = len(binary_ratios[0])
    diffs = np.zeros((B, n_sets))
    binary_ratios_array = np.array(binary_ratios)
    trinary_ratios_array = np.array(trinary_ratios)
    for i in tqdm.tqdm(range(B)):
        # sample with replacement 
        binary_subjects = np.random.choice(np.arange(len(binary_ratios)), size=len(binary_ratios), replace=True)
        trinary_subjects = np.random.choice(np.arange(len(trinary_ratios)), size=len(trinary_ratios), replace=True)
        binary_ratios_sample = binary_ratios_array[binary_subjects]
        trinary_ratios_sample = trinary_ratios_array[trinary_subjects]
        # calculate the difference between binary and trinary means
        # trinary_ratios_sample structure: subjects x sets x coulmns
        # column [-2] is A_ratio
        sample_diff = [np.mean(trinary_ratios_sample[:, set_i, -2]) - np.mean(binary_ratios_sample[:, set_i, -2]) for set_i in range(n_sets)]
        sample_diff = np.array(sample_diff)
        diffs[i, :] = sample_diff
    bootstrap_decoy_std = np.std(diffs, axis=0)
    bootstrap_decoy_std = pd.DataFrame({'decoy_std':bootstrap_decoy_std[:27]}, index=decoy_table.binary_id)
    return bootstrap_decoy_std

def plot_decoy_table(pretty_table, mturk=0):
    cvals = [-1, 0, 1]
    decoy_colors = ['#3e8a83', '#ffffff', '#C42934']
    colors_norm = plt.Normalize(min(cvals), max(cvals))
    tuples = list(zip(map(colors_norm, cvals), decoy_colors))
    DECOY_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
    fig, ax = plt.subplots(figsize=(41, 25), dpi=300)
    
    cmap = DECOY_CMAP
    negative_effects = pretty_table.decoy_effect_A[pretty_table.decoy_effect_A < 0].values
    positive_effects = pretty_table.decoy_effect_A[pretty_table.decoy_effect_A >= 0].values
    # scale negative effects from 0 to 0.4, positive effects from 0.6 to 1, for colormap
    negative_effects_scaled = 0.35 * (negative_effects - negative_effects.min()) / (negative_effects.max() - negative_effects.min())
    positive_effects_scaled = 0.35 * (positive_effects - positive_effects.min()) / (positive_effects.max() - positive_effects.min()) + 0.65
    effects_scaled = np.concatenate([negative_effects_scaled, positive_effects_scaled])

    if 'decoy_std' in pretty_table.columns:
        bar_plot = ax.bar(np.arange(len(pretty_table)), pretty_table.decoy_effect_A*100, color=cmap(effects_scaled),
            yerr=pretty_table.decoy_std*100, ecolor='#0d0d0d', capsize=10)
        max_std = pretty_table.decoy_std.max() * 100
    else:
        bar_plot = ax.bar(np.arange(len(pretty_table)), pretty_table.decoy_effect_A*100, color=cmap(effects_scaled))
        max_std = 0
    ax.set_axisbelow(True)
    plt.xlabel('Lottery set', fontsize=100)
    plt.ylabel('Decoy effect (%)', fontsize=100)
    x_tick_locs, x_ticks = plt.xticks()
    plt.xticks(np.arange(0, len(pretty_table), 5), np.arange(1, len(pretty_table)+1, 5), fontsize=75)
    y_tick_locs, y_ticks = plt.yticks()
    y_ticks_positions = [y_tick.get_position()[1] for y_tick in y_ticks]
    y_ticks_with_pct = [f'{int(tick)}%' for tick in y_ticks_positions]
    plt.yticks(y_tick_locs, y_ticks_with_pct)
    min_effect = pretty_table.decoy_effect_A.min() * 100
    max_effect = pretty_table.decoy_effect_A.max() * 100
    
    min_y = min_effect - max_std - 1
    max_y = max_effect + max_std + 1
    plt.ylim([min_y, max_y])
    min_y_tick = 5 * np.round(min_y / 5, 0)
    max_y_tick = 5 * np.round(max_y / 5, 0)
    plt.yticks(np.arange(min_y_tick, max_y_tick, 5), fontsize=75)
    plt.tick_params(axis='both', which='major', pad=60)
    plt.grid(visible=True, which='major', color='#696969', axis='y')
    [spine.set_linewidth(4) for spine in ax.spines.values()]

def read_survey(files):
    # extract survey and winning prizes
    current_os = platform.system()
    info_df = pd.DataFrame()
    i = 0
    date_pattern = re.compile(r'\d{2}-\d{2}-\d{4}')
    for f in files:
        subject_data = pd.read_csv(f)
        subject_uid = subject_data.subject_id.values[0]
        if len(subject_uid) > 10:
            behavioral_sample = True
        subject_dir = f.split('/')[-2]
        if (not behavioral_sample) and (subject_uid != subject_dir):
            subject_uid = subject_dir
        subject_date = date_pattern.search(f).group()
        is_trinary = 'trinary' in subject_data.part.unique()
        info_df.loc[i, 'UID'] = subject_uid
        info_df.loc[i, 'Date'] = subject_date
        info_df.loc[i, 'Group'] = 'Trinary' if is_trinary==1 else 'Binary'
        try:
            survey_answer = subject_data[subject_data.trial_type=='survey'].response.values[0]
            survey_answer_dict = json.loads(survey_answer)
            info_df.loc[i, 'Female'] = 1 if survey_answer_dict['Gender']=='נקבה' else 0
            info_df.loc[i, 'RightHanded'] = 1 if survey_answer_dict['Hand']=='ימין' else 0
            info_df.loc[i, 'Glasses'] = 1 if survey_answer_dict['Glasses']=='כן' else 0
            info_df.loc[i, 'Age'] = float(survey_answer_dict['Age'])
            i += 1
        except IndexError:
            i += 1
            continue
        info_df.loc[i, 'BDM_Stage'] = 1 if 'remaining_budget' in subject_data.columns else 0
        if 'remaining_budget' in subject_data.columns:
            print(f'Subject id: {subject_uid}')
            info_df.loc[i, 'Won'] = int(subject_data.total_prize[subject_data.total_prize.notna()].values[-1])
        else:
            info_df.loc[i, 'Won'] = int(subject_data.total_prize[subject_data.total_prize.notna()].values[-1])
    info_df.Date = pd.to_datetime(info_df.Date, format='%d-%m-%Y')
    return info_df