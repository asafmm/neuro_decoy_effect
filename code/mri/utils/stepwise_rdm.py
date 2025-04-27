import pandas as pd
import numpy as np
from utils import rdm_regression

def first_stepwise(rois, subjects, set_objs, verbose=False, use_explicit=False):
    if verbose:
        print('first step of stepwise regression')
    first_model_ps = np.zeros(len(rois))
    for i, roi in enumerate(rois):
        rdm_reg = rdm_regression.RDMRegression(subjects=subjects, rois=[roi], sets=set_objs)
        rdm_reg.fit(use_explicit=use_explicit)
        first_model_ps[i] = rdm_reg.pval
    if len(first_model_ps) == 0:
        best_first_roi = None
        return None, None
    else:
        best_first_roi = rois[np.argmin(first_model_ps)]
    if verbose:
        print(f'Best first ROI: {best_first_roi}')
    model = rdm_regression.RDMRegression(subjects=subjects, rois=[best_first_roi], sets=set_objs)
    model.fit(use_explicit=use_explicit)
    rois = rois.copy()
    rois.remove(best_first_roi)
    if verbose:
        print(f'Continue with: {rois}')
    previously_used_rois = []
    final_model, model_pvals = stepwise(model, rois, subjects, set_objs, previously_used_rois, step_i=2, 
                                        verbose=verbose, use_explicit=use_explicit, all_model_pvals=first_model_ps)
    if verbose:
        print('*******')
        print(f'Final model uses: {final_model.rois}')
    return final_model, final_model.rois, model_pvals

def stepwise(model, rois, subjects, set_objs, previously_used_rois, step_i=2, verbose=False, use_explicit=False, all_model_pvals=np.array([])):
    if step_i > 20:
        return model
    if len(rois)==0:
        return model
    else:
        pvals_sig = np.zeros(len(rois))
        avg_minus_log_p = np.zeros(len(rois))
        if verbose:
            print('-----')
            print(f'Step {step_i}')
            print(f'Current model uses: {model.rois}')
            print(f'Testing: {rois}')
        model_rois = model.rois.copy()
        previously_used_rois.append(sorted(list(model_rois)))
        for i, roi in enumerate(rois):
            current_rois = np.append(model_rois, roi)
            if sorted(list(current_rois)) in previously_used_rois:
                print(f'Skipping. Combination already used: {current_rois}.')
                pvals_sig[i] = 0
                avg_minus_log_p[i] = 0
            else:
                rdm_reg = rdm_regression.RDMRegression(subjects=subjects, rois=current_rois, sets=set_objs)
                rdm_reg.fit(use_explicit=use_explicit)
                all_model_pvals = np.append(all_model_pvals, rdm_reg.pval)
                #   save all p-values 
                rdm_reg_pvals = save_model_pvalues(rdm_reg)
                roi_sig = rdm_reg_pvals.loc[pd.IndexSlice[roi, :], 'significant']
                roi_pvals = rdm_reg_pvals.loc[pd.IndexSlice[roi, :], 'p_val']
                pvals_sig[i] = np.sum(roi_sig)
                avg_minus_log_p[i] = np.mean(-np.log10(roi_pvals))
        #   compare which ROI has best p-values
        pvals_df = pd.DataFrame({'roi':rois, 'pvals_sig':pvals_sig, 'avg_minus_log_p':avg_minus_log_p})
        pvals_df = pvals_df.sort_values(by=['pvals_sig', 'avg_minus_log_p'], ascending=False)
        #   choose best ROI
        best_roi = pvals_df.roi.values[0]
        best_pval = pvals_df.pvals_sig.values[0]
        if verbose:
            print(f'Best ROI: {best_roi}, # significant p-values is {best_pval}, average -log(p-value): {pvals_df.avg_minus_log_p.values[0]}')
        if best_pval < 1:
            if verbose:
                print(f'Stopping stepwise as best ROI has <1 significant p-values')
        else:
            if verbose:
                print(f'Refitting model with {best_roi}')
            rois.remove(best_roi)
            model_rois = model.rois
            model_rois = np.append(model_rois, best_roi)
            new_model = rdm_regression.RDMRegression(subjects=subjects, rois=model_rois, sets=set_objs)
            new_model.fit(use_explicit=use_explicit)
            #   test other ROIs in the chosen model
            model_pvals = save_model_pvalues(new_model)
            #   remove ROIs with p > alpha_remove
            model_sig = model_pvals.groupby('roi').sum().significant
            surviving_rois = model_sig[model_sig>0].index.values
            if verbose:
                print(f'Surviving ROIs after addition of {best_roi}: {surviving_rois}')
            removed_rois = model_sig[model_sig==0].index.values.tolist()
            rois = rois #+ removed_rois
            model = rdm_regression.RDMRegression(subjects=subjects, rois=surviving_rois, sets=set_objs)
            model.fit(use_explicit=use_explicit)
            if len(removed_rois) > 0:
                all_model_pvals = np.append(all_model_pvals, model.pval)
            model, all_model_pvals = stepwise(model, rois, subjects, set_objs, previously_used_rois, step_i+1, 
                                            verbose=verbose, use_explicit=use_explicit, all_model_pvals=all_model_pvals)
    
        return model, all_model_pvals
    
def save_model_pvalues(model):
    alpha_enter = 0.1
    alpha_remove = 0.05
    if 'schaefer' not in model.rois[0]:
        model_rois = model.lm.pvalues.index.str.split('_').str[0]
        model_rois = model_rois.tolist()
        # if len(model_rois[0]) > 3:
        #     model_rois = [model_rois[:2] for model_roi in model_rois]
    else:
        model_rois = model.lm.pvalues.index.str.split('_').str[0:2]
        model_rois = ['_'.join(roi_parts) for roi_parts in model_rois]
    model_rois.remove('intercept')
    if 'amount' in model.lm.pvalues.index:
        model_rois = [roi for roi in model_rois if roi!='amount' and roi!='prob']
        model_pvalues = model.lm.pvalues[7:] # ignore the intercept and amount/prob
    else:
        model_pvalues = model.lm.pvalues[1:] # ignore the intercept
    model_sig_p = model_pvalues < alpha_enter
    model_p_df = pd.DataFrame({'p_val':model_pvalues, 'significant':model_sig_p, 'roi':model_rois}, index=model_pvalues.index)
    model_p_df.set_index(['roi', model_p_df.index], inplace=True)
    return model_p_df