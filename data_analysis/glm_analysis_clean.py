import pandas as pd
import mne
from mne_nirs.experimental_design import longest_inter_annotation_interval
from nilearn.glm.first_level import make_first_level_design_matrix
# from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_glm
from mne_nirs.channels import get_short_channels
from pandas import DataFrame

from joblib import Parallel, delayed

from rpy2.robjects import r, globalenv
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
from mne_nirs.statistics import statsmodels_to_results

def plot_group_fir_model(betas_df, condition, design_matrix, raw_haemo=None):
    """
    Plot group FIR response (HbO & HbR) using mixed-effects model,
    reconstructed into a single haemodynamic 'wave'.
    """

    import statsmodels.formula.api as smf
    from mne_nirs.statistics import statsmodels_to_results

    df = betas_df.copy()
    dm = design_matrix.copy()
    df["isCondition"] = [condition in n for n in df["Condition"]]
    df["isDelay"] = ["delay" in n for n in df["Condition"]]
    df = df.query("isDelay in [True]").query("isCondition in [True]").copy()
    df.loc[:, "TidyCond"] = ""
    df.loc[df["isCondition"] == True, "TidyCond"] = condition  # noqa: E712
    df.loc[:, "delay"] = [n.split("_")[-1] for n in df.Condition]
    dm_cols_not_left = np.where([condition in c for c in dm.columns])[0]
    dm = dm[[dm.columns[i] for i in dm_cols_not_left]]
    lme = smf.mixedlm("theta ~ -1 + delay:TidyCond:Chroma", df, groups=df["ID"]).fit()
    
    # Create a dataframe from LME model for plotting below
    df_sum = statsmodels_to_results(lme)
    df_sum["delay"] = [int(n) for n in df_sum["delay"]]
    df_sum = df_sum.sort_values("delay")

    # Print the result for the oxyhaemoglobin data in the tapping condition
    df_sum.query(f"TidyCond in ['{condition}']").query("Chroma in ['hbo']")

    # Extract design matrix columns that correspond to the condition of interest
    dm_cond_idxs = np.where([condition in n for n in dm.columns])[0]
    dm_cond = dm[[dm.columns[i] for i in dm_cond_idxs]]

    # Extract the corresponding estimates from the lme dataframe for hbo
    df_hbo = df_sum.query(f"TidyCond in ['{condition}']").query("Chroma in ['hbo']")
    vals_hbo = [float(v) for v in df_hbo["Coef."]]
    dm_cond_scaled_hbo = dm_cond * vals_hbo

    # Extract the corresponding estimates from the lme dataframe for hbr
    df_hbr = df_sum.query(f"TidyCond in ['{condition}']").query("Chroma in ['hbr']")
    vals_hbr = [float(v) for v in df_hbr["Coef."]]
    dm_cond_scaled_hbr = dm_cond * vals_hbr

    # Extract the time scale for plotting.
    # Set time zero to be the onset of the finger tapping.
    index_values = dm_cond_scaled_hbo.index - np.ceil(raw_haemo.annotations.onset[0])
    index_values = np.asarray(index_values)
    
    # We can also extract the 95% confidence intervals of the estimates too
    l95_hbo = [float(v) for v in df_hbo["[0.025"]]  # lower estimate
    u95_hbo = [float(v) for v in df_hbo["0.975]"]]  # upper estimate
    dm_cond_scaled_hbo_l95 = dm_cond * l95_hbo
    dm_cond_scaled_hbo_u95 = dm_cond * u95_hbo
    l95_hbr = [float(v) for v in df_hbr["[0.025"]]  # lower estimate
    u95_hbr = [float(v) for v in df_hbr["0.975]"]]  # upper estimate
    dm_cond_scaled_hbr_l95 = dm_cond * l95_hbr
    dm_cond_scaled_hbr_u95 = dm_cond * u95_hbr

    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))
    
    # Plot the result
    axes[0].plot(index_values, np.asarray(dm_cond))
    axes[1].plot(index_values, np.asarray(dm_cond_scaled_hbo))
    
    axes[2].plot(index_values, np.sum(dm_cond_scaled_hbo, axis=1), "r")
    axes[2].plot(index_values, np.sum(dm_cond_scaled_hbr, axis=1), "b")
    axes[2].fill_between(
        index_values,
        np.asarray(np.sum(dm_cond_scaled_hbo_l95, axis=1)),
        np.asarray(np.sum(dm_cond_scaled_hbo_u95, axis=1)),
        facecolor="red",
        alpha=0.25,
    )
    axes[2].fill_between(
        index_values,
        np.asarray(np.sum(dm_cond_scaled_hbr_l95, axis=1)),
        np.asarray(np.sum(dm_cond_scaled_hbr_u95, axis=1)),
        facecolor="blue",
        alpha=0.25,
    )

    # Format the plot
    for ax in range(3):
        axes[ax].set_xlim(100, 150)
        axes[ax].set_xlabel("Time (s)")
    # axes[0].set_ylim(-0.5, 1.3)
    # axes[1].set_ylim(-3, 8)
    # axes[2].set_ylim(-3, 8)
    axes[0].set_title("FIR Model (Unscaled by GLM estimates)")
    axes[1].set_title("FIR Components (Scaled by Tapping/Right GLM Estimates)")
    axes[2].set_title("Evoked Response (Tapping/Right)")
    axes[0].set_ylabel("FIR Model")
    axes[1].set_ylabel("Oyxhaemoglobin (ΔμMol)")
    axes[2].set_ylabel("Haemoglobin (ΔμMol)")
    axes[2].legend(["Oyxhaemoglobin", "Deoyxhaemoglobin"])
    plt.savefig(rf"C:\Users\NKUE0003\OneDrive - Region Hovedstaden\Bachelor\results\group_fir_response.png")
    print("DONE")
    
def run_glm_analysis(subjects, class_instance, hrf_model="fir"):
    
    print("Running GLM analysis")
    def glm_subject(subject, idx, data_types, hrf_model):
        print(f"Constructing design matrix and running GLM on subject {idx+1}/{len(subjects)}")
        haemo = subject.raw_haemo.copy()
        redundant_annotations = [x for x in np.unique(haemo.annotations.description) if x not in set(data_types)]
        if len(redundant_annotations) != 0:
            for annotation in redundant_annotations:
                haemo.annotations.delete(haemo.annotations.description == annotation)
        renames = {cond: cond.replace("/", "_") if "/" in cond else cond for cond in haemo.annotations.description}
        haemo.annotations.rename(renames)
        short_channel_haemo = get_short_channels(subject.raw_haemo_unfiltered)
        haemo.resample(0.5, npad="auto")
        short_channel_haemo.resample(0.5, npad="auto")
        isis, names = longest_inter_annotation_interval(haemo)
        
        conditions = haemo.annotations.description
        
        high_pass_value = 1/(max(isis)*2)
        onsets = haemo.annotations.onset - haemo.first_time
        duration = haemo.annotations.duration
        
        frame_times = haemo.times
        events = DataFrame({'trial_type': conditions,
                    'onset': onsets,
                    'duration': duration})
        drift_model="cosine"
        hrf_model=hrf_model
        min_onset = 0 # Normally used for fMRI in case events are coded relative to a trigger that happens before scanning. Not relevant here.
        high_pass = 0.01 #high_pass_value
        add_regs = short_channel_haemo.get_data().T
        oversampling = 1 # Default value.
        drift_order = 1 # When we use the cosine drift model this parameter doesn't really matter, as the drift order is then actually determined by the high_pass argument
        add_reg_names = short_channel_haemo.ch_names
        fir_delays = range(10) # Default when we don't use a FIR model
        
        
        design_matrix = make_first_level_design_matrix(frame_times, events,
                                        drift_model=drift_model,
                                        drift_order=drift_order,
                                        hrf_model=hrf_model,
                                        min_onset=min_onset,
                                        high_pass=high_pass,
                                        add_regs=add_regs,
                                        oversampling=oversampling,
                                        add_reg_names=add_reg_names,
                                        fir_delays=fir_delays)
        
        glm_estimates = run_glm(haemo, design_matrix, n_jobs=1)
        
        # Create a single ROI that includes all channels for example
        rois = dict(AllChannels=range(len(haemo.ch_names)))
        # Calculate ROI for all conditions
        conditions = design_matrix.columns
        # Compute output metrics by ROI
        df_ind = glm_estimates.to_dataframe_region_of_interest(rois, conditions)

        df_ind["ID"] = subject.name
        df_ind["theta"] = [t * 1.0e6 for t in df_ind["theta"]]

        
        return df_ind, haemo, design_matrix
    
    results = Parallel(n_jobs=1)(
    delayed(glm_subject)(subject, idx, class_instance.data_types, hrf_model)
    for idx, subject in enumerate(subjects)
    )

    subject_dfs, glm_results, design_matrices = zip(*results)

    betas_df = pd.concat(subject_dfs, ignore_index=True)

    plot_group_fir_model(betas_df, "Tongue", design_matrices[0], raw_haemo=subjects[0].raw_haemo)