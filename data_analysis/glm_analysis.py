import pandas as pd
import mne
from mne_nirs.experimental_design import longest_inter_annotation_interval
from nilearn.glm.first_level import make_first_level_design_matrix
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

def plot_group_fir_response(glm_results, design_matrices, condition, channel_base="S1_D1"):
    """
    Plot group-average reconstructed FIR responses (HbO & HbR) with 95% CI.
    
    glm_results: list of glm_estimates (one per subject)
    design_matrices: list of design matrices (one per subject)
    condition: str, e.g. "TongueMI"
    channel_base: str, base name of channel pair (e.g. "S1_D1")
    """
    all_hbo = []
    all_hbr = []

    for glm_estimate, design_matrix in zip(glm_results, design_matrices):
        # Get FIR regressors
        fir_cols = [c for c in design_matrix.columns if condition in c and "delay" in c]
        dm_cond = design_matrix[fir_cols]

        # Channel names
        ch_hbo = f"{channel_base} hbo"
        ch_hbr = f"{channel_base} hbr"

        # Extract betas
        theta_hbo = glm_estimate.data[ch_hbo].theta.flatten()
        theta_hbr = glm_estimate.data[ch_hbr].theta.flatten()
        cond_betas_hbo = np.array([theta_hbo[design_matrix.columns.get_loc(c)] for c in fir_cols])
        cond_betas_hbr = np.array([theta_hbr[design_matrix.columns.get_loc(c)] for c in fir_cols])


        # Scale FIR
        dm_cond_scaled_hbo = dm_cond.values * cond_betas_hbo
        dm_cond_scaled_hbr = dm_cond.values * cond_betas_hbr

        # Reconstruct
        reconstructed_hbo = dm_cond_scaled_hbo.sum(axis=1)
        reconstructed_hbr = dm_cond_scaled_hbr.sum(axis=1)

        all_hbo.append(reconstructed_hbo)
        all_hbr.append(reconstructed_hbr)

    # Convert to arrays [n_subjects x n_timepoints]
    all_hbo = np.vstack(all_hbo)
    all_hbr = np.vstack(all_hbr)

    # Mean and 95% CI
    mean_hbo = all_hbo.mean(axis=0)
    mean_hbr = all_hbr.mean(axis=0)

    se_hbo = all_hbo.std(axis=0, ddof=1) / np.sqrt(all_hbo.shape[0])
    se_hbr = all_hbr.std(axis=0, ddof=1) / np.sqrt(all_hbr.shape[0])

    # 95% CI = mean ± 1.96 * SE
    l95_hbo, u95_hbo = mean_hbo - 1.96 * se_hbo, mean_hbo + 1.96 * se_hbo
    l95_hbr, u95_hbr = mean_hbr - 1.96 * se_hbr, mean_hbr + 1.96 * se_hbr

    # Time axis (use first subject’s design matrix index)
    time = dm_cond.index.values

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(time, mean_hbo, "r", label="HbO")
    ax.fill_between(time, l95_hbo, u95_hbo, color="red", alpha=0.3)

    ax.plot(time, mean_hbr, "b", label="HbR")
    ax.fill_between(time, l95_hbr, u95_hbr, color="blue", alpha=0.3)

    ax.set_title(f"Group FIR Response ({condition}, {channel_base})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Haemoglobin (ΔμMol)")
    ax.legend()
    ax.set_xlim(time.min(), time.max())

    plt.tight_layout()
    plt.show()
    plt.savefig(f"Group_FIR_Response_{condition}_{channel_base}.png")

    
def run_glm_analysis(subjects, class_instance, hrf_model="fir"):
    
    print("Running GLM analysis")
    def glm_subject(subject, idx, data_types, hrf_model):
        print(f"Constructing design matrix and running GLM on subject {idx+1}/{len(subjects)}")
        haemo = subject.raw_haemo.copy()
        redundant_annotations = [x for x in np.unique(haemo.annotations.description) if x not in set(data_types)]
        if len(redundant_annotations) != 0:
            for annotation in redundant_annotations:
                haemo.annotations.delete(haemo.annotations.description == annotation)
        renames = {cond: cond.split("/")[0] if "/" in cond else cond for cond in haemo.annotations.description}
        haemo.annotations.rename(renames)
        isis, names = longest_inter_annotation_interval(haemo)
        high_pass_value = 1/(max(isis)*2)
        
        short_channel_haemo = get_short_channels(subject.raw_haemo_unfiltered)
        
        # Here we modify the inputs just like they do in the MNE wrapper - but do it here, as the MNE wrapper doesn't support varying stimulus durations for different events
        frame_times = haemo.times
         # Create events for nilearn
        conditions = haemo.annotations.description
        onsets = haemo.annotations.onset - haemo.first_time
        duration = haemo.annotations.duration
        events = DataFrame({'trial_type': conditions,
                            'onset': onsets,
                            'duration': duration})
        drift_model="cosine"
        hrf_model=hrf_model
        min_onset = 0 # Normally used for fMRI in case events are coded relative to a trigger that happens before scanning. Not relevant here.
        high_pass = high_pass_value
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

        # Get column labels (conditions, drift terms, constant, etc.)
        # regressor_names = [regressor_name for regressor_name in design_matrix.columns if ("drift" not in regressor_name) & ("constant" not in regressor_name)]
        regressor_names = design_matrix.columns
        betas = []
        for ch_name, result in glm_estimates.data.items():
            thetas = result.theta.flatten()  # 1 beta per regressor
            for cond_name, beta in zip(regressor_names, thetas):
                if any (c in cond_name for c in np.unique(haemo.annotations.description)):
                    betas.append({
                        "participant": subject.name,
                        "channel": ch_name,
                        "condition": cond_name,
                        "beta": beta
                    })
                    
        return pd.DataFrame(betas), glm_estimates, design_matrix
    
    results = Parallel(n_jobs=1)(
    delayed(glm_subject)(subject, idx, class_instance.data_types, hrf_model)
    for idx, subject in enumerate(subjects)
    )

    subject_dfs, glm_results, design_matrices = zip(*results)
    
    # Pick the first subject’s GLM + design matrix and plot TongueMI
    # Example: plot TongueMI for the first subject, channel pair S1_D1
    conditions = ["TongueMI", "Control"]

    for cond in conditions:
        try:
            plot_group_fir_response(glm_results, design_matrices, condition=cond, channel_base="S1_D1")
        except Exception as e:
            print(f"Skipping {cond} due to error: {e}")


    
    betas_df = pd.concat(subject_dfs, ignore_index=True)

    with localconverter(pandas2ri.converter):
        globalenv["rdf"] = betas_df

    lme4 = importr("lme4")

    r('''
    library(lme4)
    model <- lmer(beta ~ condition + channel + (1 | participant), data=rdf, REML=FALSE)
    nullModel <- lmer(beta ~ channel + (1 | participant), data=rdf, REML=FALSE)
    print(summary(model))
    print(summary(nullModel))
    coefs <- as.data.frame(coef(summary(model)))
    anova_result <- anova(model, nullModel)
    print(anova_result)
    ''')
    
    # Convert to pandas
    with localconverter(pandas2ri.converter):
        coefs_df = r('coefs')

    print(coefs_df.head())

'''
Draft for plotting code Only tested for hrf_model=:

glm_est = glm_estimates
glm_hbo = glm_est.copy().pick(picks="hbo", exclude='bads')
conditions = ["TongueMI"]

left_hem_end = len(glm_hbo)//2

# Create the plot
fig, axes = plt.subplots(
    nrows=1, ncols=2, figsize=(10, 6), gridspec_kw=dict(width_ratios=[0.92, 1])
)

# Plot all channels smoothed
glm_hbo.plot_topo(axes=axes[0], colorbar=False, conditions=conditions)

# Plot left hemisphere
glm_hbo.copy().pick(picks=range(0, left_hem_end)).plot_topo(
    conditions=conditions, axes=axes[1], colorbar=False, vlim=(-16, 16)
)

# Plot right hemisphere
glm_hbo.copy().pick(picks=range(left_hem_end, len(glm_hbo.ch_names))).plot_topo(
    conditions=conditions, axes=axes[1], colorbar=False, vlim=(-16, 16)
)

axes[1].set_title("Hemispheres plotted independently")
plt.tight_layout()
plt.show()

'''