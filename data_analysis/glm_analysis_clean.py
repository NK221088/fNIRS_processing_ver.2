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
import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
from mne_nirs.statistics import statsmodels_to_results
from mne_nirs.visualisation import plot_glm_group_topo, plot_glm_surface_projection

import os
from dotenv import load_dotenv
from pathlib import Path

from sklearn.decomposition import PCA
from mne.preprocessing import ICA

load_dotenv()
save_path = Path(os.getenv(rf"data_save_path"))
Phase_1_assumptions_plot_save_path = Path(os.getenv(rf"Phase_1_assumptions_plot_save_path"))
Phase_1_ANOVA_save_path = Path(os.getenv(rf"Phase_1_ANOVA_save_path"))
Phase_2_assumptions_plot_save_path = Path(os.getenv(rf"Phase_2_assumptions_plot_save_path"))
Phase_2_ANOVA_save_path = Path(os.getenv(rf"Phase_2_ANOVA_save_path"))

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
    fir_delays = np.arange(0, 20, 1)
    delays = np.asarray(list(fir_delays), dtype=float)
    
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
        axes[ax].set_xlim(150, 250)
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
    plt.savefig(os.path.join(save_path, f"group_spm_response.png"))
    print("DONE")

from mne.io.pick import _picks_to_idx
from nilearn.glm.first_level import run_glm as nilearn_glm
from mne_nirs.statistics import RegressionResults

def run_glm(method, raw, design_matrix, noise_model="ar1", bins=0, n_jobs=1, verbose=0):
    """
    GLM fit for an MNE structure containing fNIRS data.

    This is a wrapper function for nilearn.stats.first_level_model.run_glm.

    Parameters
    ----------
    raw : instance of Raw
        The haemoglobin data.
    design_matrix : as specified in Nilearn
        The design matrix as generated by
        `mne_nirs.make_first_level_design_matrix`.
        See example ``9.5.5. Examples of design matrices`` at
        https://nilearn.github.io/auto_examples/index.html
        for details on how to specify design matrices.
    noise_model : {'ar1', 'ols', 'arN', 'auto'}, optional
        The temporal variance model. Defaults to first order
        auto regressive model 'ar1'.
        The AR model can be set to any integer value by modifying the value
        of N. E.g. use `ar5` for a fifth order model.
        If the string `auto` is provided a model with order 4 times the sample
        rate will be used.
    bins : int, optional
        Maximum number of discrete bins for the AR coef histogram/clustering.
        By default the value is 0, which will set the number of bins to the
        number of channels, effectively estimating the AR model for each
        channel.
    n_jobs : int, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs'.
    verbose : int, optional
        The verbosity level. Default is 0.

    Returns
    -------
    glm_estimates : RegressionResults
        RegressionResults class which stores the GLM results.
    """
    sum_method = lambda data: np.sum(data, axis=0)
    if method == "PCA_HbT":
        raw_copy = raw.copy()
        raw_copy_hbo = raw_copy.copy().pick("hbo")
        raw_copy_hbr = raw_copy.copy().pick("hbr")
        raw_hbo = raw_copy_hbo.get_data()
        raw_hbr = raw_copy_hbr.get_data()
        Sigma_hbo = np.cov(raw_hbo)
        Sigma_hbr = np.cov(raw_hbr)
        eigenvalues_hbo, eigenvectors_hbo = np.linalg.eigh(Sigma_hbo)
        eigenvalues_hbr, eigenvectors_hbr = np.linalg.eigh(Sigma_hbr)
        idx_hbo = np.argsort(eigenvalues_hbo)[::-1]
        eigenvalues_hbo = eigenvalues_hbo[idx_hbo]
        eigenvectors_hbo = eigenvectors_hbo[:, idx_hbo]
        idx_hbr = np.argsort(eigenvalues_hbr)[::-1]
        eigenvalues_hbr = eigenvalues_hbr[idx_hbr]
        eigenvectors_hbr = eigenvectors_hbr[:, idx_hbr]
        pca_hbo = eigenvectors_hbo[:, 0] @ raw_hbo
        pca_hbr = eigenvectors_hbr[:, 0] @ raw_hbr
        for ch in raw_copy.info["chs"]:
                if ch["kind"] == mne.io.constants.FIFF.FIFFV_FNIRS_CH:
                    ch["coil_type"] = mne.io.constants.FIFF.FIFFV_COIL_FNIRS_HBO
        groups = {"PC_channel": [i for i, ch in enumerate(raw_copy.ch_names)]}
        pca_method = lambda data: pca_hbo + pca_hbr
        raw_pca = mne.channels.combine_channels(
            raw_copy, 
            groups=groups, 
            method=pca_method)
        glm_raw = raw_pca.copy()
        picks = _picks_to_idx(glm_raw.info, "fnirs", exclude=[], allow_empty=True)
        ch_names = list(groups.keys())
    
    elif method == "PCA":
        raw_copy = raw.copy()
        raw_copy_hbo = raw_copy.copy().pick("hbo")
        raw_copy_hbr = raw_copy.copy().pick("hbr")
        raw_hbo = raw_copy_hbo.get_data()
        raw_hbr = raw_copy_hbr.get_data()
        raw_hbo -= np.mean(raw_hbo,axis = 1, keepdims=True)
        raw_hbr -= np.mean(raw_hbr,axis = 1, keepdims=True)
        pca = PCA()
        pca_hbo = pca.fit_transform(raw_hbo.T)
        pca_hbr = pca.fit_transform(raw_hbr.T)
        pca_hbo_first_PCA = pca_hbo[:, 0:1].T
        pca_hbr_first_PCA = pca_hbr[:, 0:1].T
        
        pca_channel_names = [f"PC1_hbo", f"PC1_hbr"]
        groups = {ch_name: [i for i, ch in enumerate(raw_copy.ch_names) if ch.split(" ")[1] in ch_name.split("_")[1]] for ch_name in pca_channel_names}
        for ch in raw_copy.info["chs"]:
                if ch["kind"] == mne.io.constants.FIFF.FIFFV_FNIRS_CH:
                    ch["coil_type"] = mne.io.constants.FIFF.FIFFV_COIL_FNIRS_HBO
        pca_data = np.vstack([pca_hbo_first_PCA, pca_hbr_first_PCA])
        pca_method = lambda data: pca_hbo_first_PCA if "hbo" in data[0] else pca_hbr_first_PCA
        info = mne.create_info(
        ch_names=["PC1_hbo", "PC1_hbr"],
        sfreq=raw_copy.info["sfreq"],
        ch_types=["hbo", "hbr"]
        )
        glm_raw = mne.io.RawArray(pca_data, info)
        picks = _picks_to_idx(glm_raw.info, "fnirs", exclude=[], allow_empty=True)
        ch_names = list(groups.keys())
    
    elif method == "ICA":
        raw_copy = raw.copy()
        raw_copy_hbo = raw_copy.copy().pick("hbo")
        raw_copy_hbr = raw_copy.copy().pick("hbr")
        ica_hbo = ICA(random_state=97)
        ica_hbo.fit(raw_copy_hbo)
        ica_hbr = ICA(random_state=97)
        ica_hbr.fit(raw_copy_hbr)
        raw_hbo_clean = ica_hbo.apply(raw_copy_hbo)
        raw_hbr_clean = ica_hbr.apply(raw_copy_hbr)
        glm_raw = raw_hbo_clean.add_channels([raw_hbr_clean])
        picks = _picks_to_idx(glm_raw.info, "fnirs", exclude=[], allow_empty=True)
        ch_names = list(glm_raw.ch_names)
        
    elif method == "HbT":
        raw_tmp = raw.copy()
        for ch in raw_tmp.info["chs"]:
            if ch["kind"] == mne.io.constants.FIFF.FIFFV_FNIRS_CH:
                ch["coil_type"] = mne.io.constants.FIFF.FIFFV_COIL_FNIRS_HBO # We set the channel types to HbO to allow combination
        ch_names = list(set([ch_name.strip("hbo").strip("hbr") + "HbT" for ch_name in raw_tmp.ch_names]))
        groups = {ch_name: [i for i, ch in enumerate(raw.ch_names) if ch.strip(" hbo").strip(" hbr") in ch_name.strip(" HbT")] for ch_name in ch_names}
        raw_HbT = mne.channels.combine_channels(
        raw_tmp, 
        groups=groups, 
        method=sum_method)
        glm_raw = raw_HbT.copy()
        picks = _picks_to_idx(glm_raw.info, "fnirs", exclude=[], allow_empty=True)
    elif method == "Standard":
        glm_raw = raw.copy()
        picks = _picks_to_idx(glm_raw.info, "fnirs", exclude=[], allow_empty=True)
        ch_names = raw.ch_names

    if noise_model == "auto":
        noise_model = f"ar{int(np.round(glm_raw.info['sfreq'])) * 4}"

    if bins == 0:
        bins = len(glm_raw.ch_names)

    results = dict()
    for pick in picks:
        labels, glm_estimates = nilearn_glm(
            glm_raw.get_data(pick).T,
            design_matrix.values,
            noise_model=noise_model,
            bins=bins,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        results[ch_names[pick]] = glm_estimates[labels[0]]

    return RegressionResults(glm_raw.info, results, design_matrix)
    
def run_glm_analysis(subjects, class_instance, drift_model="cosine", hrf_model="glover", number_of_subjects=[]):
    
    print("Running GLM analysis")
    def glm_subject(subject, idx, data_types, drift_model, hrf_model):
        print(f"Constructing design matrix and running GLM on subject {idx+1}/{len(subjects)}")
        haemo = subject.raw_haemo.copy()
        relevant_channels = [ch for ch in haemo.ch_names if ("S1" in ch) or ("S2" in ch) or ("S3" in ch) or ("S4" in ch)] #haemo.ch_names #
        haemo = haemo.pick(picks=relevant_channels)
        
        redundant_annotations = [x for x in np.unique(haemo.annotations.description) if x not in set(data_types)]
        if len(redundant_annotations) != 0:
            for annotation in redundant_annotations:
                haemo.annotations.delete(haemo.annotations.description == annotation)
        renames_isis = {cond: cond.split("/")[0] if "/" in cond else cond for cond in haemo.annotations.description}
        renames = {cond: cond.split("/")[1] if "/" in cond else cond for cond in haemo.annotations.description}
        haemo_isis = haemo.copy()
        haemo_isis.annotations.rename(renames_isis)
        haemo.annotations.rename(renames_isis)
        short_channel_haemo = get_short_channels(subject.raw_haemo_unfiltered)
        # haemo.resample(2.5, npad="auto")
        # short_channel_haemo.resample(2.5, npad="auto")
        isis, names = longest_inter_annotation_interval(haemo_isis)
        
        conditions = haemo.annotations.description
        
        high_pass_value = 1/(max(isis)*2)
        onsets = haemo.annotations.onset - haemo.first_time
        duration = haemo.annotations.duration
        
        frame_times = haemo.times
        events = DataFrame({'trial_type': conditions,
                    'onset': onsets,
                    'duration': duration})
        drift_model=drift_model
        hrf_model = hrf_model
        min_onset = 0 # Normally used for fMRI in case events are coded relative to a trigger that happens before scanning. Not relevant here.
        high_pass = high_pass_value #high_pass_value
        add_regs = short_channel_haemo.get_data().T * 10**6 # Scale to uM
        oversampling = 1 # Default value.
        drift_order = 1 # When we use the cosine drift model this parameter doesn't really matter, as the drift order is then actually determined by the high_pass argument
        add_reg_names = short_channel_haemo.ch_names
        fir_delays = range(21) # Default when we don't use a FIR model
        from nilearn.glm.first_level.hemodynamic_models import _calculate_tr
        from nilearn.glm.first_level.hemodynamic_models import _gamma_difference_hrf
        from functools import partial
        t_r = _calculate_tr(frame_times)    
        hrf_model_ = partial(
        _gamma_difference_hrf,
        time_length=32.0,
        onset=min_onset,
        delay=7,
        undershoot=12.0,
        dispersion=0.9,
        u_dispersion=0.9,
        ratio=0.48,
        )
        hrf_model_.__name__ = '_gamma_custom_delay_hrf'
        
        try:
            design_matrix = make_first_level_design_matrix(frame_times,
                                            events,
                                            drift_model=drift_model,
                                            drift_order=drift_order,
                                            hrf_model=hrf_model_,
                                            min_onset=min_onset,
                                            high_pass=high_pass,
                                            add_regs=add_regs,
                                            oversampling=oversampling,
                                            add_reg_names=add_reg_names,) 
            glm_estimates = run_glm("Standard", haemo, design_matrix, n_jobs=1)

        except Exception as e:
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
        # Create a single ROI that includes all channels for example
        # rois = dict(AllChannels=range(len(haemo.ch_names)))
        rois = dict(AllChannels=[i for i, ch in enumerate(haemo.ch_names) if ("S2" in ch) or ("S10" in ch)])
        # rois = dict(
        # Left=[i for i, ch in enumerate(haemo.ch_names) if "S2" in ch],
        # Right=[i for i, ch in enumerate(haemo.ch_names) if "S12" in ch]
        # )

        # Calculate ROI for all conditions
        conditions = design_matrix.columns
        
        # Compute output metrics by ROI
        # df_ind = glm_estimates.to_dataframe_region_of_interest(rois, conditions)
        cha = glm_estimates.to_dataframe()
        
        hrf_model_suffix = f"_{hrf_model_.__name__}"
        # df_ind["Condition"] = df_ind["Condition"].str.replace(hrf_model_suffix, "", regex=False) # Remove the HRF model name from the name of the conditions
        cha["Condition"] = cha["Condition"].str.replace(hrf_model_suffix, "", regex=False) # Remove the HRF model name from the name of the conditions
        #df_ind["ID"] = 
        cha["ID"] = subject.name + "_" + "HC" if idx < number_of_subjects[0] else subject.name + "_" + "Patient"
        cha["Group"] = "HC" if idx < number_of_subjects[0] else "Patient"
        
        # Convert to uM for nicer plotting below.
        # df_ind["theta"] = [t * 1.0e6 for t in df_ind["theta"]]
        cha["theta"] = [t * 1.0e6 for t in cha["theta"]]

        
        return haemo, design_matrix, cha
    
    results = Parallel(n_jobs=1)(
    delayed(glm_subject)(subject, idx, class_instance.data_types, drift_model,hrf_model)
    for idx, subject in enumerate(subjects)
    )

    haemos, design_matrices, cha_dfs = zip(*results)

    # betas_roi_df = pd.concat(subject_dfs, ignore_index=True)

    if hrf_model == "fir":
        plot_group_fir_model(betas_roi_df, "Tongue", design_matrices[0], raw_haemo=subjects[0].raw_haemo)
    
    else:
        data_types = list(np.unique(haemos[0].annotations.description))
        
        # grp_results = betas_roi_df.query("Condition in @data_types")
        # roi_model = smf.mixedlm("theta ~ -1 + ROI:Condition:Chroma", grp_results, groups=grp_results["ID"]).fit(method="nm")
        # roi_model.summary()
        # df = statsmodels_to_results(roi_model)
        
        
        # fig = sns.catplot(
        # x="Condition",
        # y="theta",
        # col="ID",
        # hue="ROI",
        # data=grp_results.query("Chroma in ['hbo']"),
        # col_wrap=5,
        # errorbar=None,
        # palette="muted",
        # height=4,
        # s=10,
        # )
        # plt.savefig(os.path.join(save_path, f"individual_results.png"))
        
        
        # sns.catplot(
        # x="Condition",
        # y="Coef.",
        # hue="ROI",
        # data=df.query("Chroma == 'hbo'"),
        # errorbar=None,
        # palette="muted",
        # height=4,
        # s=10,
        # )
        # plt.savefig(os.path.join(save_path, f"group_results.png"))

        betas_cha_df = pd.concat(cha_dfs, ignore_index=True)
        betas_cha_df = betas_cha_df.query("Condition in @data_types")
        # Cut down the dataframe just to the conditions we are interested in
        data_types = [data_types[0], data_types[-1]]
        ch_summary = betas_cha_df.query("Condition in @data_types")

        # Save the dataframe to verify it's identical
        # ch_summary.to_csv(rf"C:\Users\NKUE0003\OneDrive - Region Hovedstaden\Bachelor\Results\debug_ch_summary.csv", index=False)
        with localconverter(pandas2ri.converter):
            globalenv["rdf"] = ch_summary
        globalenv["Phase_1_assumptions_plot_save_path"] = str(Phase_1_assumptions_plot_save_path).replace("\\", "/")
        globalenv["Phase_2_assumptions_plot_save_path"] = str(Phase_2_assumptions_plot_save_path).replace("\\", "/")

        lme4 = importr("lme4")
        
        r('''
        library(lme4)
        library(lmerTest)
        library(performance)
        library(see)
        library(ggplot2)
        library(patchwork)
        library(effectsize)
        
        # Create all three models
        # modelConch_plot <- lmer(theta ~ Condition + ch_name + Condition:ch_name + (1 | ID), 
        #                         data=rdf, REML=TRUE)
        # modelchannel_plot <- lmer(theta ~ Condition + ch_name + (1 | ID), 
        #                         data=rdf, REML=TRUE)
        # modelCondition_plot <- lmer(theta ~ Condition + (1 | ID), 
        #                             data=rdf, REML=TRUE)

        # # Get diagnostic plots for all models
        # diag_conch <- plot(check_model(modelConch_plot, panel = FALSE))
        # diag_channel <- plot(check_model(modelchannel_plot, panel = FALSE))
        # diag_condition <- plot(check_model(modelCondition_plot, panel = FALSE))

        # # Define plot names
        # plot_names <- c(
        #     "posterior_predictive_check",
        #     "linearity",
        #     "homogeneity_of_variance",
        #     "influential_outliers",
        #     "multicollinearity",
        #     "normal_residuals"
        # )

        # # Combine and save each diagnostic type across all three models
        # for(i in seq_along(plot_names)) {
        #     filename <- file.path(Phase_1_assumptions_plot_save_path, 
        #                         paste0("combined_", plot_names[i], ".pdf"))
            
        #     tryCatch({
        #         # Combine the three plots horizontally with titles
        #         combined_plot <- (diag_conch[[i]] + ggtitle("Model: Condition × Channel")) | 
        #                         (diag_channel[[i]] + ggtitle("Model: Condition + Channel")) | 
        #                         (diag_condition[[i]] + ggtitle("Model: Condition Only"))
                
        #         # Save combined plot
        #         ggsave(filename, plot = combined_plot, width = 18, height = 6, device = "pdf")
        #         message(paste("✓ Created:", filename))
        #     }, error = function(e) {
        #         message(paste("✗ Could not create:", plot_names[i], "-", e$message))
        #     })
        # }

        # message("All combined diagnostic plots completed!")
                
        modelConch <- lmer(theta ~ Condition + ch_name + Condition:ch_name + (1 | ID), data=rdf, REML=FALSE)
        nullModelConch <- lmer(theta ~ Condition + ch_name + (1 | ID), data=rdf, REML=FALSE)
        anova_result_Condch <- anova(modelConch, nullModelConch)
        anova_Condch_df <- as.data.frame(anova_result_Condch)
        print(anova_result_Condch)
        # # print(isSingular(modelConch, tol = 1e-4))
        # # print(summary(modelConch)$varcor)
        # X <- model.matrix(~ Condition * ch_name, data = rdf)
        # # print(qr(X)$rank)
        # # print(ncol(X))
        
        # #Extract coefficents as dataframe:
        # coef_summary_modelConch <- as.data.frame(summary(modelConch)$coefficients)
        # coef_summary_modelConch$Parameter <- rownames(coef_summary_modelConch)
        # colnames(coef_summary_modelConch) <- c("Estimate", "Std_Error", "df", "t_value", "p_value", "Parameter")

        # #Extract coefficents for plotting:
        # coef_summary_nullModelConch <- as.data.frame(summary(nullModelConch)$coefficients)
        # coef_summary_nullModelConch$Parameter <- rownames(coef_summary_nullModelConch)
        # colnames(coef_summary_nullModelConch) <- c("Estimate", "Std_Error", "df", "t_value", "p_value", "Parameter")
        
        # ################################################################################################################
        
        modelchannel <- lmer(theta ~ Condition + ch_name + (1 | ID), data=rdf, REML=FALSE)
        nullModelchannel <- lmer(theta ~ Condition + (1 | ID), data=rdf, REML=FALSE)
        anova_result_channel <- anova(modelchannel, nullModelchannel)
        anova_channel_df <- as.data.frame(anova_result_channel)
        print(anova_result_channel)
        
        # #Extract coefficents as dataframe:
        # coef_summary_modelchannel <- as.data.frame(summary(modelchannel)$coefficients)
        # coef_summary_modelchannel$Parameter <- rownames(coef_summary_modelchannel)
        # colnames(coef_summary_modelchannel) <- c("Estimate", "Std_Error", "df", "t_value", "p_value", "Parameter")
        
        # #Extract coefficents for plotting:
        # coef_summary_nullModelchannel <- as.data.frame(summary(nullModelchannel)$coefficients)
        # coef_summary_nullModelchannel$Parameter <- rownames(coef_summary_nullModelchannel)
        # colnames(coef_summary_nullModelchannel) <- c("Estimate", "Std_Error", "df", "t_value", "p_value", "Parameter")

        # ################################################################################################################
        
        modelCondition <- lmer(theta ~ Condition + (1 | ID), data=rdf, REML=FALSE)
        nullModelCondition <- lmer(theta ~ (1 | ID), data=rdf, REML=FALSE)
        anova_result_condition <- anova(modelCondition, nullModelCondition)
        anova_condition_df <- as.data.frame(anova_result_condition)
        print(anova_result_condition)
        
        # # Fit the model
        # modelCondition_ML <- lmer(theta ~ Condition + (1 | ID), data=rdf, REML=FALSE)

        # # ---- Diagnostics ----
        # print("Column names in rdf:")
        # print(names(rdf))

        # print("Head of rdf:")
        # print(head(rdf))

        # print("Model formula:")
        # print(formula(modelCondition_ML))

        # print("Fixed effects:")
        # print(fixef(modelCondition_ML))

        # print("Model class:")
        # print(class(modelCondition_ML))

        # # ---- Now try simr ----
        # library(simr)

        # model_sim <- makeLmer(modelCondition_ML)

        # pc <- powerCurve(
        #     model_sim,
        #     along="ID",
        #     breaks=seq(20, 60, by=5),
        #     nsim=100,
        #     test=fixed("Conditionn_back", method="KR")
        # )

        # print(pc)

        #Extract coefficents as dataframe:
        coef_summary_modelCondition <- as.data.frame(summary(modelCondition)$coefficients)
        coef_summary_modelCondition$Parameter <- rownames(coef_summary_modelCondition)
        colnames(coef_summary_modelCondition) <- c("Estimate", "Std_Error", "df", "t_value", "p_value", "Parameter")
        
        #Extract coefficents for plotting:
        coef_summary_nullModelCondition<- as.data.frame(summary(modelCondition)$coefficients)
        coef_summary_nullModelCondition$Parameter <- rownames(coef_summary_nullModelCondition)
        colnames(coef_summary_nullModelCondition) <- c("Estimate", "Std_Error", "df", "t_value", "p_value", "Parameter")

        ################################################################################################################

        # Create the model
        # modelGroup_plot <- lmer(theta ~ Condition:Group:ch_name + Condition:ch_name + Condition:Group + Group:ch_name + Condition + ch_name + Group + (1 | ID), data=rdf, REML=TRUE)
        modelGroup <-  lmer(theta ~ Condition:Group:ch_name + Condition:ch_name + Condition:Group + Group:ch_name + Condition + ch_name + Group + (1 | ID), data=rdf, REML=FALSE)
        print(anova(modelGroup))

        # # Get all diagnostic plots as a list of ggplot objects
        # diagnostic_plots <- plot(check_model(modelGroup_plot, panel = FALSE))

        # # Define plot names for each position
        # plot_names <- c(
        #     "posterior_predictive_check",  # [[1]]
        #     "linearity",                   # [[2]]
        #     "homogeneity_of_variance",     # [[3]]
        #     "influential_outliers",        # [[4]]
        #     "multicollinearity",           # [[5]]
        #     "normal_residuals"             # [[6]]
        # )

        # # Save each plot individually
        # for(i in seq_along(diagnostic_plots)) {
        #     filename <- file.path(Phase_2_assumptions_plot_save_path, paste0("modelGroup_", plot_names[i], ".pdf"))
            
        #     tryCatch({
        #         ggsave(filename, plot = diagnostic_plots[[i]], width = 8, height = 6, device = "pdf")
        #         message(paste("✓ Created:", filename))
        #     }, error = function(e) {
        #         message(paste("✗ Could not create:", plot_names[i], "-", e$message))
        #     })
        # }

        # # Also save the complete panel
        # pdf(file.path(Phase_2_assumptions_plot_save_path, "modelGroup_all_diagnostics.pdf"), width = 12, height = 10)
        # diagnostic_plots <- plot(check_model(modelGroup_plot))
        # dev.off()

        # message("All diagnostic plots completed!")
        
        # modelGroup <- lmer(theta ~ Condition:Group:ch_name + Condition:ch_name + Condition:Group + Group:ch_name + Condition + ch_name + Group + (1 | ID), data=rdf, REML=FALSE)
        # nullModelGroup <- lmer(theta ~ Condition:ch_name + Condition:Group + Group:ch_name + Condition + ch_name + Group + (1 | ID), data=rdf, REML=FALSE)
        # anova_result_group <- anova(modelGroup, nullModelGroup)
        # anova_group_df <- as.data.frame(anova_result_group)
        # print(anova_result_group)
        
        # #Extract coefficents as dataframe:
        # coef_summary_modelGroup <- as.data.frame(summary(modelGroup)$coefficients)
        # coef_summary_modelGroup$Parameter <- rownames(coef_summary_modelGroup)
        # colnames(coef_summary_modelGroup) <- c("Estimate", "Std_Error", "df", "t_value", "p_value", "Parameter")
        
        # #Extract coefficents for plotting:
        # coef_summary_nullModelGroup<- as.data.frame(summary(modelCondition)$coefficients)
        # coef_summary_nullModelGroup$Parameter <- rownames(coef_summary_nullModelGroup)
        # colnames(coef_summary_nullModelGroup) <- c("Estimate", "Std_Error", "df", "t_value", "p_value", "Parameter")
        ''')
        
        with localconverter(pandas2ri.converter):
            # anova_Condch_df = globalenv["anova_Condch_df"]
            # coef_summary_modelConch = globalenv["coef_summary_modelConch"]
            # coef_summary_nullModelConch = globalenv["coef_summary_nullModelConch"]
            
            
            # anova_channel_df = globalenv["anova_channel_df"]
            # coef_summary_modelchannel = globalenv["coef_summary_modelchannel"]
            # coef_summary_nullModelchannel = globalenv["coef_summary_nullModelchannel"]
            
            anova_condition_df = globalenv["anova_condition_df"]
            coef_summary_modelCondition = globalenv["coef_summary_modelCondition"]
            coef_summary_nullModelCondition = globalenv["coef_summary_nullModelCondition"]

            # anova_Group_df = globalenv["anova_group_df"]
            # coef_summary_modelGroup = globalenv["coef_summary_modelGroup"]
            # coef_summary_nullModelGroup = globalenv["coef_summary_nullModelGroup"]
            # anova_Group_df = globalenv["anova_group_df"]
            # coef_summary_modelGroup = globalenv["coef_summary_modelGroup"]
            # coef_summary_nullModelGroup = globalenv["coef_summary_nullModelGroup"]
            
            # results = globalenv["results_for_plotting"]
        # anova_Condch_df.to_csv(os.path.join(save_path, f"anova_Condch_df.csv"))
        # anova_channel_df.to_csv(os.path.join(save_path, f"anova_channel_df.csv"))
        anova_condition_df.to_csv(os.path.join(save_path, f"anova_condition_df.csv"))
        # anova_Group_df.to_csv(os.path.join(save_path, f"anova_group_df.csv"))
        # anova_Group_df.to_csv(os.path.join(save_path, f"anova_group_df.csv"))
        
        # control_estimate = coef_summary_modelCondition[coef_summary_modelCondition['Parameter'] == '(Intercept)']['Estimate'].values[0]
        # active_estimate = control_estimate + coef_summary_modelCondition[coef_summary_modelCondition['Parameter'] == coef_summary_modelCondition["Parameter"][1]]['Estimate'].values[0]
        # plot_df = pd.DataFrame({
        # 'Condition': ['Control', coef_summary_modelCondition["Parameter"][1]],
        # 'Estimate': [control_estimate, active_estimate]
        # })
        # fig = sns.catplot(
        # x="Condition",
        # y="Estimate",
        # data=plot_df,
        # errorbar=None,
        # palette="muted",
        # height=4,
        # s=10,
        # )
        # plt.savefig(os.path.join(save_path, f"R_model_group_results.png"))
        # figs = {}
        # figs['individual_results'] = fig
        # return [anova_condition_df, coef_summary_modelCondition, coef_summary_nullModelCondition, figs]

'''
    
        # results <- data.frame(
        #     Coef = numeric(),
        #     Std_Error = numeric(),
        #     z = numeric(),
        #     P_z = numeric(),
        #     CI_lower = numeric(),
        #     CI_upper = numeric(),
        #     ch_name = character(),
        #     Chroma = character(),
        #     Condition = character(),
        #     Significant = logical(),
        #     stringsAsFactors = FALSE
        # )
        # for (ch in unique(rdf$ch_name)) {
        #     ch_data <- subset(rdf, ch_name == ch)
        #     mod <- lmer(theta ~ Condition + (1 | ID), data=ch_data, REML=FALSE)
        #     res <- summary(mod)$coefficients
            
        #     # Calculate confidence intervals
        #     conf_int <- confint(mod, parm="beta_", method="Wald")
            
        #     # Get the chromophore for this channel (assuming it's consistent within channel)
        #     chroma_val <- unique(ch_data$Chroma)[1]
            
        #     # Calculate p_adj for this row (will recalculate for all at the end)
        #     p_value <- res["ConditionTongueMI", "Pr(>|t|)"]
            
        #     results <- rbind(results, data.frame(
        #         Coef = res["ConditionTongueMI", "Estimate"],
        #         Std_Error = res["ConditionTongueMI", "Std. Error"],
        #         z = res["ConditionTongueMI", "t value"],
        #         P_z = p_value,
        #         CI_lower = conf_int["ConditionTongueMI", 1],
        #         CI_upper = conf_int["ConditionTongueMI", 2],
        #         ch_name = ch,
        #         Chroma = chroma_val,
        #         Condition = "TongueMI",
        #         Significant = FALSE  # Will update after p-adjustment
        #     ))
        # }

        # # Calculate adjusted p-values
        # results$p_adj <- p.adjust(results$P_z, method="fdr")
        # results$Significant <- results$p_adj < 0.05

        # # Reorder columns and rename to match your exact specification
        # results_for_plotting <- results[, c("Coef", "Std_Error", "z", "P_z", "CI_lower", "CI_upper", "ch_name", "Chroma", "Condition", "Significant")]

        # colnames(results_for_plotting) <- c("Coef.", "Std_Error", "z", "P>|z|", "[0.025", "0.975]", "ch_name", "Chroma", "Condition", "Significant")

        # print(results_for_plotting)
        
        channels = [ch for ch in raw_haemo.copy().ch_names if ch not in raw_haemo.copy().info["bads"]]        # Plot the two conditions
        fig = plot_glm_group_topo(
            raw_haemo.copy().pick(picks=channels).pick(picks="hbo"),
            results.query("Condition == @data_types[1]"),
            colorbar=True,
            vlim=(0, 20),
            cmap=plt.cm.Oranges,
        )
        plt.savefig(os.path.join(save_path, f"R_topo_model.png"))
        
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), gridspec_kw=dict(width_ratios=[1, 1]))
        ch_model = smf.mixedlm("theta ~ -1 + ch_name:Chroma:Condition", ch_summary, groups=ch_summary["ID"]).fit(method="nm")
        ch_model_df = statsmodels_to_results(ch_model)
        
        
        
        plot_glm_group_topo(
            raw_haemo.copy().pick(picks=channels).pick(picks="hbo"),
            ch_model_df.query("Condition == @data_types[0]"),
            colorbar=False,
            axes=axes[0, 0],
            vlim=(0, 20),
            cmap=plt.cm.Oranges,
        )

        plot_glm_group_topo(
            raw_haemo.copy().pick(picks=channels).pick(picks="hbo"),
            ch_model_df.query("Condition == @data_types[1]"),
            colorbar=True,
            axes=axes[0, 1],
            vlim=(0, 20),
            cmap=plt.cm.Oranges,
        )

        # Cut down the dataframe just to the conditions we are interested in
        ch_summary = betas_cha_df.query("Condition in @data_types")
        ch_summary = ch_summary.query("Chroma in ['hbr']")

        # Run group level model and convert to dataframe
        ch_model = smf.mixedlm("theta ~ -1 + ch_name:Chroma:Condition", ch_summary, groups=ch_summary["ID"]).fit(method="nm")
        ch_model_df = statsmodels_to_results(ch_model)

        # Plot the two conditions
        plot_glm_group_topo(
            raw_haemo.copy().pick(picks=channels).pick(picks="hbr"),
            ch_model_df.query("Condition == @data_types[0]"),
            colorbar=False,
            axes=axes[1, 0],
            vlim=(-10, 0),
            cmap=plt.cm.Blues_r,
        )
        plot_glm_group_topo(
            raw_haemo.copy().pick(picks=channels).pick(picks="hbr"),
            ch_model_df.query("Condition == @data_types[1]"),
            colorbar=True,
            axes=axes[1, 1],
            vlim=(-10, 0),
            cmap=plt.cm.Blues_r,
        )
        
        plt.savefig(os.path.join(save_path, f"group_results_topo.png"))
        
        ch_summary = betas_cha_df.query("Condition in @class_instance.data_types")
        ch_summary = ch_summary.query("Chroma in ['hbo']")

        # Run group level model and convert to dataframe
        ch_model = smf.mixedlm("theta ~ -1 + ch_name:Chroma:Condition", ch_summary, groups=ch_summary["ID"]).fit(method="nm")

        # Here we can use the order argument to ensure the channel name order
        ch_model_df = statsmodels_to_results(
            ch_model, order=raw_haemo.copy().pick(picks="hbo").ch_names
        )
        # And make the table prettier
        ch_model_df.reset_index(drop=True, inplace=True)
        ch_model_df = ch_model_df.set_index(["ch_name", "Condition"])
        print(ch_model_df)
        print("\n")
        print("Significant results:")
        print(ch_model_df[ch_model_df["Significant"] == True])
        
        largest_response_channel = ch_model_df.loc[ch_model_df["Coef."].idxmax()]
        print("\n")
        print("Largest response channel:")
        print(largest_response_channel)
        
        from mne_nirs.io.fold import fold_channel_specificity
        
        raw_channel = raw_haemo.copy().pick(largest_response_channel.name[0])
        print("\n")
        print("fold channel specificity")
        print(fold_channel_specificity(raw_channel)[0])
        print("stopklods")
'''

import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from collections import defaultdict
from preprocessing_toolbox.load_data_function import data_loaders

dataSetList = list(data_loaders.keys())
dataLoaders = [dataSetList[15], dataSetList[17]]
datasets = defaultdict(defaultdict)

for data_loader in dataLoaders:
    settings = {
        "data_set": data_loader,  # Default to first dataset
        "epoch_type": "TongueMI",
        "individual": "All Individuals",
        "short_channel_correction": True,
        "negative_correlation_enhancement": False,
        "haemo_type": "hbo",
        "baseline_correction": "Previous rest period",
        "tmin": 0,
        "stimulus_duration": 5,
        "scalp_coupling_threshold": 0.8,
        "reject_criteria": dict(hbo=80e-6),
        "unwanted": ["15.0"],
        "filter_lower_value": 0.01,
        "filter_upper_value": 0.5,
        "h_trans_bandwidth": 0.2,           
        "l_trans_bandwidth": 0.01,
        "snr_rejection": "None",  # Default to None, can be set to "SNR" or "CV"
        "snr_threshold": 8,  # Default threshold for SNR
        "Apply_TDDR": True,
        "interpolate_bad_channels": True,
    }
    current_loader = data_loaders[data_loader](
                    data_name = data_loader,
                    file_path = data_loader,
                    short_channel_correction=settings["short_channel_correction"],
                    negative_correlation_enhancement=settings["negative_correlation_enhancement"],
                    interpolate_bad_channels=settings["interpolate_bad_channels"],
                    baseline_correction=settings["baseline_correction"],
                    tmin=settings["tmin"],
                    filter_lower_value=settings["filter_lower_value"],
                    filter_upper_value=settings["filter_upper_value"],
                    l_trans_bandwidth=settings["l_trans_bandwidth"],
                    h_trans_bandwidth=settings["h_trans_bandwidth"],
                    scalp_coupling_threshold=settings["scalp_coupling_threshold"],
                    reject_criteria=settings["reject_criteria"],
                    snr_rejection=settings["snr_rejection"],
                    snr_threshold=settings["snr_threshold"],
                    apply_tddr=settings["Apply_TDDR"]
                )
    data = current_loader.load_data()
    variables = ("all_epochs", "data_name", "all_data", "freq", "data_types", "all_individuals")
    datasets[data_loader] = {key: value for key, value in zip(variables, data)}

all_participants = datasets[dataLoaders[0]]["all_individuals"] + datasets[dataLoaders[1]]["all_individuals"]
number_of_subjects = [len(datasets[dataLoaders[0]]["all_individuals"]), len((datasets[dataLoaders[1]]["all_individuals"]))]
run_glm_analysis(all_participants, current_loader, "cosine", "glover", number_of_subjects)