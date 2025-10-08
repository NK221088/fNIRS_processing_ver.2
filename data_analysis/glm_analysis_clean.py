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
load_dotenv()
save_path = Path(os.getenv("data_save_path"))

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
    
def run_glm_analysis(subjects, class_instance, hrf_model="spm"):
    
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
        # haemo.resample(1, npad="auto")
        # short_channel_haemo.resample(1, npad="auto")
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
        high_pass = high_pass_value #high_pass_value
        add_regs = short_channel_haemo.get_data().T
        oversampling = 1 # Default value.
        drift_order = 1 # When we use the cosine drift model this parameter doesn't really matter, as the drift order is then actually determined by the high_pass argument
        add_reg_names = short_channel_haemo.ch_names
        fir_delays = range(21) # Default when we don't use a FIR model
        
        design_matrix = make_first_level_design_matrix(frame_times,
                                        events,
                                        drift_model=drift_model,
                                        drift_order=drift_order,
                                        hrf_model=hrf_model,
                                        min_onset=min_onset,
                                        high_pass=high_pass,
                                        add_regs=add_regs,
                                        oversampling=oversampling,
                                        add_reg_names=add_reg_names,)
        
        glm_estimates = run_glm(haemo, design_matrix, n_jobs=1)
        
        # Create a single ROI that includes all channels for example
        # rois = dict(AllChannels=range(len(haemo.ch_names)))
        # rois = dict(AllChannels=[i for i, ch in enumerate(haemo.ch_names) if ("S2" in ch) or ("S10" in ch)])
        rois = dict(
        Left=[i for i, ch in enumerate(haemo.ch_names) if "S2" in ch],
        Right=[i for i, ch in enumerate(haemo.ch_names) if "S12" in ch]
        )

        # Calculate ROI for all conditions
        conditions = design_matrix.columns
        # Compute output metrics by ROI
        df_ind = glm_estimates.to_dataframe_region_of_interest(rois, conditions)
        cha = glm_estimates.to_dataframe()

        df_ind["ID"] = cha["ID"] = subject.name
        
        # Convert to uM for nicer plotting below.
        df_ind["theta"] = [t * 1.0e6 for t in df_ind["theta"]]
        cha["theta"] = [t * 1.0e6 for t in cha["theta"]]

        
        return df_ind, haemo, design_matrix, cha
    
    results = Parallel(n_jobs=1)(
    delayed(glm_subject)(subject, idx, class_instance.data_types, hrf_model)
    for idx, subject in enumerate(subjects)
    )

    subject_dfs, haemos, design_matrices, cha_dfs = zip(*results)

    betas_roi_df = pd.concat(subject_dfs, ignore_index=True)

    if hrf_model == "fir":
        plot_group_fir_model(betas_roi_df, "Tongue", design_matrices[0], raw_haemo=subjects[0].raw_haemo)
    
    if hrf_model == "spm":
        data_types = list(np.unique(haemos[0].annotations.description))
        
        grp_results = betas_roi_df.query("Condition in @data_types")
        roi_model = smf.mixedlm("theta ~ -1 + ROI:Condition:Chroma", grp_results, groups=grp_results["ID"]).fit(method="nm")
        roi_model.summary()
        df = statsmodels_to_results(roi_model)
        
        
        fig = sns.catplot(
        x="Condition",
        y="theta",
        col="ID",
        hue="ROI",
        data=grp_results.query("Chroma in ['hbo']"),
        col_wrap=5,
        errorbar=None,
        palette="muted",
        height=4,
        s=10,
        )
        plt.savefig(os.path.join(save_path, f"individual_results.png"))
        
        
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
        raw_haemo=subjects[0].raw_haemo
        relevant_channels = [ch for ch in raw_haemo.ch_names if ("S2" in ch) or ("S10" in ch)]
        # Cut down the dataframe just to the conditions we are interested in
        data_types = [data_types[0], data_types[-1]]
        ch_summary = betas_cha_df.query("Condition in @data_types")
        ch_summary = ch_summary.query("Chroma in ['hbo']")
        # ch_summary = ch_summary.query("ch_name in @relevant_channels")
        
        with localconverter(pandas2ri.converter):
            globalenv["rdf"] = ch_summary

        lme4 = importr("lme4")
        
        r('''
        library(lme4)
        library(lmerTest)
        
        modelCondition <- lmer(theta ~ Condition + (1 + Condition | ch_name) + (1 | ID), data=rdf, REML=FALSE)
        nullModelCondition <- lmer(theta ~ (1 | ch_name) + (1 | ID), data=rdf, REML=FALSE)
        print(summary(modelCondition))
        print(summary(nullModelCondition))
        anova_result_condition <- anova(modelCondition, nullModelCondition)
        print(anova_result_condition)
        
        #Extract coefficents for plotting:
        coef_summary <- as.data.frame(summary(modelCondition)$coefficients)
        coef_summary$Parameter <- rownames(coef_summary)
        colnames(coef_summary) <- c("Estimate", "Std_Error", "df", "t_value", "p_value", "Parameter")
    
        results <- data.frame(
            Coef = numeric(),
            Std_Error = numeric(),
            z = numeric(),
            P_z = numeric(),
            CI_lower = numeric(),
            CI_upper = numeric(),
            ch_name = character(),
            Chroma = character(),
            Condition = character(),
            Significant = logical(),
            stringsAsFactors = FALSE
        )

        for (ch in unique(rdf$ch_name)) {
            ch_data <- subset(rdf, ch_name == ch)
            mod <- lmer(theta ~ Condition + (1 | ID), data=ch_data, REML=FALSE)
            res <- summary(mod)$coefficients
            
            # Calculate confidence intervals
            conf_int <- confint(mod, parm="beta_", method="Wald")
            
            # Get the chromophore for this channel (assuming it's consistent within channel)
            chroma_val <- unique(ch_data$Chroma)[1]
            
            # Calculate p_adj for this row (will recalculate for all at the end)
            p_value <- res["ConditionTongueMI", "Pr(>|t|)"]
            
            results <- rbind(results, data.frame(
                Coef = res["ConditionTongueMI", "Estimate"],
                Std_Error = res["ConditionTongueMI", "Std. Error"],
                z = res["ConditionTongueMI", "t value"],
                P_z = p_value,
                CI_lower = conf_int["ConditionTongueMI", 1],
                CI_upper = conf_int["ConditionTongueMI", 2],
                ch_name = ch,
                Chroma = chroma_val,
                Condition = "TongueMI",
                Significant = FALSE  # Will update after p-adjustment
            ))
        }

        # Calculate adjusted p-values
        results$p_adj <- p.adjust(results$P_z, method="fdr")
        results$Significant <- results$p_adj < 0.05

        # Reorder columns and rename to match your exact specification
        results_for_plotting <- results[, c("Coef", "Std_Error", "z", "P_z", "CI_lower", "CI_upper", "ch_name", "Chroma", "Condition", "Significant")]

        colnames(results_for_plotting) <- c("Coef.", "Std_Error", "z", "P>|z|", "[0.025", "0.975]", "ch_name", "Chroma", "Condition", "Significant")

        print(results_for_plotting)
        
        ''')
        with localconverter(pandas2ri.converter):
            modelCondition = globalenv["coef_summary"]
            results = globalenv["results_for_plotting"]
        control_estimate = modelCondition[modelCondition['Parameter'] == '(Intercept)']['Estimate'].values[0]
        tongueMI_estimate = control_estimate + modelCondition[modelCondition['Parameter'] == 'ConditionTongueMI']['Estimate'].values[0]
        plot_df = pd.DataFrame({
        'Condition': ['Control', 'TongueMI'],
        'Estimate': [control_estimate, tongueMI_estimate]
        })
        sns.catplot(
        x="Condition",
        y="Estimate",
        data=plot_df,
        errorbar=None,
        palette="muted",
        height=4,
        s=10,
        )
        plt.savefig(os.path.join(save_path, f"R_model_group_results.png"))
        
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