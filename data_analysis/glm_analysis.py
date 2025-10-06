import pandas as pd
import mne
from mne_nirs.experimental_design import longest_inter_annotation_interval
# from nilearn.glm.first_level import make_first_level_design_matrix
from mne_nirs.experimental_design import make_first_level_design_matrix
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
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from mne_nirs.statistics import statsmodels_to_results

def plot_group_fir_model(betas_df, design_matrix, condition="TongueMI", raw_haemo=None):
    """
    Plot group FIR response (HbO & HbR) using mixed-effects model,
    reconstructed into a single haemodynamic 'wave'.
    """

    import statsmodels.formula.api as smf
    from mne_nirs.statistics import statsmodels_to_results

    # --------------------------------------------------------
    # 1. Filter to condition of interest (only FIR regressors)
    # --------------------------------------------------------
    df = betas_df.copy()
    dm = design_matrix.copy()
    df["isTapping"] = ["Tapping_Right" in n for n in df["Condition"]]
    df["isDelay"] = ["delay" in n for n in df["Condition"]]
    df = df.query("isDelay in [True]").query("isTapping in [True]").copy()
    df.loc[:, "TidyCond"] = ""
    df.loc[df["isTapping"] == True, "TidyCond"] = "Tapping"  # noqa: E712
    df.loc[:, "delay"] = [n.split("_")[-1] for n in df.Condition]
    dm_cols_not_left = np.where(["Right" in c for c in dm.columns])[0]
    dm = dm[[dm.columns[i] for i in dm_cols_not_left]]
    lme = smf.mixedlm("theta ~ -1 + delay:TidyCond:Chroma", df, groups=df["ID"]).fit()
    
    # Create a dataframe from LME model for plotting below
    df_sum = statsmodels_to_results(lme)
    df_sum["delay"] = [int(n) for n in df_sum["delay"]]
    df_sum = df_sum.sort_values("delay")

    # Print the result for the oxyhaemoglobin data in the tapping condition
    df_sum.query("TidyCond in ['Tapping']").query("Chroma in ['hbo']")
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))

    # Extract design matrix columns that correspond to the condition of interest
    dm_cond_idxs = np.where(["Tapping" in n for n in dm.columns])[0]
    dm_cond = dm[[dm.columns[i] for i in dm_cond_idxs]]

    # Extract the corresponding estimates from the lme dataframe for hbo
    df_hbo = df_sum.query("TidyCond in ['Tapping']").query("Chroma in ['hbo']")
    vals_hbo = [float(v) for v in df_hbo["Coef."]]
    dm_cond_scaled_hbo = dm_cond * vals_hbo

    # Extract the corresponding estimates from the lme dataframe for hbr
    df_hbr = df_sum.query("TidyCond in ['Tapping']").query("Chroma in ['hbr']")
    vals_hbr = [float(v) for v in df_hbr["Coef."]]
    dm_cond_scaled_hbr = dm_cond * vals_hbr

    # Extract the time scale for plotting.
    # Set time zero to be the onset of the finger tapping.
    index_values = dm_cond_scaled_hbo.index - np.ceil(raw_haemo.annotations.onset[0])
    index_values = np.asarray(index_values)

    # Plot the result
    axes[0].plot(index_values, np.asarray(dm_cond))
    axes[1].plot(index_values, np.asarray(dm_cond_scaled_hbo))
    axes[2].plot(index_values, np.sum(dm_cond_scaled_hbo, axis=1), "r")
    axes[2].plot(index_values, np.sum(dm_cond_scaled_hbr, axis=1), "b")

    # Format the plot
    for ax in range(3):
        axes[ax].set_xlim(-5, 30)
        axes[ax].set_xlabel("Time (s)")
    axes[0].set_ylim(-0.5, 1.3)
    axes[1].set_ylim(-3, 8)
    axes[2].set_ylim(-3, 8)
    axes[0].set_title("FIR Model (Unscaled by GLM estimates)")
    axes[1].set_title("FIR Components (Scaled by Tapping/Right GLM Estimates)")
    axes[2].set_title("Evoked Response (Tapping/Right)")
    axes[0].set_ylabel("FIR Model")
    axes[1].set_ylabel("Oyxhaemoglobin (ΔμMol)")
    axes[2].set_ylabel("Haemoglobin (ΔμMol)")
    axes[2].legend(["Oyxhaemoglobin", "Deoyxhaemoglobin"])
    
    df = df[df["condition"].str.contains(condition)]
    df = df[df["condition"].str.contains("delay")]

    # Add parsed columns
    df["delay"] = df["condition"].apply(lambda x: int(x.split("_")[-1]))
    df["Chroma"] = df["channel"].apply(lambda x: "hbo" if "hbo" in x else "hbr")
    df["TidyCond"] = condition

    # --------------------------------------------------------
    # 2. Mixed-effects model (like example script)
    # --------------------------------------------------------
    model = smf.mixedlm("beta ~ -1 + C(delay):TidyCond:Chroma",
                        df, groups=df["participant"])
    lme = model.fit()
    df_sum = statsmodels_to_results(lme)

    # Extract values
    df_sum["delay"] = df_sum.index.str.extract(r'C\(delay\)\[(\d+)\]').astype(float)
    df_sum["TidyCond"] = condition
    df_sum["Chroma"] = df_sum.index.str.extract(r'Chroma\[([^\]]+)\]')
    df_sum = df_sum.dropna(subset=["delay"])
    df_sum["delay"] = df_sum["delay"].astype(int)
    df_sum = df_sum.sort_values("delay")

    # --------------------------------------------------------
    # 3. FIR design matrix (subset to condition)
    # --------------------------------------------------------
    fir_cols = [c for c in design_matrix.columns if condition in c and "delay" in c]
    dm_cond = design_matrix[fir_cols]
    time = dm_cond.index.values

    # --------------------------------------------------------
    # 4. Reconstruct haemodynamic response
    # --------------------------------------------------------
    def reconstruct(chroma, color):
        df_chroma = df_sum.query(f"Chroma == '{chroma}'")
        vals = df_chroma["Coef."].astype(float).values
        l95 = df_chroma["[0.025"].astype(float).values
        u95 = df_chroma["0.975]"].astype(float).values
        return dict(
            total=np.sum(dm_cond.values * vals, axis=1),
            l95=np.sum(dm_cond.values * l95, axis=1),
            u95=np.sum(dm_cond.values * u95, axis=1),
            color=color
        )

    results = {
        "hbo": reconstruct("hbo", "red"),
        "hbr": reconstruct("hbr", "blue")
    }

    # --------------------------------------------------------
    # 5. Plot (3 panels like the example)
    # --------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Panel 1: FIR Basis
    axes[0].plot(time, dm_cond.values)
    axes[0].set_title("FIR Basis (Unscaled)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Basis amplitude")

    # Panel 2: FIR Components (HbO scaled)
    df_hbo = df_sum.query("Chroma == 'hbo'")
    if not df_hbo.empty:
        vals_hbo = df_hbo["Coef."].astype(float).values
        axes[1].plot(time, dm_cond.values * vals_hbo)
        axes[1].set_title(f"FIR Components Scaled ({condition}, HbO)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("ΔμMol")

    # Panel 3: Final Evoked Response with CI
    for chroma, res in results.items():
        axes[2].plot(time, res["total"], color=res["color"], label=chroma.upper())
        axes[2].fill_between(time, res["l95"], res["u95"],
                             alpha=0.3, color=res["color"])
    axes[2].set_title(f"Group Evoked Response ({condition})")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Haemoglobin (ΔμMol)")
    axes[2].legend()

    plt.tight_layout()
    plt.show()
    plt.savefig(f"Group_FIR_Response_{condition}.png")
    
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
        high_pass = 0.01 #high_pass_value
        add_regs = short_channel_haemo.get_data().T
        oversampling = 1 # Default value.
        drift_order = 1 # When we use the cosine drift model this parameter doesn't really matter, as the drift order is then actually determined by the high_pass argument
        add_reg_names = short_channel_haemo.ch_names
        fir_delays = range(10) # Default when we don't use a FIR model
        
        
        # design_matrix = make_first_level_design_matrix(frame_times, events,
        #                                 drift_model=drift_model,
        #                                 drift_order=drift_order,
        #                                 hrf_model=hrf_model,
        #                                 min_onset=min_onset,
        #                                 high_pass=high_pass,
        #                                 add_regs=add_regs,
        #                                 oversampling=oversampling,
        #                                 add_reg_names=add_reg_names,
        #                                 fir_delays=fir_delays)
        
        design_matrix = make_first_level_design_matrix(
        haemo,
        hrf_model="fir",
        stim_dur=1.0,
        fir_delays=range(10),
        drift_model="cosine",
        high_pass=0.01,
        oversampling=1,
        )
        
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
        
        # Create a single ROI that includes all channels for example
        rois = dict(AllChannels=range(len(haemo.ch_names)))
        # Calculate ROI for all conditions
        conditions = design_matrix.columns
        # Compute output metrics by ROI
        df_ind = glm_estimates.to_dataframe_region_of_interest(rois, conditions)

        df_ind["ID"] = subject.name
        df_ind["theta"] = [t * 1.0e6 for t in df_ind["theta"]]
        
        # df = betas_df.copy()
        
        return df_ind, haemo, design_matrix
    
    results = Parallel(n_jobs=2)(
    delayed(glm_subject)(subject, idx, class_instance.data_types, hrf_model)
    for idx, subject in enumerate(subjects)
    )

    subject_dfs, glm_results, design_matrices = zip(*results)
    
    # Pick the first subject’s GLM + design matrix and plot TongueMI
    # Example: plot TongueMI for the first subject, channel pair S1_D1

    betas_df = pd.concat(subject_dfs, ignore_index=True)

    plot_group_fir_model(betas_df, design_matrices[0], condition="TongueMI", raw_haemo=subjects[0].raw_haemo)
    # plot_group_fir_model(betas_df, design_matrices[0], condition="Control")
    
#     motor_cortex_roi = ['S4_D3 hbo', 'S4_D3 hbr', 'S4_D5 hbo', 'S4_D5 hbr', 'S4_D6 hbo', 'S4_D6 hbr', 'S4_D7 hbo', 'S4_D7 hbr', 'S2_D2 hbo', 'S2_D2 hbr', 'S2_D3 hbo', 'S2_D3 hbr', 'S2_D5 hbo', 'S2_D5 hbr', 'S6_D5 hbo', 'S6_D5 hbr', 'S6_D7 hbo', 'S6_D7 hbr', 'S10_D2 hbo', 'S10_D2 hbr', 'S10_D10 hbo', 'S10_D10 hbr', 'S10_D12 hbo', 'S10_D12 hbr', 'S14_D12 hbo', 'S14_D12 hbr', 'S14_D14 hbo', 'S14_D14 hbr', 'S14_D14 hbo', 'S14_D14 hbr', 'S12_D10 hbo',
# 'S12_D10 hbr',
# 'S12_D12 hbo',
# 'S12_D12 hbr',
# 'S12_D13 hbo',
# 'S12_D13 hbr',
# 'S12_D14 hbo', 
# 'S12_D14 hbr',]
    
    filtered_betas_df = betas_df[betas_df["channel"].isin(betas_df)]

    with localconverter(pandas2ri.converter):
        globalenv["rdf"] = filtered_betas_df

    lme4 = importr("lme4")

    r('''
    library(lme4)
    modelCondition <- lmer(beta ~ condition + channel + (1 | participant), data=rdf, REML=FALSE)
    nullModelCondition <- lmer(beta ~ channel + (1 | participant), data=rdf, REML=FALSE)
    print(summary(modelCondition))
    print(summary(nullModelCondition))
    coefs <- as.data.frame(coef(summary(modelCondition)))
    anova_result_condition <- anova(modelCondition, nullModelCondition)
    print(anova_result_condition)
    
    nullModelChannel <- lmer(beta ~ condition + (1 | participant), data=rdf, REML=FALSE)
    anova_result_channel <- anova(modelCondition, nullModelChannel)
    print(anova_result_channel)
    
    model_interaction <- lmer(beta ~ condition * channel + (1 | participant), data=rdf, REML=FALSE)
    print(summary(model_interaction)$coefficients)
    coefs <- as.data.frame(coef(summary(model_interaction)))
    sig_channels <- subset(coefs, abs(`t value`) > 2)
    print(sig_channels)
    
    library(lmerTest)
    modelCondition2 <- lmer(beta ~ condition * channel + (1 | participant), data=rdf, REML=FALSE)
    print(summary(modelCondition2))
        
    # Residual plots
    par(mfrow=c(1,2))  # two plots side by side
    
    # Residuals vs fitted
    plot(fitted(modelCondition), resid(modelCondition),
         main="Residuals vs Fitted",
         xlab="Fitted values", ylab="Residuals")
    abline(h=0, col="red")

    # Normal Q-Q plot
    qqnorm(resid(modelCondition), main="Normal Q-Q")
    qqline(resid(modelCondition), col="red")
    ''')
    
    # Convert to pandas
    with localconverter(pandas2ri.converter):
        coefs_df = r('coefs')

    print(coefs_df.head())