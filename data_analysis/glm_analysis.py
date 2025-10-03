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
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from mne_nirs.statistics import statsmodels_to_results

def plot_group_fir_model(betas_df, design_matrix, condition="TongueMI"):
    """
    Plot group FIR response using mixed-effects model (HbO & HbR in one plot).
    
    betas_df: DataFrame with columns [participant, channel, condition, beta]
    design_matrix: design matrix from GLM (for one subject, just to get FIR basis)
    condition: str, condition to plot (e.g., "TongueMI")
    """

    
    # --- Step 1: filter betas to FIR regressors for the condition of interest ---
    df = betas_df.copy()
    df = df[df["condition"].str.contains(condition)]
    df = df[df["condition"].str.contains("delay")]

    # Extract delay number and chromophore
    df["delay"] = df["condition"].apply(lambda x: int(x.split("_")[-1]))
    df["Chroma"] = df["channel"].apply(lambda x: "hbo" if "hbo" in x else "hbr")
    df["TidyCond"] = condition

    # --- Step 2: fit mixed-effects model ---
    # Use C(delay) to treat delay as categorical
    model = smf.mixedlm("beta ~ -1 + C(delay):TidyCond:Chroma",
                        df, groups=df["participant"])
    lme = model.fit()
    df_sum = statsmodels_to_results(lme)

    # Debug: show actual index format
    print(f"\nActual index format (first 3):")
    for idx in df_sum.index[:3]:
        print(f"  '{idx}'")

    # Extract numeric delay, TidyCond, and Chroma from the index
    # Try multiple regex patterns
    # Pattern 1: C(delay)[0]:TidyCond[TongueMI]:Chroma[hbo]
    df_sum["delay"] = df_sum.index.str.extract(r'C\(delay\)\[(\d+)\]', expand=False)
    df_sum["TidyCond"] = df_sum.index.str.extract(r'TidyCond\[([^\]]+)\]', expand=False)
    df_sum["Chroma"] = df_sum.index.str.extract(r'Chroma\[([^\]]+)\]', expand=False)
    
    print(f"\nAfter extraction:")
    print(df_sum[["delay", "TidyCond", "Chroma"]].head(10))
    
    # If still NaN, try alternative: maybe it's "delay[T.0]" format
    if df_sum["delay"].isna().all():
        print("\nFirst pattern failed, trying alternative patterns...")
        df_sum["delay"] = df_sum.index.str.extract(r'\[T\.(\d+)\]', expand=False)
        print(df_sum[["delay", "TidyCond", "Chroma"]].head(10))
    
    # Convert delay to int
    df_sum["delay"] = pd.to_numeric(df_sum["delay"], errors='coerce')
    df_sum = df_sum.dropna(subset=["delay"])
    
    if len(df_sum) == 0:
        print("ERROR: Could not extract delay values from index!")
        print("Full index:")
        print(df_sum.index.tolist())
        return
    
    df_sum["delay"] = df_sum["delay"].astype(int)
    df_sum = df_sum.sort_values("delay")

    print(f"\nFinal df_sum shape: {df_sum.shape}")
    print(f"Unique delays: {sorted(df_sum['delay'].unique())}")
    print(f"Unique TidyCond: {df_sum['TidyCond'].unique()}")
    print(f"Unique Chroma: {df_sum['Chroma'].unique()}")

    # --- Step 3: reconstruct responses for both chromas ---
    fir_cols = [c for c in design_matrix.columns if condition in c and "delay" in c]
    dm_cond = design_matrix[fir_cols]
    time = dm_cond.index.values

    results = {}
    for chroma, color in [("hbo", "red"), ("hbr", "blue")]:
        df_chroma = df_sum.query(f"TidyCond == '{condition}' and Chroma == '{chroma}'")
        
        print(f"\nFiltering for TidyCond='{condition}' and Chroma='{chroma}': {len(df_chroma)} rows")
        if len(df_chroma) == 0:
            print(f"WARNING: No data found for {chroma}")
            continue

        vals = df_chroma["Coef."].astype(float).values
        l95 = df_chroma["[0.025"].astype(float).values
        u95 = df_chroma["0.975]"].astype(float).values

        dm_scaled = dm_cond.values * vals
        dm_scaled_l95 = dm_cond.values * l95
        dm_scaled_u95 = dm_cond.values * u95

        results[chroma] = dict(
            color=color,
            total=np.sum(dm_scaled, axis=1),
            l95=np.sum(dm_scaled_l95, axis=1),
            u95=np.sum(dm_scaled_u95, axis=1),
        )

    if len(results) == 0:
        print("ERROR: No results generated. Cannot plot.")
        return

    # --- Step 4: plots ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Panel 1: FIR basis
    axes[0].plot(time, dm_cond.values)
    axes[0].set_title("FIR Basis (Unscaled)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Basis amplitude")

    # Panel 2: Scaled FIR components (HbO only)
    df_hbo = df_sum.query(f"TidyCond == '{condition}' and Chroma == 'hbo'")
    if len(df_hbo) > 0:
        vals_hbo = df_hbo["Coef."].astype(float).values
        axes[1].plot(time, dm_cond.values * vals_hbo)
        axes[1].set_title(f"FIR Components Scaled ({condition}, HbO)")
    else:
        axes[1].set_title(f"No HbO data available")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("ΔμMol")

    # Panel 3: Evoked response (HbO + HbR with CIs)
    for chroma in ["hbo", "hbr"]:
        if chroma in results:
            r = results[chroma]
            axes[2].plot(time, r["total"], color=r["color"], label=chroma.upper())
            axes[2].fill_between(time, r["l95"], r["u95"], alpha=0.3, color=r["color"])
    axes[2].set_title(f"Group Evoked Response ({condition})")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Haemoglobin (ΔμMol)")
    axes[2].legend()

    plt.tight_layout()
    plt.show()
    plt.savefig(f"Group_FIR_Response_{condition}.png")
    
def run_glm_analysis(subjects, class_instance, hrf_model="glover"):
    
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
        fir_delays = None # Default when we don't use a FIR model
        
        
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
        
        glm_est = glm_estimates
        glm_hbo = glm_est.copy().pick(picks="hbo", exclude='bads')
        conditions = ["HandMI"]

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
        
        return pd.DataFrame(betas), glm_estimates, design_matrix
    
    results = Parallel(n_jobs=1)(
    delayed(glm_subject)(subject, idx, class_instance.data_types, hrf_model)
    for idx, subject in enumerate(subjects)
    )

    subject_dfs, glm_results, design_matrices = zip(*results)
    
    # Pick the first subject’s GLM + design matrix and plot TongueMI
    # Example: plot TongueMI for the first subject, channel pair S1_D1

    betas_df = pd.concat(subject_dfs, ignore_index=True)
    
    # plot_group_fir_model(betas_df, design_matrices[0], condition="TongueMI")
    # plot_group_fir_model(betas_df, design_matrices[0], condition="Control")

    with localconverter(pandas2ri.converter):
        globalenv["rdf"] = betas_df

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