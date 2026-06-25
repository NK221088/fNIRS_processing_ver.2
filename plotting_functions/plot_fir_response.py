import matplotlib.pyplot as plt

def plot_fir_response(design_matrix, glm_estimates, condition="Math", channel=None, raw_haemo=None):
    
    cha_df = glm_estimates.to_dataframe()
    
    # Get FIR columns for condition, sorted by delay number
    fir_cols = sorted(
        [c for c in design_matrix.columns if c.startswith(f"{condition}_delay_")],
        key=lambda x: int(x.split("_delay_")[1])
    )
    dm_cond = design_matrix[fir_cols]  # (n_timepoints, n_delays)
    
    # Filter betas to this condition
    df_cond = cha_df[cha_df["Condition"].isin(fir_cols)].copy()
    df_cond["delay"] = df_cond["Condition"].apply(lambda x: int(x.split("_delay_")[1]))
    df_cond = df_cond.sort_values("delay")
    
    if channel is not None:
        df_ch = df_cond[df_cond["ch_name"] == channel]
    else:
        # Average theta across all channels per delay
        df_ch = df_cond.groupby("delay")["theta"].mean().reset_index()
    
    vals = df_ch["theta"].values  # shape: (n_delays,)
    
    dm_cond_scaled = dm_cond.values * vals  # (n_timepoints, n_delays)
    
    # Time axis
    index_values = design_matrix.index.values
    if raw_haemo is not None:
        index_values = index_values - raw_haemo.annotations.onset[0]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    axes[0].plot(index_values, dm_cond.values)
    axes[0].set_title(f"FIR Basis Functions ({condition}, unscaled)")
    axes[0].set_ylabel("Regressor value")
    
    axes[1].plot(index_values, dm_cond_scaled)
    axes[1].set_title(f"FIR Components (scaled by betas)")
    axes[1].set_ylabel("ΔHbO (μMol)")
    
    axes[2].plot(index_values, np.sum(dm_cond_scaled, axis=1), "r", label=condition)
    axes[2].set_title(f"Reconstructed HRF ({condition})")
    axes[2].set_ylabel("ΔHbO (μMol)")
    axes[2].legend()
    
    for ax in axes:
        ax.set_xlim(-5, 35)
        ax.set_xlabel("Time (s)")
    
    plt.tight_layout()
    return fig