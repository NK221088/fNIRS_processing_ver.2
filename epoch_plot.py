import mne
import matplotlib.pyplot as plt
import os
from collections import Counter
from datetime import datetime
import re
import numpy as np


def epoch_plot(epochs, picks: list, epoch_type: str, bad_channels_strategy: str, save : bool, combine_strategy: str = "mean", threshold = None, data_set : str = "data_name"):

    """Plot epochs for one or multiple patients

    PARAMETERS
    ----------
    epochs : epoch element from mne pipeline
        epochs object to be plotted.
    epoch_type : str
        Type to be plotted, e.g. "Tapping" or "Noise". Dependant on the specific dataset.
    bad_channels_strategy : str
        Way to deal with differing bad channels according different epoch elements. Choose between "delete", "threshold" or "all".
    combine_strategy : str
        Strategy for combining epochs, default is mean.
    save : bool.
        Whether to save the plot or not.
    """
    # Check if bad_channels_strategy is valid
    if bad_channels_strategy not in ("delete", "all", "threshold"):
        raise ValueError("Invalid bad_channels_strategy. Please use 'delete', 'all' or 'threshold'.")
    
    # Check if combine_strategy is valid
    if combine_strategy not in ("mean", "median", "std", "gfp"):
        raise ValueError("Invalid combine_strategy. Please use 'mean', 'median', 'std' or 'gfp'.")
    
    # Check if save is a boolean
    if not isinstance(save, bool):
        raise ValueError("Invalid value for save. Please use True or False.")

    # Force all epochs to exactly the first epochs frequency (Hz)
    target_sfreq = epochs[0].info['sfreq']
    for i in range(len(epochs)):
        current_sfreq = epochs[i].info["sfreq"]
        if current_sfreq != target_sfreq:
            # Store original data for comparison
            original_data = epochs[i].get_data().copy()
            
            # Resample to a slightly different frequency first, then to target
            temp_sfreq = target_sfreq * 1.0001  # slightly different
            epochs[i] = epochs[i].resample(temp_sfreq)
            epochs[i] = epochs[i].resample(target_sfreq)
            
            # Compare original vs resampled data
            resampled_data = epochs[i].get_data()
            max_diff = np.max(np.abs(original_data - resampled_data))
            mean_diff = np.mean(np.abs(original_data - resampled_data))
            relative_diff = max_diff / np.max(np.abs(original_data)) * 100
            
            print(f"Epoch {i}:")
            print(f"  Original sfreq: {current_sfreq}")
            print(f"  New sfreq: {epochs[i].info['sfreq']}")
            print(f"  Max absolute difference: {max_diff:.2e}")
            print(f"  Mean absolute difference: {mean_diff:.2e}")
            print(f"  Relative difference: {relative_diff:.4f}%")
            print(f"  Data shape: {original_data.shape}")
            print()
    
    # Use the baseline from the first epoch
    common_baseline = epochs[0].baseline
    for epoch in epochs:
        epoch.baseline = common_baseline

    # Function implementation:
    if bad_channels_strategy == "delete":
        for i in range(len(epochs)):
            epochs[i].info['bads'] = []
        epochs = mne.concatenate_epochs(epochs)
        
    elif bad_channels_strategy == "all":
        bad_channels = []
        for i in range(len(epochs)):
            bad_channels.extend(epochs[i].info['bads'])
        bad_channels = list(set(bad_channels))
        for i in range(len(epochs)):
                epochs[i].info['bads'] = bad_channels
        epochs = mne.concatenate_epochs(epochs)
    elif bad_channels_strategy == "threshold":
        if threshold == None:
            raise ValueError(f"When using bad_channels_strategy {bad_channels_strategy}, you must input a threshold value as an int.")
        else:
            bad_channels = []
            for i in range(len(epochs)):
                bad_channels.extend(epochs[i].info['bads'])

            # Count occurrences of each bad channel
            channel_counts = Counter(bad_channels)

            # Keep only channels that occur more than twice
            bad_channels = [channel for channel, count in channel_counts.items() if count > 2]

            # Update epochs with filtered bad channels
            for i in range(len(epochs)):
                epochs[i].info['bads'] = bad_channels
            epochs = mne.concatenate_epochs(epochs)
    
    # Helper function to reorder channels for proper pairing
    def reorder_channels_for_pairing(epochs_obj):
        """Reorder channels to pair hbo/hbr channels together"""
        try:
            ch_names = epochs_obj.ch_names
            hbo_channels = [ch for ch in ch_names if 'hbo' in ch.lower()]
            hbr_channels = [ch for ch in ch_names if 'hbr' in ch.lower()]
            
            # Create paired ordering
            paired_channels = []
            for hbo_ch in sorted(hbo_channels):
                # Find corresponding hbr channel
                hbr_ch = hbo_ch.replace('hbo', 'hbr').replace('HbO', 'HbR')
                if hbr_ch in hbr_channels:
                    paired_channels.extend([hbo_ch, hbr_ch])
            
            # Only reorder if we have paired channels and they're not already in order
            if paired_channels and paired_channels != ch_names:
                print(f"Reordering channels for proper pairing...")
                return epochs_obj.reorder_channels(paired_channels)
            else:
                return epochs_obj
        except Exception as e:
            print(f"Warning: Could not reorder channels: {e}")
            return epochs_obj
    
    # Separate channel types if picks is provided
    if picks != "all":
        # Identify the channel types in the picks
        channel_types = set(ch.split('_')[-1].lower() for ch in picks)
        
        if len(channel_types) > 1:
            # If multiple types are selected, split into separate type lists
            hbo_picks = [ch for ch in picks if ch.lower().endswith('hbo')]
            hbr_picks = [ch for ch in picks if ch.lower().endswith('hbr')]
            
            # Create separate plots for each type
            plots = []
            
            if hbo_picks:
                hbo_plots = epochs[epoch_type].plot_image(
                    picks=hbo_picks,
                    combine=combine_strategy,
                    vmin=-30,
                    vmax=30,
                    ts_args=dict(ylim=dict(hbo=[-15, 15])),
                    show=False,
                )
                plots.extend(hbo_plots if isinstance(hbo_plots, list) else [hbo_plots])
            
            if hbr_picks:
                hbr_plots = epochs[epoch_type].plot_image(
                    picks=hbr_picks,
                    combine=combine_strategy,
                    vmin=-30,
                    vmax=30,
                    ts_args=dict(ylim=dict(hbr=[-15, 15])),
                    show=False,
                )
                plots.extend(hbr_plots if isinstance(hbr_plots, list) else [hbr_plots])
        else:
            # If only one type is selected, proceed normally
            plots = epochs[epoch_type].plot_image(
                picks=picks,
                combine=combine_strategy,
                vmin=-30,
                vmax=30,
                ts_args=dict(ylim=dict(hbo=[-15, 15], hbr=[-15, 15])),
                show=False,
            )
    else:
        # If no picks (all channels), reorder channels first to ensure proper pairing
        try:
            # Reorder channels for proper pairing
            epochs_reordered = reorder_channels_for_pairing(epochs[epoch_type])
            
            plots = epochs_reordered.plot_image(
                combine=combine_strategy,
                vmin=-30,
                vmax=30,
                ts_args=dict(ylim=dict(hbo=[-15, 15], hbr=[-15, 15])),
                show=False,
            )
        except Exception as e:
            print(f"Error with automatic channel pairing: {e}")
            print("Falling back to explicit channel selection...")
            
            # Fallback: separate hbo and hbr channels
            ch_names = epochs[epoch_type].ch_names
            hbo_channels = [ch for ch in ch_names if 'hbo' in ch.lower()]
            hbr_channels = [ch for ch in ch_names if 'hbr' in ch.lower()]
            
            plots = []
            
            if hbo_channels:
                hbo_plots = epochs[epoch_type].plot_image(
                    picks=hbo_channels,
                    combine=combine_strategy,
                    vmin=-30,
                    vmax=30,
                    ts_args=dict(ylim=dict(hbo=[-15, 15])),
                    show=False,
                )
                plots.extend(hbo_plots if isinstance(hbo_plots, list) else [hbo_plots])
            
            if hbr_channels:
                hbr_plots = epochs[epoch_type].plot_image(
                    picks=hbr_channels,
                    combine=combine_strategy,
                    vmin=-30,
                    vmax=30,
                    ts_args=dict(ylim=dict(hbr=[-15, 15])),
                    show=False,
                )
                plots.extend(hbr_plots if isinstance(hbr_plots, list) else [hbr_plots])
    
    # Save each plot if save is True (same as before)
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_types = ["Oxyhemoglobin", "Deoxyhemoglobin"]

    if not os.path.exists("Plots"):
        os.makedirs("Plots")

    for i, plot in enumerate(plots):
        if save:
            label = plot_types[i] if i < len(plot_types) else f"Plot_{i}"
            filename = os.path.join("Plots", f"{epoch_type}_epochs_plot_{label}_{bad_channels_strategy}_{data_set}_{current_datetime}.pdf")
            plot.savefig(filename)
            print(f"Plot {label} saved as {filename}")
        
        plt.close(plot)  # Always close the figure

    return plots