import mne
import matplotlib.pyplot as plt
import os
from collections import Counter
from datetime import datetime

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
    
    # Separate channel types if picks is provided
    if picks is not "all":
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
        # If no picks (all channels), proceed with original method
        plots = epochs[epoch_type].plot_image(
            combine=combine_strategy,
            vmin=-30,
            vmax=30,
            ts_args=dict(ylim=dict(hbo=[-15, 15], hbr=[-15, 15])),
            show=False,
        )
    
    # Save each plot if save is True (same as before)
    plots_folder = "Plots"
    if save:
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        Plot_types = ["Oxyhemoglobin", "Deoxyhemoglobin"]
        saved_plots = []
        
        # Ensure plots is a list
        if not isinstance(plots, list):
            plots = [plots]
        
        for plot_type, plot in zip(Plot_types[:len(plots)], plots):
            filename = os.path.join(plots_folder, f"{epoch_type}_epochs_plot_{plot_type}_{bad_channels_strategy}_{data_set}_{current_datetime}.pdf")
            plot.savefig(filename)
            print(f"Plot {plot_type} saved as {filename}")
            plt.close(plot)  # Close the figure after saving
            saved_plots.append(plot)
    
    return plots

