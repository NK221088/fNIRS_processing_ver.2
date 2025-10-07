import mne
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from collections import Counter
from datetime import datetime

def standard_fNIRS_response_plot(epochs, data_types: list, bad_channels_strategy: str, save: bool, combine_strategy: str = "mean", threshold=None, data_set: str = "data_name", picks_ : list = "all"):
    """Plot standard functional near-infrared spectroscopy (fNIRS) responses.

    This function plots the standard fNIRS responses for different conditions, such as tapping or control.

    Parameters:
    -----------
    epochs : Epoch object
        Epoch object containing the fNIRS data.
    data_types : list of str
        List of data types to be plotted, e.g., ["Tapping", "Control"].
    bad_channels_strategy : str
        Strategy for handling bad channels. Options: 'delete', 'all', or 'threshold'.
    save : bool
        Whether to save the plot.
    combine_strategy : str, optional
        Strategy for combining epochs. Default is 'mean'.
    threshold : int, optional
        Threshold value for the 'threshold' bad_channels_strategy.
    data_set : str, optional
        Name of the dataset. Default is 'data_name'.

    Raises:
    -------
    ValueError
        If bad_channels_strategy or combine_strategy is invalid.

    Notes:
    ------
    The evoked data for each condition (e.g., tapping/HbO, tapping/HbR, control/HbO, control/HbR)
    is computed and plotted.

    """
    # Check if bad_channels_strategy is valid
    if bad_channels_strategy not in ("delete", "all", "threshold"):
        raise ValueError("Invalid bad_channels_strategy. Please use 'delete', 'all' or 'threshold'.")
    
    # Check if combine_strategy is valid
    if combine_strategy not in ("mean", "median", "sum", "gfp"):
        raise ValueError("Invalid combine_strategy. Please use 'mean', 'median', or 'sum'.")
    
    # Check if save is a boolean
    if not isinstance(save, bool):
        raise ValueError("Invalid value for save. Please use True or False.")
    
    # Handle bad channels
    if bad_channels_strategy == "delete":
        for i in range(len(epochs)):
            epochs[i].info['bads'] = []
        
    elif bad_channels_strategy == "all":
        bad_channels = list(set(channel for epoch in epochs for ep in epoch for channel in ep.info['bads']))
        for epoch in epochs:
            for ep in epoch:
                ep.info['bads'] = bad_channels

    elif bad_channels_strategy == "threshold":
        if threshold is None:
            raise ValueError(f"When using bad_channels_strategy '{bad_channels_strategy}', you must input a threshold value as an int.")
        bad_channels = [channel for ep in epochs for channel in ep.info['bads']]
        channel_counts = Counter(bad_channels)
        bad_channels = [ch for ch, count in channel_counts.items() if count > threshold]
        for ep in epochs:
            ep.info['bads'] = bad_channels    

    # Create evoked data dictionary for each condition
    evoked_dict = {}

    for data_type in data_types:
        for hemoglobin in ("HbO", "HbR"):
            # Compute evoked responses per subject
            evoked_list = [next(ep for ep in sub_ep if data_type in ep.event_id).average(picks=hemoglobin.lower()) for sub_ep in epochs]


            # Rename channels inside each evoked object
            for evoked in evoked_list:
                evoked.rename_channels(lambda x: x[:-4])

            # Store list of Evoked objects
            evoked_dict[f"{data_type}/{hemoglobin}"] = evoked_list  
    
    # Assign colors automatically from colormap (one per event type)
    cmap = plt.cm.get_cmap("tab10", len(data_types))
    base_colors = {evt.split("/")[0]: mpl.colors.to_hex(cmap(i)) for i, evt in enumerate(data_types)}

    # Expand to exact evoked_dict keys (event/chromophore)
    color_dict = {key: base_colors[key.split("/")[0]] for key in evoked_dict.keys()}

    # Styles fixed by chromophore
    styles_dict = {
        "HbO": dict(linestyle="-"),
        "HbR": dict(linestyle="--"),
    }

    # Prepare picks
    if picks_ != "all":
        picks_ = set([s.removesuffix(" hbo").removesuffix(" hbr") for s in picks_]) 
        picks_ = list(picks_)

    
    # Find maximum length among all evokeds
    max_len = max(evk.data.shape[1] for lst in evoked_dict.values() for evk in lst)

    # Pad all evokeds to same length
    padded_dict = {
        key: [pad_evoked(evk, max_len) for evk in evks]
        for key, evks in evoked_dict.items()
    }

    # Plot evoked data
    plot = mne.viz.plot_compare_evokeds(
        padded_dict, combine=combine_strategy, ci=0.95, colors=color_dict, styles=styles_dict, show_sensors=True, show=False, picks=picks_,
    )

    # Save the plot if specified
    if save:
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join("Plots", f"standard_fNIRS_response_plot_{current_datetime}.pdf")
        plot[0].savefig(filename)
        print(f"Plot saved as {filename}")
    plt.close(plot[0])  # Close the figure after saving
    
    return plot

import numpy as np
import mne

def pad_evoked(evoked, new_length):
    """Pad an Evoked object with zeros up to new_length (in samples)."""
    data = evoked.data
    n_ch, n_times = data.shape

    if n_times >= new_length:
        return evoked.copy()

    # Pad with zeros
    pad_width = new_length - n_times
    padded_data = np.pad(data, ((0, 0), (0, pad_width)), mode="constant")

    # Create new EvokedArray
    padded_evoked = mne.EvokedArray(padded_data, evoked.info.copy(), tmin=evoked.times[0])
    return padded_evoked