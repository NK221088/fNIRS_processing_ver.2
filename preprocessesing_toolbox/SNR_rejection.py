import numpy as np
import re
from itertools import compress

def snr_rejection(raw_intensity, method: str):
    """Calculate signal quality metric for each channel.
    
    Parameters
    ----------
    raw_intensity : mne.io.BaseRaw
        The raw fNIRS data (before conversion to optical density).
    method : str
        Either "SNR" (signal-to-noise ratio) or "CV" (coefficient of variation).
    
    Returns
    -------
    quality_metric : array of float
        Array containing quality metric for each channel.
        - For SNR: mean/std (higher = better quality)
        - For CV: std/mean (lower = better quality)
    """
    raw = raw_intensity.copy()
    raw_data = raw.get_data()
    
    # Vectorized calculation
    means = np.mean(raw_data, axis=1)
    stds = np.std(raw_data, axis=1)
    
    # Handle potential division by zero
    if np.any(stds == 0):
        print("Warning: Some channels have zero standard deviation")
        stds = np.where(stds == 0, np.finfo(float).eps, stds)
    
    if method == "SNR":
        quality_metric = means / stds
    elif method == "CV":
        # Handle potential division by zero for means
        means = np.where(means == 0, np.finfo(float).eps, means)
        quality_metric = (stds / means) * 100  # Convert to percentage
    else:
        raise ValueError(f"Unknown method: {method}. Use 'SNR' or 'CV'.")
    
    return quality_metric

def get_bad_channels_by_pairs(ch_names, snr_values, threshold, method):
    """Group channel names by source-detector pairs and determine bad channels.
    
    If any channel in a pair fails the SNR threshold, the entire pair is marked as bad.
    
    Parameters
    ----------
    ch_names : list
        List of channel names (e.g., ['S1_D1 760', 'S1_D1 850', ...])
    snr_values : array
        SNR/CV values for each channel
    threshold : float
        Threshold for determining bad channels
    
    Returns
    -------
    bad_channels : list
        List of all bad channel names
    """
    # Get individual bad channels based on SNR
    if method == "SNR":
        # For SNR: lower values are worse (use < threshold)
        individual_bad_channels = list(compress(ch_names, snr_values < threshold))
    elif method == "CV":
        # For CV: higher values are worse (use > threshold)
        individual_bad_channels = list(compress(ch_names, snr_values > threshold))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Group channels by source-detector pairs
    pairs = {}
    for ch_name in ch_names:
        # Extract the source-detector part (everything before the wavelength)
        # This handles formats like "S1_D1 760" or "S1_D1_760"
        match = re.match(r'(S\d+_D\d+)', ch_name)
        if match:
            pair_name = match.group(1)
            if pair_name not in pairs:
                pairs[pair_name] = []
            pairs[pair_name].append(ch_name)
    
    # If any channel in a pair is bad, mark the entire pair as bad
    bad_channels = []
    for pair_name, pair_channels in pairs.items():
        # Check if any channel in this pair is individually bad
        if any(ch in individual_bad_channels for ch in pair_channels):
            # Mark all channels in this pair as bad
            bad_channels.extend(pair_channels)
            print(f"Marking entire pair {pair_name} as bad: {pair_channels}")
    
    return bad_channels