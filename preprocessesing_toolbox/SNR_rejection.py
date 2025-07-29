import numpy as np

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