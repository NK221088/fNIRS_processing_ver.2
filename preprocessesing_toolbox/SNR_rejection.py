import numpy as np

def snr_rejection(
    raw_intensity: np.ndarray,
):
    r"""Calculate signal to noise ratio for each channel.

    This function calculates the signal to noise ratio (SNR) for each channel

    Parameters
    ----------
    raw_intensity : instance of raw_intensity
        The raw data (before converted to optical density).

    Returns
    -------
    snr : array of float
        Array containing signal to noise ratio for each channel.

    """

    raw = raw_intensity.copy()
    raw_data = raw.get_data()
    snr = np.zeros(raw_data.shape[0])
    for ch in range(raw_data.shape[0]):
        snr[ch] = np.mean(raw_data[ch]) / np.std(raw_data[ch])
    return snr