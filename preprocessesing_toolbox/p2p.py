import numpy as np

def compute_p2p(epochs, data_types, percentile):
    """
    Compute percentile peak-to-peak amplitudes for HbO and HbR channels.

    This function extracts HbO and HbR data from an MNE Epochs object,
    computes the peak-to-peak amplitude for each epoch and channel,
    and then returns the chosen percentile across all values.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched fNIRS data in concentration space (after Beer–Lambert).
    data_types : str | list of str
        Channel/condition selector (e.g. 'condition1', ['cond1', 'cond2']).
        Passed to epochs[...] for subsetting before splitting into HbO/HbR.
    percentile : float
        Percentile to compute (e.g., 99 for the 99th percentile).

    Returns
    -------
    dict
        Dictionary with percentile thresholds for each chromophore type:
        {
            'hbo': float,  # HbO percentile peak-to-peak amplitude (in M)
            'hbr': float   # HbR percentile peak-to-peak amplitude (in M)
        }

    Notes
    -----
    - The percentile is computed over all epoch × channel combinations
      for the selected data subset.
    - Typical usage is to define rejection thresholds for
      mne.Epochs(..., reject=...).
    """
    hbo_data = epochs.copy()[data_types].pick("hbo").get_data()
    hbr_data = epochs.copy()[data_types].pick("hbr").get_data()
    hbo_percentile_p2p = np.percentile(
        (hbo_data.max(axis=-1) - hbo_data.min(axis=-1)).ravel(), percentile
    )
    hbr_percentile_p2p = np.percentile(
        (hbr_data.max(axis=-1) - hbr_data.min(axis=-1)).ravel(), percentile
    )
    return {"hbo": hbo_percentile_p2p, "hbr": hbr_percentile_p2p}
