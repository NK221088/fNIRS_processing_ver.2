
class baselineCorrection:
    """
    Class for baseline correction of fNIRS data.

    Parameters
    ----------
    name : str
        The name of the baseline correction method.

    Attributes
    ----------
    name : str
        The name of the baseline correction method.

    Methods
    -------
    get_name()
        Returns the name of the correction method.

    useFirstBaseline(resting_baseline, epochs, raw_haemo)
        Uses the first resting baseline to subtract from all epochs.
    """
    
    def __init__(self, name):
        self.name = name
    
    def get_name(self):
        return self.name
    
    def useFirstBaseline(self, resting_baseline, epochs, raw_haemo):
        """
        Apply baseline correction using the first resting period.

        Parameters
        ----------
        resting_baseline : array-like
            Baseline values per channel.
        epochs : mne.Epochs
            Epoch data to be corrected.
        raw_haemo : mne.io.Raw
            Raw fNIRS data used for channel referencing.

        Returns
        -------
        epochs : mne.Epochs
            Baseline-corrected epochs.
        """
        self.name = "First Baseline available"
        for epoch_idx in range(len(epochs)):
                for ch_name in epochs.ch_names:
                    try:
                        epochs_ch_idx = epochs.ch_names.index(ch_name)
                        raw_ch_idx = raw_haemo.ch_names.index(ch_name)
                        epochs._data[epoch_idx, epochs_ch_idx, :] -= resting_baseline[raw_ch_idx]
                    except ValueError:
                        # Channel was removed during preprocessing - skip it
                        print(f"Skipping channel {ch_name}: not found in baseline data")
                        continue
        return epochs
