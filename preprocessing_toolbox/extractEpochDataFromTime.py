import numpy as np
import mne

class ExtractRawDataFromAbsoluteTime():
    """
    A utility class to extract fNIRS or EEG data from absolute time points in the original continuous recording.
    """
    
    @staticmethod
    def extract_data_from_absolute_time(raw, tmin_abs: float, tmax_abs: float) -> np.ndarray:
        """
        Extract data from the raw continuous recording at absolute time points.

        Parameters
        ----------
        raw : mne.io.Raw
            The original raw data object used to create the epochs.
        tmin_abs : float
            Start time (in seconds) in the original recording.
        tmax_abs : float
            End time (in seconds) in the original recording.

        Returns
        -------
        np.ndarray
            The extracted data of shape (n_channels, n_times_in_window).
        """
        # Use MNE's crop method to extract the time window
        raw_copy = raw.copy()
        raw_cropped = raw_copy.crop(tmin=tmin_abs, tmax=tmax_abs)
        data = raw_cropped.get_data()
        return data, raw_cropped.times
    
    @staticmethod
    def convert_sample_to_absolute_time(sample_index: int, sfreq: float) -> float:
        """Convert sample index to absolute time."""
        return sample_index / sfreq