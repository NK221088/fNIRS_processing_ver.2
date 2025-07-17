from preprocessesing_toolbox.extractEpochDataFromTime import ExtractRawDataFromAbsoluteTime
import numpy as np

class baselineCorrection:
    """Class for baseline correction of fNIRS data."""
    
    def __init__(self, name):
        self.name = name
        # Registry mapping method names to functions
        self.methods = {
            "First Baseline available": self.useFirstBaseline,
            "Previous rest period": self.usePreviousRest,
            "xSecondsBefore": None  # This uses MNE's built-in baseline
        }
    
    def get_available_methods(self):
        """Get list of available baseline correction methods for UI display."""
        return ["First Baseline available", "Previous rest period", "xSecondsBefore"]
    
    def apply_correction(self, method_name, *args, **kwargs):
        """Apply a specific baseline correction method."""
        if method_name not in self.methods:
            raise ValueError(f"Unknown method: {method_name}")
        
        method = self.methods[method_name]
        if method is None:
            raise ValueError(f"Method {method_name} should be handled by MNE")
        
        return method(*args, **kwargs)
    
    def useFirstBaseline(self, epochs, raw_haemo):
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
    
    def usePreviousRest(self, epochs, data_types):
        sfreq_before = epochs.info["sfreq"]
        name = "Use previous rest"
        active_id = [epochs.event_id.get(data_type) for data_type in data_types]
        rest_id = [epochs.event_id.get("Rest"), epochs.event_id.get("Control")]
        
        # Get epochs data - shape is (n_epochs, n_channels, n_times)
        epochs_data = epochs.get_data()
        
        for epoch_idx in range(len(epochs.events)):
            current_event_id = epochs.events[epoch_idx, 2]
            
            if current_event_id in active_id:
                # Check if there's a previous epoch and if it's a rest condition
                if epoch_idx > 0 and epochs.events[epoch_idx - 1, 2] in rest_id:
                    # Get the previous rest epoch data
                    previous_rest_data = epochs_data[epoch_idx - 1]  # Shape: (n_channels, n_times)
                    
                    # Calculate mean of previous rest epoch for each channel
                    mean_previous = previous_rest_data.mean(axis=1)  # Shape: (n_channels,)
                    
                    # Subtract the mean from current epoch (broadcast across time)
                    epochs_data[epoch_idx] -= mean_previous[:, np.newaxis]
                    
                    # Also subtract from the previous rest epoch if desired
                    epochs_data[epoch_idx - 1] -= mean_previous[:, np.newaxis]
        
        # Update epochs with corrected data
        epochs._data = epochs_data
        assert epochs.info["sfreq"] == sfreq_before, "Sampling frequency changed during baseline correction"
        return epochs
                    
