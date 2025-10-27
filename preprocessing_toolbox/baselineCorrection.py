from preprocessing_toolbox.extractEpochDataFromTime import ExtractRawDataFromAbsoluteTime
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
    
    def useFirstBaseline(self, epochs, data_types):
        """
        Apply baseline correction using the first resting period.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Epoch data to be corrected.
        data_types : list
            Data types (currently unused).
            
        Returns
        -------
        epochs : mne.Epochs
            Baseline-corrected epochs.
        """        
        self.name = "First Baseline available"
        sfreq_before = epochs.info["sfreq"]
        
        # Get potential resting baseline event IDs
        resting_baseline_id = [
            epochs.event_id.get("Rest"), 
            epochs.event_id.get("Control"), 
            epochs.event_id.get("Resting state"), 
            epochs.event_id.get("Pause")
        ]
        # Remove None values safely
        resting_baseline_id = [id for id in resting_baseline_id if id is not None]
        
        # Find first baseline epoch
        first_baseline_idx = None
        for epoch_idx in range(len(epochs.events)):
            if epochs.events[epoch_idx][2] in resting_baseline_id:
                first_baseline_idx = epoch_idx
                break
        
        # Validate that a baseline was found
        if first_baseline_idx is None:
            raise ValueError("No resting baseline epoch found")
        
        # Get epochs data - shape is (n_epochs, n_channels, n_times)
        epochs_data = epochs.get_data()
        baseline_data = epochs_data[first_baseline_idx]
        baseline_mean = baseline_data.mean(axis=1)
        
        # Apply baseline correction
        for epoch_idx in range(len(epochs)):
            epochs_data[epoch_idx] -= baseline_mean[:, np.newaxis]
        
        # Update epochs with corrected data
        epochs._data = epochs_data
        assert epochs.info["sfreq"] == sfreq_before, "Sampling frequency changed during baseline correction"
        
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
                    
