from preprocessesing_toolbox.extractEpochDataFromTime import ExtractRawDataFromAbsoluteTime

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
    
    def usePreviousRest(self, events, event_dict, raw_haemo, stimulusDuration):
        name = "Use previous rest"
        eventID = list(event_dict.values())
        eventID.remove(event_dict["Control"])
        raw_data = raw_haemo.get_data()
        for i in range(1, len(events)):  # start from 1 to have a "previous" row
            current_event = events[i, 2]
            if current_event in eventID:
                previous_row = events[i - 1]
                current_row = events[i]
                
                startSamplePrevious = previous_row[0]
                endSamplePrevious = startSamplePrevious + int(stimulusDuration * raw_haemo.info["sfreq"])
                
                startSampleCurrent = current_row[0]
                endSampleCurrent = startSampleCurrent + int(stimulusDuration * raw_haemo.info["sfreq"])
                
                # Extract data from previous epoch
                tminPrevious = ExtractRawDataFromAbsoluteTime.convert_sample_to_absolute_time(startSamplePrevious, raw_haemo.info["sfreq"])
                tmaxPrevious = tminPrevious + stimulusDuration
                timeCroppedDataPreviousEvent, _ = ExtractRawDataFromAbsoluteTime.extract_data_from_absolute_time(raw_haemo, tminPrevious, tmaxPrevious)
                
                # Extract data from current epoch
                tminCurrent = ExtractRawDataFromAbsoluteTime.convert_sample_to_absolute_time(startSampleCurrent, raw_haemo.info["sfreq"])
                tmaxCurrent = tminCurrent + stimulusDuration
                
                # Subtract mean of previous epoch (for each channel) from current epoch
                meanPrevious = timeCroppedDataPreviousEvent.mean(axis=1)
                
                # Subtract the mean from each channel first in the current epoch and then in the previous epoch
                for ch_idx in range(raw_data.shape[0]):
                    raw_data[ch_idx, startSampleCurrent:endSampleCurrent] -= meanPrevious[ch_idx]
                    raw_data[ch_idx, startSamplePrevious:endSamplePrevious] -= meanPrevious[ch_idx]
                
                # Update the raw object with modified data
                raw_haemo._data = raw_data
        return raw_haemo
