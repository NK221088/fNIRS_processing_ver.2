from preprocessesing_toolbox.extractEpochDataFromTime import ExtractRawDataFromAbsoluteTime

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
    
    @staticmethod
    def usePreviousRest(events, event_dict, raw_haemo, stimulusDuration):
        eventID = list(event_dict.values())
        eventID.remove(event_dict["Control"])
        raw_data = raw_haemo.get_data()
        for i in range(1, len(events)):  # start from 1 to have a "previous" row
            current_event = events[i, 2]
            if current_event in eventID:
                previous_row = events[i - 1]
                current_row = events[i]
                
                # Extract data from previous epoch
                tminPrevious = ExtractRawDataFromAbsoluteTime.convert_sample_to_absolute_time(previous_row[0], raw_haemo.info["sfreq"])
                tmaxPrevious = tminPrevious + stimulusDuration
                timeCroppedDataPreviousEvent, _ = ExtractRawDataFromAbsoluteTime.extract_data_from_absolute_time(raw_haemo, tminPrevious, tmaxPrevious)
                
                # Extract data from current epoch
                tminCurrent = ExtractRawDataFromAbsoluteTime.convert_sample_to_absolute_time(current_row[0], raw_haemo.info["sfreq"])
                tmaxCurrent = tminCurrent + stimulusDuration
                
                # Subtract mean of previous epoch (for each channel) from current epoch
                meanPrevious = timeCroppedDataPreviousEvent.mean(axis=1)
                
                startSample = current_row[0]
                endSample = startSample + int(stimulusDuration * raw_haemo.info["sfreq"])
                
                # Subtract the mean from each channel in the current epoch
                for ch_idx in range(raw_data.shape[0]):
                    raw_data[ch_idx, startSample:endSample] -= meanPrevious[ch_idx]
                
                # Update the raw object with modified data
                raw_haemo._data = raw_data
        return raw_haemo
