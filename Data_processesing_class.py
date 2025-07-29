from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
import mne
import mne_nirs
import os
from Participant_class import individual_participant_class
import glob
from pathlib import Path
from dotenv import load_dotenv
from preprocessesing_toolbox.baselineCorrection import baselineCorrection
from preprocessesing_toolbox.post_rejection import reject_if_single_event_type
from preprocessesing_toolbox.SNR_rejection import snr_rejection

load_dotenv()

class fNIRS_data_load:
    def __init__(self, file_path, number_of_participants=1, annotation_names=None, stimulus_duration=5,
                 short_channel_correction=True, negative_correlation_enhancement=True, scalp_coupling_threshold=0.8,
                 reject_criteria: dict = dict(hbo=80e-6), tmin=0, tmax=15, baseline=(None, 0), data_types=[], number_of_data_types=2,
                 data_name="None", interpolate_bad_channels=False, unwanted = ["15.0"], baseline_correction: str = "Previous rest period",
                 filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2, l_trans_bandwidth: float = 0.02,
                 snr_rejection: str = None, snr_threshold : int = 8):    
            
        self.number_of_participants = number_of_participants
        self.file_path = file_path
        self.annotation_names = annotation_names
        self.stimulus_duration = stimulus_duration
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.scalp_coupling_threshold = scalp_coupling_threshold
        self.reject_criteria = reject_criteria
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        self.all_epochs = []
        self.all_control = []
        self.data_types = data_types
        self.number_of_data_types = len(data_types)
        self.data_name = data_name
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = unwanted
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        self.snr_rejection = snr_rejection
        self.snr_threshold = snr_threshold
        setattr(self, 'Individual_participants', [])
        for name in self.data_types:
            setattr(self, f'all_{name}', [])

    def define_raw_intensity(self, sub_id):
        fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
        fnirs_cw_amplitude_dir = fnirs_data_folder / "Participant-1"
        raw_intensity = mne.io.read_raw_nirx(fnirs_cw_amplitude_dir, verbose=True)
        raw_intensity.load_data()
        return raw_intensity

    def load_data(self):
        for i in range(1, self.number_of_participants + 1):
            sub_id = str(i).zfill(2)  # Pad with zeros to get "01", "02", etc.
            raw_intensity = self.define_raw_intensity(sub_id)

            raw_intensity.annotations.set_durations(self.stimulus_duration)
            raw_intensity.annotations.rename(self.annotation_names)
            for _unwanted in self.unwanted:
                    unwanted = np.nonzero(raw_intensity.annotations.description == _unwanted)
                    raw_intensity.annotations.delete(unwanted)

            if self.snr_rejection:
                snr = snr_rejection(raw_intensity, self.snr_rejection)
                if self.snr_rejection == "SNR" and self.snr_threshold < 1:
                    raise ValueError("Currently the classic signal to noise ratio is used but the threshold for SNR is below 1 resulting in all channels being marked as bad. Please set the threshold to a value above 1.")
                if self.snr_rejection == "CV" and self.snr_threshold > 1:
                    raise ValueError("Currently the coefficient of variation is used but the threshold for CV is above 1 resulting in all channels being marked as bad. Please set the threshold to a value below 1.")
                snr_bad_channels = list(compress(raw_intensity.ch_names, snr < self.snr_threshold))
                raw_intensity.info["bads"] = snr_bad_channels
            else:
                snr_bad_channels = []
                
            raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
            
            raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)

            # Check channel name consistency
            assert raw_intensity.ch_names == raw_od.ch_names, \
                f"Channel names mismatch!\nraw_intensity: {len(raw_intensity.ch_names)} channels\nraw_od: {len(raw_od.ch_names)} channels"
            
            if self.short_channel_correction:
                raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)
            raw_od = mne_nirs.channels.get_long_channels(raw_od)

            sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)

            sci_bad_channels = list(compress(raw_od.ch_names, sci < self.scalp_coupling_threshold))
            
            # Combine bad channels from all preprocessing
            all_bad_channels = list(set(snr_bad_channels + sci_bad_channels))            
            raw_od.info["bads"] = all_bad_channels
            
            if self.interpolate_bad_channels:
                raw_od.interpolate_bads()

            raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)

            raw_haemo_unfiltered = raw_haemo.copy()
            raw_haemo.filter(self.filter_lower_value, self.filter_upper_value, h_trans_bandwidth=self.h_trans_bandwidth, l_trans_bandwidth=self.l_trans_bandwidth)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            # Set baseline parameter based on correction method
            baseline = self.baseline if self.baseline_correction == "xSecondsBefore" else None

            epochs = mne.Epochs(
                raw_haemo,
                events,
                event_id=event_dict,
                tmin=self.tmin,
                tmax=self.tmax,
                reject=self.reject_criteria,
                reject_by_annotation=True,
                proj=True,
                baseline=baseline,
                preload=True,
                detrend=None,
                verbose=True,
            )

            epochs = reject_if_single_event_type(epochs) # Reject all epochs if only one event type is present

            if len(epochs) != 0:
                # Apply custom baseline correction if needed
                if self.baseline_correction != "xSecondsBefore":
                    corrector = baselineCorrection(self.baseline_correction)
                    epochs = corrector.apply_correction(
                        self.baseline_correction,
                        epochs,
                        data_types=self.data_types,
                    )

                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
                
                Participant_i = individual_participant_class(f"Participant_{i}")
                Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                Participant_i.raw_intensity = raw_intensity
                Participant_i.raw_od = raw_od
                Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                Participant_i.raw_haemo = raw_haemo
                Participant_i.epochs = epochs
                
                for name in self.data_types:
                    getattr(self, f'all_{name}').append(epochs[name].get_data(copy=True))
                    Participant_i.events.update({name: epochs[name].get_data(copy=True)})
                
                getattr(self, 'Individual_participants').append(Participant_i)
                

        # Concatenate the control data
        self.all_control = np.concatenate(self.all_control, axis=0)

        # Concatenate the data for each data type
        for name in self.data_types:
            setattr(self, f'all_{name}', np.concatenate(getattr(self, f'all_{name}'), axis=0))

        # Create the dictionary all_data with Control and data for each data type
        all_data = {"Control": self.all_control}
        for name in self.data_types:
            all_data.update({name: getattr(self, f'all_{name}')})

        # Update all_data with control_dict
        all_freq = self.all_epochs[0].info['sfreq']
        self.data_types.append("Control")
        return self.all_epochs, self.data_name, all_data, all_freq, self.data_types, self.Individual_participants

###############################################################################################################################################################################################

class AudioSpeechNoise_data_load(fNIRS_data_load):
    def __init__(self, short_channel_correction : bool, negative_correlation_enhancement : bool, interpolate_bad_channels:bool=False, tmin:int = -5,baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2, l_trans_bandwidth: float = 0.02,
                 reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8, snr_rejection: bool = True, snr_threshold: int = 8):
        self.number_of_participants = 17
        self.all_speech = []
        self.all_noise = []
        self.annotation_names = {"1.0": "Control",
                            "2.0": "Activity/Noise",
                            "3.0": "Activity/Speech"}
        self.file_path = mne_nirs.datasets.block_speech_noise.data_path()
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 5
        self.scalp_coupling_threshold = scalp_coupling_threshold
        self.reject_criteria = reject_criteria
        self.tmin = tmin
        self.tmax = 15
        self.baseline = (None, 0)
        self.data_types = ["Speech", "Noise"]
        self.number_of_data_types = 2
        self.data_name = "AudioSpeechNoise"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = ["15.0"]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        self.snr_rejection = snr_rejection
        self.snr_threshold = snr_threshold
        super().__init__(
                        number_of_participants = self.number_of_participants,
                        file_path = self.file_path,
                        annotation_names = self.annotation_names,
                        stimulus_duration = self.stimulus_duration,
                        short_channel_correction = self.short_channel_correction,
                        negative_correlation_enhancement = self.negative_correlation_enhancement,
                        scalp_coupling_threshold = self.scalp_coupling_threshold,
                        reject_criteria = self.reject_criteria,
                        baseline = self.baseline,
                        tmin = self.tmin,
                        tmax = self.tmax,
                        data_types = self.data_types,
                        number_of_data_types = self.number_of_data_types,
                        data_name = self.data_name,
                        interpolate_bad_channels = self.interpolate_bad_channels,
                        unwanted = self.unwanted,
                        baseline_correction = self.baseline_correction,
                        filter_lower_value = self.filter_lower_value,
                        filter_upper_value = self.filter_upper_value,
                        h_trans_bandwidth = self.h_trans_bandwidth,
                        l_trans_bandwidth = self.l_trans_bandwidth,
                        snr_rejection = self.snr_rejection,
                        snr_threshold = self.snr_threshold
                    )

    def define_raw_intensity(self, sub_id):
        fnirs_snirf_file_path = os.path.join(self.file_path, f"sub-{sub_id}", "ses-01", "nirs", f"sub-{sub_id}_ses-01_task-AudioSpeechNoise_nirs.snirf")
        raw_intensity = mne.io.read_raw_snirf(fnirs_snirf_file_path, verbose=True)
        raw_intensity.load_data()
        return raw_intensity

###############################################################################################################################################################################################

class fNIRS_motor_data_load(fNIRS_data_load):
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool, interpolate_bad_channels:bool=False, tmin:int = 0, baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2, l_trans_bandwidth: float = 0.02, reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8):
        self.number_of_participants = 1
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1.0": "Control",
                                "2.0": "Tapping/Left",
                                "3.0": "Tapping/Right"}
        self.file_path = mne.datasets.fnirs_motor.data_path()
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 5
        self.scalp_coupling_threshold = scalp_coupling_threshold
        self.reject_criteria =  reject_criteria
        self.tmin = tmin
        self.tmax = 15
        self.baseline = (None, 0)
        self.data_types = ["Tapping"]
        self.number_of_data_types = 2
        self.data_name = "fnirs_motor_plus_anti"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = ["15.0"]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        super().__init__(
            number_of_participants=self.number_of_participants,
            file_path=self.file_path,
            annotation_names=self.annotation_names,
            stimulus_duration=self.stimulus_duration,
            short_channel_correction=self.short_channel_correction,
            negative_correlation_enhancement=self.negative_correlation_enhancement,
            scalp_coupling_threshold=self.scalp_coupling_threshold,
            reject_criteria=self.reject_criteria,
            baseline=self.baseline,
            tmin=self.tmin,
            tmax=self.tmax,
            data_types=self.data_types,
            number_of_data_types=self.number_of_data_types,
            data_name=self.data_name,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted,
            filter_lower_value = self.filter_lower_value,
            filter_upper_value = self.filter_upper_value,
            h_trans_bandwidth = self.h_trans_bandwidth,
            l_trans_bandwidth = self.l_trans_bandwidth
        )

    def define_raw_intensity(self, sub_id):
        fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
        fnirs_cw_amplitude_dir = fnirs_data_folder / "Participant-1"
        raw_intensity = mne.io.read_raw_nirx(fnirs_cw_amplitude_dir, verbose=True)
        raw_intensity.load_data()
        return raw_intensity

###############################################################################################################################################################################################

class fNIRS_full_motor_data_load(fNIRS_data_load):
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool, interpolate_bad_channels:bool=False, tmin:int = 0, baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2, l_trans_bandwidth: float = 0.02, reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8):
        self.number_of_participants = 5
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1.0": "Control",
                                 "2.0": "Tapping/Left",
                                 "3.0": "Tapping/Right"}
        self.file_path = mne.datasets.fnirs_motor.data_path()
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 5
        self.scalp_coupling_threshold = scalp_coupling_threshold
        self.reject_criteria = reject_criteria
        self.tmin = tmin
        self.tmax = 15
        self.baseline = (None, 0)
        self.data_types = ["Tapping"]
        self.number_of_data_types = 2
        self.data_name = "Dr. Luke: full motor data"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = ["15.0"]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        self.snr_rejection = True
        self.snr_threshold = 8
        super().__init__(
            number_of_participants=self.number_of_participants,
            file_path=self.file_path,
            annotation_names=self.annotation_names,
            stimulus_duration=self.stimulus_duration,
            short_channel_correction=self.short_channel_correction,
            negative_correlation_enhancement=self.negative_correlation_enhancement,
            scalp_coupling_threshold=self.scalp_coupling_threshold,
            reject_criteria=self.reject_criteria,
            baseline=self.baseline,
            tmin=self.tmin,
            tmax=self.tmax,
            data_types=self.data_types,
            number_of_data_types=self.number_of_data_types,
            data_name=self.data_name,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted,
            filter_lower_value = self.filter_lower_value,
            filter_upper_value = self.filter_upper_value,
            l_trans_bandwidth=self.l_trans_bandwidth,
            h_trans_bandwidth=self.h_trans_bandwidth)

    def define_raw_intensity(self, sub_id):
        raw_intensity = mne.io.read_raw_snirf(f"Dataset/rob-luke/rob-luke-BIDS-NIRS-Tapping-e262df8/sub-{sub_id}/nirs/sub-{sub_id}_task-tapping_nirs.snirf", verbose=True)
        raw_intensity.load_data()
        return raw_intensity

###############################################################################################################################################################################################

class fNIRS_Alexandros_DoC_data_load(fNIRS_data_load):
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool, interpolate_bad_channels:bool=False, tmin:int = 0, baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2, l_trans_bandwidth: float = 0.02, reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8):
        self.number_of_participants = 4
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1": "Tongue",
                                 "Rest": "Control"
                                }
        self.file_path = mne.datasets.fnirs_motor.data_path()
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 15
        self.scalp_coupling_threshold = scalp_coupling_threshold
        self.reject_criteria = reject_criteria
        self.tmin = tmin
        self.tmax = 15
        self.baseline = (None, 0)
        self.data_types = ["Tongue"]
        self.number_of_data_types = 2
        self.data_name = "fNIRS_Alexandros_DoC_data"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = ["15.0"]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        super().__init__(
            number_of_participants=self.number_of_participants,
            file_path=self.file_path,
            annotation_names=self.annotation_names,
            stimulus_duration=self.stimulus_duration,
            short_channel_correction=self.short_channel_correction,
            negative_correlation_enhancement=self.negative_correlation_enhancement,
            scalp_coupling_threshold=self.scalp_coupling_threshold,
            reject_criteria=self.reject_criteria,
            baseline=self.baseline,
            tmin=self.tmin,
            tmax=self.tmax,
            data_types=self.data_types,
            number_of_data_types=self.number_of_data_types,
            data_name=self.data_name,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted,
            baseline_correction = self.baseline_correction,
            filter_lower_value = self.filter_lower_value,
            filter_upper_value = self.filter_upper_value,
            h_trans_bandwidth = self.h_trans_bandwidth,
            l_trans_bandwidth = self.l_trans_bandwidth
        )

    def define_raw_intensity(self, sub_id):
        raw_intensity = mne.io.read_raw_snirf(f"Dataset/Alexandros/DoC/_2024-04-29_{sub_id}.snirf", verbose=True)
        raw_intensity.load_data()
        return raw_intensity
    
    def load_data(self):
        for i in range(1, self.number_of_participants + 1):
            sub_id = str(i).zfill(2)  # Pad with zeros to get "01", "02", etc.
            raw_intensity = self.define_raw_intensity(sub_id)
            raw_intensity = self.make_annotations(raw_intensity)

            raw_intensity.annotations.rename(self.annotation_names)
            for _unwanted in self.unwanted:
                    unwanted = np.nonzero(raw_intensity.annotations.description == _unwanted)
                    raw_intensity.annotations.delete(unwanted)

            raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
            
            if self.short_channel_correction:
                raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)
            raw_od = mne_nirs.channels.get_long_channels(raw_od)

            sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)

            raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < self.scalp_coupling_threshold))
            if self.interpolate_bad_channels:
                raw_od.interpolate_bads()

            raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)

            raw_haemo_unfiltered = raw_haemo.copy()
            raw_haemo.filter(self.filter_lower_value, self.filter_upper_value, h_trans_bandwidth=self.h_trans_bandwidth, l_trans_bandwidth=self.l_trans_bandwidth)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            # Set baseline parameter based on correction method
            baseline = self.baseline if self.baseline_correction == "xSecondsBefore" else None

            epochs = mne.Epochs(
                raw_haemo,
                events,
                event_id=event_dict,
                tmin=self.tmin,
                tmax=self.tmax,
                reject=self.reject_criteria,
                reject_by_annotation=True,
                proj=True,
                baseline=baseline,
                preload=True,
                detrend=None,
                verbose=True,
            )

            epochs = reject_if_single_event_type(epochs) # Reject all epochs if only one event type is present

            if len(epochs) != 0:
                # Apply custom baseline correction if needed
                if self.baseline_correction != "xSecondsBefore":
                    corrector = baselineCorrection(self.baseline_correction)
                    epochs = corrector.apply_correction(
                        self.baseline_correction,
                        epochs,
                        data_types=self.data_types,
                    )

                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
                
                Participant_i = individual_participant_class(f"Participant_{i}")
                Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                Participant_i.raw_intensity = raw_intensity
                Participant_i.raw_od = raw_od
                Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                Participant_i.raw_haemo = raw_haemo
                Participant_i.epochs = epochs
                
                for name in self.data_types:
                    getattr(self, f'all_{name}').append(epochs[name].get_data(copy=True))
                    Participant_i.events.update({name: epochs[name].get_data(copy=True)})
                
                getattr(self, 'Individual_participants').append(Participant_i)
                

        # Concatenate the control data
        self.all_control = np.concatenate(self.all_control, axis=0)

        # Concatenate the data for each data type
        for name in self.data_types:
            setattr(self, f'all_{name}', np.concatenate(getattr(self, f'all_{name}'), axis=0))

        # Create the dictionary all_data with Control and data for each data type
        all_data = {"Control": self.all_control}
        for name in self.data_types:
            all_data.update({name: getattr(self, f'all_{name}')})

        # Update all_data with control_dict
        all_freq = self.all_epochs[0].info['sfreq']
        self.data_types.append("Control")
        return self.all_epochs, self.data_name, all_data, all_freq, self.data_types, self.Individual_participants

    
    def make_annotations(self, raw_intensity):
        sampling_frequency = raw_intensity.info["sfreq"]
        events, event_dict = mne.events_from_annotations(raw_intensity)
        cropped_raw_data = raw_intensity.copy()
        cropped_raw_data.annotations.set_durations(15)
        for id,event in enumerate(events):
            cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, 14.0, "Rest")
            if id == 7:
                cropped_raw_data.annotations.append((event[0] ) / cropped_raw_data.info['sfreq'] + 29, 10, "Pause")
            if id == 15 or id == 23:
                cropped_raw_data.annotations.append((event[0] ) / cropped_raw_data.info['sfreq'] + 29, 33, "Pause")
        return cropped_raw_data

###############################################################################################################################################################################################

class fNIRS_Alexandros_Healthy_data_load(fNIRS_data_load):
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool, interpolate_bad_channels:bool=False, tmin:int = 0, baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2, l_trans_bandwidth: float = 0.02, reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8):
        self.number_of_participants = 7
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1": "Physical_movement",
                                 "2": "Control",
                                 "3": "Imagery",
                                }
        self.file_path = mne.datasets.fnirs_motor.data_path()
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 15
        self.scalp_coupling_threshold = scalp_coupling_threshold
        self.reject_criteria = reject_criteria
        self.tmin = tmin
        self.tmax = 15
        self.baseline = (None, 0)
        self.data_types = ["Imagery"]
        self.number_of_data_types = 2
        self.data_name = "fNIRS_Alexandros_Healthy_data"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = ["1"]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        super().__init__(
            number_of_participants=self.number_of_participants,
            file_path=self.file_path,
            annotation_names=self.annotation_names,
            stimulus_duration=self.stimulus_duration,
            short_channel_correction=self.short_channel_correction,
            negative_correlation_enhancement=self.negative_correlation_enhancement,
            scalp_coupling_threshold=self.scalp_coupling_threshold,
            reject_criteria=self.reject_criteria,
            baseline=self.baseline,
            tmin=self.tmin,
            tmax=self.tmax,
            data_types=self.data_types,
            number_of_data_types=self.number_of_data_types,
            data_name=self.data_name,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted,
            filter_lower_value=self.filter_lower_value,
            filter_upper_value=self.filter_upper_value,
            h_trans_bandwidth=self.h_trans_bandwidth,
            l_trans_bandwidth=self.l_trans_bandwidth
        )

    def define_raw_intensity(self, sub_id):
        raw_intensity = mne.io.read_raw_snirf(f"Dataset/Alexandros/Healthy/_2024-04-29_{sub_id}.snirf", verbose=True)
        raw_intensity.load_data()
        return raw_intensity

    

###############################################################################################################################################################################################

class fNIRS_CUH_patient_data_load(fNIRS_data_load):
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool, interpolate_bad_channels:bool=False, tmin:int = 0, baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2, l_trans_bandwidth: float = 0.02, reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8):
        self.number_of_participants = 48
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1": "Imagery",
                                 "Rest": "Control"
                                }
        self.file_path = Path(os.getenv('Alexandros_CUH_patient_data'))
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 15
        self.scalp_coupling_threshold = scalp_coupling_threshold
        self.reject_criteria = reject_criteria
        self.tmin = tmin
        self.tmax = 15
        self.baseline = (None, 0)
        self.data_types = ["Imagery"]
        self.number_of_data_types = 2
        self.data_name = "fNIRS_CUH_patient_data"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = "Pause"
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth

        super().__init__(
            number_of_participants=self.number_of_participants,
            file_path=self.file_path,
            annotation_names=self.annotation_names,
            stimulus_duration=self.stimulus_duration,
            short_channel_correction=self.short_channel_correction,
            negative_correlation_enhancement=self.negative_correlation_enhancement,
            scalp_coupling_threshold=self.scalp_coupling_threshold,
            reject_criteria=self.reject_criteria,
            baseline=self.baseline,
            tmin=self.tmin,
            tmax=self.tmax,
            data_types=self.data_types,
            number_of_data_types=self.number_of_data_types,
            data_name=self.data_name,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted,
            baseline_correction = self.baseline_correction,
            filter_lower_value = self.filter_lower_value,
            filter_upper_value = self.filter_upper_value,
            h_trans_bandwidth = self.h_trans_bandwidth,
            l_trans_bandwidth = self.l_trans_bandwidth
        )

    def define_raw_intensity(self, sub_id):
            if sub_id == 9:
                raw_intensity = mne.io.read_raw_snirf(f"{self.file_path / f'P{sub_id}_2_2.snirf'}", verbose=True)
            else:            
                raw_intensity = mne.io.read_raw_snirf(f"{self.file_path / f'P{sub_id}_1.snirf'}", verbose=True)
            raw_intensity.load_data()
            return raw_intensity
        
    def load_data(self):
        for i in range(1, self.number_of_participants + 1):
            sub_id = str(i).zfill(2)  # Pad with zeros to get "01", "02", etc.
            if self.data_name == 'fNIRS_CUH_patient_data':
                sub_id = i
            if sub_id in [3, 14, 15, 17, 18, 30, 31, 41, 46]:
                continue
            raw_intensity = self.define_raw_intensity(sub_id)
            raw_intensity = self.make_annotations(raw_intensity)

            raw_intensity.annotations.rename(self.annotation_names)
            for _unwanted in self.unwanted:
                    unwanted = np.nonzero(raw_intensity.annotations.description == _unwanted)
                    raw_intensity.annotations.delete(unwanted)

            raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
            
            if self.short_channel_correction:
                raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)
            raw_od = mne_nirs.channels.get_long_channels(raw_od)

            sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)

            raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < self.scalp_coupling_threshold))
            if self.interpolate_bad_channels:
                raw_od.interpolate_bads()

            raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)

            raw_haemo_unfiltered = raw_haemo.copy()
            raw_haemo.filter(self.filter_lower_value, self.filter_upper_value, h_trans_bandwidth=self.h_trans_bandwidth, l_trans_bandwidth=self.l_trans_bandwidth)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            # Set baseline parameter based on correction method
            baseline = self.baseline if self.baseline_correction == "xSecondsBefore" else None

            epochs = mne.Epochs(
                raw_haemo,
                events,
                event_id=event_dict,
                tmin=self.tmin,
                tmax=self.tmax,
                reject=self.reject_criteria,
                reject_by_annotation=True,
                proj=True,
                baseline=baseline,
                preload=True,
                detrend=None,
                verbose=True,
            )

            epochs = reject_if_single_event_type(epochs) # Reject all epochs if only one event type is present

            if len(epochs) != 0:
                # Apply custom baseline correction if needed
                if self.baseline_correction != "xSecondsBefore":
                    corrector = baselineCorrection(self.baseline_correction)
                    epochs = corrector.apply_correction(
                        self.baseline_correction,
                        epochs,
                        data_types=self.data_types,
                    )

                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
                
                Participant_i = individual_participant_class(f"Participant_{i}")
                Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                Participant_i.raw_intensity = raw_intensity
                Participant_i.raw_od = raw_od
                Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                Participant_i.raw_haemo = raw_haemo
                Participant_i.epochs = epochs
                
                for name in self.data_types:
                    getattr(self, f'all_{name}').append(epochs[name].get_data(copy=True))
                    Participant_i.events.update({name: epochs[name].get_data(copy=True)})
                
                getattr(self, 'Individual_participants').append(Participant_i)
                

        # Concatenate the control data
        self.all_control = np.concatenate(self.all_control, axis=0)

        # Concatenate the data for each data type
        for name in self.data_types:
            setattr(self, f'all_{name}', np.concatenate(getattr(self, f'all_{name}'), axis=0))

        # Create the dictionary all_data with Control and data for each data type
        all_data = {"Control": self.all_control}
        for name in self.data_types:
            all_data.update({name: getattr(self, f'all_{name}')})

        # Update all_data with control_dict
        all_freq = self.all_epochs[0].info['sfreq']
        self.data_types.append("Control")
        return self.all_epochs, self.data_name, all_data, all_freq, self.data_types, self.Individual_participants

    
    def make_annotations(self, raw_intensity):
        sampling_frequency = raw_intensity.info["sfreq"]
        events, event_dict = mne.events_from_annotations(raw_intensity)
        cropped_raw_data = raw_intensity.copy()
        cropped_raw_data.annotations.set_durations(15)
        for id,event in enumerate(events):
            cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + 15, 14.0, "Rest")
            if id == 7 or id == 15 or id == 23:
                cropped_raw_data.annotations.append((event[0] ) / cropped_raw_data.info['sfreq'] + 29, 33, "Pause")
            if id == 0:
                duration = event[0] / cropped_raw_data.info['sfreq']
                cropped_raw_data.annotations.append(0, duration, "Pause")
        return cropped_raw_data

###############################################################################################################################################################################################

class fNIRS_Melika_hand_data_5Hz_load(fNIRS_data_load):
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool, interpolate_bad_channels:bool=False, tmin:int = 0, baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2, l_trans_bandwidth: float = 0.02, reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8):
        self.number_of_participants = 4
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1": "HandMI",
                                 "Rest": "Control"
                                }
        self.file_path = Path(os.getenv('Melika_hand_data_5Hz'))
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 28
        self.scalp_coupling_threshold = scalp_coupling_threshold
        self.reject_criteria = reject_criteria
        self.tmin = tmin
        self.tmax = 28
        self.baseline = (None, 0)
        self.data_types = ["HandMI"]
        self.number_of_data_types = 2
        self.data_name = "fNIRS_Melika_data"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = [""]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        super().__init__(
            number_of_participants=self.number_of_participants,
            file_path=self.file_path,
            annotation_names=self.annotation_names,
            stimulus_duration=self.stimulus_duration,
            short_channel_correction=self.short_channel_correction,
            negative_correlation_enhancement=self.negative_correlation_enhancement,
            scalp_coupling_threshold=self.scalp_coupling_threshold,
            reject_criteria=self.reject_criteria,
            baseline=self.baseline,
            tmin=self.tmin,
            tmax=self.tmax,
            data_types=self.data_types,
            number_of_data_types=self.number_of_data_types,
            data_name=self.data_name,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted,
            baseline_correction = self.baseline_correction,
            filter_lower_value = self.filter_lower_value,
            filter_upper_value = self.filter_upper_value,
            h_trans_bandwidth = self.h_trans_bandwidth,
            l_trans_bandwidth = self.l_trans_bandwidth
        )

    def define_raw_intensity(self, sub_id):
        raw_intensity = mne.io.read_raw_snirf(rf"{self.file_path / rf'subj-{sub_id}.snirf'}", verbose=True)
            
        raw_intensity.load_data()
        return raw_intensity
        
    def load_data(self):
        for i in range(1, self.number_of_participants + 1):
            sub_id = str(i).zfill(2)  # Pad with zeros to get "01", "02", etc.
            raw_intensity = self.define_raw_intensity(sub_id)
            if i == 1 or i == 2 or i == 3 or i == 4:# When data for the first patient was recorded, the introduction was not added in Satori, so we add it manually
                raw_intensity = self.make_without_intro_annotations(raw_intensity)
            else: # For all other patients we just add the resting phases
                raw_intensity = self.make_annotations(raw_intensity)


            raw_intensity.annotations.rename(self.annotation_names)
            for _unwanted in self.unwanted:
                    unwanted = np.nonzero(raw_intensity.annotations.description == _unwanted)
                    raw_intensity.annotations.delete(unwanted)

            raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
            
            if self.short_channel_correction:
                raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)
            raw_od = mne_nirs.channels.get_long_channels(raw_od)

            sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)

            raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < self.scalp_coupling_threshold))
            if self.interpolate_bad_channels:
                raw_od.interpolate_bads()

            raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)

            raw_haemo_unfiltered = raw_haemo.copy()
            raw_haemo.filter(self.filter_lower_value, self.filter_upper_value, h_trans_bandwidth=self.h_trans_bandwidth, l_trans_bandwidth=self.l_trans_bandwidth)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            # Set baseline parameter based on correction method
            baseline = self.baseline if self.baseline_correction == "xSecondsBefore" else None

            epochs = mne.Epochs(
                raw_haemo,
                events,
                event_id=event_dict,
                tmin=self.tmin,
                tmax=self.tmax,
                reject=self.reject_criteria,
                reject_by_annotation=True,
                proj=True,
                baseline=baseline,
                preload=True,
                detrend=None,
                verbose=True,
            )

            epochs = reject_if_single_event_type(epochs) # Reject all epochs if only one event type is present

            if len(epochs) != 0:
                # Apply custom baseline correction if needed
                if self.baseline_correction != "xSecondsBefore":
                    corrector = baselineCorrection(self.baseline_correction)
                    epochs = corrector.apply_correction(
                        self.baseline_correction,
                        epochs,
                        data_types=self.data_types,
                    )

                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
                
                Participant_i = individual_participant_class(f"Participant_{i}")
                Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                Participant_i.raw_intensity = raw_intensity
                Participant_i.raw_od = raw_od
                Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                Participant_i.raw_haemo = raw_haemo
                Participant_i.epochs = epochs
                
                for name in self.data_types:
                    getattr(self, f'all_{name}').append(epochs[name].get_data(copy=True))
                    Participant_i.events.update({name: epochs[name].get_data(copy=True)})
                
                getattr(self, 'Individual_participants').append(Participant_i)
            
        # Concatenate the control data
        self.all_control = np.concatenate(self.all_control, axis=0)

        # Concatenate the data for each data type
        for name in self.data_types:
            setattr(self, f'all_{name}', np.concatenate(getattr(self, f'all_{name}'), axis=0))

        # Create the dictionary all_data with Control and data for each data type
        all_data = {"Control": self.all_control}
        for name in self.data_types:
            all_data.update({name: getattr(self, f'all_{name}')})

        # Update all_data with control_dict
        all_freq = self.all_epochs[0].info['sfreq']
        self.data_types.append("Control")
        return self.all_epochs, self.data_name, all_data, all_freq, self.data_types, self.Individual_participants

    
    def make_without_intro_annotations(self, raw_intensity):
        sampling_frequency = raw_intensity.info["sfreq"]
        events, event_dict = mne.events_from_annotations(raw_intensity)
        cropped_raw_data = raw_intensity.copy()
        cropped_raw_data.annotations.set_durations(self.stimulus_duration)
        cropped_raw_data.annotations.rename({"0": "End"})

        for id,event in enumerate(events):
            if id == 0:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] - 30, 30, "Resting state") # Adding resting state in the beginning
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] - 110, 80, "Introduction")
            if id == 5:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + ( 2*self.stimulus_duration), 30, "Pause")
            if id == 11:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + ( 2*self.stimulus_duration), 10, "Outro")
            cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, self.stimulus_duration, "Rest")
        
        return cropped_raw_data

    def make_annotations(self, raw_intensity):
        sampling_frequency = raw_intensity.info["sfreq"]
        events, event_dict = mne.events_from_annotations(raw_intensity)
        cropped_raw_data = raw_intensity.copy()
        cropped_raw_data.annotations.set_durations(self.stimulus_duration)
        cropped_raw_data.annotations.description[0] = "I"
        cropped_raw_data.annotations.set_durations({"I" : 80})
        cropped_raw_data.annotations.rename({"I": "Introduction"})
        cropped_raw_data.annotations.rename({"0": "End"})

        
        for id,event in enumerate(events):
            if id == 0:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + 80, 30, "Resting state") # Adding resting state in the beginning
            elif id == 6:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, self.stimulus_duration, "Rest")
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + ( 2*self.stimulus_duration), 30, "Pause")
            elif id == 12:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, self.stimulus_duration, "Rest")
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + ( 2*self.stimulus_duration), 10, "Outro")
            else:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, self.stimulus_duration, "Rest")
        return cropped_raw_data

###############################################################################################################################################################################################

class fNIRS_Melika_tongue_5Hz_data_load(fNIRS_data_load):
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool, interpolate_bad_channels:bool=False, tmin:int = 0, baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2, l_trans_bandwidth: float = 0.02, reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8):
        self.number_of_participants = 4
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1": "TongueMI",
                                 "Rest": "Control",
                                }
        self.file_path = Path(os.getenv('Melika_tongue_5Hz'))
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 28
        self.scalp_coupling_threshold = scalp_coupling_threshold
        self.reject_criteria = reject_criteria
        self.tmin = tmin
        self.tmax = 28
        self.baseline = (None, 0)
        self.data_types = ["TongueMI"]
        self.number_of_data_types = 2
        self.data_name = "fNIRS_Melika_data"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = ["2"]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        super().__init__(
            number_of_participants=self.number_of_participants,
            file_path=self.file_path,
            annotation_names=self.annotation_names,
            stimulus_duration=self.stimulus_duration,
            short_channel_correction=self.short_channel_correction,
            negative_correlation_enhancement=self.negative_correlation_enhancement,
            scalp_coupling_threshold=self.scalp_coupling_threshold,
            reject_criteria=self.reject_criteria,
            baseline=self.baseline,
            tmin=self.tmin,
            tmax=self.tmax,
            data_types=self.data_types,
            number_of_data_types=self.number_of_data_types,
            data_name=self.data_name,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted,
            baseline_correction = self.baseline_correction,
            filter_lower_value = self.filter_lower_value,
            filter_upper_value = self.filter_upper_value,
            h_trans_bandwidth = self.h_trans_bandwidth,
            l_trans_bandwidth = self.l_trans_bandwidth
        )

    def define_raw_intensity(self, sub_id):
        raw_intensity = mne.io.read_raw_snirf(rf"{self.file_path / rf'subj-{sub_id}.snirf'}", verbose=True)
        raw_intensity.load_data()
        return raw_intensity
        
    def load_data(self):
        for i in range(1, self.number_of_participants + 1):
            sub_id = str(i).zfill(2)  # Pad with zeros to get "01", "02", etc.
            raw_intensity = self.define_raw_intensity(sub_id)
            if i == 1 : # When data for the first patient was recorded, the introduction was not added in Satori, so we add it manually
                raw_intensity = self.make_without_intro_annotations(raw_intensity)
            else: # For all other patients we just add the resting phases
                raw_intensity = self.make_annotations(raw_intensity)

            raw_intensity.annotations.rename(self.annotation_names)
            for _unwanted in self.unwanted:
                    unwanted = np.nonzero(raw_intensity.annotations.description == _unwanted)
                    raw_intensity.annotations.delete(unwanted)

            raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)

            raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
            
            if self.short_channel_correction:
                raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)
            raw_od = mne_nirs.channels.get_long_channels(raw_od)

            sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)

            raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < self.scalp_coupling_threshold))
            if self.interpolate_bad_channels:
                raw_od.interpolate_bads()

            raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)

            raw_haemo_unfiltered = raw_haemo.copy()
            raw_haemo.filter(self.filter_lower_value, self.filter_upper_value, h_trans_bandwidth=self.h_trans_bandwidth, l_trans_bandwidth=self.l_trans_bandwidth)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            # Set baseline parameter based on correction method
            baseline = self.baseline if self.baseline_correction == "xSecondsBefore" else None

            epochs = mne.Epochs(
                raw_haemo,
                events,
                event_id=event_dict,
                tmin=self.tmin,
                tmax=self.tmax,
                reject=self.reject_criteria,
                reject_by_annotation=True,
                proj=True,
                baseline=baseline,
                preload=True,
                detrend=None,
                verbose=True,
            )

            epochs = reject_if_single_event_type(epochs) # Reject all epochs if only one event type is present

            if len(epochs) != 0:
                # Apply custom baseline correction if needed
                if self.baseline_correction != "xSecondsBefore":
                    corrector = baselineCorrection(self.baseline_correction)
                    epochs = corrector.apply_correction(
                        self.baseline_correction,
                        epochs,
                        data_types=self.data_types,
                    )

                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
                
                Participant_i = individual_participant_class(f"Participant_{i}")
                Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                Participant_i.raw_intensity = raw_intensity
                Participant_i.raw_od = raw_od
                Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                Participant_i.raw_haemo = raw_haemo
                Participant_i.epochs = epochs
                
                for name in self.data_types:
                    getattr(self, f'all_{name}').append(epochs[name].get_data(copy=True))
                    Participant_i.events.update({name: epochs[name].get_data(copy=True)})
                
                getattr(self, 'Individual_participants').append(Participant_i)
                

        # Concatenate the control data
        self.all_control = np.concatenate(self.all_control, axis=0)

        # Concatenate the data for each data type
        for name in self.data_types:
            setattr(self, f'all_{name}', np.concatenate(getattr(self, f'all_{name}'), axis=0))

        # Create the dictionary all_data with Control and data for each data type
        all_data = {"Control": self.all_control}
        for name in self.data_types:
            all_data.update({name: getattr(self, f'all_{name}')})

        # Update all_data with control_dict
        all_freq = self.all_epochs[0].info['sfreq']
        self.data_types.append("Control")
        return self.all_epochs, self.data_name, all_data, all_freq, self.data_types, self.Individual_participants

    
    def make_without_intro_annotations(self, raw_intensity):
        sampling_frequency = raw_intensity.info["sfreq"]
        events, event_dict = mne.events_from_annotations(raw_intensity)
        cropped_raw_data = raw_intensity.copy()
        cropped_raw_data.annotations.set_durations(self.stimulus_duration)
        cropped_raw_data.annotations.rename({"0": "End"})

        for id,event in enumerate(events):
            if id == 0:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] - 30, 30, "Resting state") # Adding resting state in the beginning
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] - 110, 80, "Introduction")
            if id == 5:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + ( 2*self.stimulus_duration), 30, "Pause")
            if id == 11:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + ( 2*self.stimulus_duration), 10, "Outro")
            cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, self.stimulus_duration, "Rest")
        
        return cropped_raw_data

    def make_annotations(self, raw_intensity):
        sampling_frequency = raw_intensity.info["sfreq"]
        events, event_dict = mne.events_from_annotations(raw_intensity)
        cropped_raw_data = raw_intensity.copy()
        cropped_raw_data.annotations.set_durations(self.stimulus_duration)
        cropped_raw_data.annotations.description[0] = "I"
        cropped_raw_data.annotations.set_durations({"I" : 80})
        cropped_raw_data.annotations.rename({"I": "Introduction"})
        cropped_raw_data.annotations.rename({"0": "End"})

        
        for id,event in enumerate(events):
            if id == 0:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + 80, 30, "Resting state") # Adding resting state in the beginning
            elif id == 6:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, self.stimulus_duration, "Rest")
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + ( 2*self.stimulus_duration), 30, "Pause")
            elif id == 12:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, self.stimulus_duration, "Rest")
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + ( 2*self.stimulus_duration), 10, "Outro")
            else:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, self.stimulus_duration, "Rest")
        return cropped_raw_data

###############################################################################################################################################################################################

class fNIRS_Melika_hand_data_10Hz_load(fNIRS_data_load):
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool, interpolate_bad_channels:bool=False, tmin:int = 0, baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2, l_trans_bandwidth: float = 0.02, reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8):
        self.number_of_participants = 9
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1": "HandMI",
                                 "Rest": "Control"
                                }
        self.file_path = Path(os.getenv('Melika_hand_data_10Hz'))
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 28
        self.scalp_coupling_threshold = scalp_coupling_threshold
        self.reject_criteria = reject_criteria
        self.tmin = tmin
        self.tmax = 28
        self.baseline = (None, 0)
        self.data_types = ["HandMI"]
        self.number_of_data_types = 2
        self.data_name = "fNIRS_Melika_data"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = [""]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        super().__init__(
            number_of_participants=self.number_of_participants,
            file_path=self.file_path,
            annotation_names=self.annotation_names,
            stimulus_duration=self.stimulus_duration,
            short_channel_correction=self.short_channel_correction,
            negative_correlation_enhancement=self.negative_correlation_enhancement,
            scalp_coupling_threshold=self.scalp_coupling_threshold,
            reject_criteria=self.reject_criteria,
            baseline=self.baseline,
            tmin=self.tmin,
            tmax=self.tmax,
            data_types=self.data_types,
            number_of_data_types=self.number_of_data_types,
            data_name=self.data_name,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted,
            baseline_correction = self.baseline_correction,
            filter_lower_value = self.filter_lower_value,
            filter_upper_value = self.filter_upper_value,
            h_trans_bandwidth = self.h_trans_bandwidth,
            l_trans_bandwidth = self.l_trans_bandwidth)

    def define_raw_intensity(self, sub_id):
        raw_intensity = mne.io.read_raw_snirf(rf"{self.file_path / rf'subj-{sub_id}.snirf'}", verbose=True)
        raw_intensity.load_data()
        return raw_intensity
        
    def load_data(self):
        for i, filename in enumerate(sorted(os.listdir(self.file_path)), start=1):
            sub_id = str(i).zfill(2)  # Pad with zeros to get "01", "02", etc.
            raw_intensity = self.define_raw_intensity(sub_id)
            raw_intensity = self.make_without_intro_annotations(raw_intensity)


            raw_intensity.annotations.rename(self.annotation_names)
            for _unwanted in self.unwanted:
                    unwanted = np.nonzero(raw_intensity.annotations.description == _unwanted)
                    raw_intensity.annotations.delete(unwanted)

            raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
            
            if self.short_channel_correction:
                raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)
            raw_od = mne_nirs.channels.get_long_channels(raw_od)

            sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)

            raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < self.scalp_coupling_threshold))
            if self.interpolate_bad_channels:
                raw_od.interpolate_bads()

            raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)

            raw_haemo_unfiltered = raw_haemo.copy()
            raw_haemo.filter(self.filter_lower_value, self.filter_upper_value, h_trans_bandwidth=self.h_trans_bandwidth, l_trans_bandwidth=self.l_trans_bandwidth)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            # Set baseline parameter based on correction method
            baseline = self.baseline if self.baseline_correction == "xSecondsBefore" else None

            epochs = mne.Epochs(
                raw_haemo,
                events,
                event_id=event_dict,
                tmin=self.tmin,
                tmax=self.tmax,
                reject=self.reject_criteria,
                reject_by_annotation=True,
                proj=True,
                baseline=baseline,
                preload=True,
                detrend=None,
                verbose=True,
            )

            epochs = reject_if_single_event_type(epochs) # Reject all epochs if only one event type is present

            if len(epochs) != 0:
                # Apply custom baseline correction if needed
                if self.baseline_correction != "xSecondsBefore":
                    corrector = baselineCorrection(self.baseline_correction)
                    epochs = corrector.apply_correction(
                        self.baseline_correction,
                        epochs,
                        data_types=self.data_types,
                    )

                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
                
                Participant_i = individual_participant_class(f"Participant_{i}")
                Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                Participant_i.raw_intensity = raw_intensity
                Participant_i.raw_od = raw_od
                Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                Participant_i.raw_haemo = raw_haemo
                Participant_i.epochs = epochs
                
                for name in self.data_types:
                    getattr(self, f'all_{name}').append(epochs[name].get_data(copy=True))
                    Participant_i.events.update({name: epochs[name].get_data(copy=True)})
                
                getattr(self, 'Individual_participants').append(Participant_i)
                

        # Concatenate the control data
        self.all_control = np.concatenate(self.all_control, axis=0)

        # Concatenate the data for each data type
        for name in self.data_types:
            setattr(self, f'all_{name}', np.concatenate(getattr(self, f'all_{name}'), axis=0))

        # Create the dictionary all_data with Control and data for each data type
        all_data = {"Control": self.all_control}
        for name in self.data_types:
            all_data.update({name: getattr(self, f'all_{name}')})

        # Update all_data with control_dict
        all_freq = self.all_epochs[0].info['sfreq']
        self.data_types.append("Control")
        return self.all_epochs, self.data_name, all_data, all_freq, self.data_types, self.Individual_participants

    
    def make_without_intro_annotations(self, raw_intensity):
        sampling_frequency = raw_intensity.info["sfreq"]
        events, event_dict = mne.events_from_annotations(raw_intensity)
        cropped_raw_data = raw_intensity.copy()
        cropped_raw_data.annotations.set_durations(self.stimulus_duration)
        cropped_raw_data.annotations.rename({"0": "End"})

        for id,event in enumerate(events):
            if id == 0:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] - 30, 30, "Resting state") # Adding resting state in the beginning
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] - 110, 80, "Introduction")
            if id == 5:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + ( 2*self.stimulus_duration), 30, "Pause")
            if id == 11:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + ( 2*self.stimulus_duration), 10, "Outro")
            cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, self.stimulus_duration, "Rest")
        
        return cropped_raw_data

    def make_annotations(self, raw_intensity):
        sampling_frequency = raw_intensity.info["sfreq"]
        events, event_dict = mne.events_from_annotations(raw_intensity)
        cropped_raw_data = raw_intensity.copy()
        cropped_raw_data.annotations.set_durations(self.stimulus_duration)
        cropped_raw_data.annotations.description[0] = "I"
        cropped_raw_data.annotations.set_durations({"I" : 80})
        cropped_raw_data.annotations.rename({"I": "Introduction"})
        cropped_raw_data.annotations.rename({"0": "End"})

        
        for id,event in enumerate(events):
            if id == 0:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + 80, 30, "Resting state") # Adding resting state in the beginning
            elif id == 6:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, self.stimulus_duration, "Rest")
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + ( 2*self.stimulus_duration), 30, "Pause")
            elif id == 12:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, self.stimulus_duration, "Rest")
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + ( 2*self.stimulus_duration), 10, "Outro")
            else:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, self.stimulus_duration, "Rest")
        return cropped_raw_data

###############################################################################################################################################################################################

class fNIRS_Melika_tongue_10Hz_data_load(fNIRS_data_load):
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool, interpolate_bad_channels:bool=False, tmin:int = 0, baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2, l_trans_bandwidth: float = 0.02, reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8):
        self.number_of_participants = 9
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1": "TongueMI",
                                 "Rest": "Control",
                                }
        self.file_path = Path(os.getenv('Melika_tongue_10Hz'))
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 28
        self.scalp_coupling_threshold = scalp_coupling_threshold
        self.reject_criteria = reject_criteria
        self.tmin = tmin
        self.tmax = 28
        self.baseline = (None, 0)
        self.data_types = ["TongueMI"]
        self.number_of_data_types = 2
        self.data_name = "fNIRS_Melika_data"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = ["2"]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth

        super().__init__(
            number_of_participants=self.number_of_participants,
            file_path=self.file_path,
            annotation_names=self.annotation_names,
            stimulus_duration=self.stimulus_duration,
            short_channel_correction=self.short_channel_correction,
            negative_correlation_enhancement=self.negative_correlation_enhancement,
            scalp_coupling_threshold=self.scalp_coupling_threshold,
            reject_criteria=self.reject_criteria,
            baseline=self.baseline,
            tmin=self.tmin,
            tmax=self.tmax,
            data_types=self.data_types,
            number_of_data_types=self.number_of_data_types,
            data_name=self.data_name,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted,
            baseline_correction = self.baseline_correction,
            filter_lower_value = self.filter_lower_value,
            filter_upper_value = self.filter_upper_value,
            h_trans_bandwidth = self.h_trans_bandwidth,
            l_trans_bandwidth = self.l_trans_bandwidth
            )

    def define_raw_intensity(self, sub_id):
        raw_intensity = mne.io.read_raw_snirf(rf"{self.file_path / rf'subj-{sub_id}.snirf'}", verbose=True)
        raw_intensity.load_data()
        return raw_intensity
        
    def load_data(self):
        for i, filename in enumerate(sorted(os.listdir(self.file_path)), start=1):
            sub_id = str(i).zfill(2)  # Pad with zeros to get "01", "02", etc.
            raw_intensity = self.define_raw_intensity(sub_id)
            raw_intensity = self.make_annotations(raw_intensity)

            raw_intensity.annotations.rename(self.annotation_names)
            for _unwanted in self.unwanted:
                    unwanted = np.nonzero(raw_intensity.annotations.description == _unwanted)
                    raw_intensity.annotations.delete(unwanted)

            raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)

            raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
            
            if self.short_channel_correction:
                raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)
            raw_od = mne_nirs.channels.get_long_channels(raw_od)

            sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)

            raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < self.scalp_coupling_threshold))
            if self.interpolate_bad_channels:
                raw_od.interpolate_bads()

            raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)

            raw_haemo_unfiltered = raw_haemo.copy()
            raw_haemo.filter(self.filter_lower_value, self.filter_upper_value, h_trans_bandwidth=self.h_trans_bandwidth, l_trans_bandwidth=self.l_trans_bandwidth)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            # Set baseline parameter based on correction method
            baseline = self.baseline if self.baseline_correction == "xSecondsBefore" else None

            epochs = mne.Epochs(
                raw_haemo,
                events,
                event_id=event_dict,
                tmin=self.tmin,
                tmax=self.tmax,
                reject=self.reject_criteria,
                reject_by_annotation=True,
                proj=True,
                baseline=baseline,
                preload=True,
                detrend=None,
                verbose=True,
            )

            epochs = reject_if_single_event_type(epochs) # Reject all epochs if only one event type is present

            if len(epochs) != 0:
                # Apply custom baseline correction if needed
                if self.baseline_correction != "xSecondsBefore":
                    corrector = baselineCorrection(self.baseline_correction)
                    epochs = corrector.apply_correction(
                        self.baseline_correction,
                        epochs,
                        data_types=self.data_types,
                    )

                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
            
                Participant_i = individual_participant_class(f"Participant_{i}")
                Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                Participant_i.raw_intensity = raw_intensity
                Participant_i.raw_od = raw_od
                Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                Participant_i.raw_haemo = raw_haemo
                Participant_i.epochs = epochs
                
                for name in self.data_types:
                    getattr(self, f'all_{name}').append(epochs[name].get_data(copy=True))
                    Participant_i.events.update({name: epochs[name].get_data(copy=True)})
                
                getattr(self, 'Individual_participants').append(Participant_i)
                
        # Concatenate the control data
        self.all_control = np.concatenate(self.all_control, axis=0)

        # Concatenate the data for each data type
        for name in self.data_types:
            setattr(self, f'all_{name}', np.concatenate(getattr(self, f'all_{name}'), axis=0))

        # Create the dictionary all_data with Control and data for each data type
        all_data = {"Control": self.all_control}
        for name in self.data_types:
            all_data.update({name: getattr(self, f'all_{name}')})

        # Update all_data with control_dict
        all_freq = self.all_epochs[0].info['sfreq']
        self.data_types.append("Control")
        return self.all_epochs, self.data_name, all_data, all_freq, self.data_types, self.Individual_participants

    def make_annotations(self, raw_intensity):
        sampling_frequency = raw_intensity.info["sfreq"]
        events, event_dict = mne.events_from_annotations(raw_intensity)
        cropped_raw_data = raw_intensity.copy()
        cropped_raw_data.annotations.set_durations(self.stimulus_duration)
        cropped_raw_data.annotations.description[0] = "I"
        cropped_raw_data.annotations.set_durations({"I" : 80})
        cropped_raw_data.annotations.rename({"I": "Introduction"})
        cropped_raw_data.annotations.rename({"0": "End"})

        
        for id,event in enumerate(events):
            if id == 0:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + 80, 30, "Resting state") # Adding resting state in the beginning
            elif id == 6:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, self.stimulus_duration, "Rest")
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + ( 2*self.stimulus_duration), 30, "Pause")
            elif id == 12:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, self.stimulus_duration, "Rest")
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + ( 2*self.stimulus_duration), 10, "Outro")
            else:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, self.stimulus_duration, "Rest")
        return cropped_raw_data

###############################################################################################################################################################################################

class fNIRS_Melika_old_data_load(fNIRS_data_load):
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool, interpolate_bad_channels:bool=False, tmin:int = 0, baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2, l_trans_bandwidth: float = 0.02, reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8):
        self.number_of_participants = 11
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1": "HandMI",
                                 "2": "TongueMI",
                                 "Rest": "Control"
                                }
        self.file_path = Path(os.getenv('Melika_old_data'))
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 20
        self.scalp_coupling_threshold = scalp_coupling_threshold
        self.reject_criteria = reject_criteria
        self.tmin = tmin
        self.tmax = 15
        self.baseline = (None, 0)
        self.data_types = ["HandMI", "TongueMI"]
        self.number_of_data_types = 2
        self.data_name = "fNIRS_Melika_data"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = ["0"]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        super().__init__(
            number_of_participants=self.number_of_participants,
            file_path=self.file_path,
            annotation_names=self.annotation_names,
            stimulus_duration=self.stimulus_duration,
            short_channel_correction=self.short_channel_correction,
            negative_correlation_enhancement=self.negative_correlation_enhancement,
            scalp_coupling_threshold=self.scalp_coupling_threshold,
            reject_criteria=self.reject_criteria,
            baseline=self.baseline,
            tmin=self.tmin,
            tmax=self.tmax,
            data_types=self.data_types,
            number_of_data_types=self.number_of_data_types,
            data_name=self.data_name,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted,
            baseline_correction = self.baseline_correction,
            filter_lower_value = self.filter_lower_value,
            filter_upper_value = self.filter_upper_value,
            h_trans_bandwidth = self.h_trans_bandwidth,
            l_trans_bandwidth = self.l_trans_bandwidth
        )

    def define_raw_intensity(self, sub_id):
        raw_intensity = mne.io.read_raw_snirf(rf"{self.file_path / rf'subj-{sub_id}.snirf'}", verbose=True)
        raw_intensity.load_data()
        return raw_intensity
        
    def load_data(self):
        for i in range(1, self.number_of_participants + 1):
            if i == 3:
                continue
            sub_id = str(i).zfill(2)  # Pad with zeros to get "01", "02", etc.
            raw_intensity = self.define_raw_intensity(sub_id)
            raw_intensity = self.make_annotations(raw_intensity)

            raw_intensity.annotations.rename(self.annotation_names)
            for _unwanted in self.unwanted:
                    unwanted = np.nonzero(raw_intensity.annotations.description == _unwanted)
                    raw_intensity.annotations.delete(unwanted)

            raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
            
            if self.short_channel_correction:
                raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)
            raw_od = mne_nirs.channels.get_long_channels(raw_od)

            sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)

            raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < self.scalp_coupling_threshold))
            if self.interpolate_bad_channels:
                raw_od.interpolate_bads()

            raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)

            raw_haemo_unfiltered = raw_haemo.copy()
            raw_haemo.filter(self.filter_lower_value, self.filter_upper_value, h_trans_bandwidth=self.h_trans_bandwidth, l_trans_bandwidth=self.l_trans_bandwidth)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            # Set baseline parameter based on correction method
            baseline = self.baseline if self.baseline_correction == "xSecondsBefore" else None

            epochs = mne.Epochs(
                raw_haemo,
                events,
                event_id=event_dict,
                tmin=self.tmin,
                tmax=self.tmax,
                reject=self.reject_criteria,
                reject_by_annotation=True,
                proj=True,
                baseline=baseline,
                preload=True,
                detrend=None,
                verbose=True,
            )

            epochs = reject_if_single_event_type(epochs) # Reject all epochs if only one event type is present

            if len(epochs) != 0:
                # Apply custom baseline correction if needed
                if self.baseline_correction != "xSecondsBefore":
                    corrector = baselineCorrection(self.baseline_correction)
                    epochs = corrector.apply_correction(
                        self.baseline_correction,
                        epochs,
                        data_types=self.data_types,
                    )

                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
                
                Participant_i = individual_participant_class(f"Participant_{i}")
                Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                Participant_i.raw_intensity = raw_intensity
                Participant_i.raw_od = raw_od
                Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                Participant_i.raw_haemo = raw_haemo
                Participant_i.epochs = epochs
                
                for name in self.data_types:
                    getattr(self, f'all_{name}').append(epochs[name].get_data(copy=True))
                    Participant_i.events.update({name: epochs[name].get_data(copy=True)})
                
                getattr(self, 'Individual_participants').append(Participant_i)
                
        # Concatenate the control data
        self.all_control = np.concatenate(self.all_control, axis=0)

        # Concatenate the data for each data type
        for name in self.data_types:
            setattr(self, f'all_{name}', np.concatenate(getattr(self, f'all_{name}'), axis=0))

        # Create the dictionary all_data with Control and data for each data type
        all_data = {"Control": self.all_control}
        for name in self.data_types:
            all_data.update({name: getattr(self, f'all_{name}')})

        # Update all_data with control_dict
        all_freq = self.all_epochs[0].info['sfreq']
        self.data_types.append("Control")
        return self.all_epochs, self.data_name, all_data, all_freq, self.data_types, self.Individual_participants

    
    def make_annotations(self, raw_intensity):
        events, event_dict = mne.events_from_annotations(raw_intensity)
        cropped_raw_data = raw_intensity.copy()
        cropped_raw_data.annotations.set_durations(self.stimulus_duration)
        for id,event in enumerate(events):
            cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, self.stimulus_duration, "Rest")
        return cropped_raw_data

###############################################################################################################################################################################################

class fNIRS_Melika_hand_data_long_load(fNIRS_data_load):
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool, interpolate_bad_channels:bool=False, tmin:int = 0, baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2, l_trans_bandwidth: float = 0.02, reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8):
        self.number_of_participants = 7
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1": "HandMI",
                                 "Rest": "Control"
                                }
        self.file_path = Path(os.getenv('Melika_hand_data_long'))
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 21
        self.scalp_coupling_threshold = scalp_coupling_threshold
        self.reject_criteria = reject_criteria
        self.tmin = tmin
        self.tmax = 21
        self.baseline = (None, 0)
        self.data_types = ["HandMI"]
        self.number_of_data_types = 2
        self.data_name = "fNIRS_Melika_data"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = [""]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        super().__init__(
            number_of_participants=self.number_of_participants,
            file_path=self.file_path,
            annotation_names=self.annotation_names,
            stimulus_duration=self.stimulus_duration,
            short_channel_correction=self.short_channel_correction,
            negative_correlation_enhancement=self.negative_correlation_enhancement,
            scalp_coupling_threshold=self.scalp_coupling_threshold,
            reject_criteria=self.reject_criteria,
            baseline=self.baseline,
            tmin=self.tmin,
            tmax=self.tmax,
            data_types=self.data_types,
            number_of_data_types=self.number_of_data_types,
            data_name=self.data_name,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted,
            baseline_correction = self.baseline_correction,
            filter_lower_value = self.filter_lower_value,
            filter_upper_value = self.filter_upper_value,
            h_trans_bandwidth = self.h_trans_bandwidth,
            l_trans_bandwidth = self.l_trans_bandwidth
        )

    def define_raw_intensity(self, sub_id):
        raw_intensity = mne.io.read_raw_snirf(rf"{self.file_path / rf'subj-{sub_id}.snirf'}", verbose=True)
        raw_intensity.load_data()
        return raw_intensity
        
    def load_data(self):
        for i, filename in enumerate(sorted(os.listdir(self.file_path)), start=1):
            sub_id = str(i).zfill(2)  # Pad with zeros to get "01", "02", etc.
            raw_intensity = self.define_raw_intensity(sub_id)
            raw_intensity = self.make_without_intro_annotations(raw_intensity)


            raw_intensity.annotations.rename(self.annotation_names)
            for _unwanted in self.unwanted:
                    unwanted = np.nonzero(raw_intensity.annotations.description == _unwanted)
                    raw_intensity.annotations.delete(unwanted)

            raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
            
            if self.short_channel_correction:
                raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)
            raw_od = mne_nirs.channels.get_long_channels(raw_od)

            sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)

            raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < self.scalp_coupling_threshold))
            if self.interpolate_bad_channels:
                raw_od.interpolate_bads()

            raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)

            raw_haemo_unfiltered = raw_haemo.copy()
            raw_haemo.filter(self.filter_lower_value, self.filter_upper_value, h_trans_bandwidth=self.h_trans_bandwidth, l_trans_bandwidth=self.l_trans_bandwidth)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            # Set baseline parameter based on correction method
            baseline = self.baseline if self.baseline_correction == "xSecondsBefore" else None

            epochs = mne.Epochs(
                raw_haemo,
                events,
                event_id=event_dict,
                tmin=self.tmin,
                tmax=self.tmax,
                reject=self.reject_criteria,
                reject_by_annotation=True,
                proj=True,
                baseline=baseline,
                preload=True,
                detrend=None,
                verbose=True,
            )

            epochs = reject_if_single_event_type(epochs) # Reject all epochs if only one event type is present

            if len(epochs) != 0:
                # Apply custom baseline correction if needed
                if self.baseline_correction != "xSecondsBefore":
                    corrector = baselineCorrection(self.baseline_correction)
                    epochs = corrector.apply_correction(
                        self.baseline_correction,
                        epochs,
                        data_types=self.data_types,
                    )

                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
                
                Participant_i = individual_participant_class(f"Participant_{i}")
                Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                Participant_i.raw_intensity = raw_intensity
                Participant_i.raw_od = raw_od
                Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                Participant_i.raw_haemo = raw_haemo
                Participant_i.epochs = epochs
                
                for name in self.data_types:
                    getattr(self, f'all_{name}').append(epochs[name].get_data(copy=True))
                    Participant_i.events.update({name: epochs[name].get_data(copy=True)})
                
                getattr(self, 'Individual_participants').append(Participant_i)
                
        # Concatenate the control data
        self.all_control = np.concatenate(self.all_control, axis=0)

        # Concatenate the data for each data type
        for name in self.data_types:
            setattr(self, f'all_{name}', np.concatenate(getattr(self, f'all_{name}'), axis=0))

        # Create the dictionary all_data with Control and data for each data type
        all_data = {"Control": self.all_control}
        for name in self.data_types:
            all_data.update({name: getattr(self, f'all_{name}')})

        # Update all_data with control_dict
        all_freq = self.all_epochs[0].info['sfreq']
        self.data_types.append("Control")
        return self.all_epochs, self.data_name, all_data, all_freq, self.data_types, self.Individual_participants

    
    def make_without_intro_annotations(self, raw_intensity):
        sampling_frequency = raw_intensity.info["sfreq"]
        events, event_dict = mne.events_from_annotations(raw_intensity)
        cropped_raw_data = raw_intensity.copy()
        cropped_raw_data.annotations.set_durations(self.stimulus_duration)
        cropped_raw_data.annotations.rename({"0": "End"})

        for id,event in enumerate(events):
            if id == 0:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] - 30, 30, "Resting state") # Adding resting state in the beginning
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] - 110, 80, "Introduction")
            if id == 5:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + ( 2*self.stimulus_duration), 30, "Pause")
            if id == 11:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + ( 2*self.stimulus_duration), 30, "Pause")
            if id == 17:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + ( 2*self.stimulus_duration), 10, "Outro")
            cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, self.stimulus_duration, "Rest")
        
        return cropped_raw_data

    def make_annotations(self, raw_intensity):
        sampling_frequency = raw_intensity.info["sfreq"]
        events, event_dict = mne.events_from_annotations(raw_intensity)
        cropped_raw_data = raw_intensity.copy()
        cropped_raw_data.annotations.set_durations(self.stimulus_duration)
        cropped_raw_data.annotations.description[0] = "I"
        cropped_raw_data.annotations.set_durations({"I" : 80})
        cropped_raw_data.annotations.rename({"I": "Introduction"})
        if "0" in cropped_raw_data.annotations.description:
            cropped_raw_data.annotations.rename({"0": "End"})

        
        for id,event in enumerate(events):
            if id == 0:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + 80, 30, "Resting state") # Adding resting state in the beginning
            elif id == 6:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, self.stimulus_duration, "Rest")
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + ( 2*self.stimulus_duration), 30, "Pause")
            elif id == 12:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, self.stimulus_duration, "Rest")
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + ( 2*self.stimulus_duration), 30, "Pause")
            elif id == 18:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, self.stimulus_duration, "Rest")
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + ( 2*self.stimulus_duration), 10, "Outro")
            else:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, self.stimulus_duration, "Rest")
        return cropped_raw_data

###############################################################################################################################################################################################

class fNIRS_Melika_tongue_long_data_load(fNIRS_data_load):
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool, interpolate_bad_channels:bool=False, tmin:int = 0, baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2, l_trans_bandwidth: float = 0.02, reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8):
        self.number_of_participants = 6
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1": "TongueMI",
                                 "Rest": "Control",
                                }
        self.standard_event_ids = {
        np.str_('Control'): 1,
        np.str_('End'): 2,
        np.str_('Introduction'): 3,
        np.str_('Outro'): 4,
        np.str_('Pause'): 5,
        np.str_('Resting state'): 6,
        np.str_('TongueMI'): 7
        }
        self.file_path = Path(os.getenv('Melika_tongue_long_data'))
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 21
        self.scalp_coupling_threshold = scalp_coupling_threshold
        self.reject_criteria = reject_criteria
        self.tmin = tmin
        self.tmax = 21
        self.baseline = (None, 0)
        self.data_types = ["TongueMI"]
        self.number_of_data_types = 2
        self.data_name = "fNIRS_Melika_tongue_long_data"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = ["2"]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        super().__init__(
            number_of_participants=self.number_of_participants,
            file_path=self.file_path,
            annotation_names=self.annotation_names,
            stimulus_duration=self.stimulus_duration,
            short_channel_correction=self.short_channel_correction,
            negative_correlation_enhancement=self.negative_correlation_enhancement,
            scalp_coupling_threshold=self.scalp_coupling_threshold,
            reject_criteria=self.reject_criteria,
            baseline=self.baseline,
            tmin=self.tmin,
            tmax=self.tmax,
            data_types=self.data_types,
            number_of_data_types=self.number_of_data_types,
            data_name=self.data_name,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted,
            baseline_correction = self.baseline_correction,
            filter_lower_value = self.filter_lower_value,
            filter_upper_value = self.filter_upper_value,
            h_trans_bandwidth = self.h_trans_bandwidth,
            l_trans_bandwidth = self.l_trans_bandwidth
        )

    def define_raw_intensity(self, sub_id):
        raw_intensity = mne.io.read_raw_snirf(rf"{self.file_path / rf'subj-{sub_id}.snirf'}", verbose=True)
        raw_intensity.load_data()
        return raw_intensity
        
    def load_data(self):
        for i, filename in enumerate(sorted(os.listdir(self.file_path)), start=1):
            sub_id = str(i).zfill(2)  # Pad with zeros to get "01", "02", etc.
            raw_intensity = self.define_raw_intensity(sub_id)
            raw_intensity = self.make_annotations(raw_intensity)

            raw_intensity.annotations.rename(self.annotation_names)
            for _unwanted in self.unwanted:
                    unwanted = np.nonzero(raw_intensity.annotations.description == _unwanted)
                    raw_intensity.annotations.delete(unwanted)

            raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)

            raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
            
            if self.short_channel_correction:
                raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)
            raw_od = mne_nirs.channels.get_long_channels(raw_od)

            sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)

            raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < self.scalp_coupling_threshold))
            if self.interpolate_bad_channels:
                raw_od.interpolate_bads()

            raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)

            raw_haemo_unfiltered = raw_haemo.copy()
            raw_haemo.filter(self.filter_lower_value, self.filter_upper_value, h_trans_bandwidth=self.h_trans_bandwidth, l_trans_bandwidth=self.l_trans_bandwidth)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)
            
            # Standardize event IDs
            reversed_event_dict = {value: key for key, value in event_dict.items()}
            for event in events:
                event[2] = self.standard_event_ids[reversed_event_dict[event[2]]]
            for key in list(event_dict.keys()):
                event_dict[key] = self.standard_event_ids[key]
                    
            # Set baseline parameter based on correction method
            baseline = self.baseline if self.baseline_correction == "xSecondsBefore" else None

            epochs = mne.Epochs(
                raw_haemo,
                events,
                event_id=event_dict,
                tmin=self.tmin,
                tmax=self.tmax,
                reject=self.reject_criteria,
                reject_by_annotation=True,
                proj=True,
                baseline=baseline,
                preload=True,
                detrend=None,
                verbose=True,
            )

            epochs = reject_if_single_event_type(epochs) # Reject all epochs if only one event type is present

            if len(epochs) != 0:
                # Apply custom baseline correction if needed
                if self.baseline_correction != "xSecondsBefore":
                    corrector = baselineCorrection(self.baseline_correction)
                    epochs = corrector.apply_correction(
                        self.baseline_correction,
                        epochs,
                        data_types=self.data_types,
                    )
                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
                
                Participant_i = individual_participant_class(f"Participant_{i}")
                Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                Participant_i.raw_intensity = raw_intensity
                Participant_i.raw_od = raw_od
                Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                Participant_i.raw_haemo = raw_haemo
                Participant_i.epochs = epochs
                
                for name in self.data_types:
                    getattr(self, f'all_{name}').append(epochs[name].get_data(copy=True))
                    Participant_i.events.update({name: epochs[name].get_data(copy=True)})
                
                getattr(self, 'Individual_participants').append(Participant_i)
                
        # Concatenate the control data
        self.all_control = np.concatenate(self.all_control, axis=0)

        # Concatenate the data for each data type
        for name in self.data_types:
            setattr(self, f'all_{name}', np.concatenate(getattr(self, f'all_{name}'), axis=0))

        # Create the dictionary all_data with Control and data for each data type
        all_data = {"Control": self.all_control}
        for name in self.data_types:
            all_data.update({name: getattr(self, f'all_{name}')})

        # Update all_data with control_dict
        all_freq = self.all_epochs[0].info['sfreq']
        self.data_types.append("Control")
        return self.all_epochs, self.data_name, all_data, all_freq, self.data_types, self.Individual_participants

    
    def make_without_intro_annotations(self, raw_intensity):
        sampling_frequency = raw_intensity.info["sfreq"]
        events, event_dict = mne.events_from_annotations(raw_intensity)
        cropped_raw_data = raw_intensity.copy()
        cropped_raw_data.annotations.set_durations(self.stimulus_duration)
        cropped_raw_data.annotations.rename({"0": "End"})

        for id,event in enumerate(events):
            if id == 0:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] - 30, 30, "Resting state") # Adding resting state in the beginning
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] - 110, 80, "Introduction")
            if id == 5:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + ( 2*self.stimulus_duration), 30, "Pause")
            if id == 11:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + ( 2*self.stimulus_duration), 30, "Pause")
            if id == 17:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + ( 2*self.stimulus_duration), 10, "Outro")
            cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, self.stimulus_duration, "Rest")
        
        return cropped_raw_data

    def make_annotations(self, raw_intensity):
        sampling_frequency = raw_intensity.info["sfreq"]
        events, event_dict = mne.events_from_annotations(raw_intensity)
        cropped_raw_data = raw_intensity.copy()
        cropped_raw_data.annotations.set_durations(self.stimulus_duration)
        cropped_raw_data.annotations.description[0] = "I"
        cropped_raw_data.annotations.set_durations({"I" : 80})
        cropped_raw_data.annotations.rename({"I": "Introduction"})
        if "0" in cropped_raw_data.annotations.description:
            cropped_raw_data.annotations.rename({"0": "End"})

        
        for id,event in enumerate(events):
            if id == 0:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + 80, 30, "Resting state") # Adding resting state in the beginning
            elif id == 6:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, self.stimulus_duration, "Rest")
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + ( 2*self.stimulus_duration), 30, "Pause")
            elif id == 12:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, self.stimulus_duration, "Rest")
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + ( 2*self.stimulus_duration), 30, "Pause")
            elif id == 18:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, self.stimulus_duration, "Rest")
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + ( 2*self.stimulus_duration), 10, "Outro")
            else:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, self.stimulus_duration, "Rest")
        return cropped_raw_data


###############################################################################################################################################################################################

class fNIRS_Pardis_DOC_data_load(fNIRS_data_load):
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool, interpolate_bad_channels:bool=False, tmin:int = 0, baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2, l_trans_bandwidth: float = 0.02, reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8):
        self.number_of_participants = 68
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1": "TongueMI",
                                 "Rest": "Control",
                                }
        self.file_path = Path(os.getenv('Pardis_DOC_data'))
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 15
        self.scalp_coupling_threshold = scalp_coupling_threshold
        self.reject_criteria = reject_criteria
        self.tmin = tmin
        self.tmax = 21
        self.baseline = (None, 0)
        self.data_types = ["TongueMI"]
        self.number_of_data_types = 2
        self.data_name = "fNIRS_Pardis_DOC_data"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = [""]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        super().__init__(
            number_of_participants=self.number_of_participants,
            file_path=self.file_path,
            annotation_names=self.annotation_names,
            stimulus_duration=self.stimulus_duration,
            short_channel_correction=self.short_channel_correction,
            negative_correlation_enhancement=self.negative_correlation_enhancement,
            scalp_coupling_threshold=self.scalp_coupling_threshold,
            reject_criteria=self.reject_criteria,
            baseline=self.baseline,
            tmin=self.tmin,
            tmax=self.tmax,
            data_types=self.data_types,
            number_of_data_types=self.number_of_data_types,
            data_name=self.data_name,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted,
            baseline_correction = self.baseline_correction)

    def find_snirf_file(self, p_folder_path):
        """
        Find the .snirf file in the nested folder structure.
        Returns the full path to the .snirf file or None if not found.
        """
        # Look for .snirf files recursively in the P folder
        snirf_files = glob.glob(os.path.join(p_folder_path, "**", "*.snirf"), recursive=True)
        
        if snirf_files:
            return snirf_files[0]  # Return the first .snirf file found
        return None

    def define_raw_intensity(self, p_folder_name):
        """
        Load raw intensity data from a P folder.
        p_folder_name: The name of the P folder (e.g., "P1_1", "P2_1", etc.)
        """
        p_folder_path = os.path.join(self.file_path, p_folder_name)
        
        # Find the .snirf file in the nested structure
        snirf_file_path = self.find_snirf_file(p_folder_path)
        
        if not snirf_file_path:
            raise FileNotFoundError(f"No .snirf file found in {p_folder_path}")
        
        raw_intensity = mne.io.read_raw_snirf(snirf_file_path, verbose=True)
        raw_intensity.load_data()
        return raw_intensity
        
    def load_data(self):
        # Get all P folders and sort them
        p_folders = [f for f in sorted(os.listdir(self.file_path)) if f.startswith("P") and "_" in f]
        
        for i, p_folder_name in enumerate(p_folders, start=1):
            try:
                raw_intensity = self.define_raw_intensity(p_folder_name)
                raw_intensity = self.make_annotations(raw_intensity)
                
                raw_intensity.annotations.rename(self.annotation_names)

                for _unwanted in self.unwanted:
                    unwanted = np.nonzero(raw_intensity.annotations.description == _unwanted)
                    raw_intensity.annotations.delete(unwanted)

                raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)

                raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
                
                if self.short_channel_correction:
                    raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)
                raw_od = mne_nirs.channels.get_long_channels(raw_od)

                sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)

                raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < self.scalp_coupling_threshold))
                if self.interpolate_bad_channels:
                    raw_od.interpolate_bads()

                raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)

                raw_haemo_unfiltered = raw_haemo.copy()
                raw_haemo.filter(self.filter_lower_value, self.filter_upper_value, h_trans_bandwidth=self.h_trans_bandwidth, l_trans_bandwidth=self.l_trans_bandwidth)

                if self.negative_correlation_enhancement:
                    raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

                events, event_dict = mne.events_from_annotations(raw_haemo)

                # Set baseline parameter based on correction method
                baseline = self.baseline if self.baseline_correction == "xSecondsBefore" else None

                epochs = mne.Epochs(
                    raw_haemo,
                    events,
                    event_id=event_dict,
                    tmin=self.tmin,
                    tmax=self.tmax,
                    reject=self.reject_criteria,
                    reject_by_annotation=True,
                    proj=True,
                    baseline=baseline,
                    preload=True,
                    detrend=None,
                    verbose=True,
                )

                epochs = reject_if_single_event_type(epochs) # Reject all epochs if only one event type is present

                if len(epochs) != 0:
                    # Apply custom baseline correction if needed
                    if self.baseline_correction != "xSecondsBefore":
                        corrector = baselineCorrection(self.baseline_correction)
                        epochs = corrector.apply_correction(
                            self.baseline_correction,
                            epochs,
                            data_types=self.data_types,
                        )

                    self.all_epochs.append(epochs)
                    self.all_control.append(epochs["Control"].get_data(copy=True))
                    
                    Participant_i = individual_participant_class(f"Participant_{i}")
                    Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                    Participant_i.raw_intensity = raw_intensity
                    Participant_i.raw_od = raw_od
                    Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                    Participant_i.raw_haemo = raw_haemo
                    Participant_i.epochs = epochs
                    
                    for name in self.data_types:
                        getattr(self, f'all_{name}').append(epochs[name].get_data(copy=True))
                        Participant_i.events.update({name: epochs[name].get_data(copy=True)})
                    
                    getattr(self, 'Individual_participants').append(Participant_i)
            except FileNotFoundError as e:
                print(f"Error loading {p_folder_name}: {e}")
            except Exception as e:
                print(f"Unexpected error with {p_folder_name}: {e}")

        # Concatenate the control data
        self.all_control = np.concatenate(self.all_control, axis=0)

        # Concatenate the data for each data type
        for name in self.data_types:
            setattr(self, f'all_{name}', np.concatenate(getattr(self, f'all_{name}'), axis=0))

        # Create the dictionary all_data with Control and data for each data type
        all_data = {"Control": self.all_control}
        for name in self.data_types:
            all_data.update({name: getattr(self, f'all_{name}')})

        # Update all_data with control_dict
        all_freq = self.all_epochs[0].info['sfreq']
        self.data_types.append("Control")
        return self.all_epochs, self.data_name, all_data, all_freq, self.data_types, self.Individual_participants

    def make_annotations(self, raw_intensity):
        sampling_frequency = raw_intensity.info["sfreq"]
        events, event_dict = mne.events_from_annotations(raw_intensity)
        cropped_raw_data = raw_intensity.copy()
        cropped_raw_data.annotations.set_durations(self.stimulus_duration)

        
        for id,event in enumerate(events):
            if id == 7:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, 14, "Rest")
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + (self.stimulus_duration + 14), 33, "Pause")
            elif id == 15:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, 14, "Rest")
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + (self.stimulus_duration + 14), 33, "Pause")
            else:
                cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, 14, "Rest")
        return cropped_raw_data

###############################################################################################################################################################################################

class fNIRS_Pardis_HC_data_load(fNIRS_data_load):
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool, interpolate_bad_channels:bool=False, tmin:int = 0, baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2, l_trans_bandwidth: float = 0.02, reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8):
        self.number_of_participants = 68
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1": "TonguePhysical",
                                 "2": "Control",
                                 "3": "TongueIM"
                                }
        self.file_path = Path(os.getenv('Pardis_HC_data'))
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 15
        self.scalp_coupling_threshold = scalp_coupling_threshold
        self.reject_criteria = reject_criteria
        self.tmin = tmin
        self.tmax = 20
        self.baseline = (None, 0)
        self.data_types = ["TonguePhysical", "TongueIM"]
        self.number_of_data_types = 2
        self.data_name = "fNIRS_Pardis_HC"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = ["5", "6", "7"]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        super().__init__(
            number_of_participants=self.number_of_participants,
            file_path=self.file_path,
            annotation_names=self.annotation_names,
            stimulus_duration=self.stimulus_duration,
            short_channel_correction=self.short_channel_correction,
            negative_correlation_enhancement=self.negative_correlation_enhancement,
            scalp_coupling_threshold=self.scalp_coupling_threshold,
            reject_criteria=self.reject_criteria,
            baseline=self.baseline,
            tmin=self.tmin,
            tmax=self.tmax,
            data_types=self.data_types,
            number_of_data_types=self.number_of_data_types,
            data_name=self.data_name,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted,
            baseline_correction = self.baseline_correction,
            filter_lower_value = self.filter_lower_value,
            filter_upper_value = self.filter_upper_value,
            h_trans_bandwidth = self.h_trans_bandwidth,
            l_trans_bandwidth = self.l_trans_bandwidth,
            )

    def find_snirf_file(self, folder_path):
        """
        Find the .snirf file in the nested folder structure.
        Returns the full path to the .snirf file or None if not found.
        """
        # Look for .snirf files recursively in the folder
        snirf_files = glob.glob(os.path.join(folder_path, "**", "*.snirf"), recursive=True)
        
        if snirf_files:
            return snirf_files[0]  # Return the first .snirf file found
        return None

    def define_raw_intensity(self, folder_name):
        """
        Load raw intensity data from a folder (handles different dataset structures).
        folder_name: The name of the folder containing the data
        """
        folder_path = os.path.join(self.file_path, folder_name)
        
        # Find the .snirf file in the nested structure
        snirf_file_path = self.find_snirf_file(folder_path)
        
        if not snirf_file_path:
            raise FileNotFoundError(f"No .snirf file found in {folder_path}")
        
        raw_intensity = mne.io.read_raw_snirf(snirf_file_path, verbose=True)
        raw_intensity.load_data()
        return raw_intensity

    def load_data(self):
        # Get all folders and sort them (works for both P folders and random named folders)
        all_folders = [f for f in sorted(os.listdir(self.file_path)) 
                    if os.path.isdir(os.path.join(self.file_path, f))]
        
        for i, folder_name in enumerate(all_folders, start=1):
            try:
                raw_intensity = self.define_raw_intensity(folder_name)
                # Process your raw_intensity data here
                print(f"Successfully loaded data from {folder_name} (iteration {i})")

                raw_intensity.annotations.rename(self.annotation_names)
                for _unwanted in self.unwanted:
                    unwanted = np.nonzero(raw_intensity.annotations.description == _unwanted)
                    raw_intensity.annotations.delete(unwanted)

                raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)

                raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
                
                if self.short_channel_correction:
                    raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)
                raw_od = mne_nirs.channels.get_long_channels(raw_od)

                sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)

                raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < self.scalp_coupling_threshold))
                if self.interpolate_bad_channels:
                    raw_od.interpolate_bads()

                raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=0.1)

                raw_haemo_unfiltered = raw_haemo.copy()
                raw_haemo.filter(self.filter_lower_value, self.filter_upper_value, h_trans_bandwidth=self.h_trans_bandwidth, l_trans_bandwidth=self.l_trans_bandwidth)

                if self.negative_correlation_enhancement:
                    raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

                events, event_dict = mne.events_from_annotations(raw_haemo)

                # Set baseline parameter based on correction method
                baseline = self.baseline if self.baseline_correction == "xSecondsBefore" else None

                epochs = mne.Epochs(
                    raw_haemo,
                    events,
                    event_id=event_dict,
                    tmin=self.tmin,
                    tmax=self.tmax,
                    reject=self.reject_criteria,
                    reject_by_annotation=True,
                    proj=True,
                    baseline=baseline,
                    preload=True,
                    detrend=None,
                    verbose=True,
                )

                epochs = reject_if_single_event_type(epochs) # Reject all epochs if only one event type is present

                if len(epochs) != 0:
                    # Apply custom baseline correction if needed
                    if self.baseline_correction != "xSecondsBefore":
                        corrector = baselineCorrection(self.baseline_correction)
                        epochs = corrector.apply_correction(
                            self.baseline_correction,
                            epochs,
                            data_types=self.data_types,
                        )

                    self.all_epochs.append(epochs)
                    self.all_control.append(epochs["Control"].get_data(copy=True))
                    
                    Participant_i = individual_participant_class(f"Participant_{i}")
                    Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                    Participant_i.raw_intensity = raw_intensity
                    Participant_i.raw_od = raw_od
                    Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                    Participant_i.raw_haemo = raw_haemo
                    Participant_i.epochs = epochs
                    
                    for name in self.data_types:
                        getattr(self, f'all_{name}').append(epochs[name].get_data(copy=True))
                        Participant_i.events.update({name: epochs[name].get_data(copy=True)})
                    
                    getattr(self, 'Individual_participants').append(Participant_i)
            except FileNotFoundError as e:
                print(f"Error loading {folder_name}: {e}")
            except Exception as e:
                print(f"Unexpected error with {folder_name}: {e}")

        # Concatenate the control data
        self.all_control = np.concatenate(self.all_control, axis=0)

        # Concatenate the data for each data type
        for name in self.data_types:
            setattr(self, f'all_{name}', np.concatenate(getattr(self, f'all_{name}'), axis=0))

        # Create the dictionary all_data with Control and data for each data type
        all_data = {"Control": self.all_control}
        for name in self.data_types:
            all_data.update({name: getattr(self, f'all_{name}')})

        # Update all_data with control_dict
        all_freq = self.all_epochs[0].info['sfreq']
        self.data_types.append("Control")
        return self.all_epochs, self.data_name, all_data, all_freq, self.data_types, self.Individual_participants
