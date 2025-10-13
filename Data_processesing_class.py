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
from preprocessesing_toolbox.SNR_rejection import snr_rejection, get_bad_channels_by_pairs
from preprocessesing_toolbox.differential_pathlength import compute_differential_pathlength
from preprocessesing_toolbox.p2p import compute_p2p
import pandas as pd

load_dotenv()

class fNIRS_data_load:
    def __init__(self, file_path, annotation_names=None, stimulus_duration=5,
                 short_channel_correction=True, negative_correlation_enhancement=True, scalp_coupling_threshold=0.8,
                 reject_criteria: dict = dict(hbo=80e-6), tmin=0, tmax=15, baseline=(None, 0), data_types=[],
                 data_name="None", interpolate_bad_channels=False, unwanted = ["15.0"], baseline_correction: str = "Previous rest period",
                 filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2, l_trans_bandwidth: float = 0.02,
                 snr_rejection: str = "None", snr_threshold : int = 8, apply_tddr: bool = False):    
        self.number_of_participants = 0    
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
        self.all_raw_epochs = []
        self.all_epochs = []
        self.drop_log = []
        self.all_control = []
        self.data_types = data_types
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
        self.apply_tddr = apply_tddr
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
        for i, filename in enumerate(sorted(os.listdir(self.file_path)), start=1):
            self.number_of_participants += 1
            sub_id = str(i).zfill(2)  # Pad with zeros to get "01", "02", etc.
            raw_intensity = self.define_raw_intensity(sub_id)

            raw_intensity.annotations.set_durations(self.stimulus_duration)
            raw_intensity.annotations.rename(self.annotation_names)
            for _unwanted in self.unwanted:
                    unwanted = np.nonzero(raw_intensity.annotations.description == _unwanted)
                    raw_intensity.annotations.delete(unwanted)

            if self.snr_rejection != "None":
                snr = snr_rejection(raw_intensity, self.snr_rejection)
                
                # Validation
                if self.snr_rejection == "SNR" and self.snr_threshold < 1:
                    raise ValueError("Currently the classic signal to noise ratio is used but the threshold for SNR is below 1 resulting in all channels being marked as bad. Please set the threshold to a value above 1.")
                if self.snr_rejection == "CV" and self.snr_threshold > 1:
                    raise ValueError("Currently the coefficient of variation is used but the threshold for CV is above 1 resulting in all channels being marked as bad. Please set the threshold to a value below 1.")
                
                # Get bad channels based on pair logic
                snr_bad_channels = get_bad_channels_by_pairs(raw_intensity.ch_names, snr, self.snr_threshold, self.snr_rejection)
                raw_intensity.info["bads"] = snr_bad_channels
            else:
                snr_bad_channels = []

            raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
            raw_od_original = raw_od.copy()
            
            # Check channel name consistency
            assert raw_intensity.ch_names == raw_od.ch_names, \
                f"Channel names mismatch!\nraw_intensity: {len(raw_intensity.ch_names)} channels\nraw_od: {len(raw_od.ch_names)} channels"
            
            if self.short_channel_correction:
                raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)
            raw_od = mne_nirs.channels.get_long_channels(raw_od)
            
            if self.apply_tddr:
                raw_od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)

            sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)

            sci_bad_channels = list(compress(raw_od.ch_names, sci < self.scalp_coupling_threshold))
            
            # Filter SNR bad channels to only include those that still exist in the long channels dataset
            snr_bad_channels_long_only = [ch for ch in snr_bad_channels if ch in raw_od.ch_names]
            
            # Combine bad channels from all preprocessing
            all_bad_channels = list(set(snr_bad_channels_long_only + sci_bad_channels))            
            raw_od.info["bads"] = all_bad_channels
            
            if self.interpolate_bad_channels:
                raw_od.interpolate_bads()
            
            dpf = compute_differential_pathlength(raw_od)
            raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=dpf)

            raw_haemo_unfiltered = mne.preprocessing.nirs.beer_lambert_law(raw_od_original, ppf=dpf).copy()
            raw_haemo.filter(self.filter_lower_value, self.filter_upper_value, h_trans_bandwidth=self.h_trans_bandwidth, l_trans_bandwidth=self.l_trans_bandwidth)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            # Set baseline parameter based on correction method
            baseline = self.baseline if self.baseline_correction == "xSecondsBefore" else None
            
            raw_epochs = epochs = mne.Epochs(raw_haemo_unfiltered, events, event_id=event_dict, tmin=self.tmin, tmax=self.tmax, reject=None, reject_by_annotation=None, proj=False, baseline=None, preload=True, detrend=None, verbose=True)

            self.reject_criteria = compute_p2p(raw_epochs, self.data_types+["Control"], 99)
            
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

            epochs = reject_if_single_event_type(epochs, self.data_types + ["Control"])

            self.drop_log.append(epochs.drop_log)
            if len(epochs) != 0:
                # Apply custom baseline correction if needed
                if self.baseline_correction != "xSecondsBefore":
                    corrector = baselineCorrection(self.baseline_correction)
                    epochs = corrector.apply_correction(
                        self.baseline_correction,
                        epochs,
                        data_types=self.data_types,
                    )

                self.all_raw_epochs.append(raw_epochs)
                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
                
                Participant_i = individual_participant_class(f"Participant_{i}")
                Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                Participant_i.raw_intensity = raw_intensity
                Participant_i.raw_od = raw_od_original
                Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                Participant_i.raw_haemo = raw_haemo
                Participant_i.raw_epochs = raw_epochs
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
    def __init__(self, data_name, file_path, short_channel_correction : bool, negative_correlation_enhancement : bool, interpolate_bad_channels:bool=False, tmin:int = -5,baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2, l_trans_bandwidth: float = 0.02,
                 reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8, snr_rejection: str = "None", snr_threshold : int = 8, apply_tddr: bool = False):
        self.number_of_participants = 0
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
        self.apply_tddr = apply_tddr
        super().__init__(
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
                        data_name = self.data_name,
                        interpolate_bad_channels = self.interpolate_bad_channels,
                        unwanted = self.unwanted,
                        baseline_correction = self.baseline_correction,
                        filter_lower_value = self.filter_lower_value,
                        filter_upper_value = self.filter_upper_value,
                        h_trans_bandwidth = self.h_trans_bandwidth,
                        l_trans_bandwidth = self.l_trans_bandwidth,
                        snr_rejection = self.snr_rejection,
                        snr_threshold = self.snr_threshold,
                        apply_tddr = self.apply_tddr
                    )

    def define_raw_intensity(self, sub_id):
        fnirs_snirf_file_path = os.path.join(self.file_path, f"sub-{sub_id}", "ses-01", "nirs", f"sub-{sub_id}_ses-01_task-AudioSpeechNoise_nirs.snirf")
        raw_intensity = mne.io.read_raw_snirf(fnirs_snirf_file_path, verbose=True)
        raw_intensity.load_data()
        return raw_intensity

###############################################################################################################################################################################################

class fNIRS_motor_data_load(fNIRS_data_load):
    def __init__(self, data_name, file_path, short_channel_correction: bool, negative_correlation_enhancement: bool, interpolate_bad_channels:bool=False, tmin:int = 0, baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2, l_trans_bandwidth: float = 0.02,
                 reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8, snr_rejection: str = "None", snr_threshold: int = 8, apply_tddr: bool = False):
        self.number_of_participants = 0
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
        self.data_name = "fnirs motor"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = ["15.0"]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        self.snr_rejection = snr_rejection
        self.snr_threshold = snr_threshold
        self.apply_tddr = apply_tddr
        super().__init__(
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
            data_name=self.data_name,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted,
            filter_lower_value = self.filter_lower_value,
            filter_upper_value = self.filter_upper_value,
            h_trans_bandwidth = self.h_trans_bandwidth,
            l_trans_bandwidth = self.l_trans_bandwidth,
            snr_rejection = self.snr_rejection,
            snr_threshold = self.snr_threshold,
            apply_tddr = self.apply_tddr
        )

    def define_raw_intensity(self, sub_id):
        fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
        fnirs_cw_amplitude_dir = fnirs_data_folder / "Participant-1"
        raw_intensity = mne.io.read_raw_nirx(fnirs_cw_amplitude_dir, verbose=True)
        raw_intensity.load_data()
        return raw_intensity

###############################################################################################################################################################################################

class fNIRS_full_motor_data_load(fNIRS_data_load):
    def __init__(self, data_name, file_path, short_channel_correction: bool, negative_correlation_enhancement: bool, interpolate_bad_channels:bool=False, tmin:int = 0, baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2, l_trans_bandwidth: float = 0.02,
                 reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8, snr_rejection: str = "None", snr_threshold: int = 8, apply_tddr: bool = False):
        self.number_of_participants = 0
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1.0": "Control",
                                 "2.0": "Tapping/Left",
                                 "3.0": "Tapping/Right"}
        self.file_path = Path(os.getenv('Luke_full_motor'))
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 5
        self.scalp_coupling_threshold = scalp_coupling_threshold
        self.reject_criteria = reject_criteria
        self.tmin = tmin
        self.tmax = 15
        self.baseline = (None, 0)
        self.data_types = ["Tapping/Left", "Tapping/Right"]
        self.data_name = "Dr. Luke: full motor data"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = ["15.0"]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        self.snr_rejection = snr_rejection
        self.snr_threshold = snr_threshold
        self.apply_tddr = apply_tddr
        super().__init__(
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
            data_name=self.data_name,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted,
            filter_lower_value = self.filter_lower_value,
            filter_upper_value = self.filter_upper_value,
            l_trans_bandwidth=self.l_trans_bandwidth,
            h_trans_bandwidth=self.h_trans_bandwidth,
            snr_rejection=self.snr_rejection,
            snr_threshold=self.snr_threshold,
            apply_tddr=self.apply_tddr
        )
    
    def define_raw_intensity(self, sub_id):
        raw_intensity = mne.io.read_raw_snirf(rf"{self.file_path / rf'sub-{sub_id}/nirs/sub-{sub_id}_task-tapping_nirs.snirf'}", preload=True, verbose=True)
            
        raw_intensity.load_data()
        return raw_intensity


###############################################################################################################################################################################################

class fNIRS_Alexandros_DoC_data_load(fNIRS_data_load):
    def __init__(self, data_name, file_path, short_channel_correction: bool, negative_correlation_enhancement: bool, interpolate_bad_channels:bool=False, tmin:int = 0, baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2, l_trans_bandwidth: float = 0.02,
                 reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8, snr_rejection: str = "None", snr_threshold: int = 8, apply_tddr: bool = False):
        self.number_of_participants = 0
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
        self.data_name = "Alexandros DoC data"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = ["15.0"]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        self.snr_rejection = snr_rejection
        self.snr_threshold = snr_threshold
        self.apply_tddr = apply_tddr
        super().__init__(
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
            data_name=self.data_name,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted,
            baseline_correction = self.baseline_correction,
            filter_lower_value = self.filter_lower_value,
            filter_upper_value = self.filter_upper_value,
            h_trans_bandwidth = self.h_trans_bandwidth,
            l_trans_bandwidth = self.l_trans_bandwidth,
            snr_rejection = self.snr_rejection,
            snr_threshold = self.snr_threshold,
            apply_tddr = self.apply_tddr
        )

    def define_raw_intensity(self, sub_id):
        raw_intensity = mne.io.read_raw_snirf(f"Dataset/Alexandros/DoC/_2024-04-29_{sub_id}.snirf", verbose=True)
        raw_intensity.load_data()
        return raw_intensity
    
    def load_data(self):
        for i, filename in enumerate(sorted(os.listdir(self.file_path)), start=1):
            self.number_of_participants += 1
            sub_id = str(i).zfill(2)  # Pad with zeros to get "01", "02", etc.
            raw_intensity = self.define_raw_intensity(sub_id)
            raw_intensity = self.make_annotations(raw_intensity)

            raw_intensity.annotations.rename(self.annotation_names)
            for _unwanted in self.unwanted:
                    unwanted = np.nonzero(raw_intensity.annotations.description == _unwanted)
                    raw_intensity.annotations.delete(unwanted)

            if self.snr_rejection != "None":
                snr = snr_rejection(raw_intensity, self.snr_rejection)
                
                # Validation
                if self.snr_rejection == "SNR" and self.snr_threshold < 1:
                    raise ValueError("Currently the classic signal to noise ratio is used but the threshold for SNR is below 1 resulting in all channels being marked as bad. Please set the threshold to a value above 1.")
                if self.snr_rejection == "CV" and self.snr_threshold > 1:
                    raise ValueError("Currently the coefficient of variation is used but the threshold for CV is above 1 resulting in all channels being marked as bad. Please set the threshold to a value below 1.")
                
                # Get bad channels based on pair logic
                snr_bad_channels = get_bad_channels_by_pairs(raw_intensity.ch_names, snr, self.snr_threshold, self.snr_rejection)
                raw_intensity.info["bads"] = snr_bad_channels
            else:
                snr_bad_channels = []

            raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
            raw_od_original = raw_od.copy()

            # Check channel name consistency
            assert raw_intensity.ch_names == raw_od.ch_names, \
                f"Channel names mismatch!\nraw_intensity: {len(raw_intensity.ch_names)} channels\nraw_od: {len(raw_od.ch_names)} channels"
            
            if self.short_channel_correction:
                raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)
            raw_od = mne_nirs.channels.get_long_channels(raw_od)

            if self.apply_tddr:
                raw_od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)
                
            sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)

            sci_bad_channels = list(compress(raw_od.ch_names, sci < self.scalp_coupling_threshold))
            
            # Filter SNR bad channels to only include those that still exist in the long channels dataset
            snr_bad_channels_long_only = [ch for ch in snr_bad_channels if ch in raw_od.ch_names]
            
            # Combine bad channels from all preprocessing
            all_bad_channels = list(set(snr_bad_channels_long_only + sci_bad_channels))         
            raw_od.info["bads"] = all_bad_channels
            
            if self.interpolate_bad_channels:
                raw_od.interpolate_bads()

            dpf = compute_differential_pathlength(raw_od)
            raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=dpf)

            raw_haemo_unfiltered = mne.preprocessing.nirs.beer_lambert_law(raw_od_original, ppf=0.1).copy()
            raw_haemo.filter(self.filter_lower_value, self.filter_upper_value, h_trans_bandwidth=self.h_trans_bandwidth, l_trans_bandwidth=self.l_trans_bandwidth)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            # Set baseline parameter based on correction method
            baseline = self.baseline if self.baseline_correction == "xSecondsBefore" else None
            
            raw_epochs = epochs = mne.Epochs(raw_haemo_unfiltered, events, event_id=event_dict, tmin=self.tmin, tmax=self.tmax, reject=None, reject_by_annotation=None, proj=False, baseline=None, preload=True, detrend=None, verbose=True)

            self.reject_criteria = compute_p2p(raw_epochs, self.data_types+["Control"], 90)

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

            epochs = reject_if_single_event_type(epochs, self.data_types + ["Control"])

            self.drop_log.append(epochs.drop_log)
            if len(epochs) != 0:
                # Apply custom baseline correction if needed
                if self.baseline_correction != "xSecondsBefore":
                    corrector = baselineCorrection(self.baseline_correction)
                    epochs = corrector.apply_correction(
                        self.baseline_correction,
                        epochs,
                        data_types=self.data_types,
                    )

                self.all_raw_epochs.append(raw_epochs)
                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
                
                Participant_i = individual_participant_class(f"Participant_{i}")
                Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                Participant_i.raw_intensity = raw_intensity
                Participant_i.raw_od = raw_od
                Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                Participant_i.raw_haemo = raw_haemo
                Participant_i.raw_epochs = raw_epochs
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
    def __init__(self, data_name, file_path, short_channel_correction: bool, negative_correlation_enhancement: bool, interpolate_bad_channels:bool=False, tmin:int = 0, baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2, l_trans_bandwidth: float = 0.02,
                 reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8, snr_rejection: str = "None", snr_threshold: int = 8, apply_tddr: bool = False):
        self.number_of_participants = 0
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
        self.data_name = "Alexandros Healthy data"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = ["1"]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        self.snr_rejection = snr_rejection
        self.snr_threshold = snr_threshold
        self.apply_tddr = apply_tddr
        super().__init__(
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
            data_name=self.data_name,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted,
            filter_lower_value=self.filter_lower_value,
            filter_upper_value=self.filter_upper_value,
            h_trans_bandwidth=self.h_trans_bandwidth,
            l_trans_bandwidth=self.l_trans_bandwidth,
            apply_tddr=self.apply_tddr
        )

    def define_raw_intensity(self, sub_id):
        raw_intensity = mne.io.read_raw_snirf(f"Dataset/Alexandros/Healthy/_2024-04-29_{sub_id}.snirf", verbose=True)
        raw_intensity.load_data()
        return raw_intensity

    

###############################################################################################################################################################################################

class fNIRS_CUH_patient_data_load(fNIRS_data_load):
    def __init__(self, data_name, file_path, short_channel_correction: bool, negative_correlation_enhancement: bool, interpolate_bad_channels:bool=False, tmin:int = 0, baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2, l_trans_bandwidth: float = 0.02,
                reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8, snr_rejection: str = "None", snr_threshold: int = 8, apply_tddr: bool = False):
        self.number_of_participants = 0
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
        self.data_name = "Alexandros CUH patient data"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = "Pause"
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        self.snr_rejection = snr_rejection
        self.snr_threshold = snr_threshold
        self.apply_tddr = apply_tddr

        super().__init__(
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
            data_name=self.data_name,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted,
            baseline_correction = self.baseline_correction,
            filter_lower_value = self.filter_lower_value,
            filter_upper_value = self.filter_upper_value,
            h_trans_bandwidth = self.h_trans_bandwidth,
            l_trans_bandwidth = self.l_trans_bandwidth,
            snr_rejection = self.snr_rejection,
            snr_threshold = self.snr_threshold,
            apply_tddr = self.apply_tddr
        )

    def define_raw_intensity(self, sub_id):
            if sub_id == 9:
                raw_intensity = mne.io.read_raw_snirf(f"{self.file_path / f'P{sub_id}_2_2.snirf'}", verbose=True)
            else:            
                raw_intensity = mne.io.read_raw_snirf(f"{self.file_path / f'P{sub_id}_1.snirf'}", verbose=True)
            raw_intensity.load_data()
            return raw_intensity
        
    def load_data(self):
        for i, filename in enumerate(sorted(os.listdir(self.file_path)), start=1):
            sub_id = str(i).zfill(2)  # Pad with zeros to get "01", "02", etc.
            if self.data_name == 'fNIRS_CUH_patient_data':
                sub_id = i
            if sub_id in [3, 14, 15, 17, 18, 30, 31, 41, 46]:
                continue
            self.number_of_participants += 1
            raw_intensity = self.define_raw_intensity(sub_id)
            raw_intensity = self.make_annotations(raw_intensity)

            raw_intensity.annotations.rename(self.annotation_names)
            for _unwanted in self.unwanted:
                    unwanted = np.nonzero(raw_intensity.annotations.description == _unwanted)
                    raw_intensity.annotations.delete(unwanted)

            if self.snr_rejection != "None":
                snr = snr_rejection(raw_intensity, self.snr_rejection)
                
                # Validation
                if self.snr_rejection == "SNR" and self.snr_threshold < 1:
                    raise ValueError("Currently the classic signal to noise ratio is used but the threshold for SNR is below 1 resulting in all channels being marked as bad. Please set the threshold to a value above 1.")
                if self.snr_rejection == "CV" and self.snr_threshold > 1:
                    raise ValueError("Currently the coefficient of variation is used but the threshold for CV is above 1 resulting in all channels being marked as bad. Please set the threshold to a value below 1.")
                
                # Get bad channels based on pair logic
                snr_bad_channels = get_bad_channels_by_pairs(raw_intensity.ch_names, snr, self.snr_threshold, self.snr_rejection)
                raw_intensity.info["bads"] = snr_bad_channels
            else:
                snr_bad_channels = []

            raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
            raw_od_original = raw_od.copy()

            # Check channel name consistency
            assert raw_intensity.ch_names == raw_od.ch_names, \
                f"Channel names mismatch!\nraw_intensity: {len(raw_intensity.ch_names)} channels\nraw_od: {len(raw_od.ch_names)} channels"
            
            if self.short_channel_correction:
                raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)
            raw_od = mne_nirs.channels.get_long_channels(raw_od)

            if self.apply_tddr:
                raw_od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)
                
            sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)

            sci_bad_channels = list(compress(raw_od.ch_names, sci < self.scalp_coupling_threshold))
            
            # Filter SNR bad channels to only include those that still exist in the long channels dataset
            snr_bad_channels_long_only = [ch for ch in snr_bad_channels if ch in raw_od.ch_names]
            
            # Combine bad channels from all preprocessing
            all_bad_channels = list(set(snr_bad_channels_long_only + sci_bad_channels))         
            raw_od.info["bads"] = all_bad_channels

            if self.interpolate_bad_channels:
                raw_od.interpolate_bads()
            
            dpf = compute_differential_pathlength(raw_od)
            raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=dpf)

            raw_haemo_unfiltered = mne.preprocessing.nirs.beer_lambert_law(raw_od_original, ppf=0.1).copy()
            raw_haemo.filter(self.filter_lower_value, self.filter_upper_value, h_trans_bandwidth=self.h_trans_bandwidth, l_trans_bandwidth=self.l_trans_bandwidth)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            # Set baseline parameter based on correction method
            baseline = self.baseline if self.baseline_correction == "xSecondsBefore" else None
            
            raw_epochs = epochs = mne.Epochs(raw_haemo_unfiltered, events, event_id=event_dict, tmin=self.tmin, tmax=self.tmax, reject=None, reject_by_annotation=None, proj=False, baseline=None, preload=True, detrend=None, verbose=True)

            self.reject_criteria = compute_p2p(raw_epochs, self.data_types+["Control"], 90)

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

            epochs = reject_if_single_event_type(epochs, self.data_types + ["Control"])

            self.drop_log.append(epochs.drop_log)
            if len(epochs) != 0:
                # Apply custom baseline correction if needed
                if self.baseline_correction != "xSecondsBefore":
                    corrector = baselineCorrection(self.baseline_correction)
                    epochs = corrector.apply_correction(
                        self.baseline_correction,
                        epochs,
                        data_types=self.data_types,
                    )

                self.all_raw_epochs.append(raw_epochs)
                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
                
                Participant_i = individual_participant_class(f"Participant_{i}")
                Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                Participant_i.raw_intensity = raw_intensity
                Participant_i.raw_od = raw_od
                Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                Participant_i.raw_haemo = raw_haemo
                Participant_i.raw_epochs = raw_epochs
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
    def __init__(self, data_name, file_path, short_channel_correction: bool, negative_correlation_enhancement: bool, interpolate_bad_channels:bool=False, tmin:int = 0, baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2, l_trans_bandwidth: float = 0.02,
                 reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8, snr_rejection: str = "None", snr_threshold: int = 8, apply_tddr: bool = False):
        self.number_of_participants = 0
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1.0": "HandMI",
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
        self.data_name = "Melika hand 5 Hz"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = [""]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        self.snr_rejection = snr_rejection
        self.snr_threshold = snr_threshold
        self.apply_tddr = apply_tddr
        super().__init__(
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
            data_name=self.data_name,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted,
            baseline_correction = self.baseline_correction,
            filter_lower_value = self.filter_lower_value,
            filter_upper_value = self.filter_upper_value,
            h_trans_bandwidth = self.h_trans_bandwidth,
            l_trans_bandwidth = self.l_trans_bandwidth,
            snr_rejection = self.snr_rejection,
            snr_threshold = self.snr_threshold,
            apply_tddr = self.apply_tddr
        )

    def define_raw_intensity(self, filename):
        raw_intensity = mne.io.read_raw_nirx(rf"{self.file_path / filename}", preload=True, verbose=True)
        return raw_intensity
        
    def load_data(self):
        for i, filename in enumerate(sorted(os.listdir(self.file_path)), start=1):
            self.number_of_participants += 1
            raw_intensity = self.define_raw_intensity(filename)
            raw_intensity = self.make_annotations(raw_intensity)
            if i == 1 or i == 2 or i == 3 or i == 4:# When data for the first patient was recorded, the introduction was not added in Satori, so we add it manually
                raw_intensity = self.make_without_intro_annotations(raw_intensity)
            else: # For all other patients we just add the resting phases
                raw_intensity = self.make_annotations(raw_intensity)


            raw_intensity.annotations.rename(self.annotation_names)
            for _unwanted in self.unwanted:
                    unwanted = np.nonzero(raw_intensity.annotations.description == _unwanted)
                    raw_intensity.annotations.delete(unwanted)

            if self.snr_rejection != "None":
                snr = snr_rejection(raw_intensity, self.snr_rejection)
                
                # Validation
                if self.snr_rejection == "SNR" and self.snr_threshold < 1:
                    raise ValueError("Currently the classic signal to noise ratio is used but the threshold for SNR is below 1 resulting in all channels being marked as bad. Please set the threshold to a value above 1.")
                if self.snr_rejection == "CV" and self.snr_threshold > 1:
                    raise ValueError("Currently the coefficient of variation is used but the threshold for CV is above 1 resulting in all channels being marked as bad. Please set the threshold to a value below 1.")
                
                # Get bad channels based on pair logic
                snr_bad_channels = get_bad_channels_by_pairs(raw_intensity.ch_names, snr, self.snr_threshold, self.snr_rejection)
                raw_intensity.info["bads"] = snr_bad_channels
            else:
                snr_bad_channels = []

            raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
            raw_od_original = raw_od.copy()

            # Check channel name consistency
            assert raw_intensity.ch_names == raw_od.ch_names, \
                f"Channel names mismatch!\nraw_intensity: {len(raw_intensity.ch_names)} channels\nraw_od: {len(raw_od.ch_names)} channels"
            
            if self.short_channel_correction:
                raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)
            raw_od = mne_nirs.channels.get_long_channels(raw_od)
            
            if self.apply_tddr:
                raw_od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)
                
            sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)

            sci_bad_channels = list(compress(raw_od.ch_names, sci < self.scalp_coupling_threshold))
            
            # Filter SNR bad channels to only include those that still exist in the long channels dataset
            snr_bad_channels_long_only = [ch for ch in snr_bad_channels if ch in raw_od.ch_names]
            
            # Combine bad channels from all preprocessing
            all_bad_channels = list(set(snr_bad_channels_long_only + sci_bad_channels))         
            raw_od.info["bads"] = all_bad_channels
            
            if self.interpolate_bad_channels:
                raw_od.interpolate_bads()
                
            dpf = compute_differential_pathlength(raw_od)
            raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=dpf)

            raw_haemo_unfiltered = mne.preprocessing.nirs.beer_lambert_law(raw_od_original, ppf=0.1).copy()
            raw_haemo.filter(self.filter_lower_value, self.filter_upper_value, h_trans_bandwidth=self.h_trans_bandwidth, l_trans_bandwidth=self.l_trans_bandwidth)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            # Set baseline parameter based on correction method
            baseline = self.baseline if self.baseline_correction == "xSecondsBefore" else None
            
            raw_epochs = epochs = mne.Epochs(raw_haemo_unfiltered, events, event_id=event_dict, tmin=self.tmin, tmax=self.tmax, reject=None, reject_by_annotation=None, proj=False, baseline=None, preload=True, detrend=None, verbose=True)

            self.reject_criteria = compute_p2p(raw_epochs, self.data_types+["Control"], 90)
            
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

            epochs = reject_if_single_event_type(epochs, self.data_types + ["Control"])

            self.drop_log.append(epochs.drop_log)
            if len(epochs) != 0:
                # Apply custom baseline correction if needed
                if self.baseline_correction != "xSecondsBefore":
                    corrector = baselineCorrection(self.baseline_correction)
                    epochs = corrector.apply_correction(
                        self.baseline_correction,
                        epochs,
                        data_types=self.data_types,
                    )

                self.all_raw_epochs.append(raw_epochs)
                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
                
                Participant_i = individual_participant_class(f"Participant_{i}")
                Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                Participant_i.raw_intensity = raw_intensity
                Participant_i.raw_od = raw_od
                Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                Participant_i.raw_haemo = raw_haemo
                Participant_i.raw_epochs = raw_epochs
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
        cropped_raw_data.annotations.rename({"0.0": "End"})

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
        cropped_raw_data.annotations.rename({"0.0": "End"})

        
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
    def __init__(self, data_name, file_path, short_channel_correction: bool, negative_correlation_enhancement: bool, interpolate_bad_channels:bool=False, tmin:int = 0,
                 baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2,
                 l_trans_bandwidth: float = 0.02, reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8, snr_rejection: str = "None",
                 snr_threshold: int = 8, apply_tddr: bool = False):
        self.number_of_participants = 0
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1.0": "TongueMI",
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
        self.data_name = "Melika tongue 5Hz"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = ["2.0"]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        self.snr_rejection = snr_rejection
        self.snr_threshold = snr_threshold
        self.apply_tddr = apply_tddr
        super().__init__(
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
            data_name=self.data_name,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted,
            baseline_correction = self.baseline_correction,
            filter_lower_value = self.filter_lower_value,
            filter_upper_value = self.filter_upper_value,
            h_trans_bandwidth = self.h_trans_bandwidth,
            l_trans_bandwidth = self.l_trans_bandwidth,
            snr_rejection = self.snr_rejection,
            snr_threshold = self.snr_threshold,
            apply_tddr = self.apply_tddr
        )

    def define_raw_intensity(self, filename):
        raw_intensity = mne.io.read_raw_nirx(rf"{self.file_path / filename}", preload=True, verbose=True)
        return raw_intensity
        
    def load_data(self):
        for i, filename in enumerate(sorted(os.listdir(self.file_path)), start=1):
            self.number_of_participants += 1
            raw_intensity = self.define_raw_intensity(filename)
            raw_intensity = self.make_annotations(raw_intensity)
            if i == 1 : # When data for the first patient was recorded, the introduction was not added in Satori, so we add it manually
                raw_intensity = self.make_without_intro_annotations(raw_intensity)
            else: # For all other patients we just add the resting phases
                raw_intensity = self.make_annotations(raw_intensity)

            raw_intensity.annotations.rename(self.annotation_names)
            for _unwanted in self.unwanted:
                    unwanted = np.nonzero(raw_intensity.annotations.description == _unwanted)
                    raw_intensity.annotations.delete(unwanted)

            if self.snr_rejection != "None":
                snr = snr_rejection(raw_intensity, self.snr_rejection)
                
                # Validation
                if self.snr_rejection == "SNR" and self.snr_threshold < 1:
                    raise ValueError("Currently the classic signal to noise ratio is used but the threshold for SNR is below 1 resulting in all channels being marked as bad. Please set the threshold to a value above 1.")
                if self.snr_rejection == "CV" and self.snr_threshold > 1:
                    raise ValueError("Currently the coefficient of variation is used but the threshold for CV is above 1 resulting in all channels being marked as bad. Please set the threshold to a value below 1.")
                
                # Get bad channels based on pair logic
                snr_bad_channels = get_bad_channels_by_pairs(raw_intensity.ch_names, snr, self.snr_threshold, self.snr_rejection)
                raw_intensity.info["bads"] = snr_bad_channels
            else:
                snr_bad_channels = []

            raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
            raw_od_original = raw_od.copy()

            # Check channel name consistency
            assert raw_intensity.ch_names == raw_od.ch_names, \
                f"Channel names mismatch!\nraw_intensity: {len(raw_intensity.ch_names)} channels\nraw_od: {len(raw_od.ch_names)} channels"
            
            if self.short_channel_correction:
                raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)
            raw_od = mne_nirs.channels.get_long_channels(raw_od)
            
            if self.apply_tddr:
                raw_od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)

            sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)

            sci_bad_channels = list(compress(raw_od.ch_names, sci < self.scalp_coupling_threshold))
            
            # Filter SNR bad channels to only include those that still exist in the long channels dataset
            snr_bad_channels_long_only = [ch for ch in snr_bad_channels if ch in raw_od.ch_names]
            
            # Combine bad channels from all preprocessing
            all_bad_channels = list(set(snr_bad_channels_long_only + sci_bad_channels))         
            raw_od.info["bads"] = all_bad_channels
            
            if self.interpolate_bad_channels:
                raw_od.interpolate_bads()
                
            dpf = compute_differential_pathlength(raw_od)
            raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=dpf)

            raw_haemo_unfiltered = mne.preprocessing.nirs.beer_lambert_law(raw_od_original, ppf=0.1).copy()
            raw_haemo.filter(self.filter_lower_value, self.filter_upper_value, h_trans_bandwidth=self.h_trans_bandwidth, l_trans_bandwidth=self.l_trans_bandwidth)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            # Set baseline parameter based on correction method
            baseline = self.baseline if self.baseline_correction == "xSecondsBefore" else None
            
            raw_epochs = epochs = mne.Epochs(raw_haemo_unfiltered, events, event_id=event_dict, tmin=self.tmin, tmax=self.tmax, reject=None, reject_by_annotation=None, proj=False, baseline=None, preload=True, detrend=None, verbose=True)

            self.reject_criteria = compute_p2p(raw_epochs, self.data_types+["Control"], 90)
            
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

            epochs = reject_if_single_event_type(epochs, self.data_types + ["Control"])

            self.drop_log.append(epochs.drop_log)
            if len(epochs) != 0:
                # Apply custom baseline correction if needed
                if self.baseline_correction != "xSecondsBefore":
                    corrector = baselineCorrection(self.baseline_correction)
                    epochs = corrector.apply_correction(
                        self.baseline_correction,
                        epochs,
                        data_types=self.data_types,
                    )

                self.all_raw_epochs.append(raw_epochs)
                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
                
                Participant_i = individual_participant_class(f"Participant_{i}")
                Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                Participant_i.raw_intensity = raw_intensity
                Participant_i.raw_od = raw_od
                Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                Participant_i.raw_haemo = raw_haemo
                Participant_i.raw_epochs = raw_epochs
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
        # cropped_raw_data.annotations.rename({"0.0": "End"})

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
        # cropped_raw_data.annotations.rename({"0.0": "End"})

        
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
    def __init__(self, data_name, file_path, short_channel_correction: bool, negative_correlation_enhancement: bool, interpolate_bad_channels:bool=False, tmin:int = 0,
                 baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2,
                 l_trans_bandwidth: float = 0.02, reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8, snr_rejection: str = "None",
                 snr_threshold: int = 8, apply_tddr: bool = False):
        self.number_of_participants = 0
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1.0": "HandMI",
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
        self.data_name = "Melika hand 10Hz"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = [""]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        self.snr_rejection = snr_rejection
        self.snr_threshold = snr_threshold
        self.apply_tddr = apply_tddr
        super().__init__(
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
            data_name=self.data_name,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted,
            baseline_correction = self.baseline_correction,
            filter_lower_value = self.filter_lower_value,
            filter_upper_value = self.filter_upper_value,
            h_trans_bandwidth = self.h_trans_bandwidth,
            l_trans_bandwidth = self.l_trans_bandwidth,
            snr_rejection = self.snr_rejection,
            snr_threshold = self.snr_threshold,
            apply_tddr = self.apply_tddr
        )

    def define_raw_intensity(self, filename):
        raw_intensity = mne.io.read_raw_nirx(rf"{self.file_path / filename}", preload=True, verbose=True)
        return raw_intensity
        
    def load_data(self):
        for i, filename in enumerate(sorted(os.listdir(self.file_path)), start=1):
            self.number_of_participants += 1
            raw_intensity = self.define_raw_intensity(filename)
            raw_intensity = self.make_annotations(raw_intensity)


            raw_intensity.annotations.rename(self.annotation_names)
            for _unwanted in self.unwanted:
                    unwanted = np.nonzero(raw_intensity.annotations.description == _unwanted)
                    raw_intensity.annotations.delete(unwanted)

            if self.snr_rejection != "None":
                snr = snr_rejection(raw_intensity, self.snr_rejection)
                
                # Validation
                if self.snr_rejection == "SNR" and self.snr_threshold < 1:
                    raise ValueError("Currently the classic signal to noise ratio is used but the threshold for SNR is below 1 resulting in all channels being marked as bad. Please set the threshold to a value above 1.")
                if self.snr_rejection == "CV" and self.snr_threshold > 1:
                    raise ValueError("Currently the coefficient of variation is used but the threshold for CV is above 1 resulting in all channels being marked as bad. Please set the threshold to a value below 1.")
                
                # Get bad channels based on pair logic
                snr_bad_channels = get_bad_channels_by_pairs(raw_intensity.ch_names, snr, self.snr_threshold, self.snr_rejection)
                raw_intensity.info["bads"] = snr_bad_channels
            else:
                snr_bad_channels = []

            raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
            raw_od_original = raw_od.copy()

            # Check channel name consistency
            assert raw_intensity.ch_names == raw_od.ch_names, \
                f"Channel names mismatch!\nraw_intensity: {len(raw_intensity.ch_names)} channels\nraw_od: {len(raw_od.ch_names)} channels"
            
            if self.short_channel_correction:
                raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)
            raw_od = mne_nirs.channels.get_long_channels(raw_od)
            
            if self.apply_tddr:
                raw_od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)

            sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)

            sci_bad_channels = list(compress(raw_od.ch_names, sci < self.scalp_coupling_threshold))
            
            # Filter SNR bad channels to only include those that still exist in the long channels dataset
            snr_bad_channels_long_only = [ch for ch in snr_bad_channels if ch in raw_od.ch_names]
            
            # Combine bad channels from all preprocessing
            all_bad_channels = list(set(snr_bad_channels_long_only + sci_bad_channels))         
            raw_od.info["bads"] = all_bad_channels
            
            if self.interpolate_bad_channels:
                raw_od.interpolate_bads()
                
            dpf = compute_differential_pathlength(raw_od)
            raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=dpf)

            raw_haemo_unfiltered = mne.preprocessing.nirs.beer_lambert_law(raw_od_original, ppf=0.1).copy()
            raw_haemo.filter(self.filter_lower_value, self.filter_upper_value, h_trans_bandwidth=self.h_trans_bandwidth, l_trans_bandwidth=self.l_trans_bandwidth)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            # Set baseline parameter based on correction method
            baseline = self.baseline if self.baseline_correction == "xSecondsBefore" else None
            
            raw_epochs = epochs = mne.Epochs(raw_haemo_unfiltered, events, event_id=event_dict, tmin=self.tmin, tmax=self.tmax, reject=None, reject_by_annotation=None, proj=False, baseline=None, preload=True, detrend=None, verbose=True)

            self.reject_criteria = compute_p2p(raw_epochs, self.data_types+["Control"], 90)

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

            epochs = reject_if_single_event_type(epochs, self.data_types + ["Control"])

            self.drop_log.append(epochs.drop_log)
            if len(epochs) != 0:
                # Apply custom baseline correction if needed
                if self.baseline_correction != "xSecondsBefore":
                    corrector = baselineCorrection(self.baseline_correction)
                    epochs = corrector.apply_correction(
                        self.baseline_correction,
                        epochs,
                        data_types=self.data_types,
                    )

                self.all_raw_epochs.append(raw_epochs)
                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
                
                Participant_i = individual_participant_class(f"Participant_{i}")
                Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                Participant_i.raw_intensity = raw_intensity
                Participant_i.raw_od = raw_od
                Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                Participant_i.raw_haemo = raw_haemo
                Participant_i.raw_epochs = raw_epochs
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
        cropped_raw_data.annotations.rename({"0.0": "End"})

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
        cropped_raw_data.annotations.rename({"0.0": "End"})

        
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
    def __init__(self, data_name, file_path, short_channel_correction: bool, negative_correlation_enhancement: bool, interpolate_bad_channels:bool=False, tmin:int = 0,
                 baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2,
                 l_trans_bandwidth: float = 0.02, reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8, snr_rejection: str = "None",
                 snr_threshold: int = 8, apply_tddr: bool = False):
        self.number_of_participants = 0
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1.0": "TongueMI",
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
        self.data_name = "Melika tongue 10Hz"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = ["2.0", "0.0"]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        self.snr_rejection = snr_rejection
        self.snr_threshold = snr_threshold
        self.apply_tddr = apply_tddr

        super().__init__(
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
            data_name=self.data_name,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted,
            baseline_correction = self.baseline_correction,
            filter_lower_value = self.filter_lower_value,
            filter_upper_value = self.filter_upper_value,
            h_trans_bandwidth = self.h_trans_bandwidth,
            l_trans_bandwidth = self.l_trans_bandwidth,
            snr_rejection = self.snr_rejection,
            snr_threshold = self.snr_threshold,
            apply_tddr = self.apply_tddr
            )

    def define_raw_intensity(self, filename):
        raw_intensity = mne.io.read_raw_nirx(rf"{self.file_path / filename}", preload=True, verbose=True)
        return raw_intensity
        
    def load_data(self):
        for i, filename in enumerate(sorted(os.listdir(self.file_path)), start=1):
            self.number_of_participants += 1
            raw_intensity = self.define_raw_intensity(filename)
            raw_intensity = self.make_annotations(raw_intensity)

            raw_intensity.annotations.rename(self.annotation_names)
            for _unwanted in self.unwanted:
                    unwanted = np.nonzero(raw_intensity.annotations.description == _unwanted)
                    raw_intensity.annotations.delete(unwanted)

            if self.snr_rejection != "None":
                snr = snr_rejection(raw_intensity, self.snr_rejection)
                
                # Validation
                if self.snr_rejection == "SNR" and self.snr_threshold < 1:
                    raise ValueError("Currently the classic signal to noise ratio is used but the threshold for SNR is below 1 resulting in all channels being marked as bad. Please set the threshold to a value above 1.")
                if self.snr_rejection == "CV" and self.snr_threshold > 1:
                    raise ValueError("Currently the coefficient of variation is used but the threshold for CV is above 1 resulting in all channels being marked as bad. Please set the threshold to a value below 1.")
                
                # Get bad channels based on pair logic
                snr_bad_channels = get_bad_channels_by_pairs(raw_intensity.ch_names, snr, self.snr_threshold, self.snr_rejection)
                raw_intensity.info["bads"] = snr_bad_channels
            else:
                snr_bad_channels = []

            raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
            raw_od_original = raw_od.copy()

            # Check channel name consistency
            assert raw_intensity.ch_names == raw_od.ch_names, \
                f"Channel names mismatch!\nraw_intensity: {len(raw_intensity.ch_names)} channels\nraw_od: {len(raw_od.ch_names)} channels"
            
            if self.short_channel_correction:
                raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)
            raw_od = mne_nirs.channels.get_long_channels(raw_od)
            
            if self.apply_tddr:
                raw_od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)

            sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)

            sci_bad_channels = list(compress(raw_od.ch_names, sci < self.scalp_coupling_threshold))
            
            # Filter SNR bad channels to only include those that still exist in the long channels dataset
            snr_bad_channels_long_only = [ch for ch in snr_bad_channels if ch in raw_od.ch_names]
            
            # Combine bad channels from all preprocessing
            all_bad_channels = list(set(snr_bad_channels_long_only + sci_bad_channels))         
            raw_od.info["bads"] = all_bad_channels
            
            if self.interpolate_bad_channels:
                raw_od.interpolate_bads()
                
            dpf = compute_differential_pathlength(raw_od)
            raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=dpf)

            raw_haemo_unfiltered = mne.preprocessing.nirs.beer_lambert_law(raw_od_original, ppf=0.1).copy()
            raw_haemo.filter(self.filter_lower_value, self.filter_upper_value, h_trans_bandwidth=self.h_trans_bandwidth, l_trans_bandwidth=self.l_trans_bandwidth)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            # Set baseline parameter based on correction method
            baseline = self.baseline if self.baseline_correction == "xSecondsBefore" else None
            
            raw_epochs = epochs = mne.Epochs(raw_haemo_unfiltered, events, event_id=event_dict, tmin=self.tmin, tmax=self.tmax, reject=None, reject_by_annotation=None, proj=False, baseline=None, preload=True, detrend=None, verbose=True)

            self.reject_criteria = compute_p2p(raw_epochs, self.data_types+["Control"], 90)
            
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

            epochs = reject_if_single_event_type(epochs, self.data_types + ["Control"])

            self.drop_log.append(epochs.drop_log)
            if len(epochs) != 0:
                # Apply custom baseline correction if needed
                if self.baseline_correction != "xSecondsBefore":
                    corrector = baselineCorrection(self.baseline_correction)
                    epochs = corrector.apply_correction(
                        self.baseline_correction,
                        epochs,
                        data_types=self.data_types,
                    )

                self.all_raw_epochs.append(raw_epochs)
                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
            
                Participant_i = individual_participant_class(f"Participant_{i}")
                Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                Participant_i.raw_intensity = raw_intensity
                Participant_i.raw_od = raw_od
                Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                Participant_i.raw_haemo = raw_haemo
                Participant_i.raw_epochs = raw_epochs
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
        cropped_raw_data.annotations.rename({"0.0": "End"})

        
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
    def __init__(self, data_name, file_path, short_channel_correction: bool, negative_correlation_enhancement: bool, interpolate_bad_channels:bool=False, tmin:int = 0,
                 baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2,
                 l_trans_bandwidth: float = 0.02, reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8, snr_rejection: str = "None",
                 snr_threshold: int = 8, apply_tddr: bool = False):
        self.number_of_participants = 0
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
        self.data_name = "Melika old data"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = ["0"]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        self.snr_rejection = snr_rejection
        self.snr_threshold = snr_threshold
        self.apply_tddr = apply_tddr
        
        super().__init__(
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
            data_name=self.data_name,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted,
            baseline_correction = self.baseline_correction,
            filter_lower_value = self.filter_lower_value,
            filter_upper_value = self.filter_upper_value,
            h_trans_bandwidth = self.h_trans_bandwidth,
            l_trans_bandwidth = self.l_trans_bandwidth,
            snr_rejection = self.snr_rejection,
            snr_threshold = self.snr_threshold,
            apply_tddr = self.apply_tddr
        )

    def define_raw_intensity(self, sub_id):
        raw_intensity = mne.io.read_raw_snirf(rf"{self.file_path / rf'subj-{sub_id}.snirf'}", preload=True, verbose=True)
        raw_intensity.load_data()
        return raw_intensity
        
    def load_data(self):
        for i, filename in enumerate(sorted(os.listdir(self.file_path)), start=1):
            if i == 3:
                continue
            self.number_of_participants += 1
            sub_id = str(i).zfill(2)  # Pad with zeros to get "01", "02", etc.
            raw_intensity = self.define_raw_intensity(sub_id)
            raw_intensity = self.make_annotations(raw_intensity)

            raw_intensity.annotations.rename(self.annotation_names)
            for _unwanted in self.unwanted:
                    unwanted = np.nonzero(raw_intensity.annotations.description == _unwanted)
                    raw_intensity.annotations.delete(unwanted)

            if self.snr_rejection != "None":
                snr = snr_rejection(raw_intensity, self.snr_rejection)
                
                # Validation
                if self.snr_rejection == "SNR" and self.snr_threshold < 1:
                    raise ValueError("Currently the classic signal to noise ratio is used but the threshold for SNR is below 1 resulting in all channels being marked as bad. Please set the threshold to a value above 1.")
                if self.snr_rejection == "CV" and self.snr_threshold > 1:
                    raise ValueError("Currently the coefficient of variation is used but the threshold for CV is above 1 resulting in all channels being marked as bad. Please set the threshold to a value below 1.")
                
                # Get bad channels based on pair logic
                snr_bad_channels = get_bad_channels_by_pairs(raw_intensity.ch_names, snr, self.snr_threshold, self.snr_rejection)
                raw_intensity.info["bads"] = snr_bad_channels
            else:
                snr_bad_channels = []

            raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
            raw_od_original = raw_od.copy()

            # Check channel name consistency
            assert raw_intensity.ch_names == raw_od.ch_names, \
                f"Channel names mismatch!\nraw_intensity: {len(raw_intensity.ch_names)} channels\nraw_od: {len(raw_od.ch_names)} channels"
            
            if self.short_channel_correction:
                raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)
            raw_od = mne_nirs.channels.get_long_channels(raw_od)
            
            if self.apply_tddr:
                raw_od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)

            sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)

            sci_bad_channels = list(compress(raw_od.ch_names, sci < self.scalp_coupling_threshold))
            
            # Filter SNR bad channels to only include those that still exist in the long channels dataset
            snr_bad_channels_long_only = [ch for ch in snr_bad_channels if ch in raw_od.ch_names]
            
            # Combine bad channels from all preprocessing
            all_bad_channels = list(set(snr_bad_channels_long_only + sci_bad_channels))         
            raw_od.info["bads"] = all_bad_channels
            
            if self.interpolate_bad_channels:
                raw_od.interpolate_bads()
                
            dpf = compute_differential_pathlength(raw_od)
            raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=dpf)

            raw_haemo_unfiltered = mne.preprocessing.nirs.beer_lambert_law(raw_od_original, ppf=0.1).copy()
            raw_haemo.filter(self.filter_lower_value, self.filter_upper_value, h_trans_bandwidth=self.h_trans_bandwidth, l_trans_bandwidth=self.l_trans_bandwidth)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            # Set baseline parameter based on correction method
            baseline = self.baseline if self.baseline_correction == "xSecondsBefore" else None
            
            raw_epochs = epochs = mne.Epochs(raw_haemo_unfiltered, events, event_id=event_dict, tmin=self.tmin, tmax=self.tmax, reject=None, reject_by_annotation=None, proj=False, baseline=None, preload=True, detrend=None, verbose=True)

            self.reject_criteria = compute_p2p(raw_epochs, self.data_types+["Control"], 90)
            
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

            epochs = reject_if_single_event_type(epochs, self.data_types + ["Control"])

            self.drop_log.append(epochs.drop_log)
            if len(epochs) != 0:
                # Apply custom baseline correction if needed
                if self.baseline_correction != "xSecondsBefore":
                    corrector = baselineCorrection(self.baseline_correction)
                    epochs = corrector.apply_correction(
                        self.baseline_correction,
                        epochs,
                        data_types=self.data_types,
                    )

                self.all_raw_epochs.append(raw_epochs)
                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
                
                Participant_i = individual_participant_class(f"Participant_{i}")
                Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                Participant_i.raw_intensity = raw_intensity
                Participant_i.raw_od = raw_od
                Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                Participant_i.raw_haemo = raw_haemo
                Participant_i.raw_epochs = raw_epochs
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
    def __init__(self, data_name, file_path, short_channel_correction: bool, negative_correlation_enhancement: bool, interpolate_bad_channels:bool=False, tmin:int = 0,
                 baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2,
                 l_trans_bandwidth: float = 0.02, reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8, snr_rejection: str = "None",
                 snr_threshold: int = 8, apply_tddr: bool = False):
        self.number_of_participants = 0
        self.all_tapping = []
        self.all_Control = []
        self.annotation_names = {"1.0": "HandMI",
                                 "Rest": "Control"
                                }
        self.file_path = Path(os.getenv(file_path.replace(":","").replace(" ", "_").replace("-", "_")).encode('latin-1').decode('utf-8'))
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 21
        self.scalp_coupling_threshold = scalp_coupling_threshold
        self.reject_criteria = reject_criteria
        self.tmin = tmin
        self.tmax = 21
        self.baseline = (None, 0)
        self.data_types = ["HandMI"]
        self.data_name = "Melika hand data long"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = ["0.0"]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        self.snr_rejection = snr_rejection
        self.snr_threshold = snr_threshold
        self.apply_tddr = apply_tddr
        super().__init__(
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
            data_name=self.data_name,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted,
            baseline_correction = self.baseline_correction,
            filter_lower_value = self.filter_lower_value,
            filter_upper_value = self.filter_upper_value,
            h_trans_bandwidth = self.h_trans_bandwidth,
            l_trans_bandwidth = self.l_trans_bandwidth,
            snr_rejection = self.snr_rejection,
            snr_threshold = self.snr_threshold,
            apply_tddr = self.apply_tddr
        )

    def define_raw_intensity(self, filename):
        raw_intensity = mne.io.read_raw_nirx(rf"{self.file_path / filename}", preload=True, verbose=True)
        return raw_intensity
        
    def load_data(self):
        for i, filename in enumerate(sorted(os.listdir(self.file_path)), start=1):
            self.number_of_participants += 1
            raw_intensity = self.define_raw_intensity(filename)
            raw_intensity = self.make_annotations(raw_intensity)


            raw_intensity.annotations.rename(self.annotation_names)
            for _unwanted in self.unwanted:
                    unwanted = np.nonzero(raw_intensity.annotations.description == _unwanted)
                    raw_intensity.annotations.delete(unwanted)

            if self.snr_rejection != "None":
                snr = snr_rejection(raw_intensity, self.snr_rejection)
                
                # Validation
                if self.snr_rejection == "SNR" and self.snr_threshold < 1:
                    raise ValueError("Currently the classic signal to noise ratio is used but the threshold for SNR is below 1 resulting in all channels being marked as bad. Please set the threshold to a value above 1.")
                if self.snr_rejection == "CV" and self.snr_threshold > 1:
                    raise ValueError("Currently the coefficient of variation is used but the threshold for CV is above 1 resulting in all channels being marked as bad. Please set the threshold to a value below 1.")
                
                # Get bad channels based on pair logic
                snr_bad_channels = get_bad_channels_by_pairs(raw_intensity.ch_names, snr, self.snr_threshold, self.snr_rejection)
                raw_intensity.info["bads"] = snr_bad_channels
            else:
                snr_bad_channels = []

            raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
            raw_od_original = raw_od.copy()

            # Check channel name consistency
            assert raw_intensity.ch_names == raw_od.ch_names, \
                f"Channel names mismatch!\nraw_intensity: {len(raw_intensity.ch_names)} channels\nraw_od: {len(raw_od.ch_names)} channels"
            
            if self.short_channel_correction:
                raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)
            raw_od = mne_nirs.channels.get_long_channels(raw_od)
            
            if self.apply_tddr:
                raw_od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)

            sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)

            sci_bad_channels = list(compress(raw_od.ch_names, sci < self.scalp_coupling_threshold))
            
            # Filter SNR bad channels to only include those that still exist in the long channels dataset
            snr_bad_channels_long_only = [ch for ch in snr_bad_channels if ch in raw_od.ch_names]
            
            # Combine bad channels from all preprocessing
            all_bad_channels = list(set(snr_bad_channels_long_only + sci_bad_channels))         
            raw_od.info["bads"] = all_bad_channels
            
            if self.interpolate_bad_channels:
                raw_od.interpolate_bads()
                
            dpf = compute_differential_pathlength(raw_od)
            raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=dpf)

            raw_haemo_unfiltered = mne.preprocessing.nirs.beer_lambert_law(raw_od_original, ppf=0.1).copy()
            raw_haemo.filter(self.filter_lower_value, self.filter_upper_value, h_trans_bandwidth=self.h_trans_bandwidth, l_trans_bandwidth=self.l_trans_bandwidth)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            # Set baseline parameter based on correction method
            baseline = self.baseline if self.baseline_correction == "xSecondsBefore" else None
            
            raw_epochs = epochs = mne.Epochs(raw_haemo_unfiltered, events, event_id=event_dict, tmin=self.tmin, tmax=self.tmax, reject=None, reject_by_annotation=None, proj=False, baseline=None, preload=True, detrend=None, verbose=True)

            self.reject_criteria = compute_p2p(raw_epochs, self.data_types+["Control"], 99)
            
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

            epochs = reject_if_single_event_type(epochs, self.data_types + ["Control"])

            self.drop_log.append(epochs.drop_log)
            if len(epochs) != 0:
                # Apply custom baseline correction if needed
                if self.baseline_correction != "xSecondsBefore":
                    corrector = baselineCorrection(self.baseline_correction)
                    epochs = corrector.apply_correction(
                        self.baseline_correction,
                        epochs,
                        data_types=self.data_types,
                    )

                self.all_raw_epochs.append(raw_epochs)
                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
                
                Participant_i = individual_participant_class(epochs.info["subject_info"]["his_id"])
                Participant_i.raw_intensity = raw_intensity
                Participant_i.raw_od = raw_od
                Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                Participant_i.raw_haemo = raw_haemo
                for name in self.data_types + ["Control"]:
                    # Crop to stimulus duration per condition
                    if len(epochs[name]) != 0:
                        epochs_cond = epochs[name].copy().crop(tmin=self.tmin, tmax=self.tmax)
                        
                        # Store in Participant_i as a separate attribute
                        setattr(Participant_i, f'epochs_{name}', epochs_cond)
                        
                        # Store raw data for later extraction
                        Participant_i.events[name] = epochs_cond.get_data(copy=True)
                        Participant_i.epochs.append(epochs_cond)
                        
                        # Append to global lists
                        if not hasattr(self, f'all_{name}_epochs'):
                            setattr(self, f'all_{name}_epochs', [])
                        else:
                            getattr(self, f'all_{name}_epochs').append(epochs_cond)
                        getattr(self, f'all_{name}').append(epochs_cond.get_data(copy=True))

                
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
    def __init__(self, data_name, file_path, short_channel_correction: bool, negative_correlation_enhancement: bool, interpolate_bad_channels:bool=False, tmin:int = 0,
                 baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2,
                 l_trans_bandwidth: float = 0.02, reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8, snr_rejection: str = "None",
                 snr_threshold: int = 8, apply_tddr: bool = False):
        self.number_of_participants = 0
        self.all_Control = []
        self.annotation_names = {"1.0": "TongueMI",
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
        self.file_path = Path(os.getenv('Melika_tongue_long_data').encode('latin-1').decode('utf-8'))
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 21
        self.scalp_coupling_threshold = scalp_coupling_threshold
        self.reject_criteria = reject_criteria
        self.tmin = tmin
        self.tmax = 21
        self.baseline = (None, 0)
        self.data_types = ["TongueMI"]
        self.data_name = "Melika tongue long data"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = ["2.0", "0.0"]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        self.snr_rejection = snr_rejection
        self.snr_threshold = snr_threshold
        self.apply_tddr = apply_tddr

        super().__init__(
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
            data_name=self.data_name,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted,
            baseline_correction = self.baseline_correction,
            filter_lower_value = self.filter_lower_value,
            filter_upper_value = self.filter_upper_value,
            h_trans_bandwidth = self.h_trans_bandwidth,
            l_trans_bandwidth = self.l_trans_bandwidth,
            snr_rejection = self.snr_rejection,
            snr_threshold = self.snr_threshold,
            apply_tddr = self.apply_tddr
        )

    def define_raw_intensity(self, filename):
        raw_intensity = mne.io.read_raw_nirx(rf"{self.file_path / filename}", preload=True, verbose=True)
        return raw_intensity
        
    def load_data(self):
        for i, filename in enumerate(sorted(os.listdir(self.file_path)), start=1):
            self.number_of_participants += 1
            raw_intensity = self.define_raw_intensity(filename)
            raw_intensity = self.make_annotations(raw_intensity)
            
            # #Fix the coordinate frame
            # for dig_point in raw_intensity.info['dig']:
            #     if dig_point['coord_frame'] == 0:  # FIFFV_COORD_UNKNOWN
            #         dig_point['coord_frame'] = 4   # FIFFV_COORD_HEAD

            raw_intensity.annotations.rename(self.annotation_names)
            for _unwanted in self.unwanted:
                    unwanted = np.nonzero(raw_intensity.annotations.description == _unwanted)
                    raw_intensity.annotations.delete(unwanted)

            if self.snr_rejection != "None":
                snr = snr_rejection(raw_intensity, self.snr_rejection)
                
                # Validation
                if self.snr_rejection == "SNR" and self.snr_threshold < 1:
                    raise ValueError("Currently the classic signal to noise ratio is used but the threshold for SNR is below 1 resulting in all channels being marked as bad. Please set the threshold to a value above 1.")
                if self.snr_rejection == "CV" and self.snr_threshold > 1:
                    raise ValueError("Currently the coefficient of variation is used but the threshold for CV is above 1 resulting in all channels being marked as bad. Please set the threshold to a value below 1.")
                
                # Get bad channels based on pair logic
                snr_bad_channels = get_bad_channels_by_pairs(raw_intensity.ch_names, snr, self.snr_threshold, self.snr_rejection)
                raw_intensity.info["bads"] = snr_bad_channels
            else:
                snr_bad_channels = []

            raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
            raw_od_original = raw_od.copy()

            # Check channel name consistency
            assert raw_intensity.ch_names == raw_od.ch_names, \
                f"Channel names mismatch!\nraw_intensity: {len(raw_intensity.ch_names)} channels\nraw_od: {len(raw_od.ch_names)} channels"
            
            if self.short_channel_correction:
                raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)
            raw_od = mne_nirs.channels.get_long_channels(raw_od)
            
            if self.apply_tddr:
                raw_od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)

            sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)

            sci_bad_channels = list(compress(raw_od.ch_names, sci < self.scalp_coupling_threshold))
            
            # Filter SNR bad channels to only include those that still exist in the long channels dataset
            snr_bad_channels_long_only = [ch for ch in snr_bad_channels if ch in raw_od.ch_names]
            
            # Combine bad channels from all preprocessing
            all_bad_channels = list(set(snr_bad_channels_long_only + sci_bad_channels))         
            raw_od.info["bads"] = all_bad_channels
            
            if self.interpolate_bad_channels:
                raw_od.interpolate_bads()
                
            dpf = compute_differential_pathlength(raw_od)
            raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=dpf)

            raw_haemo_unfiltered = mne.preprocessing.nirs.beer_lambert_law(raw_od_original, ppf=dpf).copy()
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
            
            raw_epochs = epochs = mne.Epochs(raw_haemo_unfiltered, events, event_id=event_dict, tmin=self.tmin, tmax=self.tmax, reject=None, reject_by_annotation=None, proj=False, baseline=None, preload=True, detrend=None, verbose=True)

            self.reject_criteria = compute_p2p(raw_epochs, self.data_types+["Control"], 90)
            
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

            epochs = reject_if_single_event_type(epochs, self.data_types + ["Control"])
            
            self.drop_log.append(epochs.drop_log)
            if len(epochs) != 0:
                # Apply custom baseline correction if needed
                if self.baseline_correction != "xSecondsBefore":
                    corrector = baselineCorrection(self.baseline_correction)
                    epochs = corrector.apply_correction(
                        self.baseline_correction,
                        epochs,
                        data_types=self.data_types,
                    )
                    
                self.all_raw_epochs.append(raw_epochs)
                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
                
                Participant_i = individual_participant_class(epochs.info["subject_info"]["his_id"])
                Participant_i.raw_intensity = raw_intensity
                Participant_i.raw_od = raw_od
                Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                Participant_i.raw_haemo = raw_haemo
                for name in self.data_types + ["Control"]:
                    # Crop to stimulus duration per condition
                    if len(epochs[name]) != 0:
                        epochs_cond = epochs[name].copy().crop(tmin=self.tmin, tmax=self.tmax)
                        
                        # Store in Participant_i as a separate attribute
                        setattr(Participant_i, f'epochs_{name}', epochs_cond)
                        
                        # Store raw data for later extraction
                        Participant_i.events[name] = epochs_cond.get_data(copy=True)
                        Participant_i.epochs.append(epochs_cond)
                        
                        # Append to global lists
                        if not hasattr(self, f'all_{name}_epochs'):
                            setattr(self, f'all_{name}_epochs', [])
                        else:
                            getattr(self, f'all_{name}_epochs').append(epochs_cond)
                        getattr(self, f'all_{name}').append(epochs_cond.get_data(copy=True))

                
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
        cropped_raw_data.annotations.description[0] = "I"
        cropped_raw_data.annotations.set_durations({"I" : 80})
        cropped_raw_data.annotations.rename({"I": "Introduction"})

        
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
                if id != 19 and id !=20:
                    cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, self.stimulus_duration, "Rest")
        return cropped_raw_data


###############################################################################################################################################################################################

class fNIRS_Pardis_DOC_data_load(fNIRS_data_load):
    def __init__(self, data_name, file_path, short_channel_correction: bool, negative_correlation_enhancement: bool, interpolate_bad_channels:bool=False, tmin:int = 0,
                 baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2,
                 l_trans_bandwidth: float = 0.02, reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8, snr_rejection: str = "None",
                 snr_threshold: int = 8, apply_tddr: bool = False):
        self.number_of_participants = 0
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
        self.data_name = "Pardis DOC data"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = [""]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        self.snr_rejection = snr_rejection
        self.snr_threshold = snr_threshold
        self.apply_tddr = apply_tddr
        super().__init__(
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
            data_name=self.data_name,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted,
            baseline_correction = self.baseline_correction,
            filter_lower_value = self.filter_lower_value,
            filter_upper_value = self.filter_upper_value,
            h_trans_bandwidth = self.h_trans_bandwidth,
            l_trans_bandwidth = self.l_trans_bandwidth,
            snr_rejection = self.snr_rejection,
            snr_threshold = self.snr_threshold,
            apply_tddr = self.apply_tddr
        )

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
                self.number_of_participants += 1
                raw_intensity = self.make_annotations(raw_intensity)
                
                raw_intensity.annotations.rename(self.annotation_names)

                for _unwanted in self.unwanted:
                    unwanted = np.nonzero(raw_intensity.annotations.description == _unwanted)
                    raw_intensity.annotations.delete(unwanted)

                if self.snr_rejection != "None":
                    snr = snr_rejection(raw_intensity, self.snr_rejection)
                    
                    # Validation
                    if self.snr_rejection == "SNR" and self.snr_threshold < 1:
                        raise ValueError("Currently the classic signal to noise ratio is used but the threshold for SNR is below 1 resulting in all channels being marked as bad. Please set the threshold to a value above 1.")
                    if self.snr_rejection == "CV" and self.snr_threshold > 1:
                        raise ValueError("Currently the coefficient of variation is used but the threshold for CV is above 1 resulting in all channels being marked as bad. Please set the threshold to a value below 1.")
                    
                    # Get bad channels based on pair logic
                    snr_bad_channels = get_bad_channels_by_pairs(raw_intensity.ch_names, snr, self.snr_threshold, self.snr_rejection)
                    raw_intensity.info["bads"] = snr_bad_channels
                else:
                    snr_bad_channels = []

                raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
                raw_od_original = raw_od.copy()

                # Check channel name consistency
                assert raw_intensity.ch_names == raw_od.ch_names, \
                    f"Channel names mismatch!\nraw_intensity: {len(raw_intensity.ch_names)} channels\nraw_od: {len(raw_od.ch_names)} channels"
                
                if self.short_channel_correction:
                    raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)
                raw_od = mne_nirs.channels.get_long_channels(raw_od)
                
                if self.apply_tddr:
                    raw_od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)

                sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)

                sci_bad_channels = list(compress(raw_od.ch_names, sci < self.scalp_coupling_threshold))
                
                # Filter SNR bad channels to only include those that still exist in the long channels dataset
                snr_bad_channels_long_only = [ch for ch in snr_bad_channels if ch in raw_od.ch_names]
                
                # Combine bad channels from all preprocessing
                all_bad_channels = list(set(snr_bad_channels_long_only + sci_bad_channels))         
                raw_od.info["bads"] = all_bad_channels
            
                if self.interpolate_bad_channels:
                    raw_od.interpolate_bads()
                
                dpf = compute_differential_pathlength(raw_od)
                raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=dpf)

                raw_haemo_unfiltered = mne.preprocessing.nirs.beer_lambert_law(raw_od_original, ppf=0.1).copy()
                raw_haemo.filter(self.filter_lower_value, self.filter_upper_value, h_trans_bandwidth=self.h_trans_bandwidth, l_trans_bandwidth=self.l_trans_bandwidth)

                if self.negative_correlation_enhancement:
                    raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

                events, event_dict = mne.events_from_annotations(raw_haemo)

                # Set baseline parameter based on correction method
                baseline = self.baseline if self.baseline_correction == "xSecondsBefore" else None
                
                raw_epochs = epochs = mne.Epochs(raw_haemo_unfiltered, events, event_id=event_dict, tmin=self.tmin, tmax=self.tmax, reject=None, reject_by_annotation=None, proj=False, baseline=None, preload=True, detrend=None, verbose=True)

                self.reject_criteria = compute_p2p(raw_epochs, self.data_types+["Control"], 90)
                
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

                epochs = reject_if_single_event_type(epochs, self.data_types + ["Control"])

                self.drop_log.append(epochs.drop_log)
                if len(epochs) != 0:
                    # Apply custom baseline correction if needed
                    if self.baseline_correction != "xSecondsBefore":
                        corrector = baselineCorrection(self.baseline_correction)
                        epochs = corrector.apply_correction(
                            self.baseline_correction,
                            epochs,
                            data_types=self.data_types,
                        )

                    self.all_raw_epochs.append(raw_epochs)
                    self.all_epochs.append(epochs)
                    self.all_control.append(epochs["Control"].get_data(copy=True))
                    
                    Participant_i = individual_participant_class(f"Participant_{i}")
                    Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                    Participant_i.raw_intensity = raw_intensity
                    Participant_i.raw_od = raw_od
                    Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                    Participant_i.raw_haemo = raw_haemo
                    Participant_i.raw_epochs = raw_epochs
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
    def __init__(self, data_name, file_path, short_channel_correction: bool, negative_correlation_enhancement: bool, interpolate_bad_channels:bool=False, tmin:int = 0,
                 baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2,
                 l_trans_bandwidth: float = 0.02, reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8, snr_rejection: bool = False,
                 snr_threshold: float = 8.0, apply_tddr: bool = False):
        self.number_of_participants = 0
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
        self.data_name = "Pardis HC data"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = ["5", "6", "7"]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        self.snr_rejection = snr_rejection
        self.snr_threshold = snr_threshold
        self.apply_tddr = apply_tddr
        super().__init__(
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
            data_name=self.data_name,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted,
            baseline_correction = self.baseline_correction,
            filter_lower_value = self.filter_lower_value,
            filter_upper_value = self.filter_upper_value,
            h_trans_bandwidth = self.h_trans_bandwidth,
            l_trans_bandwidth = self.l_trans_bandwidth,
            snr_rejection = self.snr_rejection,
            snr_threshold = self.snr_threshold,
            apply_tddr = self.apply_tddr
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
                self.number_of_participants += 1

                raw_intensity.annotations.rename(self.annotation_names)
                for _unwanted in self.unwanted:
                    unwanted = np.nonzero(raw_intensity.annotations.description == _unwanted)
                    raw_intensity.annotations.delete(unwanted)

                if self.snr_rejection != "None":
                    snr = snr_rejection(raw_intensity, self.snr_rejection)
                    
                    # Validation
                    if self.snr_rejection == "SNR" and self.snr_threshold < 1:
                        raise ValueError("Currently the classic signal to noise ratio is used but the threshold for SNR is below 1 resulting in all channels being marked as bad. Please set the threshold to a value above 1.")
                    if self.snr_rejection == "CV" and self.snr_threshold > 1:
                        raise ValueError("Currently the coefficient of variation is used but the threshold for CV is above 1 resulting in all channels being marked as bad. Please set the threshold to a value below 1.")
                    
                    # Get bad channels based on pair logic
                    snr_bad_channels = get_bad_channels_by_pairs(raw_intensity.ch_names, snr, self.snr_threshold, self.snr_rejection)
                    raw_intensity.info["bads"] = snr_bad_channels
                else:
                    snr_bad_channels = []

                raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
                raw_od_original = raw_od.copy()

                # Check channel name consistency
                assert raw_intensity.ch_names == raw_od.ch_names, \
                    f"Channel names mismatch!\nraw_intensity: {len(raw_intensity.ch_names)} channels\nraw_od: {len(raw_od.ch_names)} channels"
                
                if self.short_channel_correction:
                    raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)
                raw_od = mne_nirs.channels.get_long_channels(raw_od)
                
                if self.apply_tddr:
                    raw_od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)

                sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od)

                sci_bad_channels = list(compress(raw_od.ch_names, sci < self.scalp_coupling_threshold))
                
                # Filter SNR bad channels to only include those that still exist in the long channels dataset
                snr_bad_channels_long_only = [ch for ch in snr_bad_channels if ch in raw_od.ch_names]
                
                # Combine bad channels from all preprocessing
                all_bad_channels = list(set(snr_bad_channels_long_only + sci_bad_channels))         
                raw_od.info["bads"] = all_bad_channels
            
                if self.interpolate_bad_channels:
                    raw_od.interpolate_bads()
                
                dpf = compute_differential_pathlength(raw_od)
                raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=dpf)

                raw_haemo_unfiltered = mne.preprocessing.nirs.beer_lambert_law(raw_od_original, ppf=6).copy()
                raw_haemo.filter(self.filter_lower_value, self.filter_upper_value, h_trans_bandwidth=self.h_trans_bandwidth, l_trans_bandwidth=self.l_trans_bandwidth)

                if self.negative_correlation_enhancement:
                    raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

                events, event_dict = mne.events_from_annotations(raw_haemo)

                # Set baseline parameter based on correction method
                baseline = self.baseline if self.baseline_correction == "xSecondsBefore" else None
                
                raw_epochs = epochs = mne.Epochs(raw_haemo_unfiltered, events, event_id=event_dict, tmin=self.tmin, tmax=self.tmax, reject=None, reject_by_annotation=None, proj=False, baseline=None, preload=True, detrend=None, verbose=True)

                self.reject_criteria = compute_p2p(raw_epochs, self.data_types+["Control"], 90)
                
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

                epochs = reject_if_single_event_type(epochs, self.data_types + ["Control"], min_number_of_conditions_events=1)

                self.drop_log.append(epochs.drop_log)
                if len(epochs) != 0:
                    # Apply custom baseline correction if needed
                    if self.baseline_correction != "xSecondsBefore":
                        corrector = baselineCorrection(self.baseline_correction)
                        epochs = corrector.apply_correction(
                            self.baseline_correction,
                            epochs,
                            data_types=self.data_types,
                        )

                    self.all_raw_epochs.append(raw_epochs)
                    self.all_epochs.append(epochs)
                    self.all_control.append(epochs["Control"].get_data(copy=True))
                    
                    Participant_i = individual_participant_class(f"Participant_{i}")
                    Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                    Participant_i.raw_intensity = raw_intensity
                    Participant_i.raw_od = raw_od
                    Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                    Participant_i.raw_haemo = raw_haemo
                    Participant_i.raw_epochs = raw_epochs
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

###############################################################################################################################################################################################

class fNIRS_EEG_HC_baseline_data_load(fNIRS_data_load):
    def __init__(self, data_name: str = "EEG fNIRS HC baseline data", file_path: str = 'EEG_fNIRS_HC_baseline_data', short_channel_correction: bool = True, negative_correlation_enhancement: bool = False, interpolate_bad_channels:bool=False, tmin:int = 0,
                 baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2,
                 l_trans_bandwidth: float = 0.02, reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8, snr_rejection: bool = False,
                 snr_threshold: float = 8.0, apply_tddr: bool = False):
        self.number_of_participants = 0
        self.all_tapping = []
        self.all_Control = []
        self.annotation_names = {"1": "TonguePhysical",
                                 "2": "Control",
                                 "3": "TongueIM",
                                 "4": "n_back/0_back",
                                 "5": "n_back/1_back",
                                 "6": "n_back/2_back",
                                 "7": "n_back/3_back"
                                }
        self.file_path = Path(os.getenv(file_path.replace(" ", "_").replace("-", "_")))
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = {"1": 15,
                                 "2": 15,
                                 "3": 15,
                                 "4": 0,
                                 "5": 0,
                                 "6": 0,
                                 "7": 0,
                                }
        self.scalp_coupling_threshold = scalp_coupling_threshold
        self.reject_criteria = reject_criteria
        self.tmin = tmin
        self.tmax = 20
        self.baseline = (None, 0)
        self.data_types = ['n_back/0_back', 'n_back/1_back', 'n_back/2_back', 'n_back/3_back']
        self.data_name = data_name
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = ["TonguePhysical", "TongueIM"]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        self.snr_rejection = snr_rejection
        self.snr_threshold = snr_threshold
        self.apply_tddr = apply_tddr
        super().__init__(
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
            data_name=self.data_name,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted,
            baseline_correction = self.baseline_correction,
            filter_lower_value = self.filter_lower_value,
            filter_upper_value = self.filter_upper_value,
            h_trans_bandwidth = self.h_trans_bandwidth,
            l_trans_bandwidth = self.l_trans_bandwidth,
            snr_rejection = self.snr_rejection,
            snr_threshold = self.snr_threshold,
            apply_tddr = self.apply_tddr
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
    
    def find_excel_file(self, folder_path):
        """
        Find the .xlsx file in the nested folder structure.
        Returns the full path to the .xlsx file or None if not found.
        """
        # Look for .xlsx files recursively in the folder
        excel_files = glob.glob(os.path.join(folder_path, "**", "*.xlsx"), recursive=True)

        if excel_files:
            return excel_files[0]  # Return the first .snirf file found
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
    
    def get_actual_event(self, df, sheets, events, sfreq):
        actual_events = np.empty((0, 3), dtype=int)
        for sheet in sheets:
            df[sheet] = df[sheet].sort_values(by=df[sheet].columns[-1]).reset_index(drop=True)
            generator = (item for item in df[sheet].columns if "started_mean" in item)
            time_column = next(generator, None)
            times = list(df[sheet][time_column])
            times = [time for time in times if str(time) != 'nan']
            time_corrected_events = events[:,0] / sfreq
            if sheet == 'Tongue_Loop':
                offset = times[0] - time_corrected_events[0]
            times -= offset
            try:
                markers = df[sheet]["marker"]
                markers = [marker for marker in markers if type(marker) != str and str(marker) != 'nan']
                for time, marker in zip(times, markers):
                    if "back" in self.annotation_names[str(int(markers[0]))]:
                        actual_events = np.vstack([actual_events, np.array([int((time-15)*sfreq), int(0), int(2)])]) # Add control/baseline/rest before active task
                    actual_events = np.vstack([actual_events, np.array([int(time*sfreq), int(0), int(marker)])])
            except:
                print("No markers available for the events")
                if self.stimulus_duration[str(actual_events[:,2][-1])] == 0:
                    self.stimulus_duration[str(actual_events[:,2][-1])] = times[-1] - (actual_events[:,0] / sfreq)[-1] + 3 # We add 3 as the compute the time from the last trigger, but the duration of this event has to be accounted for

                
            print("Sucessfully added all events")

        return actual_events
    
    def make_annotations(self, excel_path, raw_intensity):
        df = pd.read_excel(excel_path, sheet_name=None)
        sheets = list(df.keys())
        sfreq = raw_intensity.info["sfreq"]
        events, event_dict = mne.events_from_annotations(raw_intensity)
        actual_events = self.get_actual_event(df, sheets, events, sfreq)
        onset = actual_events[:, 0] / sfreq
        duration = np.array([self.stimulus_duration[str(event)] for event in actual_events[:,2]])
        description = actual_events[:, 2].astype(str)
        new_annotations = mne.Annotations(onset=onset, 
                                 duration=duration, 
                                 description=description)

        raw_intensity.set_annotations(new_annotations)
        return raw_intensity
    
    def crop_data(self, raw_intensity):
        events, event_dict = mne.events_from_annotations(raw_intensity)
        sfreq = raw_intensity.info["sfreq"]
        new_tmin = events[0][0] / sfreq - 10
        new_tmax = events[-1][0] / sfreq + self.stimulus_duration[str(events[-1][2])] + 3
        raw_intensity.crop(tmin=new_tmin, tmax=new_tmax)
        return raw_intensity

    def load_data(self):
        # Get all folders and sort them (works for both P folders and random named folders)
        all_folders = [f for f in sorted(os.listdir(self.file_path)) 
                    if os.path.isdir(os.path.join(self.file_path, f))]
        
        for i, folder_name in enumerate(all_folders, start=1):
            try:
                
                
                excel_path = self.find_excel_file(os.path.join(self.file_path, folder_name))
                raw_intensity = self.define_raw_intensity(folder_name)
                self.number_of_participants += 1
                raw_intensity = self.make_annotations(excel_path, raw_intensity)
                self.tmax = max(self.stimulus_duration.values())
                raw_intensity.annotations.rename(self.annotation_names)
                for _unwanted in self.unwanted:
                        unwanted = np.nonzero(raw_intensity.annotations.description == _unwanted)[0]
                        before_after_unwanted = np.append(unwanted - 1, unwanted + 1)
                        before_after_unwanted_control = before_after_unwanted[np.isin(before_after_unwanted, np.nonzero(raw_intensity.annotations.description == "Control")[0])]
                        raw_intensity.annotations.delete(np.append(unwanted, before_after_unwanted_control))
                raw_intensity = self.crop_data(raw_intensity)
                
                if self.snr_rejection != "None":
                    snr = snr_rejection(raw_intensity, self.snr_rejection)
                    
                    # Validation
                    if self.snr_rejection == "SNR" and self.snr_threshold < 1:
                        raise ValueError("Currently the classic signal to noise ratio is used but the threshold for SNR is below 1 resulting in all channels being marked as bad. Please set the threshold to a value above 1.")
                    if self.snr_rejection == "CV" and self.snr_threshold > 1:
                        raise ValueError("Currently the coefficient of variation is used but the threshold for CV is above 1 resulting in all channels being marked as bad. Please set the threshold to a value below 1.")
                    
                    # Get bad channels based on pair logic
                    snr_bad_channels = get_bad_channels_by_pairs(raw_intensity.ch_names, snr, self.snr_threshold, self.snr_rejection)
                    raw_intensity.info["bads"] = snr_bad_channels
                else:
                    snr_bad_channels = []

                raw_od = mne.preprocessing.nirs.optical_density(raw_intensity)
                raw_od_original = raw_od.copy()

                # Check channel name consistency
                assert raw_intensity.ch_names == raw_od.ch_names, \
                    f"Channel names mismatch!\nraw_intensity: {len(raw_intensity.ch_names)} channels\nraw_od: {len(raw_od.ch_names)} channels"
                
                if self.short_channel_correction:
                    raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)
                raw_od = mne_nirs.channels.get_long_channels(raw_od)
                
                if self.apply_tddr:
                    raw_od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)

                sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od, l_freq=0.5, h_freq=2.5)

                sci_bad_channels = list(compress(raw_od.ch_names, sci < self.scalp_coupling_threshold))
                
                # Filter SNR bad channels to only include those that still exist in the long channels dataset
                snr_bad_channels_long_only = [ch for ch in snr_bad_channels if ch in raw_od.ch_names]
                
                # Combine bad channels from all preprocessing
                all_bad_channels = list(set(snr_bad_channels_long_only + sci_bad_channels))         
                raw_od.info["bads"] = all_bad_channels
            
                if self.interpolate_bad_channels:
                    raw_od.interpolate_bads()
                
                dpf = compute_differential_pathlength(raw_od)
                raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=dpf)

                raw_haemo_unfiltered = mne.preprocessing.nirs.beer_lambert_law(raw_od_original, ppf=6).copy()
                raw_haemo.filter(self.filter_lower_value, self.filter_upper_value, h_trans_bandwidth=self.h_trans_bandwidth, l_trans_bandwidth=self.l_trans_bandwidth)

                if self.negative_correlation_enhancement:
                    raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

                event_dict_trans = {val: int(key) for key, val in self.annotation_names.items()}
                events, event_dict = mne.events_from_annotations(raw_haemo, event_dict_trans)

                # Set baseline parameter based on correction method
                baseline = self.baseline if self.baseline_correction == "xSecondsBefore" else None
                
                raw_epochs = epochs = mne.Epochs(raw_haemo_unfiltered, events, event_id=event_dict, tmin=self.tmin, tmax=self.tmax, reject=None, reject_by_annotation=None, proj=False, baseline=None, preload=True, detrend=None, verbose=True)
                
                self.reject_criteria = compute_p2p(raw_epochs, self.data_types+["Control"], 95)
                
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
                

                self.drop_log.append(epochs.drop_log)
                if len(epochs) != 0:
                    # Apply custom baseline correction if needed
                    if self.baseline_correction != "xSecondsBefore":
                        corrector = baselineCorrection(self.baseline_correction)
                        epochs = corrector.apply_correction(
                            self.baseline_correction,
                            epochs,
                            data_types=self.data_types,
                        )
                    
                    Participant_i = individual_participant_class(f"Participant_{i}")
                    Participant_i.raw_intensity = raw_intensity
                    Participant_i.raw_od = raw_od
                    Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                    Participant_i.raw_haemo = raw_haemo
                    for name in self.data_types + ["Control"]:
                        # Crop to stimulus duration per condition
                        if len(epochs[name]) != 0:
                            dur = np.floor(self.stimulus_duration[str(event_dict[name])])
                            epochs_cond = epochs[name].copy().crop(tmin=self.tmin, tmax=dur)
                            
                            # Store in Participant_i as a separate attribute
                            setattr(Participant_i, f'epochs_{name}', epochs_cond)
                            
                            # Store raw data for later extraction
                            Participant_i.events[name] = epochs_cond.get_data(copy=True)
                            Participant_i.epochs.append(epochs_cond)
                            
                            # Append to global lists
                            if not hasattr(self, f'all_{name}_epochs'):
                                setattr(self, f'all_{name}_epochs', [])
                            else:
                                getattr(self, f'all_{name}_epochs').append(epochs_cond)
                            getattr(self, f'all_{name}').append(epochs_cond.get_data(copy=True))

                    
                    getattr(self, 'Individual_participants').append(Participant_i)
            except FileNotFoundError as e:
                print(f"Error loading {folder_name}: {e}")
            except Exception as e:
                print(f"Unexpected error with {folder_name}: {e}")

        # Concatenate the control data
        self.all_control = np.concatenate(self.all_Control, axis=0)
        all_data = {"Control": self.all_control}

        # Concatenate the data for each data type
        for name in self.data_types:
            setattr(self, f'all_{name}', np.concatenate(getattr(self, f'all_{name}'), axis=0))
            self.all_epochs.append(getattr(self, f'all_{name}_epochs'))
            all_data.update({name: getattr(self, f'all_{name}')})

        # Update all_data with control_dict
        all_freq = raw_intensity.info["sfreq"]
        self.data_types.append("Control")
        return self.all_epochs, self.data_name, all_data, all_freq, self.data_types, self.Individual_participants
    
###############################################################################################################################################################################################

fNIRS_EEG_HC_follow_up_data_load = fNIRS_EEG_HC_baseline_data_load

###############################################################################################################################################################################################

fNIRS_EEG_patient_baseline_data_load = fNIRS_EEG_HC_baseline_data_load

###############################################################################################################################################################################################

fNIRS_EEG_patient_follow_up_data_load = fNIRS_EEG_HC_baseline_data_load