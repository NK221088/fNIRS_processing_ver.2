from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
import mne
import mne_nirs
import os
from Participant_class import individual_participant_class
import glob
from pathlib import Path
from preprocessing_toolbox.baselineCorrection import baselineCorrection
from preprocessing_toolbox.post_rejection import reject_if_single_event_type
from preprocessing_toolbox.SNR_rejection import snr_rejection, get_bad_channels_by_pairs
from preprocessing_toolbox.differential_pathlength import compute_differential_pathlength
from preprocessing_toolbox.p2p import compute_p2p
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime

from dotenv import dotenv_values

# Enable Python UTF-8 mode (handles Windows cp1252 locale issues)
os.environ['PYTHONUTF8'] = '1'

# Load environment variables as dictionary with explicit UTF-8 encoding
config = dotenv_values(".env", encoding='utf-8')

# def apply_baseline_correction(channel_values, times, sfreq, events, stimulus_duration, annotations):
#     previous_event = np.array([None, None, None])
#     for idx, event in enumerate(events):
#         try:
#             if str(previous_event[2]) in annotations.keys():
#                 if annotations[str(previous_event[2])] == "Control":
#                     control_event_start = previous_event[0]
#                     control_event_end = control_event_start + int(stimulus_duration[annotations[str(previous_event[2])]]*sfreq)
#                     event_start = event[0]
#                     event_end = event_start + int(stimulus_duration[annotations[str(event[2])]]*sfreq)
#                     control_average = np.mean(channel_values[control_event_start:control_event_end])
#                     channel_values[event_start:event_end] -= control_average
#                     previous_event = event
#                 else:
#                     previous_event = event
#             else:
#                 previous_event = event
#                 continue
#         except Exception as e:
#             print(f"Error processing event {event} at index {idx}: {e}")
#     return channel_values

def apply_baseline_correction(channel_values, times, sfreq, events, stimulus_duration, annotations):
    previous_event = np.array([None, None, None])
    for idx, event in enumerate(events):
        try:
            if str(previous_event[2]) in annotations.keys():
                if annotations[str(previous_event[2])] == "Control":
                    control_event_start = previous_event[0]
                    control_event_end = control_event_start + int(stimulus_duration[str(previous_event[2])]*sfreq)
                    event_start = event[0]
                    event_end = event_start + int(stimulus_duration[str(event[2])]*sfreq)
                    control_average = np.mean(channel_values[control_event_start:control_event_end])
                    channel_values[event_start:event_end] -= control_average
                    channel_values[control_event_start:control_event_end] -= control_average
                    previous_event = event
                else:
                    previous_event = event
            else:
                previous_event = event
                continue
        except Exception as e:
            print(f"Error processing event {event} at index {idx}: {e}")
    return channel_values

def find_snirf_file(folder_path):
    """
    Find the .snirf file in the nested folder structure.
    Returns the full path to the .snirf file or None if not found.
    """
    # Look for .snirf files recursively in the folder
    snirf_files = glob.glob(os.path.join(folder_path, "**", "*.snirf"), recursive=True)
    
    if snirf_files:                
        creation_times = [snirf_file.split("\\")[-1].replace(".snirf", "")[-3:] for snirf_file in snirf_files]
        snirf_file = snirf_files[np.argmax(creation_times)]  #  Find the last created .snirf file found
        snirf_file_folder = snirf_file[:-(len(snirf_file.split("\\")[-1])+1)]
        return snirf_file_folder
    return None
    
def define_raw_intensity(file_path, folder_name):
    """
    Load raw intensity data from a folder (handles different dataset structures).
    folder_name: The name of the folder containing the data
    """
    folder_path = os.path.join(file_path, folder_name)
    
    # Find the .snirf file in the nested structure
    snirf_file_path = find_snirf_file(folder_path)
    
    if not snirf_file_path:
        raise FileNotFoundError(f"No .snirf file found in {folder_path}")
    
    raw_intensity = mne.io.read_raw_nirx(snirf_file_path, verbose=True, preload=True)
    
    return raw_intensity

def load_ages(age_file):
    all_ages = {}
    df = pd.read_excel(age_file, sheet_name=None)
    sheets = list(df.keys())
    for sheet in sheets:
        try:
            ID_generator = (item for item in df[sheet].columns if "id" in item.lower())
            ID_column = next(ID_generator, None)
            age_generator = (item for item in df[sheet].columns if "cpr" in item.lower())
            age_column = next(age_generator, None)
            for id, age in zip(df[sheet][ID_column], df[sheet][age_column]):
                all_ages[id] = age
        except:
            ValueError("Data is not available")
    return all_ages
                
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
        all_folders = [f for f in sorted(os.listdir(self.file_path)) 
                    if os.path.isdir(os.path.join(self.file_path, f))]
        
        for i, folder_name in enumerate(all_folders, start=1):
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
            all_bad_channels = sorted(list(set(snr_bad_channels_long_only + sci_bad_channels)))
            raw_od.info["bads"] = all_bad_channels
            
            if self.interpolate_bad_channels:
                raw_od.interpolate_bads(method={"fnirs":"nearest"})
            
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
        self.file_path = Path(os.getenv(data_name.replace(" ", "_").replace(".", "").replace(":", "").replace("-", "_")))
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
            all_bad_channels = sorted(list(set(snr_bad_channels_long_only + sci_bad_channels)))
            raw_od.info["bads"] = all_bad_channels
            
            if self.interpolate_bad_channels:
                raw_od.interpolate_bads(method={"fnirs":"nearest"})

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
            all_bad_channels = sorted(list(set(snr_bad_channels_long_only + sci_bad_channels)))
            raw_od.info["bads"] = all_bad_channels

            if self.interpolate_bad_channels:
                raw_od.interpolate_bads(method={"fnirs":"nearest"})
            
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
                # Get your value
        key = file_path.replace(":", "").replace(" ", "_").replace("-", "_")
        env_value = config.get(key)
        if env_value:
            self.file_path = Path(env_value)
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
            all_bad_channels = sorted(list(set(snr_bad_channels_long_only + sci_bad_channels)))
            raw_od.info["bads"] = all_bad_channels
            
            if self.interpolate_bad_channels:
                raw_od.interpolate_bads(method={"fnirs":"nearest"})
                
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
            all_bad_channels = sorted(list(set(snr_bad_channels_long_only + sci_bad_channels)))
            raw_od.info["bads"] = all_bad_channels
            
            if self.interpolate_bad_channels:
                raw_od.interpolate_bads(method={"fnirs":"nearest"})
                
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
            all_bad_channels = sorted(list(set(snr_bad_channels_long_only + sci_bad_channels)))
            raw_od.info["bads"] = all_bad_channels
            
            if self.interpolate_bad_channels:
                raw_od.interpolate_bads(method={"fnirs":"nearest"})
                
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
            all_bad_channels = sorted(list(set(snr_bad_channels_long_only + sci_bad_channels)))
            raw_od.info["bads"] = all_bad_channels
            
            if self.interpolate_bad_channels:
                raw_od.interpolate_bads(method={"fnirs":"nearest"})
                
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
            all_bad_channels = sorted(list(set(snr_bad_channels_long_only + sci_bad_channels)))
            raw_od.info["bads"] = all_bad_channels
            
            if self.interpolate_bad_channels:
                raw_od.interpolate_bads(method={"fnirs":"nearest"})
                
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
        self.all_Control = []
        self.annotation_names = {
                                "0": "Control",
                                "1": "HandMI",
                                "2": "Outro",
                                "3": "Introduction",
                                "4": "Resting state",
                                }
        self.standard_event_ids = {
        }
        key = file_path.replace(":", "").replace(" ", "_").replace("-", "_")
        env_value = config.get(key)
        if env_value:
            self.file_path = Path(env_value)
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = {
                            "Introduction": 80,
                            "Resting state": 30,
                            'Control': 21,
                            'HandMI': 21,
                            "Outro": 10,
                        }
        self.scalp_coupling_threshold = scalp_coupling_threshold
        self.reject_criteria = reject_criteria
        self.tmin = tmin
        self.tmax = 21
        self.baseline = (None, 0)
        self.data_types = ["HandMI"]
        self.data_name = "Melika hand data long"
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
        self.subjects_to_exclude = {self.data_name: []}
        self.folder_errors = []
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
        
    def load_data(self):
        all_folders = [f for f in sorted(os.listdir(self.file_path)) 
            if os.path.isdir(os.path.join(self.file_path, f))]
        for i, folder_name in enumerate(all_folders, start=1):
            patient_name = folder_name.split("_")[0]
            if patient_name in self.subjects_to_exclude[self.data_name]:
                continue
            try:
                raw_intensity = define_raw_intensity(self.file_path, folder_name)
                raw_intensity.annotations.description = np.array([anno.split(".")[0] for anno in raw_intensity.annotations.description])
                for _unwanted in self.unwanted:
                    unwanted = np.nonzero(raw_intensity.annotations.description == _unwanted)
                    raw_intensity.annotations.delete(unwanted)
                raw_intensity = self.make_annotations(raw_intensity)
                self.standard_event_ids = {value: int(float(key)) for key, value in self.annotation_names.items()}

                raw_intensity_long = mne_nirs.channels.get_long_channels(raw_intensity)
                raw_intensity_short = mne_nirs.channels.get_short_channels(raw_intensity)
                
                if self.snr_rejection != "None":
                    snr = snr_rejection(raw_intensity_long, self.snr_rejection)
                    # Validation
                    if self.snr_rejection == "SNR" and self.snr_threshold < 1:
                        raise ValueError("Currently the classic signal to noise ratio is used but the threshold for SNR is below 1 resulting in all channels being marked as bad. Please set the threshold to a value above 1.")
                    if self.snr_rejection == "CV" and self.snr_threshold > 1:
                        raise ValueError("Currently the coefficient of variation is used but the threshold for CV is above 1 resulting in all channels being marked as bad. Please set the threshold to a value below 1.")
                    # Get bad channels based on pair logic
                    snr_bad_channels = get_bad_channels_by_pairs(raw_intensity_long.ch_names, snr, self.snr_threshold, self.snr_rejection)
                    raw_intensity_long.info["bads"] = snr_bad_channels
                else:
                    snr_bad_channels = []

                raw_od = mne.preprocessing.nirs.optical_density(raw_intensity_long)
                dpf = compute_differential_pathlength(raw_od)
                raw_od_short = mne.preprocessing.nirs.optical_density(raw_intensity_short)
                raw_od_original = raw_od.copy()

                sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od_short, l_freq=0.5, h_freq=2.5)
                sci_bad_channels = list(compress(raw_od_short.ch_names, sci < self.scalp_coupling_threshold))
                raw_od_short.info["bads"] = sci_bad_channels
                
                if self.interpolate_bad_channels:
                    raw_od.interpolate_bads(method={"fnirs":"nearest"})
                
                raw_od.add_channels([raw_od_short])

                if self.apply_tddr:
                    raw_od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)

                raw_haemo_unfiltered = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=dpf).copy()
                
                if self.short_channel_correction:
                    raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)

                raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=dpf)
                
                raw_haemo.filter(self.filter_lower_value, self.filter_upper_value, h_trans_bandwidth=self.h_trans_bandwidth, l_trans_bandwidth=self.l_trans_bandwidth)
                
                if self.negative_correlation_enhancement:
                    raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)
                
                events, event_dict = mne.events_from_annotations(raw_haemo, self.standard_event_ids)
                        
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

                epochs = reject_if_single_event_type(epochs, self.data_types + ["Control"])
                
                first_samp_correct_events = events.copy()
                first_samp_correct_events[:,0] = events[:,0] - raw_haemo._first_samps
                raw_haemo.apply_function(apply_baseline_correction, picks="hbo", times=raw_haemo.times, sfreq=raw_haemo.info["sfreq"], events=first_samp_correct_events, stimulus_duration=self.stimulus_duration, annotations = self.annotation_names)
                raw_haemo.apply_function(apply_baseline_correction, picks="hbr", times=raw_haemo.times, sfreq=raw_haemo.info["sfreq"], events=first_samp_correct_events, stimulus_duration=self.stimulus_duration, annotations = self.annotation_names)
                
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
                    
                    Participant_i = individual_participant_class(f"{patient_name}".replace("-", ""))
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
                    self.number_of_participants += 1
                
            except FileNotFoundError as e:
                print(f"Error loading {folder_name}: {e}")
                self.folder_errors.append(f"Unexpected error with {folder_name}: {e}")
            except Exception as e:
                print(f"Unexpected error with {folder_name}: {e}")
                self.folder_errors.append(f"Unexpected error with {folder_name}: {e}")
                
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
        cropped_raw_data = raw_intensity.copy()
        for key, value in self.annotation_names.items():
            if key in np.unique(cropped_raw_data.annotations.description):
                cropped_raw_data.annotations.rename({key: value})
                cropped_raw_data.annotations.set_durations({value: self.stimulus_duration[value]})
        new_onsets = list(cropped_raw_data.annotations.onset.copy())
        new_durations = list(cropped_raw_data.annotations.duration.copy())
        new_descriptions = list(cropped_raw_data.annotations.description.copy())
        
        # Add resting state period:
        if new_descriptions[0] != "Introduction":
            first_onset = new_onsets[0]
            new_onsets.append(first_onset - self.stimulus_duration["Resting state"] - self.stimulus_duration["Introduction"])
            new_durations.append(self.stimulus_duration["Introduction"])
            new_descriptions.append("Introduction")
            new_onsets.append(first_onset - self.stimulus_duration["Resting state"])
            new_durations.append(self.stimulus_duration["Resting state"])
            new_descriptions.append("Resting state")       
        else:
            new_onsets.append(new_onsets[0] + new_durations[0])
            new_durations.append(self.stimulus_duration["Resting state"])
            new_descriptions.append("Resting state")
        
        for annotation in cropped_raw_data.annotations:
            if annotation["description"] in self.data_types:
                new_onsets.append(annotation["onset"] + self.stimulus_duration[annotation["description"]])
                new_durations.append(self.stimulus_duration["Control"])
                new_descriptions.append("Control")
                if np.sum(np.array(new_descriptions) == "Control") == 6 or np.sum(np.array(new_descriptions) == "Control") == 12:
                    new_onsets.append(new_onsets[-1] + new_durations[-1])
                    new_durations.append(self.stimulus_duration["Resting state"])
                    new_descriptions.append("Resting state")                    
        new_annotations = mne.Annotations(onset = new_onsets, duration = new_durations, description = new_descriptions)
        cropped_raw_data.set_annotations(new_annotations)
        return cropped_raw_data
###############################################################################################################################################################################################

class fNIRS_Melika_tongue_long_data_load(fNIRS_data_load):
    def __init__(self, data_name, file_path, short_channel_correction: bool, negative_correlation_enhancement: bool, interpolate_bad_channels:bool=False, tmin:int = 0,
                 baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2,
                 l_trans_bandwidth: float = 0.02, reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8, snr_rejection: str = "None",
                 snr_threshold: int = 8, apply_tddr: bool = False):
        self.number_of_participants = 0
        self.all_Control = []
        self.annotation_names = {
                                "0": "Control",
                                "1": "TongueMI",
                                "2": "Outro",
                                "3": "Introduction",
                                "4": "Resting state",
                                }
        self.standard_event_ids = {
        }
        key = file_path.replace(":", "").replace(" ", "_").replace("-", "_")
        env_value = config.get(key)
        if env_value:
            self.file_path = Path(env_value)
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = {
                            "Introduction": 80,
                            "Resting state": 30,
                            'Control': 21,
                            'TongueMI': 21,
                            "Outro": 10,
                        }
        self.scalp_coupling_threshold = scalp_coupling_threshold
        self.reject_criteria = reject_criteria
        self.tmin = tmin
        self.tmax = 21
        self.baseline = (None, 0)
        self.data_types = ["TongueMI"]
        self.data_name = "Melika tongue long data"
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = ["2", "0"]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        self.snr_rejection = snr_rejection
        self.snr_threshold = snr_threshold
        self.apply_tddr = apply_tddr
        self.subjects_to_exclude = {self.data_name: []}
        self.folder_errors = []

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
    def make_annotations(self, raw_intensity):
        cropped_raw_data = raw_intensity.copy()
        for key, value in self.annotation_names.items():
            if key in np.unique(cropped_raw_data.annotations.description):
                cropped_raw_data.annotations.rename({key: value})
                cropped_raw_data.annotations.set_durations({value: self.stimulus_duration[value]})
        new_onsets = list(cropped_raw_data.annotations.onset.copy())
        new_durations = list(cropped_raw_data.annotations.duration.copy())
        new_descriptions = list(cropped_raw_data.annotations.description.copy())
        
        # Add resting state period:
        new_onsets.append(new_onsets[0] + new_durations[0])
        new_durations.append(self.stimulus_duration["Resting state"])
        new_descriptions.append("Resting state")
        
        for annotation in cropped_raw_data.annotations:
            if annotation["description"] in self.data_types:
                new_onsets.append(annotation["onset"] + self.stimulus_duration[annotation["description"]])
                new_durations.append(self.stimulus_duration["Control"])
                new_descriptions.append("Control")
                if np.sum(np.array(new_descriptions) == "Control") == 6 or np.sum(np.array(new_descriptions) == "Control") == 12:
                    new_onsets.append(new_onsets[-1] + new_durations[-1])
                    new_durations.append(self.stimulus_duration["Resting state"])
                    new_descriptions.append("Resting state")                    
        new_annotations = mne.Annotations(onset = new_onsets, duration = new_durations, description = new_descriptions)
        cropped_raw_data.set_annotations(new_annotations)
        return cropped_raw_data
        
    def load_data(self):
        all_folders = [f for f in sorted(os.listdir(self.file_path)) 
            if os.path.isdir(os.path.join(self.file_path, f))]
        for i, folder_name in enumerate(all_folders, start=1):
            patient_name = folder_name.split("_")[0]
            if patient_name in self.subjects_to_exclude[self.data_name]:
                continue
            try:
                raw_intensity = define_raw_intensity(self.file_path, folder_name)
                raw_intensity.annotations.description = np.array([anno.split(".")[0] for anno in raw_intensity.annotations.description])
                for _unwanted in self.unwanted:
                    unwanted = np.nonzero(raw_intensity.annotations.description == _unwanted)
                    raw_intensity.annotations.delete(unwanted)
                raw_intensity = self.make_annotations(raw_intensity)
                self.standard_event_ids = {value: int(float(key)) for key, value in self.annotation_names.items()}
                
                # #Fix the coordinate frame
                # for dig_point in raw_intensity.info['dig']:
                #     if dig_point['coord_frame'] == 0:  # FIFFV_COORD_UNKNOWN
                #         dig_point['coord_frame'] = 4   # FIFFV_COORD_HEAD

                raw_intensity_long = mne_nirs.channels.get_long_channels(raw_intensity)
                raw_intensity_short = mne_nirs.channels.get_short_channels(raw_intensity)
                
                if self.snr_rejection != "None":
                    snr = snr_rejection(raw_intensity_long, self.snr_rejection)
                    # Validation
                    if self.snr_rejection == "SNR" and self.snr_threshold < 1:
                        raise ValueError("Currently the classic signal to noise ratio is used but the threshold for SNR is below 1 resulting in all channels being marked as bad. Please set the threshold to a value above 1.")
                    if self.snr_rejection == "CV" and self.snr_threshold > 1:
                        raise ValueError("Currently the coefficient of variation is used but the threshold for CV is above 1 resulting in all channels being marked as bad. Please set the threshold to a value below 1.")
                    # Get bad channels based on pair logic
                    snr_bad_channels = get_bad_channels_by_pairs(raw_intensity_long.ch_names, snr, self.snr_threshold, self.snr_rejection)
                    raw_intensity_long.info["bads"] = snr_bad_channels
                else:
                    snr_bad_channels = []

                raw_od = mne.preprocessing.nirs.optical_density(raw_intensity_long)
                dpf = compute_differential_pathlength(raw_od)
                raw_od_short = mne.preprocessing.nirs.optical_density(raw_intensity_short)
                raw_od_original = raw_od.copy()

                sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od_short, l_freq=0.5, h_freq=2.5)
                sci_bad_channels = list(compress(raw_od_short.ch_names, sci < self.scalp_coupling_threshold))
                raw_od_short.info["bads"] = sci_bad_channels
                
                if self.interpolate_bad_channels:
                    raw_od.interpolate_bads(method={"fnirs":"nearest"})
                
                raw_od.add_channels([raw_od_short])

                if self.apply_tddr:
                    raw_od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)

                raw_haemo_unfiltered = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=dpf).copy()
                
                if self.short_channel_correction:
                    raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)

                raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=dpf)
                
                raw_haemo.filter(self.filter_lower_value, self.filter_upper_value, h_trans_bandwidth=self.h_trans_bandwidth, l_trans_bandwidth=self.l_trans_bandwidth)
                
                if self.negative_correlation_enhancement:
                    raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)
                                    
                events, event_dict = mne.events_from_annotations(raw_haemo, self.standard_event_ids)
                        
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
                
                epochs = reject_if_single_event_type(epochs, self.data_types + ["Control"])
                
                # first_samp_correct_events = events.copy()
                # first_samp_correct_events[:,0] = events[:,0] - raw_haemo._first_samps
                # raw_haemo.apply_function(apply_baseline_correction, picks="hbo", times=raw_haemo.times, sfreq=raw_haemo.info["sfreq"], events=first_samp_correct_events, stimulus_duration=self.stimulus_duration, annotations = self.annotation_names)
                # raw_haemo.apply_function(apply_baseline_correction, picks="hbr", times=raw_haemo.times, sfreq=raw_haemo.info["sfreq"], events=first_samp_correct_events, stimulus_duration=self.stimulus_duration, annotations = self.annotation_names)
                
                self.drop_log.append(epochs.drop_log)
                if len(epochs) != 0:
                    # Apply custom baseline correction if needed
                    # if self.baseline_correction != "xSecondsBefore":
                    #     corrector = baselineCorrection(self.baseline_correction)
                    #     epochs = corrector.apply_correction(
                    #         self.baseline_correction,
                    #         epochs,
                    #         data_types=self.data_types,
                    #     )

                    self.all_raw_epochs.append(raw_epochs)
                    self.all_epochs.append(epochs)
                    self.all_control.append(epochs["Control"].get_data(copy=True))
                    
                    Participant_i = individual_participant_class(f"{patient_name}".replace("-", ""))
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
                    self.number_of_participants += 1
                
                else:
                    print(f"No valid epochs for participant {patient_name}, skipping.")
                    self.folder_errors.append(f"No epochs remaining for {folder_name}.")
                
            except FileNotFoundError as e:
                print(f"Error loading {folder_name}: {e}")
                self.folder_errors.append(f"Unexpected error with {folder_name}: {e}")
            except Exception as e:
                print(f"Unexpected error with {folder_name}: {e}")
                self.folder_errors.append(f"Unexpected error with {folder_name}: {e}")

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
        print(self.folder_errors)
        print(len(self.Individual_participants))
        print(np.mean(all_data["Control"]))
        return self.all_epochs, self.data_name, all_data, all_freq, self.data_types, self.Individual_participants

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
                all_bad_channels = sorted(list(set(snr_bad_channels_long_only + sci_bad_channels)))       
                raw_od.info["bads"] = all_bad_channels
            
                if self.interpolate_bad_channels:
                    raw_od.interpolate_bads(method={"fnirs":"nearest"})
                
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
                all_bad_channels = sorted(list(set(snr_bad_channels_long_only + sci_bad_channels))) 
                raw_od.info["bads"] = all_bad_channels
            
                if self.interpolate_bad_channels:
                    raw_od.interpolate_bads(method={"fnirs":"nearest"})
                
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
        self.annotation_names = {
                                "0": "Control",
                                "1": "TonguePhysical",
                                "3": "TongueIM",
                                "4": "n_back/0_back",
                                "5": "n_back/1_back",
                                "6": "n_back/2_back",
                                "7": "n_back/3_back",
                                }
        self.standard_event_ids = {
        }
        self.file_path = Path(os.getenv(data_name.replace(" ", "_").replace("-", "_").replace(":", "")))
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = {
                                "0": 15,
                                "1": 15,
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
        self.unwanted = ["1", "3"]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        self.snr_rejection = snr_rejection
        self.snr_threshold = snr_threshold
        self.apply_tddr = apply_tddr
        self.subjects_to_exclude = {"EEG fNIRS HC baseline data": ["C5", "C7", "C8", "C9"], # "C27", "C19"
                                    "EEG fNIRS HC follow up data": [],
                                    "EEG fNIRS patient baseline data": ["P6", "P9", "P10", "P11", "P27"], # , , "P28", "P12""P15"
                                    "EEG fNIRS patient follow up data": []
                                    }
        self.folder_errors = []
        self.age_file = Path(os.getenv("demographic_data_path".replace(" ", "_").replace("-", "_")))
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
            creation_times = [snirf_file.split("\\")[-1].replace(".snirf", "")[-3:] for snirf_file in snirf_files]
            snirf_file = snirf_files[np.argmax(creation_times)]  #  Find the last created .snirf file found
            snirf_file_folder = snirf_file[:-(len(snirf_file.split("\\")[-1])+1)]
            return snirf_file_folder
        return None
    
    def find_excel_file(self, folder_path):
        """
        Find the .xlsx file in the nested folder structure.
        Returns the full path to the .xlsx file or None if not found.
        """
        # Look for .xlsx files recursively in the folder
        excel_files = glob.glob(os.path.join(folder_path, "**", "*.xlsx"), recursive=True)
        if excel_files:
            creation_times = [excel_file.split("\\")[-1].replace(".xlsx", "")[-4:] for excel_file in excel_files]
            return excel_files[np.argmax(creation_times)]  # Return the last created  .xlsx file found
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
        
        raw_intensity = mne.io.read_raw_nirx(snirf_file_path, verbose=True, preload=True)
        
        return raw_intensity
    
    def get_actual_event(self, df, sheets, events, sfreq):
        actual_events = np.empty((0, 3), dtype=int)
        is_sorted = []
        for sheet in sheets:
            generator = (item for item in df[sheet].columns if "started_mean" in item)
            time_column = next(generator, None)
            df[sheet] = df[sheet].sort_values(by=time_column).reset_index(drop=True)
            assert df[sheet]["order"].dropna().is_monotonic_increasing
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
                    if "0_back" in self.annotation_names[str(int(markers[0]))]:
                        actual_events = np.vstack([actual_events, np.array([int(int(time*sfreq)-15*sfreq), int(0), int(0)])]) # Add control/baseline/rest before active task
                    if marker == 2: marker = 0
                    actual_events = np.vstack([actual_events, np.array([int(time*sfreq), int(0), int(marker)])])
            except:
                print("No markers available for the events")
                # Ensure that there is maximally 3 seconds between the onset trigger and the first letter shown
                
                differences = [times[0]-actual_events[:,0][-1] / sfreq, 3] # Compute the actual time between the onset trigger and the first letter shown
                actual_events[-1][0] = int((times[0] - differences[np.argmin(differences)]) * sfreq) # Ensure maxmially 3 sec. between onset trigger and first letter shown
                # Add stimulus duration
                if self.stimulus_duration[str(actual_events[:,2][-1])] == 0:
                    self.stimulus_duration[str(actual_events[:,2][-1])] = times[-1] - (actual_events[:,0] / sfreq)[-1] + 3 # We add 3 as the compute the time from the last trigger, but the duration of this event has to be accounted for
                if "0_back" not in self.annotation_names[str(actual_events[-1][2])]:
                    actual_events = np.vstack([actual_events, np.array([int(actual_events[-1][0]-15*sfreq), int(0), int(0)])]) # Add control/baseline/rest before active task
            print("Sucessfully added all events")
        return actual_events
    
    def make_annotations(self, excel_path, raw_intensity, events):
        df = pd.read_excel(excel_path, sheet_name=None)
        sheets = list(df.keys())
        sfreq = raw_intensity.info["sfreq"]
        actual_events = self.get_actual_event(df, sheets, events, sfreq)
        onset = actual_events[:, 0] / sfreq
        duration = np.array([self.stimulus_duration[str(event)] for event in actual_events[:,2]])
        description = actual_events[:, 2].astype(str)
        new_annotations = mne.Annotations(onset=onset, 
                                 duration=duration, 
                                 description=description,
                                 orig_time=raw_intensity.annotations.orig_time
                                 )
        for key, value in self.annotation_names.items():
            if key in np.unique(new_annotations.description):
                new_annotations.rename({key: value})
        raw_intensity.set_annotations(new_annotations)
        return raw_intensity
    
    def crop_data(self, raw_intensity):
        events, event_dict = mne.events_from_annotations(raw_intensity, self.standard_event_ids)
        sfreq = raw_intensity.info["sfreq"]
        new_tmin = max(events[0][0] / sfreq - 10, 0) #Always ensure the tmin is non-negative.
        new_tmax = None #events[-1][0] / sfreq + self.stimulus_duration[str(events[-1][2])] + 3
        raw_intensity = raw_intensity.crop(tmin=new_tmin, tmax=new_tmax)
        assert len(events) == len(raw_intensity.annotations)
        return raw_intensity

    def load_data(self):
        ages = load_ages(self.age_file)
        all_folders = [f for f in sorted(os.listdir(self.file_path)) 
                    if os.path.isdir(os.path.join(self.file_path, f))]
        
        for i, folder_name in enumerate(all_folders, start=1):
            patient_name = f"{folder_name[:3]}".replace("-", "")
            if patient_name in self.subjects_to_exclude[self.data_name]:
                continue
            try:
                excel_path = self.find_excel_file(os.path.join(self.file_path, folder_name))
                raw_intensity = self.define_raw_intensity(folder_name)
                raw_intensity.annotations.description = np.array([anno.split(".")[0] for anno in raw_intensity.annotations.description])
                events, event_dict = mne.events_from_annotations(raw_intensity) # We extract original events before removing unwanted, as we need the original for making new annotations
                raw_intensity = self.make_annotations(excel_path, raw_intensity, events)
                self.standard_event_ids = {value: int(float(key)) for key, value in self.annotation_names.items()}
                try:
                    birthday = datetime.strptime(ages[folder_name[:2]][:6], "%d%m%y")
                    if birthday.year > datetime.now().year:
                        birthday = birthday.replace(birthday.year - 100)
                    raw_intensity.info["subject_info"]["birthday"] = birthday
                except:
                    print("No age data available for participant")
                self.tmax = max(self.stimulus_duration.values())     
                raw_intensity = self.crop_data(raw_intensity)
                
                # fig = raw_intensity.plot_sensors()
                # plt.savefig(f"sensors_{folder_name}.jpg")
                
                raw_intensity_long = mne_nirs.channels.get_long_channels(raw_intensity)
                raw_intensity_short = mne_nirs.channels.get_short_channels(raw_intensity)
                
                if self.snr_rejection != "None":
                    snr = snr_rejection(raw_intensity_long, self.snr_rejection)
                    # Validation
                    if self.snr_rejection == "SNR" and self.snr_threshold < 1:
                        raise ValueError("Currently the classic signal to noise ratio is used but the threshold for SNR is below 1 resulting in all channels being marked as bad. Please set the threshold to a value above 1.")
                    if self.snr_rejection == "CV" and self.snr_threshold > 1:
                        raise ValueError("Currently the coefficient of variation is used but the threshold for CV is above 1 resulting in all channels being marked as bad. Please set the threshold to a value below 1.")
                    # Get bad channels based on pair logic
                    snr_bad_channels = get_bad_channels_by_pairs(raw_intensity_long.ch_names, snr, self.snr_threshold, self.snr_rejection)
                    raw_intensity_long.info["bads"] = snr_bad_channels
                else:
                    snr_bad_channels = []

                raw_od = mne.preprocessing.nirs.optical_density(raw_intensity_long)
                dpf = compute_differential_pathlength(raw_od)
                raw_od_short = mne.preprocessing.nirs.optical_density(raw_intensity_short)
                raw_od_original = raw_od.copy()

                sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od_short, l_freq=0.5, h_freq=2.5)
                sci_bad_channels = list(compress(raw_od_short.ch_names, sci < self.scalp_coupling_threshold))
                raw_od_short.info["bads"] = sci_bad_channels
                
                if self.apply_tddr:
                    raw_od_short = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od_short)
                    raw_od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)
                
                if self.interpolate_bad_channels:
                    raw_od.interpolate_bads(method={"fnirs":"nearest"})
                
                raw_od.add_channels([raw_od_short])

                raw_haemo_unfiltered = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=dpf).copy()
                # raw_haemo_unfiltered._data *= 1e6
                                
                # from mne.io.constants import FIFF
                # for ch in raw_haemo_unfiltered.info['chs']:
                #     if ch['kind'] == FIFF.FIFFV_FNIRS_CH:
                #         ch['unit_mul'] = FIFF.FIFF_UNITM_MU
                
                if self.short_channel_correction:
                    raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)

                raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=dpf)
                # raw_haemo._data *= 1e6

                # for ch in raw_haemo_unfiltered.info['chs']:
                #     if ch['kind'] == FIFF.FIFFV_FNIRS_CH:
                #         ch['unit_mul'] = FIFF.FIFF_UNITM_MU
                
                raw_haemo.filter(self.filter_lower_value, self.filter_upper_value, h_trans_bandwidth=self.h_trans_bandwidth, l_trans_bandwidth=self.l_trans_bandwidth)
                
                if self.negative_correlation_enhancement:
                    raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)
                
                events, event_dict = mne.events_from_annotations(raw_haemo, self.standard_event_ids)
                        
                # Set baseline parameter based on correction method
                baseline = self.baseline if self.baseline_correction == "xSecondsBefore" else None
                
                raw_epochs = epochs = mne.Epochs(raw_haemo_unfiltered, events, event_id=event_dict, tmin=self.tmin, tmax=self.tmax, reject=None, reject_by_annotation=None, proj=False, baseline=None, preload=True, detrend=None, verbose=True)

                sum_method = lambda data: np.sum(data, axis=0)
                raw_tmp = raw_haemo.copy()
                chromophores = ["hbo", "hbr"]
                groups = {chromo: [i for i, ch in enumerate(raw_tmp.ch_names) if ch[-3:] == chromo] for chromo in chromophores}
                raw_mean = mne.channels.combine_channels(
                raw_tmp, 
                groups=groups, 
                method="mean")
                for ch in raw_mean.info["chs"]:
                    if ch["kind"] == mne.io.constants.FIFF.FIFFV_FNIRS_CH:
                        ch["coil_type"] = mne.io.constants.FIFF.FIFFV_COIL_FNIRS_HBO # We set the channel types to HbO to allow combination
                groups = {"hbt": [0, 1]}
                raw_HbT = mne.channels.combine_channels(
                raw_mean, 
                groups=groups, 
                method=sum_method)
                glm_raw = raw_HbT.copy()
                # Get the HbT data
                hbt_data = glm_raw.get_data()
                hbt_data = np.tile(hbt_data, (8, 1))

                # Create channel info for HbT
                orig_ch_names = [ch for ch in mne_nirs.channels.get_long_channels(raw_haemo.copy().pick("hbo")).ch_names]
                hbt_ch_names = [name.replace("hbo", "hbt") for name in orig_ch_names]
                hbt_info = mne.create_info(
                    ch_names=hbt_ch_names,
                    sfreq=raw_haemo.info['sfreq'],
                )

                # Copy relevant info from raw_haemo
                hbt_info['subject_info'] = raw_haemo.info.get('subject_info', None)

                # Create a new Raw object with HbT
                info_channel = mne_nirs.channels.get_long_channels(raw_haemo.copy()).info["chs"][0]
                hbt_raw = mne.io.RawArray(hbt_data, hbt_info)
                for ch in hbt_raw.info["chs"]:
                    ch["coil_type"] = info_channel["coil_type"]
                    ch["unit"] = info_channel["unit"]
                    ch["unit_mul"] = info_channel["unit_mul"]
                    ch["kind"] = info_channel["kind"]                    

                # Now add it to raw_haemo
                raw_haemo = raw_haemo.add_channels([hbt_raw], force_update_info=True)

                Participant_i = individual_participant_class(f"{patient_name}".replace("-", ""))
                Participant_i.raw_intensity = raw_intensity
                Participant_i.raw_od = raw_od
                Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                Participant_i.raw_haemo = raw_haemo
                self.number_of_participants += 1
                self.reject_criteria = compute_p2p(raw_epochs, self.data_types+["Control"], 95)
                epochs = mne.Epochs(
                    raw_haemo,
                    events,
                    event_id=event_dict,
                    tmin=self.tmin,
                    tmax=self.tmax,
                    reject=None,#self.reject_criteria,
                    reject_by_annotation=None,
                    proj=False,
                    baseline=None,
                    preload=True,
                    detrend=None,
                    verbose=True,
                )

                # first_samp_correct_events = events.copy()
                # first_samp_correct_events[:,0] = events[:,0] - raw_haemo._first_samps
                # raw_haemo.apply_function(apply_baseline_correction, picks="hbo", times=raw_haemo.times, sfreq=raw_haemo.info["sfreq"], events=first_samp_correct_events, stimulus_duration=self.stimulus_duration, annotations = self.annotation_names)
                # raw_haemo.apply_function(apply_baseline_correction, picks="hbr", times=raw_haemo.times, sfreq=raw_haemo.info["sfreq"], events=first_samp_correct_events, stimulus_duration=self.stimulus_duration, annotations = self.annotation_names)
                        
                self.drop_log.append(epochs.drop_log)
                if len(epochs) == 0:
                    print("Debug")
                if len(epochs) != 0:
                    # Apply custom baseline correction if needed
                    # if self.baseline_correction != "xSecondsBefore":
                    #     corrector = baselineCorrection(self.baseline_correction)
                        # epochs = corrector.apply_correction(
                        #     self.baseline_correction,
                        #     epochs,
                        #     data_types=self.data_types,
                        # )

                    self.all_raw_epochs.append(raw_epochs)
                    self.all_epochs.append(epochs)
                    self.all_control.append(epochs["Control"].get_data(copy=True))
                    Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                    Participant_i.raw_epochs = raw_epochs
                    Participant_i.epochs = epochs
                    
                    for name in self.data_types:
                        getattr(self, f'all_{name}').append(epochs[name].get_data(copy=True))
                        Participant_i.events.update({name: epochs[name].get_data(copy=True)})
                                                
                else:
                    print(f"No valid epochs for participant {patient_name}, skipping.")
                    self.folder_errors.append(f"No epochs remaining for {folder_name}.")
                getattr(self, 'Individual_participants').append(Participant_i)
            
            except FileNotFoundError as e:
                print(f"Error loading {folder_name}: {e}")
                self.folder_errors.append(f"Unexpected error with {folder_name}: {e}")
            except Exception as e:
                print(f"Unexpected error with {folder_name}: {e}")
                self.folder_errors.append(f"Unexpected error with {folder_name}: {e}")                

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

fNIRS_EEG_HC_follow_up_data_load = fNIRS_EEG_HC_baseline_data_load

###############################################################################################################################################################################################

fNIRS_EEG_patient_baseline_data_load = fNIRS_EEG_HC_baseline_data_load

###############################################################################################################################################################################################

fNIRS_EEG_patient_follow_up_data_load = fNIRS_EEG_HC_baseline_data_load

###############################################################################################################################################################################################

class fNIRS_EEG_Marwan_data_load(fNIRS_data_load):
    def __init__(self, data_name: str = "Marwan fNIRS data", file_path: str = 'Marwan_fNIRS_data', short_channel_correction: bool = True, negative_correlation_enhancement: bool = False, interpolate_bad_channels:bool=False, tmin:int = 0,
                 baseline_correction: str = "Previous rest period", filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2,
                 l_trans_bandwidth: float = 0.02, reject_criteria: dict = dict(hbo=80e-6), scalp_coupling_threshold: float = 0.8, snr_rejection: bool = False,
                 snr_threshold: float = 8.0, apply_tddr: bool = False):
        self.number_of_participants = 0
        self.all_tapping = []
        self.all_Control = []
        self.annotation_names = {
                                "0": "Control",
                                '4': 'Math',
                                '5': 'Math',
                                '6': 'Math',
                                '7': 'Math',
                                '8': 'Math',
                                '9': 'Hard_Math',
                                '10': 'Hard_Math',
                                '11': 'Hard_Math',
                                '12': 'Hard_Math',
                                '13': 'Hard_Math'}
        self.standard_event_ids = {
        }
        key = file_path.replace(":", "").replace(" ", "_").replace("-", "_")
        env_value = config.get(key)
        if env_value:
            self.file_path = Path(env_value)
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = {
                                'Control': 20,
                                'Math': 25,
                                'Hard_Math': 25,
                                }
        self.scalp_coupling_threshold = scalp_coupling_threshold
        self.reject_criteria = reject_criteria
        self.tmin = tmin
        self.tmax = 20
        self.baseline = (None, 0)
        self.data_types = ['Math', 'Hard_Math']
        self.data_name = data_name
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = ["1", "2", "3"]
        self.baseline_correction = baseline_correction
        self.filter_lower_value = filter_lower_value
        self.filter_upper_value = filter_upper_value
        self.h_trans_bandwidth = h_trans_bandwidth
        self.l_trans_bandwidth = l_trans_bandwidth
        self.snr_rejection = snr_rejection
        self.snr_threshold = snr_threshold
        self.apply_tddr = apply_tddr
        self.subjects_to_exclude = {"fNIRS EEG Marwan data load": ["P20", "P29", "P9"], 
                                    }
        self.folder_errors = []
        self.age_file = Path(os.getenv("demographic_data_path_Marwan".replace(" ", "_").replace("-", "_")))
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
            creation_times = [snirf_file.split("\\")[-1].replace(".snirf", "")[-3:] for snirf_file in snirf_files]
            snirf_file = snirf_files[np.argmax(creation_times)]  #  Find the last created .snirf file found
            snirf_file_folder = snirf_file[:-(len(snirf_file.split("\\")[-1])+1)]
            return snirf_file_folder
        return None
    
    def find_excel_file(self, folder_path):
        """
        Find the .xlsx file in the nested folder structure.
        Returns the full path to the .xlsx file or None if not found.
        """
        # Look for .xlsx files recursively in the folder
        excel_files = glob.glob(os.path.join(folder_path, "**", "*.xlsx"), recursive=True)
        if excel_files:
            creation_times = [excel_file.split("\\")[-1].replace(".xlsx", "")[-4:] for excel_file in excel_files]
            return excel_files[np.argmax(creation_times)]  # Return the last created  .xlsx file found
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
        
        raw_intensity = mne.io.read_raw_nirx(snirf_file_path, verbose=True, preload=True)
        
        return raw_intensity
    
    def make_annotations(self, raw_intensity):
        cropped_raw_data = raw_intensity.copy()
        for key, value in self.annotation_names.items():
            if key in np.unique(cropped_raw_data.annotations.description):
                cropped_raw_data.annotations.rename({key: value})
                cropped_raw_data.annotations.set_durations({value: self.stimulus_duration[value]})
        new_onsets = list(cropped_raw_data.annotations.onset.copy())
        new_durations = list(cropped_raw_data.annotations.duration.copy())
        new_descriptions = list(cropped_raw_data.annotations.description.copy())
        last_math_onset = np.max([an["onset"] for an in cropped_raw_data.annotations if an["description"] == "Math"])
        last_hard_math_onset = np.max([an["onset"] for an in cropped_raw_data.annotations if an["description"] == "Hard_Math"])
        new_onsets.append(last_math_onset + self.stimulus_duration["Math"])
        new_durations.append(self.stimulus_duration["Control"])
        new_descriptions.append("Control")
        new_onsets.append(last_hard_math_onset + self.stimulus_duration["Hard_Math"])
        new_durations.append(self.stimulus_duration["Control"])
        new_descriptions.append("Control")
        for annotation in cropped_raw_data.annotations:
            if annotation["description"] in list(self.annotation_names.values()):
                new_onsets.append(annotation["onset"] - self.stimulus_duration["Control"]) #+ self.stimulus_duration[annotation["description"]])
                new_durations.append(self.stimulus_duration["Control"])
                new_descriptions.append("Control")
        new_annotations = mne.Annotations(onset = new_onsets, duration = new_durations, description = new_descriptions)
        cropped_raw_data.set_annotations(new_annotations)
        return cropped_raw_data
    
    def crop_data(self, raw_intensity):
        events, event_dict = mne.events_from_annotations(raw_intensity, self.standard_event_ids)
        sfreq = raw_intensity.info["sfreq"]
        new_tmin = max(events[0][0] / sfreq - 10, 0) #Always ensure the tmin is non-negative.
        new_tmax = None #events[-1][0] / sfreq + self.stimulus_duration[str(events[-1][2])] + 3
        raw_intensity = raw_intensity.crop(tmin=new_tmin, tmax=new_tmax)
        assert len(events) == len(raw_intensity.annotations)
        return raw_intensity
    
    def replace(self, raw_intensity):

        # Create a mapping from incorrect to correct channel names
        channel_mapping = {
            'S4_D4': 'S4_D6',   # Raw -> Standard
            'S5_D5': 'S5_D4',
            'S6_D5': 'S6_D4',
            'S6_D6': 'S6_D5',
            'S7_D6': 'S7_D5',
            'S8_D6': 'S8_D5',
            'S8_D4': 'S8_D6',
        }

        # Rename channels in your raw data
        raw_intensity_corrected = raw_intensity.copy()

        # Build new channel names
        new_ch_names = []
        for ch_name in raw_intensity_corrected.ch_names:
            base_name = ch_name.split()[0]  # Get 'S1_D1' part
            wavelength = ch_name.split()[1]  # Get '760' or '850' part
            
            # Apply mapping if needed
            if base_name in channel_mapping:
                base_name = channel_mapping[base_name]
            
            new_ch_names.append(f"{base_name} {wavelength}")

        # Rename the channels
        raw_intensity_corrected.rename_channels(dict(zip(raw_intensity.ch_names, new_ch_names)))

        print("New channel names:")
        print([ch.split()[0] for ch in raw_intensity_corrected.ch_names[::2]])

        # Now create and apply the montage
        sources = {}
        detectors = {}

        optodes_file_name = config.get("standard_optodes")
        with open(optodes_file_name, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                label = parts[0]
                coords = np.array([float(parts[1]), float(parts[2]), float(parts[3])]) / 1000
                
                if label.startswith('S'):
                    new_label = 'S' + str(int(label[1:]))
                    sources[new_label] = coords
                elif label.startswith('D'):
                    new_label = 'D' + str(int(label[1:]))
                    detectors[new_label] = coords

        fiducials = {}

        digpts_filename = config.get("digpts")
        with open(digpts_filename, 'r') as f:
            for line in f:
                if ':' in line:
                    parts = line.strip().split(':')
                    label = parts[0].strip().lower()
                    coords = np.array([float(x) for x in parts[1].strip().split()]) / 1000
                    
                    if label in ['nz', 'al', 'ar', 'cz', 'iz']:
                        fiducials[label] = coords

        montage = mne.channels.make_dig_montage(
            ch_pos={**sources, **detectors},
            nasion=fiducials.get('nz'),
            lpa=fiducials.get('al'),
            rpa=fiducials.get('ar'),
            coord_frame='unknown'
        )

        # Apply montage to corrected data
        raw_intensity_corrected.set_montage(montage)
        print("\nMontage successfully applied!")
        return raw_intensity_corrected

    def load_data(self):
        ages = load_ages(self.age_file)
        all_folders = list(np.concatenate([
                sorted([patient_path.name + "/" + session_path.name  + "/" +  subsubfolder.name
                for subsubfolder in (self.file_path / Path(patient_path) / Path(session_path)).iterdir()
                if subsubfolder.is_dir()
                ])
                for patient_path in self.file_path.iterdir()
                if patient_path.is_dir()
                for session_path in (self.file_path / Path(patient_path)).iterdir()
                if session_path.is_dir()
                ]).flatten())
        
        for i, folder_name in enumerate(all_folders, start=1):
            patient_name = folder_name[0] + folder_name[folder_name.find("ID")+2:folder_name.find("ID")+4].replace("_", "") + "_" + folder_name.split("/")[1][0] + folder_name.split("/")[1][-1] + "_" + folder_name.split("/")[2][0]
            if folder_name.split("/")[2].split("_")[-1] in ["1", "2", "3"]:
                patient_name += folder_name.split("/")[2].split("_")[-1]
            if patient_name.endswith("P") or patient_name[1] == ("P"):
                print("ERROR")
            if patient_name[:3] in self.subjects_to_exclude[self.data_name]:
                continue
            if not patient_name[:3] in ["P3_", "P9_"]:
                continue
            try:
                raw_intensity = self.define_raw_intensity(folder_name)
                if len(raw_intensity.annotations.description) < 13:
                    if raw_intensity.annotations.description.astype(float).astype(int).min() != 4 or raw_intensity.annotations.description.astype(float).astype(int).max() != 13:
                        print(f"Unexpected error with {folder_name}: Not all repetitions are present")
                        self.folder_errors.append(f"Unexpected error with {folder_name}: Not all repetitions are present")
                        continue
                raw_intensity = self.replace(raw_intensity)
                raw_intensity.annotations.description = np.array([anno.split(".")[0] for anno in raw_intensity.annotations.description])
                for _unwanted in self.unwanted:
                    unwanted = np.nonzero(raw_intensity.annotations.description == _unwanted)
                    raw_intensity.annotations.delete(unwanted)

                raw_intensity = self.make_annotations(raw_intensity)
                self.standard_event_ids = {value: int(float(key)) for key, value in self.annotation_names.items()}
                if len(np.unique(raw_intensity.annotations.description)) < (len(self.data_types)+1):
                    print(f"Unexpected error with {folder_name}: Not all events are present")
                    self.folder_errors.append(f"Unexpected error with {folder_name}: Not all events are present")
                    continue
                if len(raw_intensity.ch_names) != 46:
                    print(f"Unexpected error with {folder_name}: Different number of channels ({len(raw_intensity.ch_names)}) present")
                    self.folder_errors.append(f"Unexpected error with {folder_name}: Different number of channels ({len(raw_intensity.ch_names)}) present")
                    continue
                try:
                    birthday = datetime.strptime(str(ages[folder_name[0] + folder_name[folder_name.find("ID")+2:folder_name.find("ID")+4].replace("_", "")]), "%d%m%y")
                    if birthday.year > datetime.now().year:
                        birthday = birthday.replace(birthday.year - 100)
                    raw_intensity.info["subject_info"]["birthday"] = birthday
                except:
                    print("No age data available for participant")
                self.tmax = max(self.stimulus_duration.values())
                if len(raw_intensity.annotations.description) < 22:
                    print(f"WAIT WHAT WHAT?!")
                # raw_intensity = self.crop_data(raw_intensity)
                if len(raw_intensity.annotations.description) < 22:
                    print(f"WAIT WHAT?!")

                raw_intensity_long = mne_nirs.channels.get_long_channels(raw_intensity)
                raw_intensity_short = mne_nirs.channels.get_short_channels(raw_intensity)
                
                if self.snr_rejection != "None":
                    snr = snr_rejection(raw_intensity_long, self.snr_rejection)
                    # Validation
                    if self.snr_rejection == "SNR" and self.snr_threshold < 1:
                        raise ValueError("Currently the classic signal to noise ratio is used but the threshold for SNR is below 1 resulting in all channels being marked as bad. Please set the threshold to a value above 1.")
                    if self.snr_rejection == "CV" and self.snr_threshold > 1:
                        raise ValueError("Currently the coefficient of variation is used but the threshold for CV is above 1 resulting in all channels being marked as bad. Please set the threshold to a value below 1.")
                    # Get bad channels based on pair logic
                    snr_bad_channels = get_bad_channels_by_pairs(raw_intensity_long.ch_names, snr, self.snr_threshold, self.snr_rejection)
                    raw_intensity_long.info["bads"] = snr_bad_channels
                else:
                    snr_bad_channels = []

                raw_od = mne.preprocessing.nirs.optical_density(raw_intensity_long)
                dpf = compute_differential_pathlength(raw_od)
                raw_od_short = mne.preprocessing.nirs.optical_density(raw_intensity_short)
                raw_od_original = raw_od.copy()

                sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od_short, l_freq=0.5, h_freq=2.5)
                sci_bad_channels = list(compress(raw_od_short.ch_names, sci < self.scalp_coupling_threshold))
                raw_od_short.info["bads"] = sci_bad_channels

                if self.apply_tddr:
                    raw_od_short = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od_short)
                    raw_od = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)
                    
                if self.interpolate_bad_channels:
                    raw_od.interpolate_bads(method={"fnirs":"nearest"})
                
                raw_od.add_channels([raw_od_short])

                raw_haemo_unfiltered = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=dpf).copy()

                # raw_haemo_unfiltered._data *= 1e6
                
                # from mne.io.constants import FIFF
                # for ch in raw_haemo_unfiltered.info['chs']:
                #     if ch['kind'] == FIFF.FIFFV_FNIRS_CH:
                #         ch['unit_mul'] = FIFF.FIFF_UNITM_MU  # Set unit to micromolar
                                
                if self.short_channel_correction:
                    raw_od = mne_nirs.signal_enhancement.short_channel_regression(raw_od)

                raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=dpf)
                # raw_haemo._data *= 1e6
                # for ch in raw_haemo.info['chs']:
                #     if ch['kind'] == FIFF.FIFFV_FNIRS_CH:
                #         ch['unit_mul'] = FIFF.FIFF_UNITM_MU  # Set unit to micromolar
                
                raw_haemo.filter(self.filter_lower_value, self.filter_upper_value, h_trans_bandwidth=self.h_trans_bandwidth, l_trans_bandwidth=self.l_trans_bandwidth)
                
                if self.negative_correlation_enhancement:
                    raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)
                
                events, event_dict = mne.events_from_annotations(raw_haemo, self.standard_event_ids)
                        
                # Set baseline parameter based on correction method
                baseline = self.baseline if self.baseline_correction == "xSecondsBefore" else None
                
                raw_epochs = mne.Epochs(raw_haemo_unfiltered, events, event_id=event_dict, tmin=self.tmin, tmax=self.tmax, reject=None, reject_by_annotation=None, proj=False, baseline=None, preload=True, detrend=None, verbose=True)


                sum_method = lambda data: np.sum(data, axis=0)
                raw_tmp = raw_haemo.copy()
                chromophores = ["hbo", "hbr"]
                groups = {chromo: [i for i, ch in enumerate(raw_tmp.ch_names) if ch[-3:] == chromo] for chromo in chromophores}
                raw_mean = mne.channels.combine_channels(
                raw_tmp, 
                groups=groups, 
                method="mean")
                for ch in raw_mean.info["chs"]:
                    if ch["kind"] == mne.io.constants.FIFF.FIFFV_FNIRS_CH:
                        ch["coil_type"] = mne.io.constants.FIFF.FIFFV_COIL_FNIRS_HBO # We set the channel types to HbO to allow combination
                groups = {"hbt": [0, 1]}
                raw_HbT = mne.channels.combine_channels(
                raw_mean, 
                groups=groups, 
                method=sum_method)
                glm_raw = raw_HbT.copy()
                # Get the HbT data
                hbt_data = glm_raw.get_data()
                hbt_data = np.tile(hbt_data, (15, 1))

                # Create channel info for HbT
                orig_ch_names = [ch for ch in mne_nirs.channels.get_long_channels(raw_haemo.copy().pick("hbo")).ch_names]
                hbt_ch_names = [name.replace("hbo", "hbt") for name in orig_ch_names]
                hbt_info = mne.create_info(
                    ch_names=hbt_ch_names,
                    sfreq=raw_haemo.info['sfreq'],
                )

                # Copy relevant info from raw_haemo
                hbt_info['subject_info'] = raw_haemo.info.get('subject_info', None)

                # Create a new Raw object with HbT
                info_channel = mne_nirs.channels.get_long_channels(raw_haemo.copy()).info["chs"][0]
                hbt_raw = mne.io.RawArray(hbt_data, hbt_info)
                for ch in hbt_raw.info["chs"]:
                    ch["coil_type"] = info_channel["coil_type"]
                    ch["unit"] = info_channel["unit"]
                    ch["unit_mul"] = info_channel["unit_mul"]
                    ch["kind"] = info_channel["kind"]                    

                # # Now add it to raw_haemo
                raw_haemo = raw_haemo.add_channels([hbt_raw], force_update_info=True)
                names = [ind.name for ind in self.Individual_participants]
                if f"{patient_name}".replace("-", "") in names:
                    print("debug")
                Participant_i = individual_participant_class(f"{patient_name}".replace("-", ""))
                Participant_i.raw_intensity = raw_intensity
                Participant_i.raw_od = raw_od
                Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                Participant_i.raw_haemo = raw_haemo
                self.number_of_participants += 1
                # self.reject_criteria = compute_p2p(raw_epochs, self.data_types+["Control"], 95)
                epochs = mne.Epochs(
                    raw_haemo,
                    events,
                    event_id=event_dict,
                    tmin=self.tmin,
                    tmax=self.tmax,
                    reject=None,#self.reject_criteria,
                    reject_by_annotation=None,
                    proj=False,
                    baseline=None,
                    preload=True,
                    detrend=None,
                    verbose=True,
                )

                # first_samp_correct_events = events.copy()
                # first_samp_correct_events[:,0] = events[:,0] - raw_haemo._first_samps
                # raw_haemo.apply_function(apply_baseline_correction, picks="hbo", times=raw_haemo.times, sfreq=raw_haemo.info["sfreq"], events=first_samp_correct_events, stimulus_duration=self.stimulus_duration, annotations = self.annotation_names)
                # raw_haemo.apply_function(apply_baseline_correction, picks="hbr", times=raw_haemo.times, sfreq=raw_haemo.info["sfreq"], events=first_samp_correct_events, stimulus_duration=self.stimulus_duration, annotations = self.annotation_names)
                if len(epochs) < 22:
                    print(f"WAIT WHAT WHAT?!")
                self.drop_log.append(epochs.drop_log)
                if len(epochs) != 0:
                    # Apply custom baseline correction if needed
                    # if self.baseline_correction != "xSecondsBefore":
                    #     corrector = baselineCorrection(self.baseline_correction)
                        # epochs = corrector.apply_correction(
                        #     self.baseline_correction,
                        #     epochs,
                        #     data_types=self.data_types,
                        # )

                    self.all_raw_epochs.append(raw_epochs)
                    self.all_epochs.append(epochs)
                    self.all_control.append(epochs["Control"].get_data(copy=True))
                    Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                    Participant_i.raw_epochs = raw_epochs
                    Participant_i.epochs = epochs
                    
                    for name in self.data_types:
                        getattr(self, f'all_{name}').append(epochs[name].get_data(copy=True))
                        Participant_i.events.update({name: epochs[name].get_data(copy=True)})
                                                
                else:
                    print(f"No valid epochs for participant {patient_name}, skipping.")
                    self.folder_errors.append(f"No epochs remaining for {folder_name}.")
                getattr(self, 'Individual_participants').append(Participant_i)
            
            except FileNotFoundError as e:
                print(f"Error loading {folder_name}: {e}")
                self.folder_errors.append(f"Unexpected error with {folder_name}: {e}")
            except Exception as e:
                print(f"Unexpected error with {folder_name}: {e}")
                self.folder_errors.append(f"Unexpected error with {folder_name}: {e}")                

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
        names = [ind.name for ind in self.Individual_participants]
        patient_ids = np.unique([name.split("_")[0] for name in names])
        patient_sessions = np.unique([name.replace("_" + name.split("_")[2], "") for name in names])
        patient_recordings = [name.split("_")[0] for name in names]

        # Count sessions per patient
        session_counts = pd.Series(patient_sessions).str.split('_').str[0].value_counts()

        # Count recordings per patient
        recording_counts = pd.Series(patient_recordings).value_counts()

        # Combine into a DataFrame
        df = pd.DataFrame({
            'sessions': session_counts,
            'recordings': recording_counts
        }).sort_index()

        # Fill any missing values with 0 (in case a patient has sessions but no recordings or vice versa)
        df = df.fillna(0).astype(int)

        # if df["recordings"].sum() == len(self.all_epochs):
        #     df.to_csv(os.path.join(rf"C:\Users\NTres\OneDrive - Danmarks Tekniske Universitet\Bachelor_projekt\Results\Study_2" f"\data_overview.csv"))
        excluded_recordings = []
        for recording in self.folder_errors:
            folder_name = recording[22:].split(":")[0]
            patient_name = folder_name[0] + folder_name[folder_name.find("ID")+2:folder_name.find("ID")+4].replace("_", "") + "_" + folder_name.split("/")[1][0] + folder_name.split("/")[1][-1] + "_" + folder_name.split("/")[2][0]
            if folder_name.split("/")[2].split("_")[-1] in ["1", "2", "3"]:
                patient_name += folder_name.split("/")[2].split("_")[-1]
            excluded_recordings.append(patient_name)
        names = excluded_recordings
        patient_ids = np.unique([name.split("_")[0] for name in names])
        patient_sessions = np.unique([name.replace("_" + name.split("_")[2], "") for name in names])
        patient_recordings = [name.split("_")[0] for name in names]

        # # Count sessions per patient
        # session_counts = pd.Series(patient_sessions).str.split('_').str[0].value_counts()

        # # Count recordings per patient
        # recording_counts = pd.Series(patient_recordings).value_counts()

        # # Combine into a DataFrame
        # df_ex = pd.DataFrame({
        #     'sessions': session_counts,
        #     'recordings': recording_counts
        # }).sort_index()

        # # Fill any missing values with 0 (in case a patient has sessions but no recordings or vice versa)
        # df_ex = df_ex.fillna(0).astype(int)

        # df_ex.to_csv(os.path.join(rf"C:\Users\NTres\OneDrive - Danmarks Tekniske Universitet\Bachelor_projekt\Results\Study_2" f"\excluded_data_overview.csv"))
        return self.all_epochs, self.data_name, all_data, all_freq, self.data_types, self.Individual_participants