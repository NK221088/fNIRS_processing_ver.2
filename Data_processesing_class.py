from itertools import compress
import matplotlib.pyplot as plt
import numpy as np
import mne
import mne_nirs
import mne_bids
import os
from Participant_class import individual_participant_class
import os
import glob

class fNIRS_data_load:
    def __init__(self, file_path, number_of_participants=1, annotation_names=None, stimulus_duration=5,
                 short_channel_correction=True, negative_correlation_enhancement=True, scalp_coupling_threshold=0.8,
                 reject_criteria=dict(hbo=80e-6), tmin=-5, tmax=15, baseline=(None, 0), data_types=[], number_of_data_types=2,
                 data_name="None", individuals = False, interpolate_bad_channels=False, unwanted = "15.0"):
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
        self.individuals = individuals
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = unwanted
        if individuals:
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
            unwanted = np.nonzero(raw_intensity.annotations.description == self.unwanted)
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
            raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            epochs = mne.Epochs(
                raw_haemo,
                events,
                event_id=event_dict,
                tmin=self.tmin,
                tmax=self.tmax,
                reject=self.reject_criteria,
                reject_by_annotation=True,
                proj=True,
                baseline=self.baseline,
                preload=True,
                detrend=None,
                verbose=True,
            )

            if len(epochs) != 0:
                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
                
                if self.individuals:
                    Participant_i = individual_participant_class(f"Participant_{i}")
                    Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                    Participant_i.raw_intensity = raw_intensity
                    Participant_i.raw_od = raw_od
                    Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                    Participant_i.raw_haemo = raw_haemo
                    Participant_i.epochs = epochs
                
                for name in self.data_types:
                    getattr(self, f'all_{name}').append(epochs[name].get_data(copy=True))
                    if self.individuals:
                        Participant_i.events.update({name: epochs[name].get_data(copy=True)})
                
                if self.individuals:
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
        return self.all_epochs, self.data_name, all_data, all_freq, self.data_types, self.Individual_participants if self.individuals else None

###############################################################################################################################################################################################

class AudioSpeechNoise_data_load(fNIRS_data_load):
    def __init__(self, short_channel_correction : bool, negative_correlation_enhancement : bool, individuals :bool = False, interpolate_bad_channels:bool=False):
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
        self.scalp_coupling_threshold = 0.5
        self.reject_criteria = dict(hbo=80e-6)
        self.tmin = -5
        self.tmax = 15
        self.baseline = (None, 0)
        self.data_types = ["Speech", "Noise"]
        self.number_of_data_types = 2
        self.data_name = "AudioSpeechNoise"
        self.individuals = individuals
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = "15.0"
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
                        individuals = self.individuals,
                        interpolate_bad_channels = self.interpolate_bad_channels,
                        unwanted = self.unwanted)

    def define_raw_intensity(self, sub_id):
        fnirs_snirf_file_path = os.path.join(self.file_path, f"sub-{sub_id}", "ses-01", "nirs", f"sub-{sub_id}_ses-01_task-AudioSpeechNoise_nirs.snirf")
        raw_intensity = mne.io.read_raw_snirf(fnirs_snirf_file_path, verbose=True)
        raw_intensity.load_data()
        return raw_intensity

###############################################################################################################################################################################################

class fNIRS_motor_data_load(fNIRS_data_load):
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool, individuals :bool = False, interpolate_bad_channels:bool=False):
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
        self.scalp_coupling_threshold = 0.5  # Change this value if needed
        self.reject_criteria = dict(hbo=80e-6)
        self.tmin = -5
        self.tmax = 15
        self.baseline = (None, 0)
        self.data_types = ["Tapping"]
        self.number_of_data_types = 2
        self.data_name = "fnirs_motor_plus_anti"
        self.individuals = individuals
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = "15.0"
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
            individuals = self.individuals,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted)

    def define_raw_intensity(self, sub_id):
        fnirs_data_folder = mne.datasets.fnirs_motor.data_path()
        fnirs_cw_amplitude_dir = fnirs_data_folder / "Participant-1"
        raw_intensity = mne.io.read_raw_nirx(fnirs_cw_amplitude_dir, verbose=True)
        raw_intensity.load_data()
        return raw_intensity

###############################################################################################################################################################################################

class fNIRS_full_motor_data_load(fNIRS_data_load):
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool, individuals : bool = False, interpolate_bad_channels:bool=False):
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
        self.scalp_coupling_threshold = 0.5  # Change this value if needed
        self.reject_criteria = dict(hbo=80e-6)
        self.tmin = -5
        self.tmax = 15
        self.baseline = (None, 0)
        self.data_types = ["Tapping"]
        self.number_of_data_types = 2
        self.data_name = "fnirs_full_motor"
        self.individuals = individuals
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = "15.0"
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
            individuals = self.individuals,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted)

    def define_raw_intensity(self, sub_id):
        raw_intensity = mne.io.read_raw_snirf(f"Dataset/rob-luke/rob-luke-BIDS-NIRS-Tapping-e262df8/sub-{sub_id}/nirs/sub-{sub_id}_task-tapping_nirs.snirf", verbose=True)
        raw_intensity.load_data()
        return raw_intensity

###############################################################################################################################################################################################

class fNIRS_Alexandros_DoC_data_load(fNIRS_data_load):
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool, individuals : bool = False, interpolate_bad_channels:bool=False):
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
        self.scalp_coupling_threshold = 0.5  # Change this value if needed
        self.reject_criteria = dict(hbo=80e-6)
        self.tmin = -5
        self.tmax = 15
        self.baseline = (None, 0)
        self.data_types = ["Tongue"]
        self.number_of_data_types = 2
        self.data_name = "fNIRS_Alexandros_DoC_data"
        self.individuals = individuals
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = "15.0"
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
            individuals = self.individuals,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted)

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
            unwanted = np.nonzero(raw_intensity.annotations.description == self.unwanted)
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
            raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            epochs = mne.Epochs(
                raw_haemo,
                events,
                event_id=event_dict,
                tmin=self.tmin,
                tmax=self.tmax,
                reject=self.reject_criteria,
                reject_by_annotation=True,
                proj=True,
                baseline=self.baseline,
                preload=True,
                detrend=None,
                verbose=True,
            )

            if len(epochs) != 0:
                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
                
                if self.individuals:
                    Participant_i = individual_participant_class(f"Participant_{i}")
                    Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                    Participant_i.raw_intensity = raw_intensity
                    Participant_i.raw_od = raw_od
                    Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                    Participant_i.raw_haemo = raw_haemo
                    Participant_i.epochs = epochs
                
                for name in self.data_types:
                    getattr(self, f'all_{name}').append(epochs[name].get_data(copy=True))
                    if self.individuals:
                        Participant_i.events.update({name: epochs[name].get_data(copy=True)})
                
                if self.individuals:
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
        return self.all_epochs, self.data_name, all_data, all_freq, self.data_types, self.Individual_participants if self.individuals else None

    
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
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool, individuals : bool = False, interpolate_bad_channels:bool=False):
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
        self.scalp_coupling_threshold = 0.5  # Change this value if needed
        self.reject_criteria = dict(hbo=80e-6)
        self.tmin = -5
        self.tmax = 15
        self.baseline = (None, 0)
        self.data_types = ["Imagery"]
        self.number_of_data_types = 2
        self.data_name = "fNIRS_Alexandros_Healthy_data"
        self.individuals = individuals
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = "1"
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
            individuals = self.individuals,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted)

    def define_raw_intensity(self, sub_id):
        raw_intensity = mne.io.read_raw_snirf(f"Dataset/Alexandros/Healthy/_2024-04-29_{sub_id}.snirf", verbose=True)
        raw_intensity.load_data()
        return raw_intensity

    

###############################################################################################################################################################################################

class fNIRS_CUH_patient_data_load(fNIRS_data_load):
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool, individuals : bool = False, interpolate_bad_channels:bool=False):
        self.number_of_participants = 48
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1": "Imagery",
                                 "Rest": "Control"
                                }
        self.file_path = mne.datasets.fnirs_motor.data_path()
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 15
        self.scalp_coupling_threshold = 0.5  # Change this value if needed
        self.reject_criteria = dict(hbo=80e-6)
        self.tmin = 0
        self.tmax = 15
        self.baseline = (0, 0)
        self.data_types = ["Imagery"]
        self.number_of_data_types = 2
        self.data_name = "fNIRS_CUH_patient_data"
        self.individuals = individuals
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = "Pause"
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
            individuals = self.individuals,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted)

    def define_raw_intensity(self, sub_id):
            if sub_id == 9:
                raw_intensity = mne.io.read_raw_snirf(f"L:\LovbeskyttetMapper\CONNECT-ME\DTU\Alex_Data\DoC\data_initial\P{sub_id}_2_2.snirf", verbose=True)
            else:            
                raw_intensity = mne.io.read_raw_snirf(f"L:\LovbeskyttetMapper\CONNECT-ME\DTU\Alex_Data\DoC\data_initial\P{sub_id}_1.snirf", verbose=True)
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
            unwanted = np.nonzero(raw_intensity.annotations.description == self.unwanted)
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
            raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            epochs = mne.Epochs(
                raw_haemo,
                events,
                event_id=event_dict,
                tmin=self.tmin,
                tmax=self.tmax,
                reject=self.reject_criteria,
                reject_by_annotation=True,
                proj=True,
                baseline=self.baseline,
                preload=True,
                detrend=None,
                verbose=True,
            )

            if len(epochs) != 0:
                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
                
                if self.individuals:
                    Participant_i = individual_participant_class(f"Participant_{i}")
                    Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                    Participant_i.raw_intensity = raw_intensity
                    Participant_i.raw_od = raw_od
                    Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                    Participant_i.raw_haemo = raw_haemo
                    Participant_i.epochs = epochs
                
                for name in self.data_types:
                    getattr(self, f'all_{name}').append(epochs[name].get_data(copy=True))
                    if self.individuals:
                        Participant_i.events.update({name: epochs[name].get_data(copy=True)})
                
                if self.individuals:
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
        return self.all_epochs, self.data_name, all_data, all_freq, self.data_types, self.Individual_participants if self.individuals else None

    
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
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool, individuals : bool = False, interpolate_bad_channels:bool=False):
        self.number_of_participants = 4
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1": "HandMI",
                                 "Rest": "Control"
                                }
        self.file_path = mne.datasets.fnirs_motor.data_path()
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 28
        self.scalp_coupling_threshold = 0.8  # Change this value if needed
        self.reject_criteria = dict(hbo=80e-6)  # Change this value if needed
        self.tmin = -5
        self.tmax = 28
        self.baseline = (None, 0)
        self.data_types = ["HandMI"]
        self.number_of_data_types = 2
        self.data_name = "fNIRS_Melika_data"
        self.individuals = individuals
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = ""
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
            individuals = self.individuals,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted)

    def define_raw_intensity(self, sub_id):
        raw_intensity = mne.io.read_raw_snirf(rf"L:\LovbeskyttetMapper\CONNECT-ME\Melika\Målinger_kopi\snirf_hand_5hz\subj-{sub_id}.snirf", verbose=True)
        
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
            unwanted = np.nonzero(raw_intensity.annotations.description == self.unwanted)
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
            raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            epochs = mne.Epochs(
                raw_haemo,
                events,
                event_id=event_dict,
                tmin=self.tmin,
                tmax=self.tmax,
                reject=self.reject_criteria,
                reject_by_annotation=True,
                proj=True,
                baseline=self.baseline,
                preload=True,
                detrend=None,
                verbose=True,
            )

            if len(epochs) != 0:
                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
                
                if self.individuals:
                    Participant_i = individual_participant_class(f"Participant_{i}")
                    Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                    Participant_i.raw_intensity = raw_intensity
                    Participant_i.raw_od = raw_od
                    Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                    Participant_i.raw_haemo = raw_haemo
                    Participant_i.epochs = epochs
                
                for name in self.data_types:
                    getattr(self, f'all_{name}').append(epochs[name].get_data(copy=True))
                    if self.individuals:
                        Participant_i.events.update({name: epochs[name].get_data(copy=True)})
                
                if self.individuals:
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
        return self.all_epochs, self.data_name, all_data, all_freq, self.data_types, self.Individual_participants if self.individuals else None

    
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
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool, individuals : bool = False, interpolate_bad_channels:bool=False):
        self.number_of_participants = 4
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1": "TongueMI",
                                 "Rest": "Control",
                                }
        self.file_path = mne.datasets.fnirs_motor.data_path()
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 28
        self.scalp_coupling_threshold = 0.8  # Change this value if needed
        self.reject_criteria = dict(hbo=90e-6)  # Change this value if needed
        self.tmin = -5
        self.tmax = 28
        self.baseline = (None, 0)
        self.data_types = ["TongueMI"]
        self.number_of_data_types = 2
        self.data_name = "fNIRS_Melika_data"
        self.individuals = individuals
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = "2"
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
            individuals = self.individuals,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted)

    def define_raw_intensity(self, sub_id):
        raw_intensity = mne.io.read_raw_snirf(rf"L:\LovbeskyttetMapper\CONNECT-ME\Melika\Målinger_kopi\snirf_tongue_5hz\subj-{sub_id}.snirf", verbose=True)
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
            unwanted = np.nonzero(raw_intensity.annotations.description == self.unwanted)
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
            raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            epochs = mne.Epochs(
                raw_haemo,
                events,
                event_id=event_dict,
                tmin=self.tmin,
                tmax=self.tmax,
                reject=self.reject_criteria,
                reject_by_annotation=True,
                proj=True,
                baseline=self.baseline,
                preload=True,
                detrend=None,
                verbose=True,
            )

            if len(epochs) != 0:
                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
                
                if self.individuals:
                    Participant_i = individual_participant_class(f"Participant_{i}")
                    Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                    Participant_i.raw_intensity = raw_intensity
                    Participant_i.raw_od = raw_od
                    Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                    Participant_i.raw_haemo = raw_haemo
                    Participant_i.epochs = epochs
                
                for name in self.data_types:
                    getattr(self, f'all_{name}').append(epochs[name].get_data(copy=True))
                    if self.individuals:
                        Participant_i.events.update({name: epochs[name].get_data(copy=True)})
                
                if self.individuals:
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
        return self.all_epochs, self.data_name, all_data, all_freq, self.data_types, self.Individual_participants if self.individuals else None

    
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
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool, individuals : bool = False, interpolate_bad_channels:bool=False):
        self.number_of_participants = 9
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1": "HandMI",
                                 "Rest": "Control"
                                }
        self.file_path = rf"L:\LovbeskyttetMapper\CONNECT-ME\Melika\Målinger_kopi\snirf_files_hand_10hz"
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 28
        self.scalp_coupling_threshold = 0.8  # Change this value if needed
        self.reject_criteria = dict(hbo=90e-6)  # Change this value if needed
        self.tmin = -5
        self.tmax = 28
        self.baseline = (None, 0)
        self.data_types = ["HandMI"]
        self.number_of_data_types = 2
        self.data_name = "fNIRS_Melika_data"
        self.individuals = individuals
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = ""
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
            individuals = self.individuals,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted)

    def define_raw_intensity(self, sub_id):
        file_path = os.path.join(self.file_path, f"subj-{sub_id}.snirf")  # Correct formatting
        raw_intensity = mne.io.read_raw_snirf(file_path, verbose=True)
        raw_intensity.load_data()
        return raw_intensity
        
    def load_data(self):
        for i, filename in enumerate(sorted(os.listdir(self.file_path)), start=1):
            sub_id = str(i).zfill(2)  # Pad with zeros to get "01", "02", etc.
            raw_intensity = self.define_raw_intensity(sub_id)
            raw_intensity = self.make_without_intro_annotations(raw_intensity)


            raw_intensity.annotations.rename(self.annotation_names)
            unwanted = np.nonzero(raw_intensity.annotations.description == self.unwanted)
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
            raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            epochs = mne.Epochs(
                raw_haemo,
                events,
                event_id=event_dict,
                tmin=self.tmin,
                tmax=self.tmax,
                reject=self.reject_criteria,
                reject_by_annotation=True,
                proj=True,
                baseline=self.baseline,
                preload=True,
                detrend=None,
                verbose=True,
            )

            if len(epochs) != 0:
                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
                
                if self.individuals:
                    Participant_i = individual_participant_class(f"Participant_{i}")
                    Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                    Participant_i.raw_intensity = raw_intensity
                    Participant_i.raw_od = raw_od
                    Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                    Participant_i.raw_haemo = raw_haemo
                    Participant_i.epochs = epochs
                
                for name in self.data_types:
                    getattr(self, f'all_{name}').append(epochs[name].get_data(copy=True))
                    if self.individuals:
                        Participant_i.events.update({name: epochs[name].get_data(copy=True)})
                
                if self.individuals:
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
        return self.all_epochs, self.data_name, all_data, all_freq, self.data_types, self.Individual_participants if self.individuals else None

    
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
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool, individuals : bool = False, interpolate_bad_channels:bool=False):
        self.number_of_participants = 9
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1": "TongueMI",
                                 "Rest": "Control",
                                }
        self.file_path = rf"L:\LovbeskyttetMapper\CONNECT-ME\Melika\Målinger_kopi\snirf_files_tongue10hz"
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 28
        self.scalp_coupling_threshold = 0.8  # Change this value if needed
        self.reject_criteria = dict(hbo=90e-6)  # Change this value if needed
        self.tmin = -5
        self.tmax = 28
        self.baseline = (None, 0)
        self.data_types = ["TongueMI"]
        self.number_of_data_types = 2
        self.data_name = "fNIRS_Melika_data"
        self.individuals = individuals
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = "2"
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
            individuals = self.individuals,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted)

    def define_raw_intensity(self, sub_id):
        file_path = os.path.join(self.file_path, f"subj-{sub_id}.snirf")  # Correct formatting
        raw_intensity = mne.io.read_raw_snirf(file_path, verbose=True)
        raw_intensity.load_data()
        return raw_intensity
        
    def load_data(self):
        for i, filename in enumerate(sorted(os.listdir(self.file_path)), start=1):
            sub_id = str(i).zfill(2)  # Pad with zeros to get "01", "02", etc.
            raw_intensity = self.define_raw_intensity(sub_id)
            if i == 1 : # When data for the first patient was recorded, the introduction was not added in Satori, so we add it manually
                raw_intensity = self.make_without_intro_annotations(raw_intensity)
            else: # For all other patients we just add the resting phases
                raw_intensity = self.make_annotations(raw_intensity)

            raw_intensity.annotations.rename(self.annotation_names)
            unwanted = np.nonzero(raw_intensity.annotations.description == self.unwanted)
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
            raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            epochs = mne.Epochs(
                raw_haemo,
                events,
                event_id=event_dict,
                tmin=self.tmin,
                tmax=self.tmax,
                reject=self.reject_criteria,
                reject_by_annotation=True,
                proj=True,
                baseline=self.baseline,
                preload=True,
                detrend=None,
                verbose=True,
            )

            if len(epochs) != 0:
                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
                
                if self.individuals:
                    Participant_i = individual_participant_class(f"Participant_{i}")
                    Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                    Participant_i.raw_intensity = raw_intensity
                    Participant_i.raw_od = raw_od
                    Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                    Participant_i.raw_haemo = raw_haemo
                    Participant_i.epochs = epochs
                
                for name in self.data_types:
                    getattr(self, f'all_{name}').append(epochs[name].get_data(copy=True))
                    if self.individuals:
                        Participant_i.events.update({name: epochs[name].get_data(copy=True)})
                
                if self.individuals:
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
        return self.all_epochs, self.data_name, all_data, all_freq, self.data_types, self.Individual_participants if self.individuals else None

    
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

class fNIRS_Melika_old_data_load(fNIRS_data_load):
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool, individuals : bool = False, interpolate_bad_channels:bool=False):
        self.number_of_participants = 11
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1": "HandMI",
                                 "2": "TongueMI",
                                 "Rest": "Control"
                                }
        self.file_path = mne.datasets.fnirs_motor.data_path()
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 20
        self.scalp_coupling_threshold = 0.8  # Change this value if needed
        self.reject_criteria = dict(hbo=90e-6)  # Change this value if needed
        self.tmin = -5
        self.tmax = 15
        self.baseline = (None, 0)
        self.data_types = ["HandMI", "TongueMI"]
        self.number_of_data_types = 2
        self.data_name = "fNIRS_Melika_data"
        self.individuals = individuals
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = "Pause"
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
            individuals = self.individuals,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted)

    def define_raw_intensity(self, sub_id):
        raw_intensity = mne.io.read_raw_snirf(rf"L:\LovbeskyttetMapper\CONNECT-ME\Melika\Målinger_kopi\snirf_files_old\subj-{sub_id}.snirf", verbose=True)
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
            unwanted = np.nonzero(raw_intensity.annotations.description == self.unwanted)
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
            raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            epochs = mne.Epochs(
                raw_haemo,
                events,
                event_id=event_dict,
                tmin=self.tmin,
                tmax=self.tmax,
                reject=self.reject_criteria,
                reject_by_annotation=True,
                proj=True,
                baseline=self.baseline,
                preload=True,
                detrend=None,
                verbose=True,
            )

            if len(epochs) != 0:
                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
                
                if self.individuals:
                    Participant_i = individual_participant_class(f"Participant_{i}")
                    Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                    Participant_i.raw_intensity = raw_intensity
                    Participant_i.raw_od = raw_od
                    Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                    Participant_i.raw_haemo = raw_haemo
                    Participant_i.epochs = epochs
                
                for name in self.data_types:
                    getattr(self, f'all_{name}').append(epochs[name].get_data(copy=True))
                    if self.individuals:
                        Participant_i.events.update({name: epochs[name].get_data(copy=True)})
                
                if self.individuals:
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
        return self.all_epochs, self.data_name, all_data, all_freq, self.data_types, self.Individual_participants if self.individuals else None

    
    def make_annotations(self, raw_intensity):
        sampling_frequency = raw_intensity.info["sfreq"]
        events, event_dict = mne.events_from_annotations(raw_intensity)
        cropped_raw_data = raw_intensity.copy()
        cropped_raw_data.annotations.set_durations(self.stimulus_duration)
        for id,event in enumerate(events):
            cropped_raw_data.annotations.append((event[0]) / cropped_raw_data.info['sfreq'] + self.stimulus_duration, self.stimulus_duration, "Rest")
        # cropped_raw_data.plot(n_channels=len(cropped_raw_data.ch_names), duration=600, show_scrollbars=True)
        # plt.show()
        # events, event_dict = mne.events_from_annotations(cropped_raw_data)
        # print(events)
        return cropped_raw_data

###############################################################################################################################################################################################

class fNIRS_Melika_hand_data_long_load(fNIRS_data_load):
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool, individuals : bool = False, interpolate_bad_channels:bool=False):
        self.number_of_participants = 7
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1": "HandMI",
                                 "Rest": "Control"
                                }
        self.file_path = rf"L:\LovbeskyttetMapper\CONNECT-ME\Melika\Målinger_kopi\snirf_files_hand_long"
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 21
        self.scalp_coupling_threshold = 0.8  # Change this value if needed
        self.reject_criteria = dict(hbo=90e-6)  # Change this value if needed
        self.tmin = 0
        self.tmax = 21
        self.baseline = (0, 0)
        self.data_types = ["HandMI"]
        self.number_of_data_types = 2
        self.data_name = "fNIRS_Melika_data"
        self.individuals = individuals
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = ""
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
            individuals = self.individuals,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted)

    def define_raw_intensity(self, sub_id):
        file_path = os.path.join(self.file_path, f"subj-{sub_id}.snirf")  # Correct formatting
        raw_intensity = mne.io.read_raw_snirf(file_path, verbose=True)
        raw_intensity.load_data()
        return raw_intensity
        
    def load_data(self):
        for i, filename in enumerate(sorted(os.listdir(self.file_path)), start=1):
            sub_id = str(i).zfill(2)  # Pad with zeros to get "01", "02", etc.
            raw_intensity = self.define_raw_intensity(sub_id)
            raw_intensity = self.make_without_intro_annotations(raw_intensity)


            raw_intensity.annotations.rename(self.annotation_names)
            unwanted = np.nonzero(raw_intensity.annotations.description == self.unwanted)
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
            raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            resting_state_sample = events[events[:, 2] == 7, 0][0]  # Get the sample number
            resting_state_time = resting_state_sample / raw_haemo.info['sfreq']  # Convert to seconds

            # Extract resting state data separately for baseline calculation
            resting_state_start = resting_state_time
            resting_state_end = resting_state_time + 30

            # Get the mean signal during resting state (per channel)
            resting_data = raw_haemo.copy().crop(resting_state_start, resting_state_end)
            resting_baseline = resting_data.get_data().mean(axis=1)  # Mean across time for each channel

            epochs = mne.Epochs(
                raw_haemo,
                events,
                event_id=event_dict,
                tmin=self.tmin,
                tmax=self.tmax,
                reject=self.reject_criteria,
                reject_by_annotation=True,
                proj=True,
                baseline=None,
                preload=True,
                detrend=None,
                verbose=True,
            )

            # Apply baseline correction per channel with error handling for removed channels
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

            if len(epochs) != 0:
                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
                
                if self.individuals:
                    Participant_i = individual_participant_class(f"Participant_{i}")
                    Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                    Participant_i.raw_intensity = raw_intensity
                    Participant_i.raw_od = raw_od
                    Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                    Participant_i.raw_haemo = raw_haemo
                    Participant_i.epochs = epochs
                
                for name in self.data_types:
                    getattr(self, f'all_{name}').append(epochs[name].get_data(copy=True))
                    if self.individuals:
                        Participant_i.events.update({name: epochs[name].get_data(copy=True)})
                
                if self.individuals:
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
        return self.all_epochs, self.data_name, all_data, all_freq, self.data_types, self.Individual_participants if self.individuals else None

    
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
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool, individuals : bool = False, interpolate_bad_channels:bool=False):
        self.number_of_participants = 9
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1": "TongueMI",
                                 "Rest": "Control",
                                }
        self.file_path = rf"L:\LovbeskyttetMapper\CONNECT-ME\Melika\Målinger_kopi\snirf_files_tongue_long"
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 21
        self.scalp_coupling_threshold = 0.8  # Change this value if needed
        self.reject_criteria = dict(hbo=90e-6)  # Change this value if needed
        self.tmin = 0
        self.tmax = 21
        self.baseline = (0, 0)
        self.data_types = ["TongueMI"]
        self.number_of_data_types = 2
        self.data_name = "fNIRS_Melika_data"
        self.individuals = individuals
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = "2"
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
            individuals = self.individuals,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted)

    def define_raw_intensity(self, sub_id):
        file_path = os.path.join(self.file_path, f"subj-{sub_id}.snirf")  # Correct formatting
        raw_intensity = mne.io.read_raw_snirf(file_path, verbose=True)
        raw_intensity.load_data()
        return raw_intensity
        
    def load_data(self):
        for i, filename in enumerate(sorted(os.listdir(self.file_path)), start=1):
            sub_id = str(i).zfill(2)  # Pad with zeros to get "01", "02", etc.
            raw_intensity = self.define_raw_intensity(sub_id)
            raw_intensity = self.make_annotations(raw_intensity)

            raw_intensity.annotations.rename(self.annotation_names)
            unwanted = np.nonzero(raw_intensity.annotations.description == self.unwanted)
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
            raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)

            if self.negative_correlation_enhancement:
                raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

            events, event_dict = mne.events_from_annotations(raw_haemo)

            epochs = mne.Epochs(
                raw_haemo,
                events,
                event_id=event_dict,
                tmin=self.tmin,
                tmax=self.tmax,
                reject=self.reject_criteria,
                reject_by_annotation=True,
                proj=True,
                baseline=None,
                preload=True,
                detrend=None,
                verbose=True,
            )

            if len(epochs) != 0:
                self.all_epochs.append(epochs)
                self.all_control.append(epochs["Control"].get_data(copy=True))
                
                if self.individuals:
                    Participant_i = individual_participant_class(f"Participant_{i}")
                    Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                    Participant_i.raw_intensity = raw_intensity
                    Participant_i.raw_od = raw_od
                    Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                    Participant_i.raw_haemo = raw_haemo
                    Participant_i.epochs = epochs
                
                for name in self.data_types:
                    getattr(self, f'all_{name}').append(epochs[name].get_data(copy=True))
                    if self.individuals:
                        Participant_i.events.update({name: epochs[name].get_data(copy=True)})
                
                if self.individuals:
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
        return self.all_epochs, self.data_name, all_data, all_freq, self.data_types, self.Individual_participants if self.individuals else None

    
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
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool, individuals : bool = False, interpolate_bad_channels:bool=False):
        self.number_of_participants = 68
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1": "TongueMI",
                                 "Rest": "Control",
                                }
        self.file_path = rf"L:\LovbeskyttetMapper\CONNECT-ME\Pardis\DoC_TongueMI\Patient Recordings"
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 15
        self.scalp_coupling_threshold = 0.8  # Change this value if needed
        self.reject_criteria = dict(hbo=90e-6)  # Change this value if needed
        self.tmin = 0
        self.tmax = 21
        self.baseline = (None, 0)
        self.data_types = ["TongueMI"]
        self.number_of_data_types = 2
        self.data_name = "fNIRS_Pardis_DOC_data"
        self.individuals = individuals
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = ""
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
            individuals = self.individuals,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted)

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
                unwanted = np.nonzero(raw_intensity.annotations.description == self.unwanted)
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
                raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)

                if self.negative_correlation_enhancement:
                    raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

                events, event_dict = mne.events_from_annotations(raw_haemo)

                epochs = mne.Epochs(
                    raw_haemo,
                    events,
                    event_id=event_dict,
                    tmin=self.tmin,
                    tmax=self.tmax,
                    reject=self.reject_criteria,
                    reject_by_annotation=True,
                    proj=True,
                    baseline=None,
                    preload=True,
                    detrend=None,
                    verbose=True,
                )

                if len(epochs) != 0:
                    self.all_epochs.append(epochs)
                    self.all_control.append(epochs["Control"].get_data(copy=True))
                    
                    if self.individuals:
                        Participant_i = individual_participant_class(f"Participant_{i}")
                        Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                        Participant_i.raw_intensity = raw_intensity
                        Participant_i.raw_od = raw_od
                        Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                        Participant_i.raw_haemo = raw_haemo
                        Participant_i.epochs = epochs
                    
                    for name in self.data_types:
                        getattr(self, f'all_{name}').append(epochs[name].get_data(copy=True))
                        if self.individuals:
                            Participant_i.events.update({name: epochs[name].get_data(copy=True)})
                    
                    if self.individuals:
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
        return self.all_epochs, self.data_name, all_data, all_freq, self.data_types, self.Individual_participants if self.individuals else None

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
    def __init__(self, short_channel_correction: bool, negative_correlation_enhancement: bool, individuals : bool = False, interpolate_bad_channels:bool=False):
        self.number_of_participants = 68
        self.all_tapping = []
        self.all_control = []
        self.annotation_names = {"1": "TonguePhysical",
                                 "2": "Control",
                                 "3": "TongueIM"
                                }
        self.file_path = rf"L:\LovbeskyttetMapper\CONNECT-ME\Pardis\HC_ICU_TongueMI\Data\HC\Follow-up"
        self.short_channel_correction = short_channel_correction
        self.negative_correlation_enhancement = negative_correlation_enhancement
        self.stimulus_duration = 15
        self.scalp_coupling_threshold = 0.8  # Change this value if needed
        self.reject_criteria = dict(hbo=90e-6)  # Change this value if needed
        self.tmin = -5
        self.tmax = 20
        self.baseline = (None, 0)
        self.data_types = ["TonguePhysical", "TongueIM"]
        self.number_of_data_types = 2
        self.data_name = "fNIRS_Pardis_HC"
        self.individuals = individuals
        self.interpolate_bad_channels = interpolate_bad_channels
        self.unwanted = ["5", "6", "7"]
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
            individuals = self.individuals,
            interpolate_bad_channels = self.interpolate_bad_channels,
            unwanted = self.unwanted)

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
                raw_haemo.filter(0.05, 0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)

                if self.negative_correlation_enhancement:
                    raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

                events, event_dict = mne.events_from_annotations(raw_haemo)

                epochs = mne.Epochs(
                    raw_haemo,
                    events,
                    event_id=event_dict,
                    tmin=self.tmin,
                    tmax=self.tmax,
                    reject=self.reject_criteria,
                    reject_by_annotation=True,
                    proj=True,
                    baseline=None,
                    preload=True,
                    detrend=None,
                    verbose=True,
                )

                if len(epochs) != 0:
                    self.all_epochs.append(epochs)
                    self.all_control.append(epochs["Control"].get_data(copy=True))
                    
                    if self.individuals:
                        Participant_i = individual_participant_class(f"Participant_{i}")
                        Participant_i.events.update({"Control": epochs["Control"].get_data(copy=True)})
                        Participant_i.raw_intensity = raw_intensity
                        Participant_i.raw_od = raw_od
                        Participant_i.raw_haemo_unfiltered = raw_haemo_unfiltered
                        Participant_i.raw_haemo = raw_haemo
                        Participant_i.epochs = epochs
                    
                    for name in self.data_types:
                        getattr(self, f'all_{name}').append(epochs[name].get_data(copy=True))
                        if self.individuals:
                            Participant_i.events.update({name: epochs[name].get_data(copy=True)})
                    
                    if self.individuals:
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
        return self.all_epochs, self.data_name, all_data, all_freq, self.data_types, self.Individual_participants if self.individuals else None
