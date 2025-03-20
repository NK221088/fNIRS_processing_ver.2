from Data_processesing_class import AudioSpeechNoise_data_load, fNIRS_full_motor_data_load, fNIRS_motor_data_load, fNIRS_Alexandros_DoC_data_load, fNIRS_Alexandros_Healthy_data_load, fNIRS_CUH_patient_data_load, fNIRS_Melika_data_load

data_loaders = {
    "fNIrs_motor": fNIRS_motor_data_load,
    "AudioSpeechNoise": AudioSpeechNoise_data_load,
    "fNirs_motor_full_data": fNIRS_full_motor_data_load,
    "fNIRS_Alexandros_DoC_data": fNIRS_Alexandros_DoC_data_load,
    "fNIRS_Alexandros_Healthy_data": fNIRS_Alexandros_Healthy_data_load,
    "fNIRS_CUH_patient_data": fNIRS_CUH_patient_data_load,
    "fNIRS_Melika_data": fNIRS_Melika_data_load,
}

def load_data(data_set : str, short_channel_correction : bool = None, negative_correlation_enhancement : bool = None, individuals :bool = False, interpolate_bad_channels:bool=False):
    if data_set not in data_loaders:
        raise ValueError("Dataset does not exist.")
    loader = data_loaders[data_set](short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement, individuals = individuals, interpolate_bad_channels=interpolate_bad_channels)
    return loader.load_data()