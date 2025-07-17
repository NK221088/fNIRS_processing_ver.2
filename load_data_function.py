from Data_processesing_class import AudioSpeechNoise_data_load, fNIRS_full_motor_data_load, fNIRS_motor_data_load, fNIRS_Alexandros_DoC_data_load, fNIRS_Alexandros_Healthy_data_load, fNIRS_CUH_patient_data_load, fNIRS_Melika_tongue_5Hz_data_load, fNIRS_Melika_hand_data_5Hz_load, fNIRS_Melika_old_data_load, fNIRS_Melika_hand_data_10Hz_load, fNIRS_Melika_tongue_10Hz_data_load, fNIRS_Melika_hand_data_long_load, fNIRS_Melika_tongue_long_data_load, fNIRS_Pardis_DOC_data_load, fNIRS_Pardis_HC_data_load

data_loaders = {
    "Dr. Luke: motor": fNIRS_motor_data_load,
    "Dr. Luke: AudioSpeechNoise": AudioSpeechNoise_data_load,
    "Dr. Luke: full motor data": fNIRS_full_motor_data_load,
    "Alexandros: DoC data": fNIRS_Alexandros_DoC_data_load,
    "Alexandros: HC data": fNIRS_Alexandros_Healthy_data_load,
    "CUH patient data": fNIRS_CUH_patient_data_load,
    "Melika: Hand 5 Hz": fNIRS_Melika_hand_data_5Hz_load,
    "Melika: Tongue 5 Hz": fNIRS_Melika_tongue_5Hz_data_load,
    "Melika: Old data": fNIRS_Melika_old_data_load,
    "Melika: Hand 10 Hz": fNIRS_Melika_hand_data_10Hz_load,
    "Melika: Tongue 10 Hz": fNIRS_Melika_tongue_10Hz_data_load,
    "Melika: Hand long paradigme": fNIRS_Melika_hand_data_long_load,
    "Melika: Tongue long paradigme": fNIRS_Melika_tongue_long_data_load,
    "Pardis: DOC data": fNIRS_Pardis_DOC_data_load,
    "Pardis: HC data": fNIRS_Pardis_HC_data_load,
}

def load_data(data_set : str, short_channel_correction : bool = None, negative_correlation_enhancement : bool = None, interpolate_bad_channels:bool=False, baseline_correction: str = "Previous rest period", tmin : int = 0, filter_lower_value: float = 0.05, filter_upper_value: float = 0.7, h_trans_bandwidth: float = 0.2, l_trans_bandwidth: float = 0.02):
    if data_set not in data_loaders:
        raise ValueError("Dataset does not exist.")
    loader = data_loaders[data_set](short_channel_correction = short_channel_correction,
                                    negative_correlation_enhancement = negative_correlation_enhancement,
                                    interpolate_bad_channels=interpolate_bad_channels,
                                    baseline_correction = baseline_correction,
                                    tmin = tmin,
                                    filter_lower_value = filter_lower_value,
                                    filter_upper_value = filter_upper_value,
                                    h_trans_bandwidth = h_trans_bandwidth,
                                    l_trans_bandwidth = l_trans_bandwidth)
    return loader.load_data()