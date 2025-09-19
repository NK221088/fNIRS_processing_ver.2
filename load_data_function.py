from Data_processesing_class import AudioSpeechNoise_data_load, fNIRS_EEG__HC_baseline_data_load, fNIRS_full_motor_data_load, fNIRS_motor_data_load, fNIRS_Alexandros_DoC_data_load, fNIRS_Alexandros_Healthy_data_load, fNIRS_CUH_patient_data_load, fNIRS_Melika_tongue_5Hz_data_load, fNIRS_Melika_hand_data_5Hz_load, fNIRS_Melika_old_data_load, fNIRS_Melika_hand_data_10Hz_load, fNIRS_Melika_tongue_10Hz_data_load, fNIRS_Melika_hand_data_long_load, fNIRS_Melika_tongue_long_data_load, fNIRS_Pardis_DOC_data_load, fNIRS_Pardis_HC_data_load

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
    "fNIRS-EEG HC data": fNIRS_EEG__HC_baseline_data_load
}