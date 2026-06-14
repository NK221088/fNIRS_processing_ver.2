import numpy as np
import mne
import mne_nirs

def extract_data(patient_data):
    names = [ind.name for ind in patient_data]
    first_names = {name.split("_")[0] for name in names}
    name_idx = {first_name: [idx for idx, n in enumerate(names) if n.split("_")[0] == first_name] for first_name in first_names}
    patient_raw_haemo = {name: [patient_data[i].raw_haemo for i in idxs] for name, idxs in name_idx.items()}
    for name, haemos in patient_raw_haemo.items():
        bad_channels = list(set(channel for haemo in haemos for channel in haemo.info['bads']))
        for haemo in haemos:
            haemo.info['bads'] = bad_channels
        patient_raw_haemo[name] = mne_nirs.channels.get_long_channels(mne.concatenate_raws(haemos).copy()).pick("hbo").get_data()
    return patient_raw_haemo

def construct_PCA_components(patient_data):
    patient_raw_haemo = extract_data(patient_data)
    PCA_components = {}
    return PCA_components

        
        