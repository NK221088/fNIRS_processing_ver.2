import sys
import os
from dotenv import load_dotenv
import mne
import mne_nirs
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from collections import defaultdict
from preprocessing_toolbox.load_data_function import data_loaders

load_dotenv()

dataSetList = list(data_loaders.keys())
dataLoaders = [dataSetList[-1]] #, dataSetList[17]]
datasets = defaultdict(defaultdict)

for data_loader in dataLoaders:
    settings = {
        "data_set": data_loader,  # Default to first dataset
        "epoch_type": "TongueMI",
        "individual": "All Individuals",
        "short_channel_correction": True,
        "negative_correlation_enhancement": False,
        "haemo_type": "hbo",
        "baseline_correction": "Previous rest period",
        "tmin": 0,
        "stimulus_duration": 5,
        "scalp_coupling_threshold": 0.8,
        "reject_criteria": dict(hbo=80e-6),
        "unwanted": ["15.0"],
        "filter_lower_value": 0.01,
        "filter_upper_value": 0.5,
        "h_trans_bandwidth": 0.2,           
        "l_trans_bandwidth": 0.01,
        "snr_rejection": "SNR",  # Default to None, can be set to "SNR" or "CV"
        "snr_threshold": 8,  # Default threshold for SNR
        "Apply_TDDR": True,  
        "interpolate_bad_channels": True,
    }
    current_loader = data_loaders[data_loader](
                    data_name = data_loader,
                    file_path = data_loader,
                    short_channel_correction=settings["short_channel_correction"],
                    negative_correlation_enhancement=settings["negative_correlation_enhancement"],
                    interpolate_bad_channels=settings["interpolate_bad_channels"],
                    baseline_correction=settings["baseline_correction"],
                    tmin=settings["tmin"],
                    filter_lower_value=settings["filter_lower_value"],
                    filter_upper_value=settings["filter_upper_value"],
                    l_trans_bandwidth=settings["l_trans_bandwidth"],
                    h_trans_bandwidth=settings["h_trans_bandwidth"],
                    scalp_coupling_threshold=settings["scalp_coupling_threshold"],
                    reject_criteria=settings["reject_criteria"],
                    snr_rejection=settings["snr_rejection"],
                    snr_threshold=settings["snr_threshold"],
                    apply_tddr=settings["Apply_TDDR"]
                )
    data = current_loader.load_data()
    variables = ("all_epochs", "data_name", "all_data", "freq", "data_types", "all_individuals")
    datasets[data_loader] = {key: value for key, value in zip(variables, data)}

all_participants = datasets[dataLoaders[0]]["all_individuals"] #+ datasets[dataLoaders[1]]["all_individuals"]
number_of_subjects = [len(datasets[dataLoaders[0]]["all_individuals"])]#, len((datasets[dataLoaders[1]]["all_individuals"]))]

long_channels = mne_nirs.channels.get_long_channels(all_participants[0].raw_haemo).ch_names

names = [ind.name for ind in all_participants]
all_epochs = [ind.epochs for ind in all_participants]
first_names = [name.split("_")[0] for name in names]
name_indices = {first_name: [ind for ind, name in enumerate(names) if name.split("_")[0] == first_name] for first_name in first_names}
individual_epochs = {first_name: [all_epochs[i] for i in name_indices[first_name]] for first_name in first_names}

df = pd.DataFrame(columns=["ID", "Mean_Math_HbO", "Mean_Hard_Math_HbO", "Mean_Control_HbO", "Math_Control_p-value", "Hard_Math_Control_p-value"])

for ind, epochs in individual_epochs.items():
    bad_channels = list(set(channel for epoch in epochs for channel in epoch.info['bads']))
    for epoch in epochs:
        epoch.info['bads'] = bad_channels
    individual_epochs[ind] = mne.concatenate_epochs(epochs)
    
    
    math_HbO = individual_epochs[ind].copy()["Math"].pick(long_channels).get_data().mean(axis=2).mean(axis=1)
    Hard_math_HbO = individual_epochs[ind].copy()["Hard_Math"].pick(long_channels).get_data().mean(axis=2).mean(axis=1)
    Control_HbO = individual_epochs[ind].copy()["Control"].pick(long_channels).get_data().mean(axis=2).mean(axis=1)

    math_AUC = individual_epochs[ind].copy()["Math"].pick(long_channels).get_data().mean(axis=2).mean(axis=1)
    Hard_math_HbO = individual_epochs[ind].copy()["Hard_Math"].pick(long_channels).get_data().mean(axis=2).mean(axis=1)
    Control_HbO = individual_epochs[ind].copy()["Control"].pick(long_channels).get_data().mean(axis=2).mean(axis=1)
    
    
    t_stat_math_control, p_value_math_control = ttest_ind(math_HbO, Control_HbO,equal_var=False)
    t_stat_hard_math_control, p_value_hard_math_control = ttest_ind(Hard_math_HbO, Control_HbO,equal_var=False)
    new_row = {
    "ID": ind,
    "Mean_Math_HbO": np.mean(math_HbO),
    "Mean_Hard_Math_HbO": np.mean(Hard_math_HbO),
    "Mean_Control_HbO": np.mean(Control_HbO),
    "Math_Control_p-value": p_value_math_control,
    "Hard_Math_Control_p-value": p_value_hard_math_control,
    }

    df.loc[len(df)] = new_row

df.to_csv()
print("debug")