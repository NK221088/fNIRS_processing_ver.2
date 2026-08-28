import sys
import os
from dotenv import load_dotenv
from pathlib import Path
import mne
import mne_nirs
import pandas as pd
import numpy as np
import rpy2.robjects as robjects
print(robjects.r('1 + 1'))

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from collections import defaultdict
from preprocessing_toolbox.load_data_function import data_loaders
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from collections import Counter
from mne.stats import permutation_t_test
from scipy.stats import chi2

load_dotenv()
save_path = Path(os.getenv(rf"Evoked_plots_path"))
consciousness_states_path = Path(os.getenv(rf"Consciousness_states_path"))


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
        "filter_upper_value": 0.2,
        "h_trans_bandwidth": 0.05,           
        "l_trans_bandwidth": 0.01,
        "snr_rejection": "SNR",  # Default to None, can be set to "SNR" or "CV"
        "snr_threshold": 8,  # Default threshold for SNR
        "Apply_TDDR": True,  
        "interpolate_bad_channels": False,
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

chromophore = "hbo" # "hbt" # "hbr" #
channel_names = [channel for channel in all_participants[0].raw_haemo.ch_names if chromophore in channel]

long_channels = mne_nirs.channels.get_long_channels(all_participants[0].raw_haemo.copy().pick(channel_names)).ch_names

individual_recording_analysis = True
names = [ind.name for ind in all_participants]
all_epochs = [ind.epochs for ind in all_participants]
first_names = [name.split("_")[0] for name in names]
name_indices = {first_name: [ind for ind, name in enumerate(names) if name.split("_")[0] == first_name] for first_name in first_names}

individual_epochs = {first_name: [all_epochs[i].copy().pick(long_channels) for i in name_indices[first_name]] for first_name in first_names}


channel_counts = {}
math_lenghts = []
hard_math_lengths = []
control_lengths = []

math_mean = []
hard_math_means = []
control_means = []

df = pd.DataFrame(columns=["Subject", "Recording", "Condition", "Mean_Response"])
if individual_recording_analysis:
    individual_epochs = {ind.name: ind.epochs for ind in all_participants}
    # individual_epochs = {
    # f"{subject}_{session}": values
    # for subject, sessions in 
    #     session_epoch_map.items()
    # for session, values in sessions.items()
    # }
for ind, epochs in individual_epochs.items():
    if individual_recording_analysis:
    #     # bad_channels = session_epoch_bad_channels[ind.split("_")[0]][ind.split("_")[1]]
        bad_channels = epochs.info["bads"]
    #     # individual_epochs[ind] = epochs
    else:
        n_epochs = len(epochs)
        min_fraction = round(0.5 * n_epochs)
        bad_channel_counts = Counter(ch for epoch in epochs for ch in epoch.copy().info['bads'])
        bad_channel_indices = np.array([list(bad_channel_counts.values())]).flatten() > min_fraction
        bad_channels = list(set(np.array([list(bad_channel_counts.keys())]).flatten()[bad_channel_indices]))
        for epoch in epochs:
            epoch.info['bads'] = bad_channels

        epochs = [epoch.drop_channels(epoch.info["bads"]) for epoch in epochs]
        individual_epochs[ind] = mne.concatenate_epochs(epochs)
        bad_channels = list(set(channel for epoch in epochs for channel in epoch.info['bads']))
        good_long_channels = [ch for ch in long_channels if ch in good_channel_names[first_name]]
    good_long_channels = [ch for ch in long_channels if ch not in bad_channels]

    print(f"{ind}: {len(bad_channels)} bad channels dropped, {len(good_long_channels)} good long channels remaining")
    channel_counts[ind] = [len(bad_channels), len(good_long_channels)]

    math_t_start = 0
    math_t_end = 24.9
    control_t_start = 5
    control_t_end = 20
    math_HbO_mean = individual_epochs[ind].copy()["Math"].pick(good_long_channels).crop(math_t_start, math_t_end, True).get_data().mean(axis=2).mean(axis=1)
    Hard_math_HbO_mean = individual_epochs[ind].copy()["Hard_Math"].pick(good_long_channels).crop(math_t_start, math_t_end, True).get_data().mean(axis=2).mean(axis=1)
    Control_HbO_mean = individual_epochs[ind].copy()["Control"].pick(good_long_channels).crop(control_t_start, control_t_end, True).get_data().mean(axis=2).mean(axis=1)

    ID_prefix, Session, Recording = ind.split("_")
    Recording = (int(Session[1]) - 1) * 3 + int(Recording[1]) if Recording[0] != "B" else (int(Session[1]) - 1) * 3 + 1

    for response in math_HbO_mean:
        new_row = {"Subject": ID_prefix, "Recording": Recording, "Condition": "Math", "Mean_Response": response}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    for response in Hard_math_HbO_mean:
        new_row = {"Subject": ID_prefix, "Recording": Recording, "Condition": "Hard_Math", "Mean_Response": response}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    for response in Control_HbO_mean:
        new_row = {"Subject": ID_prefix, "Recording": Recording, "Condition": "Control", "Mean_Response": response}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)


import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from scipy.stats import chi2
import pandas as pd

pandas2ri.activate()

lme4 = importr("lme4")
lmerTest = importr("lmerTest")
stats = importr("stats")

results = []

for subject, group in df.groupby("Subject"):
    for condition in ["Math", "Hard_Math"]:
        sub_df = group[group["Condition"].isin([condition, "Control"])].copy()
        sub_df["Condition"] = pd.Categorical(sub_df["Condition"], categories=["Control", condition])

        r_df = pandas2ri.py2rpy(sub_df)

        try:
            model1 = lmerTest.lmer("Mean_Response ~ Condition + (1|Recording)", data=r_df, REML=False)
            model2 = lmerTest.lmer("Mean_Response ~ (1|Recording)", data=r_df, REML=False)

            loglik1 = stats.logLik(model1)[0]
            loglik2 = stats.logLik(model2)[0]

            lrt_stat = 2 * (loglik1 - loglik2)
            df_diff = 1  # Condition contributes exactly one parameter (2 levels)
            p_value = chi2.sf(lrt_stat, df_diff)

            results.append({
                "Subject": subject,
                "Condition": condition,
                "loglik_full": loglik1,
                "loglik_null": loglik2,
                "lrt_stat": lrt_stat,
                "p_value": p_value,
            })
        except Exception as e:
            print(f"Failed for {subject}, {condition}: {e}")
            results.append({
                "Subject": subject, "Condition": condition,
                "loglik_full": None, "loglik_null": None, "lrt_stat": None, "p_value": None,
            })

results_df = pd.DataFrame(results)

print("debug")