import pandas as pd
import mne
from mne_nirs.experimental_design import longest_inter_annotation_interval
from nilearn.glm.first_level import make_first_level_design_matrix
# from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_glm
from pandas import DataFrame

from joblib import Parallel, delayed

from rpy2.robjects import r, globalenv
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
from mne_nirs.statistics import statsmodels_to_results
from mne_nirs.visualisation import plot_glm_group_topo, plot_glm_surface_projection

import os
from dotenv import load_dotenv
from pathlib import Path

from sklearn.decomposition import PCA
from mne.preprocessing import ICA

from collections import Counter

import mne_nirs

load_dotenv()
save_path = Path(os.getenv(rf"data_save_path"))
Phase_1_assumptions_plot_save_path = Path(os.getenv(rf"Phase_1_assumptions_plot_save_path"))
Phase_1_ANOVA_save_path = Path(os.getenv(rf"Phase_1_ANOVA_save_path"))
Phase_2_assumptions_plot_save_path = Path(os.getenv(rf"Phase_2_assumptions_plot_save_path"))

Phase_2_ANOVA_save_path = Path(os.getenv(rf"Phase_2_ANOVA_save_path"))
Phase_3_assumptions_plot_save_path = Path(os.getenv(rf"Study_2_Phase_2_assumptions_plot_save_path"))
Phase_3_ANOVA_save_path = Path(os.getenv(rf"Study_2_Phase_2_ANOVA_save_path"))
drug_path = Path(os.getenv(rf"Marwan_drug_data"))

from mne.io.pick import _picks_to_idx
from nilearn.glm.first_level import run_glm as nilearn_glm
from mne_nirs.statistics import RegressionResults

import sys
import os
import mne
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from collections import defaultdict
from preprocessing_toolbox.load_data_function import data_loaders

from Thesis_plotting.Significant_responders_plot import covert_responders



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
number_of_subjects = [len(datasets[dataLoaders[0]]["all_individuals"])] #, len((datasets[dataLoaders[1]]["all_individuals"]))]

if "Marwan" in dataLoaders[0]:
    study = 2
else:
    study = 1
if len(number_of_subjects) > 1 or "patient" in dataLoaders[0].lower() or study == 2:
    phase = 2
else:
    phase = 1

long_channels = mne_nirs.channels.get_long_channels(datasets[dataLoaders[0]]["all_individuals"][0].raw_haemo.copy()).ch_names

if study == 1 and phase == 2:
    all_conditions = list(np.unique(datasets[dataLoaders[1]]["all_epochs"][0].annotations.description))
else:
    all_conditions = list(np.unique(datasets[dataLoaders[0]]["all_epochs"][0].annotations.description))
if study == 1:
    if phase == 1:
        epochs = datasets[dataLoaders[0]]["all_epochs"]
    elif phase == 2:
        all_epochs = [datasets[dataLoaders[0]]["all_epochs"], datasets[dataLoaders[1]]["all_epochs"]]
        epochs_ = []
    tmax = 50
    times = np.arange(0, tmax+1, 10.0)
    relevant_data_types = list(np.unique([condition for condition in all_conditions if "n_back" in condition]))
    picks_ =  [ch for ch in long_channels if ("S1" in ch) or ("S2" in ch) or ("S3" in ch) or ("S4" in ch)] + [ch for ch in datasets[dataLoaders[0]]["all_individuals"][0].raw_haemo.copy().ch_names if (("S1" in ch) or ("S2" in ch) or ("S3" in ch) or ("S4" in ch)) and "hbt" in ch]
elif study == 2:
    if Only_responders:
        epochs = [ind.epochs for ind in datasets[dataLoaders[0]]["all_individuals"] if ind.name.split("_")[0] in covert_responders]
    else:
        epochs = [ind.epochs for ind in datasets[dataLoaders[0]]["all_individuals"] if ind.name.split("_")[0] not in covert_responders]
    tmax = 26
    times = np.arange(0, tmax, 5)

    relevant_data_types = list(np.unique([condition for condition in all_conditions if "Math" in condition]))
    picks_ =  [ch for ch in long_channels]
    
Only_responders = True
all_n_back_ = ["0"] #, "n_back", "0""n_back""Control"
for n_back_ in all_n_back_:
    
    delimiter = "/" if study == 1  else "_"
    for data_number in range(len(number_of_subjects)):
        if study == 1:
            if data_number == 0:
                end = "HC"
            elif data_number == 1:
                end = "Pa"
        if study == 1 and phase == 2:
            epochs = all_epochs[data_number]
        for index, ep in enumerate(epochs):
            # Update annotations.description
            for i, idx in enumerate(ep.annotations.description):
                if len(idx.split(delimiter)) > 1:
                    new_anno = idx.split(delimiter)
                    if n_back_ == "n_back":
                        if study == 1:
                            ep.annotations.description[i] = end + "/" + new_anno[0][0:]
                        if study == 2:
                            ep.annotations.description[i] = "All" + "/" + new_anno[1]
                    else:
                        if study == 1:
                            ep.annotations.description[i] = end + "/" + new_anno[1][0] + new_anno[0][1:]
                else:
                    if study == 1:
                        ep.annotations.description[i] = end + "/" + ep.annotations.description[i]
                    elif study == 2:
                        if n_back_ == "n_back":
                            ep.annotations.description[i] = "All" + "/" + ep.annotations.description[i]
                        else:
                            ep.annotations.description[i] = ep.annotations.description[i]
            
            # Update event_id dictionary
            old_event_id = ep.event_id.copy()
            new_event_id = {}
            for old_name, event_code in old_event_id.items():
                if len(old_name.split(delimiter)) > 1:
                    new_anno = old_name.split(delimiter)
                    if n_back_ == "n_back":
                        if study == 1:
                            new_name = end + "/" + new_anno[0][0:]
                        elif study == 2:
                            new_name = "All" + "/" + new_anno[1]
                    else:
                        if study == 1:
                            new_name = end + "/" + new_anno[1][0] + new_anno[0][1:]
                        if study == 2:
                            new_name = old_name
                else:
                    if study == 1:
                        new_name = end + "/" + old_name
                    elif study == 2:
                        if n_back_ == "n_back":
                            new_name = "All" + "/" + old_name
                        else:
                            new_name = old_name
                new_event_id[new_name] = event_code
            
            ep.event_id = new_event_id
            
        if study == 1 and phase == 2:
            epochs_.append(epochs)
    if study == 1 and phase == 2:
        epochs = epochs_[0] + epochs_[1]
    
    
    # if study == 1 and phase == 2:
    #     epochs = datasets[dataLoaders[0]]["all_epochs"] + datasets[dataLoaders[1]]["all_epochs"]
    # elif study == 1 and phase == 1:
    #     epochs = datasets[dataLoaders[0]]["all_epochs"]
    # else:
    #     if Only_responders:
    #         epochs = [ind.epochs for ind in datasets[dataLoaders[0]]["all_individuals"] if ind.name.split("_")[0] in covert_responders]
    #     else:
    #         epochs = [ind.epochs for ind in datasets[dataLoaders[0]]["all_individuals"] if ind.name.split("_")[0] not in covert_responders]

    if study == 1:
        n_backs = np.arange(0, 4)
        control_time = 16
        tmax = 61
    else:
        n_backs = ["Math", "Hard_Math"]
        control_time = 20.005
        tmax = 24.9

    for n_back in n_backs:
        if study == 1:
            picks_ =  [ch for ch in long_channels if ("S1" in ch) or ("S2" in ch) or ("S3" in ch) or ("S4" in ch)] + [ch for ch in datasets[dataLoaders[0]]["all_individuals"][0].raw_haemo.copy().ch_names if (("S1" in ch) or ("S2" in ch) or ("S3" in ch) or ("S4" in ch)) and "hbt" in ch]
        else:
            picks_ =  [ch for ch in long_channels] + [ch for ch in datasets[dataLoaders[0]]["all_individuals"][0].raw_haemo.copy().ch_names if "hbt" in ch]

        if n_back_ == "n_back":
            if study == 1:
                n_back = n_back_
            else:
                n_back = "Arithmetic"
        if n_back_ == "Control":
            n_back = n_back_

        bad_channels = list(set(channel for epoch in epochs for channel in epoch.info['bads']))
        for epoch in epochs:
            epoch.info['bads'] = bad_channels
        # Create evoked data dictionary for each condition
        evoked_dict = {}
        if study == 1:
            if phase == 1:
                all_conditions = list(np.unique(datasets[dataLoaders[0]]["all_epochs"][0].annotations.description))
            elif phase == 2:
                all_conditions = list(np.unique(datasets[dataLoaders[0]]["all_epochs"][0].annotations.description)) + list(np.unique(datasets[dataLoaders[1]]["all_epochs"][0].annotations.description))
        if study == 2:
            all_conditions = list(np.unique(epochs[0].annotations.description))
        if n_back_ == "n_back":
            data_types = list(np.unique([condition.split("/")[0] + "/" + condition.split("/")[1] for condition in all_conditions if str(n_back) in condition]))
        else:
            data_types = list(np.unique([condition for condition in all_conditions if str(n_back) in condition]))
        for data_type in data_types:
            for hemoglobin in ("HbO", "HbR", "HbT"):
                if hemoglobin == "HbO":
                    picks = [ch for ch in picks_ if "hbo" in ch]
                elif hemoglobin == "HbR":
                    picks = [ch for ch in picks_ if "hbr" in ch]
                else:
                    picks = [ch for ch in picks_ if "hbt" in ch]
                # Compute evoked responses per subject
                if n_back == 2 or n_back == "n_back":
                    evoked_list = [epoch.copy()[data_type].copy().pick(picks).crop(tmin=0 , tmax=50).average(picks=picks) for epoch in epochs if data_type in epoch.event_id]
                elif n_back == "Control":
                    evoked_list = [epoch.copy()[data_type].copy().pick(picks).crop(tmin=0 , tmax=control_time).average(picks=picks) for epoch in epochs if data_type in epoch.event_id]
                else: # n_back == "Arithmetic":
                    evoked_list = [epoch.copy()[data_type].copy().pick(picks).crop(tmin=0 , tmax=tmax).average(picks=picks) for epoch in epochs if data_type in epoch.event_id]

                # Rename channels inside each evoked object
                for evoked in evoked_list:
                    evoked.rename_channels(lambda x: x[:-4])

                # Store list of Evoked objects
                evoked_dict[f"{data_type}/{hemoglobin}"] = evoked_list
            
        base_colors = {"hbo": "#AA3377", "hbr": "b", "hbt": "#228833"} 
        color_dict = {key: base_colors[key.split('/')[-1].lower()] for i, key in enumerate(evoked_dict.keys())}

        if study == 1:
            styles_dict = {key: dict(linestyle="-") if "HC" in key else dict(linestyle="--") for key in evoked_dict.keys()}
        else:
            styles_dict = {key: dict(linestyle="-") for key in evoked_dict.keys()}

        # Prepare picks
        if picks_ != "all":
            plotting_picks_ = list(set([s.removesuffix(" hbo").removesuffix(" hbr").removesuffix(" hbt") for s in picks_]))

        combine_strategy: str = "mean"
        hbt_evoked = {key: value for key, value in evoked_dict.items() if "HbT" in key}
        hbt_style = {key: value for key, value in styles_dict.items() if "HbT" in key}
        hbt_color = {key: value for key, value in color_dict.items() if "HbT" in key}
        
        hbo_hbr_evoked = {key: value for key, value in evoked_dict.items() if "HbT" not in key}
        hbo_hbr_style = {key: value for key, value in styles_dict.items() if "HbT" not in key}
        hbo_hbr_color = {key: value for key, value in color_dict.items() if "HbT" not in key}
        
        evoked_dicts = [hbt_evoked, hbo_hbr_evoked]
        styles_dicts = [hbt_style, hbo_hbr_style]
        color_dicts = [hbt_color, hbo_hbr_color]
        for i in range (len(evoked_dicts)):
            evoked_dict = evoked_dicts[i]
            styles_dict = styles_dicts[i]
            color_dict = color_dicts[i]
            chromo = list(evoked_dict.keys())[0].split('/')[-1]
            # Plot evoked data
            if study == 1:
                ylim = dict(hbo=(-0.10, 0.21), hbr=(-0.10, 0.21)) if chromo != "HbT" else dict(hbo=(-0.4, 0.4))
                plot = mne.viz.plot_compare_evokeds(
                    evoked_dict, combine=combine_strategy, ci=0.95, colors=color_dict, styles=styles_dict, show_sensors=True, show=False, picks=plotting_picks_, title="", ylim=ylim)
                if n_back_ == "n_back":
                    filename = os.path.join(rf"C:\Users\NTres\OneDrive - Danmarks Tekniske Universitet\Bachelor_projekt\Results\Study_{study}\Phase_{phase}\Neural_correlates", f"standard_fNIRS_response_plot_n_back_{chromo}.pdf")
                elif n_back == "Control":
                    filename = os.path.join(rf"C:\Users\NTres\OneDrive - Danmarks Tekniske Universitet\Bachelor_projekt\Results\Study_{study}\Phase_{phase}\Neural_correlates", f"standard_fNIRS_response_plot_control_{chromo}.pdf")
                else:
                    filename = os.path.join(rf"C:\Users\NTres\OneDrive - Danmarks Tekniske Universitet\Bachelor_projekt\Results\Study_{study}\Phase_{phase}\Neural_correlates", f"standard_fNIRS_response_plot_{n_back}_back_{chromo}.pdf")

            elif study == 2:
                ylim = dict(hbo=(-0.01, 0.01), hbr=(-0.01, 0.01)) if chromo != "HbT" else None
                plot = mne.viz.plot_compare_evokeds(
                    evoked_dict, combine=combine_strategy, ci=0.95, colors=color_dict, styles=styles_dict, show_sensors=True, show=False, picks=plotting_picks_, title="", ylim=ylim)
                if n_back_ == "n_back":
                    if Only_responders:
                        filename = os.path.join(rf"C:\Users\NTres\OneDrive - Danmarks Tekniske Universitet\Bachelor_projekt\Results\Study_{study}\Phase_{phase}\Neural_correlates", f"standard_fNIRS_response_plot_all_math_responders_{chromo}.pdf")
                    else:
                        filename = os.path.join(rf"C:\Users\NTres\OneDrive - Danmarks Tekniske Universitet\Bachelor_projekt\Results\Study_{study}\Phase_{phase}\Neural_correlates", f"standard_fNIRS_response_plot_all_math_non-responders_{chromo}.pdf")
                elif n_back == "Control":
                    if Only_responders:
                        filename = os.path.join(rf"C:\Users\NTres\OneDrive - Danmarks Tekniske Universitet\Bachelor_projekt\Results\Study_{study}\Phase_{phase}\Neural_correlates", f"standard_fNIRS_response_plot_control_responders_{chromo}.pdf")
                    else:
                        filename = os.path.join(rf"C:\Users\NTres\OneDrive - Danmarks Tekniske Universitet\Bachelor_projekt\Results\Study_{study}\Phase_{phase}\Neural_correlates", f"standard_fNIRS_response_plot_control_non-responders_{chromo}.pdf")
                else:
                    if Only_responders:
                        filename = os.path.join(rf"C:\Users\NTres\OneDrive - Danmarks Tekniske Universitet\Bachelor_projekt\Results\Study_{study}\Phase_{phase}\Neural_correlates", f"standard_fNIRS_response_plot_{n_back}_responders_{chromo}.pdf")
                    else:
                        filename = os.path.join(rf"C:\Users\NTres\OneDrive - Danmarks Tekniske Universitet\Bachelor_projekt\Results\Study_{study}\Phase_{phase}\Neural_correlates", f"standard_fNIRS_response_plot_{n_back}_non-responders_{chromo}.pdf")
            
            plot[0].savefig(filename)
            print(f"Plot saved as {filename}")
            plt.close(plot[0])  # Close the figure after saving
            


# bad_channels = list(set(channel for epoch in epochs for channel in epoch.info['bads']))
# for epoch in epochs:
#     epoch.info['bads'] = bad_channels
# epochs = mne.concatenate_epochs(epochs)
# epochs_ = epochs.copy().pick(long_channels)

# topomap_args = dict(extrapolate="local")

# fig = epochs_.copy()[relevant_data_types].pick(picks_).pick("hbo").crop(tmin=0, tmax=tmax).average().plot_joint(
#     times=times, topomap_args=topomap_args
# )
# if study == 1:
#     filename = os.path.join(rf"C:\Users\NTres\OneDrive - Danmarks Tekniske Universitet\Bachelor_projekt\Results\Study_{study}\Phase_{phase}\Neural_correlates", f"Topographic_time_representation.pdf")
# else:
#     if Only_responders:
#         filename = os.path.join(rf"C:\Users\NTres\OneDrive - Danmarks Tekniske Universitet\Bachelor_projekt\Results\Study_{study}\Phase_{phase}\Neural_correlates", f"Topographic_time_representation_responders.pdf")
#     else:
#         filename = os.path.join(rf"C:\Users\NTres\OneDrive - Danmarks Tekniske Universitet\Bachelor_projekt\Results\Study_{study}\Phase_{phase}\Neural_correlates", f"Topographic_time_representation_non_responders.pdf")
# fig.savefig(filename)
# plt.close(fig)

# if study == 1:
#     fig, axes = plt.subplots(
#         nrows=2,
#         ncols=6,
#         figsize=(9, 5),
#         gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1, 0.1]),
#         layout="constrained",
#     )
#     if phase == 1:
#         vlim = (-.1, .1)
#     else:
#         vlim = (-.15, .15)
#     ts = tmax

#     evoked_0_back = epochs_["n_back/0_back"].average()
#     evoked_1_back = epochs_["n_back/1_back"].average()
#     evoked_2_back = epochs_["n_back/2_back"].average()
#     evoked_3_back = epochs_["n_back/3_back"].average()

#     evoked_0_back.plot_topomap(
#         ch_type="hbo", times=ts, axes=axes[0, 0], vlim=vlim, colorbar=False, **topomap_args
#     )
#     evoked_0_back.plot_topomap(
#         ch_type="hbr", times=ts, axes=axes[1, 0], vlim=vlim, colorbar=False, **topomap_args
#     )
#     evoked_1_back.plot_topomap(
#         ch_type="hbo", times=ts, axes=axes[0, 1], vlim=vlim, colorbar=False, **topomap_args
#     )
#     evoked_1_back.plot_topomap(
#         ch_type="hbr", times=ts, axes=axes[1, 1], vlim=vlim, colorbar=False, **topomap_args
#     )
#     evoked_2_back.plot_topomap(
#         ch_type="hbo", times=ts, axes=axes[0, 2], vlim=vlim, colorbar=False, **topomap_args
#     )
#     evoked_2_back.plot_topomap(
#         ch_type="hbr", times=ts, axes=axes[1, 2], vlim=vlim, colorbar=False, **topomap_args
#     )
#     evoked_3_back.plot_topomap(
#         ch_type="hbo", times=ts, axes=axes[0, 3], vlim=vlim, colorbar=False, **topomap_args
#     )
#     evoked_3_back.plot_topomap(
#         ch_type="hbr", times=ts, axes=axes[1, 3], vlim=vlim, colorbar=False, **topomap_args
#     )

#     evoked_diff = mne.combine_evoked([evoked_0_back, evoked_1_back, evoked_2_back, evoked_3_back], weights="equal")

#     evoked_diff.plot_topomap(
#         ch_type="hbo", times=ts, axes=axes[0, 4:], vlim=vlim, colorbar=True, **topomap_args
#     )
#     evoked_diff.plot_topomap(
#         ch_type="hbr", times=ts, axes=axes[1, 4:], vlim=vlim, colorbar=True, **topomap_args
#     )

#     for column, condition in enumerate(["0-back", "1-back", "2-back", "3-back", "n-back"]):
#         for row, chroma in enumerate(["HbO", "HbR"]):
#             axes[row, column].set_title(f"{chroma}: {condition}")
#     filename = os.path.join(rf"C:\Users\NTres\OneDrive - Danmarks Tekniske Universitet\Bachelor_projekt\Results\Study_{study}\Phase_{phase}\Neural_correlates", f"Topographic_representation_n_back.pdf")

# elif study == 2:
#     tmax = 25
#     fig, axes = plt.subplots(
#     nrows=2,
#     ncols=4,
#     figsize=(9, 5),
#     gridspec_kw=dict(width_ratios=[1, 1, 1, 0.1]),
#     layout="constrained",
#     )
#     vlim = (-.014, .014)
#     ts = tmax

#     evoked_math = epochs_["Math"].average()
#     evoked_hard_math = epochs_["Hard_Math"].average()

#     evoked_math.plot_topomap(
#         ch_type="hbo", times=ts, axes=axes[0, 0], vlim=vlim, colorbar=False, **topomap_args
#     )
#     evoked_math.plot_topomap(
#         ch_type="hbr", times=ts, axes=axes[1, 0], vlim=vlim, colorbar=False, **topomap_args
#     )
#     evoked_hard_math.plot_topomap(
#         ch_type="hbo", times=ts, axes=axes[0, 1], vlim=vlim, colorbar=False, **topomap_args
#     )
#     evoked_hard_math.plot_topomap(
#         ch_type="hbr", times=ts, axes=axes[1, 1], vlim=vlim, colorbar=False, **topomap_args
#     )

#     evoked_diff = mne.combine_evoked([evoked_math, evoked_hard_math], weights="equal")

#     evoked_diff.plot_topomap(
#         ch_type="hbo", times=ts, axes=axes[0, 2:], vlim=vlim, colorbar=True, **topomap_args
#     )
#     evoked_diff.plot_topomap(
#         ch_type="hbr", times=ts, axes=axes[1, 2:], vlim=vlim, colorbar=True, **topomap_args
#     )

#     for cbar in fig.get_axes():
#         # Check if this axis is a colorbar axis (they typically have different properties)
#         if cbar.get_title() == 'µM':
#             cbar.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%.3f'))

#     for column, condition in enumerate(["Math", "Hard_Math", "All Math"]):
#         for row, chroma in enumerate(["HbO", "HbR"]):
#             axes[row, column].set_title(f"{chroma}: {condition}")
#     if Only_responders:
#         filename = os.path.join(rf"C:\Users\NTres\OneDrive - Danmarks Tekniske Universitet\Bachelor_projekt\Results\Study_{study}\Phase_{phase}\Neural_correlates", f"Topographic_representation_Math_responders.pdf")
#     else:
#         filename = os.path.join(rf"C:\Users\NTres\OneDrive - Danmarks Tekniske Universitet\Bachelor_projekt\Results\Study_{study}\Phase_{phase}\Neural_correlates", f"Topographic_representation_Math_non_responders.pdf")
# plt.savefig(filename)

