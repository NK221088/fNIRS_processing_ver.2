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

from mne import Annotations

load_dotenv()
save_path = Path(os.getenv(rf"data_save_path"))
# Study_1_phase_1_neural_correlates_save_path = Path(os.getenv(rf"Study_1_phase_1_neural_correlates_save_path"))
# Phase_1_assumptions_plot_save_path = Path(os.getenv(rf"S1RQ1_assumptions_plot_save_path"))
# Phase_1_ANOVA_save_path = Path(os.getenv(rf"S1RQ1_ANOVA_save_path"))
# Phase_2_assumptions_plot_save_path = Path(os.getenv(rf"Phase_2_assumptions_plot_save_path"))

# Phase_2_ANOVA_save_path = Path(os.getenv(rf"Phase_2_ANOVA_save_path"))
# Phase_3_assumptions_plot_save_path = Path(os.getenv(rf"Study_2_Phase_2_assumptions_plot_save_path"))
# Phase_3_ANOVA_save_path = Path(os.getenv(rf"Study_2_Phase_2_ANOVA_save_path"))
drug_path = Path(os.getenv(rf"Marwan_drug_data"))


# follow_up_results_save_path = Path(os.getenv(rf"follow_up_results_save_path"))

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

load_dotenv()
responders_count_path = Path(os.getenv("Marwan_responders_count"))
df = pd.read_csv(rf"{responders_count_path}", index_col=0)
count_data = df['count'].to_dict()
total_data = df['Total count'].to_dict()
total_number_of_patients = 50
threshold = 0.4
print("Responders:")
all_responders = list(count_data.keys())
covert_responders = [ID for ID, count in count_data.items() if count / total_data[ID] >= threshold]
non_covert_responders = [ID for ID, count in count_data.items() if count / total_data[ID] < threshold]
non_responders = [ID for ID in all_responders if ID not in covert_responders and ID not in non_covert_responders]

dataSetList = list(data_loaders.keys())
dataLoaders = [dataSetList[19]] #, dataSetList[17]]
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

long_channels = mne_nirs.channels.get_long_channels(datasets[dataLoaders[0]]["all_individuals"][0].raw_haemo.copy()).ch_names

all_conditions = list(np.unique(datasets[dataLoaders[0]]["all_epochs"][0].annotations.description))

_epochs = [ind.epochs for ind in datasets[dataLoaders[0]]["all_individuals"]]
names = [ind.name for ind in datasets[dataLoaders[0]]["all_individuals"]]

relevant_data_types = list(np.unique([condition for condition in all_conditions if "Math" in condition]))
picks_ =  [ch for ch in long_channels]
all_n_back_ = ["Math", "Hard_Math", "Control"] #"single", 

max_plot = True
first_time = True

ratio_data = {name: count_data[name] / total_data[name] for name in list(count_data.keys())}
if max_plot:
    indices_sorted = np.array([list(ratio_data.values())]).flatten().argsort()
    max_ind = [list(ratio_data.keys())[indices_sorted[0]], list(ratio_data.keys())[indices_sorted[-1]]]
    max_idx = [idx for idx, name in enumerate(names) if name.split("_")[0] in max_ind]
    _epochs = [_epochs[i] for i in max_idx]
    names = [names[i] for i in max_idx]
    

for n_back_ in all_n_back_:
    epochs = _epochs.copy()
    if first_time:
        for data_number in range(len(number_of_subjects)):
            for index, ep in enumerate(epochs):
                group = "Covert" if names[index].split("_")[0] in covert_responders else "Non-Covert"
                new_descriptions = []
                for i, idx in enumerate(ep.annotations.description):
                    new_desc = group + "/" + ep.annotations.description[i]
                    new_descriptions.append(new_desc)
                
                new_annotations = Annotations(onset=ep.annotations.onset,
                                            duration=ep.annotations.duration,
                                            description=new_descriptions)
                ep.set_annotations(new_annotations)
                
                # Update event_id dictionary
                old_event_id = ep.event_id.copy()
                new_event_id = {}
                for old_name, event_code in old_event_id.items():
                    new_name = group + "/" + old_name
                    new_event_id[new_name] = event_code
                
                ep.event_id = new_event_id
                
        first_time = False
    if n_back_ == "single":
        n_backs = ["Covert/Arithmetic", "Non-Covert/Arithmetic"] # 
    elif n_back_ == "Control":
        n_backs = ["Covert/Control", "Non-Covert/Control"]
    elif n_back_ == "Math":
        n_backs = ['Covert/Math', 'Non-Covert/Math']
    elif n_back_ == "Hard_Math":
        n_backs = ['Covert/Hard_Math', 'Non-Covert/Hard_Math']
    control_time = 20.005
    tmax = 24.9

    picks_ =  [ch for ch in long_channels] + [ch for ch in datasets[dataLoaders[0]]["all_individuals"][0].raw_haemo.copy().ch_names if "hbt" in ch]

    bad_channels = list(set(channel for epoch in epochs for channel in epoch.info['bads']))
    for epoch in epochs:
        epoch.info['bads'] = bad_channels

    # Create evoked data dictionary for each condition
    evoked_dict = {}
    for data_type in n_backs:
        for hemoglobin in ("HbO", "HbR", "HbT"):
            picks = [ch for ch in picks_ if hemoglobin.lower() in ch]
            # Compute evoked responses per subject
            if n_back_ == "single":
                evoked_list = [epoch.copy()[data_type].copy().pick(picks).crop(tmin=0 , tmax=tmax).average(picks=picks) for epoch in epochs if data_type in [id.split("/")[0] + "/" + id.split("/")[1] for id in epoch.event_id]]
            elif n_back_ == "Math" or n_back_ == "Hard_Math":
                evoked_list = [epoch.copy()[data_type].copy().pick(picks).crop(tmin=0 , tmax=tmax).average(picks=picks) for epoch in epochs if data_type in epoch.event_id]
            elif n_back_ == "Control":
                evoked_list = [epoch.copy()[data_type].copy().pick(picks).crop(tmin=0 , tmax=control_time).average(picks=picks) for epoch in epochs if data_type in epoch.event_id]

            # Rename channels inside each evoked object
            for evoked in evoked_list:
                evoked.rename_channels(lambda x: x[:-4])

            # Store list of Evoked objects
            evoked_dict[f"{data_type}/{hemoglobin}"] = evoked_list
        
    base_colors = {"Pa": "#AA3377", "HC": "b", "Covert": "b", "Non-Covert": "#AA3377"} 
    color_dict = {key: base_colors[key.split('/')[0]] for i, key in enumerate(evoked_dict.keys())}

    base_styles = {"hbo": "-", "hbr": "--", "hbt": "-."} 
    styles_dict = {key: {"linestyle": base_styles[key.split('/')[-1].lower()]} for i, key in enumerate(evoked_dict.keys())}

    # Prepare picks
    if picks_ != "all":
        plotting_picks_ = list(set([s.removesuffix(" hbo").removesuffix(" hbr").removesuffix(" hbt") for s in picks_]))

    combine_strategy: str = "mean"
    hbt_evoked = {key: value for key, value in evoked_dict.items() if "HbT" in key}
    hbt_style = {key: value for key, value in styles_dict.items() if "HbT" in key}
    hbt_color = {key: value for key, value in color_dict.items() if "HbT" in key}

    hbo_evoked = {key: value for key, value in evoked_dict.items() if "HbO" in key}
    hbo_style = {key: value for key, value in styles_dict.items() if "HbO" in key}
    hbo_color = {key: value for key, value in color_dict.items() if "HbO" in key}

    hbr_evoked = {key: value for key, value in evoked_dict.items() if "HbR" in key}
    hbr_style = {key: value for key, value in styles_dict.items() if "HbR" in key}
    hbr_color = {key: value for key, value in color_dict.items() if "HbR" in key}

    evoked_dicts = [hbt_evoked, hbo_evoked, hbr_evoked]
    styles_dicts = [hbt_style, hbo_style, hbr_style]
    color_dicts = [hbt_color, hbo_color, hbr_color]
    
    from scipy import stats
    def manual_conf_int(data):
        confidence_level = 0.95
        mean = np.mean(data, axis=0)
        sem = stats.sem(data, axis=0)  # Standard error of the mean across channels

        # Number of channels (samples) - this is what we're averaging over
        n_channels = data.shape[0]
        ci = sem * stats.t.ppf((1 + confidence_level) / 2, n_channels - 1)
        lower_bound = mean - ci
        upper_bound = mean + ci
        conf_int = (lower_bound, upper_bound)
        return conf_int
    
    for i in range (len(evoked_dicts)):
        evoked_dict = evoked_dicts[i]
        styles_dict = styles_dicts[i]
        color_dict = color_dicts[i]
        chromo = list(evoked_dict.keys())[0].split('/')[-1]
        # Plot evoked data

        ylim = dict(hbo=(-0.015, 0.015), hbr=(-0.01, 0.01)) if chromo != "HbT" else dict(hbo=(-0.06, 0.1))
        plot = mne.viz.plot_compare_evokeds(
            evoked_dict, combine=combine_strategy, ci=0.95, colors=color_dict, styles=styles_dict, show_sensors=True, show=False, picks=plotting_picks_, title="", ylim=ylim)
        if n_back_ == "single":
            filename = os.path.join(rf"C:\Users\NTres\OneDrive - Danmarks Tekniske Universitet\Arbejde_Rigshospitalet\fNIRS\Test_plots", f"standard_fNIRS_response_plot_arithmetic_{chromo}.pdf")
        elif n_back_ == "Control":
            filename = os.path.join(rf"C:\Users\NTres\OneDrive - Danmarks Tekniske Universitet\Arbejde_Rigshospitalet\fNIRS\Test_plots", f"standard_fNIRS_response_plot_control_{chromo}.pdf")
        else:
            n_back_file_name = data_type.split("/")[-1]
            filename = os.path.join(rf"C:\Users\NTres\OneDrive - Danmarks Tekniske Universitet\Arbejde_Rigshospitalet\fNIRS\Test_plots", f"standard_fNIRS_response_plot_{n_back_file_name}_{chromo}.pdf")
        if chromo != "HbT":
            # Get the figure and axes
            fig = plot[0]
            axes = fig.get_axes()
            ax = axes[0]  # The main plot axis (axes[1] is usually the sensor plot)

            # Now compute and add confidence intervals for each condition
            for key, evoked_list in evoked_dict.items():
                # Stack the data from all evoked objects: (n_channels, n_times)
                data = np.array([evoked.get_data() for evoked in evoked_list]).squeeze()
                
                # Scale the data first
                data_scaled = data * 10**6
                
                # Compute statistics across channels (axis=0) on scaled data
                mean = np.mean(data_scaled, axis=0)
                sem = stats.sem(data_scaled, axis=0)  # SEM on scaled data
                n_channels = data_scaled.shape[0]
                ci_width = sem * stats.t.ppf(0.975, n_channels - 1)
                
                lower = mean - ci_width
                upper = mean + ci_width
                
                # Get time axis
                times = evoked_list[0].times
                
                # Add shaded confidence interval
                color = color_dict[key]
                # ax.fill_between(times, lower[0], upper[0], alpha=0.2, color=color)
            fig.tight_layout()
            fig.savefig(filename)
            print(f"Plot saved as {filename}")
            plt.close(fig)  # Close the figure after saving
        else:
            plot[0].savefig(filename)
            print(f"Plot saved as {filename}")
            plt.close(plot[0])  # Close the figure after saving
            

# Only_responders = True
# epochs = _epochs.copy()
# bad_channels = list(set(channel for epoch in epochs for channel in epoch.info['bads']))
# for epoch in epochs:
#     epoch.info['bads'] = bad_channels
# epochs_ = mne.concatenate_epochs(epochs)
# epochs_ = epochs_.copy().pick(long_channels)
# topomap_args = dict(extrapolate="local")

# for hemoglobin in ("HbO", "HbR"):
#     if hemoglobin == "HbO":
#         picks = [ch for ch in epochs_.ch_names if "hbo" in ch]
#     elif hemoglobin == "HbR":
#         picks = [ch for ch in epochs_.ch_names if "hbr" in ch]
#     if study == 1:
#         fig = epochs_.copy()[relevant_data_types].pick(picks).crop(tmin=0, tmax=50).average().plot_joint(
#         times=times, topomap_args=topomap_args
#         )
#         filename = os.path.join(rf"C:\Users\NTres\OneDrive - Danmarks Tekniske Universitet\Bachelor_projekt\Results\Study_{study}\Phase_{phase}\Neural_correlates", f"Topographic_time_representation_{hemoglobin}.pdf")
#     elif study == 2:
#         if Only_responders:
#             relevant_data_types = [cond for cond in list(epochs_.event_id.keys()) if "Non-Covert" not in cond and "Math" in cond]
#             epochs_ = epochs.copy()
#             epochs_ = [epoch.copy()[data_type].copy() for epoch in epochs_ for data_type in relevant_data_types if data_type in epoch.event_id]
#             epochs_ = mne.concatenate_epochs(epochs_)
#             epochs_ = epochs_.copy().pick(long_channels)
#             fig = epochs_.copy()[relevant_data_types].pick(picks).crop(tmin=0, tmax=25).average().plot_joint(
#             times=times, topomap_args=topomap_args
#             )
#             filename = os.path.join(rf"C:\Users\NTres\OneDrive - Danmarks Tekniske Universitet\Bachelor_projekt\Results\Study_{study}\Phase_{phase}\Neural_correlates", f"Topographic_time_representation_responders_{hemoglobin}.pdf")
#         else:
#             relevant_data_types = [cond for cond in list(epochs_.event_id.keys()) if "Non-Covert" in cond and "Math" in cond]
#             epochs_ = epochs.copy()
#             epochs_ = [epoch.copy()[data_type].copy() for epoch in epochs_ for data_type in relevant_data_types if data_type in epoch.event_id]
#             epochs_ = mne.concatenate_epochs(epochs_)
#             epochs_ = epochs_.copy().pick(long_channels)
#             fig = epochs_.copy()[relevant_data_types].pick(picks).crop(tmin=0, tmax=25).average().plot_joint(
#             times=times, topomap_args=topomap_args
#             )
#             filename = os.path.join(rf"C:\Users\NTres\OneDrive - Danmarks Tekniske Universitet\Bachelor_projekt\Results\Study_{study}\Phase_{phase}\Neural_correlates", f"Topographic_time_representation_non_responders_{hemoglobin}.pdf")
#     fig.savefig(filename)
#     plt.close(fig)

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
    
#     if Only_responders:
#         relevant_data_types = [cond for cond in list(epochs_.event_id.keys()) if "Non-Covert" not in cond and "Math" in cond]
#         epochs_ = epochs.copy()
#         epochs_ = [epoch.copy()[data_type].copy() for epoch in epochs_ for data_type in relevant_data_types if data_type in epoch.event_id]
#         epochs_ = mne.concatenate_epochs(epochs_)
#         epochs_ = epochs_.copy().pick(long_channels)
#     else:
#         relevant_data_types = [cond for cond in list(epochs_.event_id.keys()) if "Non-Covert" in cond and "Math" in cond]
#         epochs_ = epochs.copy()
#         epochs_ = [epoch.copy()[data_type].copy() for epoch in epochs_ for data_type in relevant_data_types if data_type in epoch.event_id]
#         epochs_ = mne.concatenate_epochs(epochs_)
#         epochs_ = epochs_.copy().pick(long_channels)
#     evoked_math = epochs_.copy()["Arithmetic/Math"].average()
#     evoked_hard_math = epochs_.copy()["Arithmetic/Hard_Math"].average()
        
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

