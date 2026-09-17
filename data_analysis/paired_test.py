import sys
import os
from dotenv import load_dotenv
from pathlib import Path
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
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from collections import Counter
from mne.stats import permutation_t_test

load_dotenv()
save_path = Path(os.getenv(rf"Evoked_plots_path_paired"))
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
        "filter_upper_value": 0.8,
        "h_trans_bandwidth": 0.01,           
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

long_channels = mne_nirs.channels.get_long_channels(all_participants[0].raw_haemo.copy().pick(channel_names)).ch_names #  [ch for ch in all_participants[0].raw_haemo.ch_names if "hbt" in ch] # 

individual_recording_analysis = False
session_analysis = True
names = [ind.name for ind in all_participants] # List of all patient names
all_epochs = [ind.epochs for ind in all_participants] # All epochs
first_names = [name.split("_")[0] for name in names] # First names of all patients
name_indices = {first_name: [ind for ind, name in enumerate(names) if name.split("_")[0] == first_name] for first_name in first_names} # Find indices of recordings belonging to same patient
name_epoch_map = {first_name: [[name, ind] for ind, name in enumerate(names) if name.split("_")[0] == first_name] for first_name in first_names} # Use indicies above to find the corresponding epochs for each patient
session_epoch_map = {first_name: defaultdict(list) for first_name in first_names}
session_epoch_bad_channels = {first_name: defaultdict(list) for first_name in first_names}

color_dict = {
    "Math": "#AA3377",
    "Hard Math": "g",
    "Control": "b"
}

for key, value in name_epoch_map.items():
    for subvalue in value:
        session_epoch_map[key][subvalue[0].split("_")[1]].append(all_epochs[subvalue[1]]) # Collect all epochs from same sessions for each patient
        session_epoch_bad_channels[key][subvalue[0].split("_")[1]].extend(all_epochs[subvalue[1]].copy().pick(long_channels).info["bads"]) # Collect all bad channels from same sessions for each patient

if session_analysis:
    all_updated = {}
    for ind, sessions in session_epoch_bad_channels.items():
        for session, bad_channels in sessions.items():
            n_recordings = len(session_epoch_map[ind][session])
            min_fraction = round(1/3 * n_recordings)
            bad_channel_counts = Counter(bad_channels)
            session_epoch_bad_channels[ind][session] = [ch for ch, count in bad_channel_counts.items() if count > min_fraction]
            for recording in session_epoch_map[ind][session]:
                recording.info["bads"] = session_epoch_bad_channels[ind][session]
            session_epoch_map[ind][session] = mne.concatenate_epochs(session_epoch_map[ind][session])
            all_updated[ind + "_" + session] = session_epoch_map[ind][session]

individual_epochs = {first_name: [all_epochs[i].copy().pick(long_channels) for i in name_indices[first_name]] for first_name in first_names}
channel_counts = {}
math_lenghts = []
hard_math_lengths = []
control_lengths = []

math_mean = []
hard_math_means = []
control_means = []

from mne.stats import permutation_cluster_test

cluster_results = []
paired_mean_results = []
mean_results = []

if individual_recording_analysis:
    individual_epochs = {ind.name: ind.epochs for ind in all_participants}
if session_analysis:
    individual_epochs = all_updated
for ind, epochs in individual_epochs.items():
    if individual_recording_analysis:
    #     # bad_channels = session_epoch_bad_channels[ind.split("_")[0]][ind.split("_")[1]]
        bad_channels = epochs.info["bads"]
    #     # individual_epochs[ind] = epochs
    elif session_analysis:
        bad_channels = session_epoch_bad_channels[ind.split("_")[0]][ind.split("_")[1]]
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

    math_HbO = individual_epochs[ind].copy()["Math"].pick(good_long_channels).crop(math_t_start, math_t_end, True).get_data().mean(axis=1)
    Hard_math_HbO = individual_epochs[ind].copy()["Hard_Math"].pick(good_long_channels).crop(math_t_start, math_t_end, True).get_data().mean(axis=1)
    Control_HbO = individual_epochs[ind].copy()["Control"].pick(good_long_channels).crop(control_t_start, control_t_end, True).get_data().mean(axis=1)

    from scipy.stats import permutation_test

    def paired_diff_statistic(x, y, axis=0):
        """Mean of paired differences (x - y)."""
        return np.mean(x - y, axis=axis)


    def run_paired_test(cond_means, preceding_control_means, label, n_resamples=10000, seed=42):
        """
        Approach 1: paired test — each condition epoch vs. its immediately 
        preceding Control epoch. Sign-flip permutation test on the differences.
        """
        result = permutation_test(
            (cond_means, preceding_control_means),
            paired_diff_statistic,
            permutation_type='samples',   # paired/sign-flip permutations
            n_resamples=n_resamples,
            alternative='greater',      # switch to 'greater' for one-sided cond > control
            random_state=seed,
        )
        return {
            "n_pairs": len(cond_means),
            "mean_diff": np.mean(cond_means - preceding_control_means),
            "p_value": result.pvalue,
        }

    n_blocks = len(math_HbO_mean) // 5
    block = np.array([0, 1, 2, 3, 4])
    math_controls = np.concatenate([block + 10 * i for i in range(n_blocks)])
    hard_math_controls = [i for i in range(len(Control_HbO_mean)) if i not in math_controls]

    math_paired_result      = run_paired_test(math_HbO_mean, Control_HbO_mean[math_controls], "Math_vs_Control_paired")
    hardmath_paired_result  = run_paired_test(Hard_math_HbO_mean, Control_HbO_mean[hard_math_controls], "HardMath_vs_Control_paired")

    paired_mean_results.append({"ID": ind, **{f"math_{k}": v for k, v in math_paired_result.items()},
                                          **{f"hardmath_{k}": v for k, v in hardmath_paired_result.items()}})

    def epochs_to_evoked_list(epochs, picks, hbo_pick="hbo"):
        """Convert an Epochs object to a list of single-trial Evoked objects."""
        epochs_picked = epochs.pick(picks, verbose=False).pick(hbo_pick, verbose=False)
        return [epochs_picked[i].average() for i in range(len(epochs_picked))]

    evoked_dict = {
        "Math": epochs_to_evoked_list(individual_epochs[ind]["Math"], good_long_channels),
        "Hard Math": epochs_to_evoked_list(individual_epochs[ind]["Hard_Math"], good_long_channels),
        "Control": epochs_to_evoked_list(individual_epochs[ind]["Control"], good_long_channels),
    }

    fig = mne.viz.plot_compare_evokeds(
        evoked_dict,
        combine="mean",
        ci=0.95,
        colors=color_dict,
        show=False,
        title=f"Patient: {ind}"
    )


    ax = fig[0].axes[0]

    fig_new, ax_new = plt.subplots(figsize=(8, 6))
    ax_new.spines['top'].set_visible(False)
    ax_new.spines['right'].set_visible(False)

    data = {}
    for line in ax.lines:
        ax_new.plot(line.get_xdata(), line.get_ydata(), 
                    color=line.get_color(), 
                    linestyle=line.get_linestyle(),
                    linewidth=line.get_linewidth(),
                    label=line.get_label())
        data[line.get_label()] = [line.get_xdata(), line.get_ydata()]


    for collection in ax.collections:
        if isinstance(collection, PolyCollection):
            new_col = PolyCollection(
                [p.vertices for p in collection.get_paths()],
                facecolor=collection.get_facecolor(),
                edgecolor=collection.get_edgecolor(),
                alpha=collection.get_alpha(),
            )
            ax_new.add_collection(new_col)
        else:
            ax_new.add_collection(collection)  # your original line, still fine for anything else

    ax_new.set_xlim(ax.get_xlim()[0], 25)
    ax_new.set_ylim(ax.get_ylim())
    ax_new.set_xlabel(ax.get_xlabel())
    ax_new.set_ylabel(ax.get_ylabel())
    ax_new.set_title(ax.get_title())
    ax_new.axvline(x=20, color='black', linestyle='--', linewidth=1, label='End of Control Epochs')
    ax_new.legend()
    ax_new.axvline(x=0, color='black', linestyle='--', linewidth=1)

    filename = os.path.join(save_path, f"standard_fNIRS_response_plot_{ind}.pdf")
    fig_new.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close(fig_new)
    plt.close(fig[0])

    
    plt.figure(figsize=(12, 5))
    control_counter = 0
    math_counter = 0
    labels = []

    for i in range(len(math_HbO_mean) + len(Control_HbO_mean[math_controls])):
        pos = i + 1
        if i % 2 == 0:
            bp = plt.boxplot(Control_HbO[math_controls][control_counter], positions=[pos], patch_artist=True)
            bp['boxes'][0].set_facecolor('yellow')
            labels.append(f"Control {control_counter}")
            control_counter += 1
        else:
            bp = plt.boxplot(math_HbO[math_counter], positions=[pos], patch_artist=True)
            bp['boxes'][0].set_facecolor('green')
            labels.append(f"Math {math_counter}")
            math_counter += 1

    plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha='right')

    from matplotlib.patches import Patch
    plt.legend(handles=[
        Patch(facecolor='yellow', label='Control'),
        Patch(facecolor='green', label='Math')
    ])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"math_boxplot_{ind}.pdf"), format="pdf", bbox_inches="tight")
    plt.close()


    plt.figure(figsize=(12, 5))
    control_counter = 0
    math_counter = 0
    labels = []

    for i in range(len(Hard_math_HbO_mean) + len(Control_HbO_mean[hard_math_controls])):
        pos = i + 1
        if i % 2 == 0:
            bp = plt.boxplot(Control_HbO[hard_math_controls][control_counter], positions=[pos], patch_artist=True)
            bp['boxes'][0].set_facecolor('yellow')
            labels.append(f"Control {control_counter}")
            control_counter += 1
        else:
            bp = plt.boxplot(Hard_math_HbO[math_counter], positions=[pos], patch_artist=True)
            bp['boxes'][0].set_facecolor('green')
            labels.append(f"Math {math_counter}")
            math_counter += 1

    plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha='right')

    from matplotlib.patches import Patch
    plt.legend(handles=[
        Patch(facecolor='yellow', label='Control'),
        Patch(facecolor='green', label='Math')
    ])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"hard_math_boxplot_{ind}.pdf"), format="pdf", bbox_inches="tight")
    plt.close()

paired_mean_df = pd.DataFrame(paired_mean_results)

if individual_recording_analysis:
    paired_mean_df[["ID_prefix", "Session", "Recording"]] = paired_mean_df["ID"].str.split("_", expand=True)
else:
    paired_mean_df[["ID_prefix", "Session"]] = paired_mean_df["ID"].str.split("_", expand=True)

states = pd.read_excel(consciousness_states_path)
states = states[["Subject", "Consciousness"]]
states["Session"] = states.groupby("Subject").cumcount() + 1
states["Subject"] = "P" + states["Subject"].astype(str)
states["Session"] = "S" + states["Session"].astype(str)

# Merge the state
paired_mean_df = paired_mean_df.merge(
    states[["Subject", "Session", "Consciousness"]],
    left_on=["ID_prefix", "Session"],
    right_on=["Subject", "Session"],
    how="left"
)

# paired_mean_df = paired_mean_df.drop(columns=["Subject", "Session", "recording"])


from statsmodels.stats.multitest import fdrcorrection
# Apply FDR correction within each ID
def fdr_combined(g):
    pvals = pd.concat(
        [g["math_p_value"], g["hardmath_p_value"]],
        keys=["math", "hardmath"]
    )
    corrected = fdrcorrection(pvals.values)[1]
    corrected_series = pd.Series(corrected, index=pvals.index)
    return pd.DataFrame({
        "math_p_value_fdr": corrected_series.loc["math"].values,
        "hardmath_p_value_fdr": corrected_series.loc["hardmath"].values,
    }, index=g.index)

result = paired_mean_df.groupby("ID_prefix", group_keys=False).apply(fdr_combined)
paired_mean_df[["math_p_value_fdr", "hardmath_p_value_fdr"]] = result
paired_mean_df = paired_mean_df.drop(columns=['Subject'])
print("debug")