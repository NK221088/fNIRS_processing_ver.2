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
save_path = Path(os.getenv(rf"Evoked_plots_path"))


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

long_channels = mne_nirs.channels.get_long_channels(all_participants[0].raw_haemo.copy().pick(channel_names)).ch_names #  [ch for ch in all_participants[0].raw_haemo.ch_names if "hbt" in ch] # 

names = [ind.name for ind in all_participants]
all_epochs = [ind.epochs for ind in all_participants]
first_names = [name.split("_")[0] for name in names]
name_indices = {first_name: [ind for ind, name in enumerate(names) if name.split("_")[0] == first_name] for first_name in first_names}
individual_epochs = {first_name: [all_epochs[i].copy().pick(long_channels) for i in name_indices[first_name]] for first_name in first_names}

PCA_epochs = False
if PCA_epochs:
    from construct_PCA_components import construct_PCA_components
    PCA_components, good_channel_names = construct_PCA_components(datasets[dataLoaders[0]]["all_individuals"], channel_names=channel_names)
    individual_epochs = {first_name: [all_epochs[i].copy().pick(good_channel_names[first_name]) for i in name_indices[first_name]] for first_name in first_names}
    PCA_long_components = PCA_components["long"]
    PCA_short_components = PCA_components["short"]
    all_raws = [ind.raw_haemo for ind in all_participants]
    individual_raws = {first_name: [mne_nirs.channels.get_long_channels(all_raws[i].copy().pick(good_channel_names[first_name])) for i in name_indices[first_name]] for first_name in first_names}
    sfreq = all_participants[0].raw_haemo.info["sfreq"]
    color_dict = {
    "Math": "#AA3377",
    "Hard Math": "g",
    "Control": "b"
    }

    for participant in all_participants:
        name = participant.name.split("_")[0]
        loadings = PCA_long_components[name]
        ch_names = [f"PC{i+1}_hbo" for i in range(loadings.shape[0])]
        ch_types = ["hbo"] * len(ch_names)
        individual_raw_PCA_projected = [loadings @ raw.get_data() for raw in individual_raws[name]]
        
        info = mne.create_info(
                ch_names = ch_names,
                sfreq = sfreq,
                ch_types = ch_types
                )
        
        individual_glm_raw = [mne.io.RawArray(pc_data, info) for pc_data in individual_raw_PCA_projected]

        events, event_dict = mne.events_from_annotations(participant.raw_haemo)

        individual_epochs[name] = [mne.Epochs(
                                            glm_raw, events, event_id=event_dict,
                                            tmin=0, tmax=20.1, baseline=None, preload=True)
                                    for glm_raw in individual_glm_raw]

        individual_epochs[name] = mne.concatenate_epochs(individual_epochs[name])
        
        fig, axes = plt.subplots(len(ch_names) // 3 + (len(ch_names) % 3 > 0), 3, figsize=(15, 12))
        fig.suptitle(f"{name} — PC evoked responses", fontsize=14)

        for pc_idx, ax in enumerate(axes.flatten()):
            if pc_idx < len(ch_names):
                pc_name = f"PC{pc_idx + 1}_{chromophore}"
                for cond, color in color_dict.items():
                    cond_key = cond.replace(" ", "_")
                    epochs_cond = individual_epochs[name][cond_key].get_data(picks=[pc_name])
                    mean = epochs_cond[:, 0, :].mean(axis=0)
                    sem = epochs_cond[:, 0, :].std(axis=0) / np.sqrt(len(epochs_cond))
                    times = individual_epochs[name].times
                    ax.plot(times, mean, color=color, label=cond)
                    ax.fill_between(times, mean - sem, mean + sem, color=color, alpha=0.2)

            ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
            ax.axvline(x=20, color='black', linestyle='--', linewidth=0.8)
            ax.set_title(pc_name, fontsize=10)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel("PC score (a.u.)", fontsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if pc_idx == 0:
                ax.legend(fontsize=7)


            plt.tight_layout()
            filename = os.path.join(
                rf"L:\Auditdata\CONNECT-ME\Nikolai\fNIRS\Marwans_project\PCA_evoked",
                f"PC_evoked_{name}.pdf"
            )
            fig.savefig(filename, format="pdf", bbox_inches="tight")
            plt.close(fig)


df = pd.DataFrame(columns=["ID", "No. Math", "No. Hard Math", "No. Control", "Mean Math AUC", "Mean Hard Math AUC", "Mean Control AUC", "Std. Math AUC", "Std. Hard Math AUC", "Std. Control AUC", "Math / Control p-value", "Hard Math / Control p-value"])

color_dict = {
    "Math": "#AA3377",
    "Hard Math": "g",
    "Control": "b"
}

channel_counts = {}
math_lenghts = []
hard_math_lengths = []
control_lengths = []

math_mean = []
hard_math_means = []
control_means = []

from mne.stats import permutation_cluster_test

cluster_results = []

for ind, epochs in individual_epochs.items():
    n_epochs = len(epochs)
    min_fraction = round(0.5 * n_epochs)
    bad_channel_counts = Counter(ch for epoch in epochs for ch in epoch.copy().info['bads'])
    bad_channel_indices = np.array([list(bad_channel_counts.values())]).flatten() > min_fraction
    bad_channels = list(set(np.array([list(bad_channel_counts.keys())]).flatten()[bad_channel_indices]))
    # bad_channels = list(set(channel for epoch in epochs for channel in epoch.info['bads']))
    # good_long_channels = [ch for ch in long_channels if ch in good_channel_names[first_name]]
    good_long_channels = [ch for ch in long_channels if ch not in bad_channels]


    print(f"{ind}: {len(bad_channels)} bad channels dropped, {len(good_long_channels)} good long channels remaining")
    channel_counts[ind] = [len(bad_channels), len(good_long_channels)]

    for epoch in epochs:
        epoch.info['bads'] = bad_channels
    
    epochs = [epoch.drop_channels(epoch.info["bads"]) for epoch in epochs]
    individual_epochs[ind] = mne.concatenate_epochs(epochs)

    
    t_start = 0
    t_end = 20.1
    math_HbO = individual_epochs[ind].copy()["Math"].pick(good_long_channels).crop(t_start, t_end, True).get_data() #.mean(axis=2).mean(axis=1)
    Hard_math_HbO = individual_epochs[ind].copy()["Hard_Math"].pick(good_long_channels).crop(t_start, t_end, True).get_data() #.mean(axis=2).mean(axis=1)
    Control_HbO = individual_epochs[ind].copy()["Control"].pick(good_long_channels).crop(t_start, t_end, True).get_data() #.mean(axis=2).mean(axis=1)
    times = individual_epochs[ind].copy()["Math"].pick(good_long_channels).crop(t_start, t_end, True).times

    # collapse channel dimension -> shape (n_epochs, n_times) per condition
    math_ts = math_HbO.mean(axis=1)
    hard_math_ts = Hard_math_HbO.mean(axis=1)
    control_ts = Control_HbO.mean(axis=1)

    def run_cluster_test(cond_ts, control_ts, label):
        T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
            [cond_ts, control_ts],
            n_permutations=5000,
            threshold=None,   # auto F-threshold at p<0.05; tune if needed
            tail=0,
            out_type='mask',
            seed=42,          # reproducibility across reruns
        )
        sig = [(times[cl][0], times[cl][-1], p) 
               for cl, p in zip(clusters, cluster_p_values)]
        return {
            "condition": label,
            "n_clusters_total": len(clusters),
            "n_clusters_sig": len(sig),
            "sig_windows": sig,          # list of (start_s, end_s, p)
            "min_p": min(cluster_p_values) if len(cluster_p_values) else np.nan,
        }

    math_result = run_cluster_test(math_ts, control_ts, "Math_vs_Control")
    hard_math_result = run_cluster_test(hard_math_ts, control_ts, "HardMath_vs_Control")

    cluster_results.append({"ID": ind, **{f"math_{k}": v for k, v in math_result.items()},
                                          **{f"hardmath_{k}": v for k, v in hard_math_result.items()}})




    times = individual_epochs[ind].copy()["Math"].crop(t_start, t_end, True).times
    math_AUC = np.trapezoid(math_HbO, x=times, axis=2).mean(axis=1)
    Hard_math_AUC = np.trapezoid(Hard_math_HbO, x=times, axis=2).mean(axis=1)
    Control_AUC = np.trapezoid(Control_HbO, x=times, axis=2).mean(axis=1)
    
    
    from scipy.stats import permutation_test

    def statistic(x, y):
        return np.mean(x) - np.mean(y)

    Math_result = permutation_test(
        (math_AUC, Control_AUC),
        statistic,
        permutation_type='independent',
        n_resamples=10000,
        alternative='two-sided'  # math > control
    )

    Hard_Math_result = permutation_test(
        (Hard_math_AUC, Control_AUC),
        statistic,
        permutation_type='independent',
        n_resamples=10000,
        alternative='two-sided'
    )

    p_value_math_control = Math_result.pvalue
    p_value_hard_math_control = Hard_Math_result.pvalue

    # t_stat_math_control, p_value_math_control = ttest_ind(math_AUC, Control_AUC,equal_var=False)
    # t_stat_hard_math_control, p_value_hard_math_control = ttest_ind(Hard_math_AUC, Control_AUC,equal_var=False)   

    new_row = {
    "ID": ind,
    "No. Math": len(math_AUC),
    "No. Hard Math": len(Hard_math_AUC),
    "No. Control": len(Control_AUC),
    "Mean Math AUC": np.mean(math_AUC),
    "Mean Hard Math AUC": np.mean(Hard_math_AUC),
    "Mean Control AUC": np.mean(Control_AUC),
    "Std. Math AUC": np.std(math_AUC),
    "Std. Hard Math AUC": np.std(Hard_math_AUC),
    "Std. Control AUC": np.std(Control_AUC),
    "Math / Control p-value": p_value_math_control,
    "Hard Math / Control p-value": p_value_hard_math_control,
    }

    df.loc[len(df)] = new_row
    
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

    # Copy everything from the MNE axes to the new one
    for line in ax.lines:
        ax_new.plot(line.get_xdata(), line.get_ydata(), 
                    color=line.get_color(), 
                    linestyle=line.get_linestyle(),
                    linewidth=line.get_linewidth(),
                    label=line.get_label())



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

    if math_result["n_clusters_total"] > 0:
        for start, end, p in math_result["sig_windows"]:
            ax_new.axvline(x=start, color=color_dict["Math"], linestyle='-', linewidth=2, label=f'Math vs Control p={p:.3f}')
            ax_new.axvline(x=end, color=color_dict["Math"], linestyle='-', linewidth=2)
    if hard_math_result["n_clusters_total"] > 0:
        for start, end, p in hard_math_result["sig_windows"]:
            ax_new.axvline(x=start, color=color_dict["Hard Math"], linestyle='-', linewidth=2, label=f'Hard Math vs Control p={p:.3f}')
            ax_new.axvline(x=end, color=color_dict["Hard Math"], linestyle='-', linewidth=2)
    filename = os.path.join(save_path, f"standard_fNIRS_response_plot_{ind}.pdf")
    fig_new.savefig(filename, format="pdf", bbox_inches="tight")
    plt.close(fig_new)

cluster_df = pd.DataFrame(cluster_results)
cluster_df.to_csv(os.path.join(save_path, "cluster_permutation_results.csv"), index=False)

# Manual Bonferroni correction for multiple comparisons (2 comparisons)
df["Math / Control p-value"] = np.minimum(
df["Math / Control p-value"] * 2, 1
)

df["Hard Math / Control p-value"] = np.minimum(
    df["Hard Math / Control p-value"] * 2, 1
)
df.to_csv(os.path.join(save_path, "wavelet_analysis_results.csv"), index=False)
print("debug")