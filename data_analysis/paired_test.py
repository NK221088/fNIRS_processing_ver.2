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
drug_path = Path(os.getenv(rf"drugs_path"))
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
name_session_map = defaultdict(list)
name_session_recording_map = defaultdict(list)
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
        name_session_map[key + "_" + subvalue[0].split("_")[1]].append(all_participants[subvalue[1]].raw_haemo) # Collect all indices of epochs from same sessions for each patient
        name_session_recording_map[key + "_" + subvalue[0].split("_")[1]].append(subvalue[0].split("_")[2]) # Collect all indices of epochs from same sessions for each patient

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

    recording_name_dict = {"B": "T0", "P1": "T15", "P2": "T60"}
    
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    # Publication-friendly defaults
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    
    CONTROL_COLOR = "#F2C14E"  # muted amber (swap back to 'yellow' if you prefer the original)
    ACTIVE_COLOR = "#4C9A6A"   # muted green
    LINE_COLOR = "#555555"
    
    
    def plot_paired_boxplot(control_data, active_data, active_label, save_path, filename,
                            unit_scale=1e6, unit_label="\u0394[HbO] (\u00b5M)",
                            pair_gap=1.0, link_stat="mean", recording_sizes=None, recordings_names=None):
        """
        control_data, active_data : list of 1D arrays, one array per pair (assumed in Molar;
            set unit_scale=1 if your data is already in the desired unit).
        active_label : legend label for the active condition, e.g. "Math" or "Hard Math".
        pair_gap : extra horizontal spacing inserted between pairs (in box-width units),
            on top of the normal 1-unit gap within a pair. Larger = more breathing room
            between pairs.
        link_stat : "mean" or "median" — which summary statistic the connecting line/marker
            tracks between Control and active. Defaults to "mean" to match a mean-based
            (e.g. signed t-statistic / permutation) test; the boxplot itself still shows
            the median, as usual. Set to "median" only if your test statistic is
            median-based.
        recording_sizes : optional list of ints, e.g. [5, 5, 5] for 3 recordings of 5 pairs
            each, summing to n_pairs. When given, draws a dashed vertical line (labeled
            "Rec i -> Rec i+1", matching the session scatter plots) at each recording
            boundary, so pairs from different recordings are visually distinguishable.
    
        Draws Control/active boxplot pairs side by side, connects each pair with a line
        at `link_stat`, and labels the x-axis "Pair 1", "Pair 2", ... at each pair's
        midpoint.
        """
        n_pairs = len(control_data)
        assert n_pairs == len(active_data), "control_data and active_data must have the same length"
        assert link_stat in ("mean", "median")
        if recording_sizes is not None:
            assert sum(recording_sizes) == n_pairs, (
                f"recording_sizes sums to {sum(recording_sizes)}, expected n_pairs={n_pairs}"
            )
        link_fn = np.mean if link_stat == "mean" else np.median
    
        fig, ax = plt.subplots(figsize=(12, 5))
        pair_centers = []
        step = 2 + pair_gap  # distance from one pair's control position to the next pair's
    
        for i in range(n_pairs):
            control_pos = i * step + 1
            active_pos = control_pos + 1
            pair_centers.append((control_pos + active_pos) / 2)
    
            control_vals = np.asarray(control_data[i]) * unit_scale
            active_vals = np.asarray(active_data[i]) * unit_scale
    
            bp_c = ax.boxplot(control_vals, positions=[control_pos], widths=0.6, patch_artist=True)
            bp_a = ax.boxplot(active_vals, positions=[active_pos], widths=0.6, patch_artist=True)
    
            bp_c["boxes"][0].set_facecolor(CONTROL_COLOR)
            bp_a["boxes"][0].set_facecolor(ACTIVE_COLOR)
            for bp in (bp_c, bp_a):
                bp["boxes"][0].set_edgecolor("black")
                bp["medians"][0].set_color("black")
    
            # connecting line between the pair's link_stat (mean by default) — kept
            # distinct from the boxes' own median line so the two aren't conflated
            control_link = link_fn(control_vals)
            active_link = link_fn(active_vals)
            ax.plot(
                [control_pos, active_pos], [control_link, active_link],
                color=LINE_COLOR, linewidth=1.2, zorder=3,
            )
            ax.scatter(
                [control_pos, active_pos], [control_link, active_link],
                marker="D", s=18, color=LINE_COLOR, zorder=4,
            )
    
        if recording_sizes is not None:
            cumulative = np.cumsum(recording_sizes)[:-1]  # pair-count boundaries, e.g. [5, 10]
            for rec_i, boundary in enumerate(cumulative):
                # boundary = number of pairs in recordings up to and including this one;
                # the line sits between pair `boundary-1` (0-indexed, last of this
                # recording) and pair `boundary` (0-indexed, first of the next)
                last_active_pos = (boundary - 1) * step + 2
                next_control_pos = boundary * step + 1
                line_x = (last_active_pos + next_control_pos) / 2
                ax.axvline(line_x, color="black", linestyle="--", linewidth=1)
                ax.text(
                    line_x, 1.01, f"{recording_name_dict.get(recordings_names[rec_i])} \u2192 {recording_name_dict.get(recordings_names[rec_i + 1])}",
                    transform=ax.get_xaxis_transform(),
                    rotation=90, ha="center", va="bottom", fontsize=8, color="#555555",
                )
    
        ax.set_xticks(pair_centers)
        ax.set_xticklabels([f"Pair {i + 1}" for i in range(n_pairs)], rotation=45, ha="right")
        ax.set_ylabel(unit_label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
        ax.legend(
            handles=[
                Patch(facecolor=CONTROL_COLOR, edgecolor="black", label="Rest"),
                Patch(facecolor=ACTIVE_COLOR, edgecolor="black", label=active_label),
                plt.Line2D([0], [0], color=LINE_COLOR, marker="D", markersize=5,
                        linewidth=1.2, label=f"{link_stat.capitalize()} (linked)"),
            ],
            frameon=False, loc="upper right",
        )
    
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, filename), format="pdf", bbox_inches="tight")
        plt.close(fig)
    
    
    # --- Moderate mental arithmetic ---
    n_recordings = len(name_session_map[ind])
    plot_paired_boxplot(
        control_data=[Control_HbO[math_controls][j] for j in range(len(math_HbO_mean))],
        active_data=[math_HbO[j] for j in range(len(math_HbO_mean))],
        active_label="Math",
        save_path=save_path,
        filename=f"math_boxplot_{ind}.pdf",
        recording_sizes=[len(math_HbO_mean) // n_recordings] * n_recordings,
        recordings_names = name_session_recording_map[ind]
    )
    
    # --- Hard mental arithmetic ---
    plot_paired_boxplot(
        control_data=[Control_HbO[hard_math_controls][j] for j in range(len(Hard_math_HbO_mean))],
        active_data=[Hard_math_HbO[j] for j in range(len(Hard_math_HbO_mean))],
        active_label="Hard Math",
        save_path=save_path,
        filename=f"hard_math_boxplot_{ind}.pdf",
        recording_sizes=[len(Hard_math_HbO_mean) // n_recordings] * n_recordings,
        recordings_names = name_session_recording_map[ind]
    )


    import os
    import numpy as np
    import mne
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    CONTROL_COLOR = "#F2C14E"
    ACTIVE_COLOR = "#4C9A6A"
    BACKGROUND_COLOR = "#EEEEEE"  # samples not covered by any task/Control window


    def build_condition_colors(n_samples, boundaries, sfreq,
                                active_duration_s=21, control_wait_s=5, control_duration_s=15,
                                control_color=CONTROL_COLOR, active_color=ACTIVE_COLOR,
                                background_color=BACKGROUND_COLOR):
        """
        boundaries : event start indices, alternating Control/Active, starting with
            Control. Active window starts AT the boundary; Control window starts
            `control_wait_s` seconds after the boundary.
        Returns an array of length n_samples with one color per sample.
        """
        n_active = int(np.ceil(active_duration_s * sfreq))
        n_wait = int(np.ceil(control_wait_s * sfreq))
        n_control = int(np.ceil(control_duration_s * sfreq))

        colors = np.full(n_samples, background_color, dtype=object)
        for i, start in enumerate(boundaries):
            is_control = (i % 2 == 0)  # first boundary = Control, alternating
            if is_control:
                win_start, win_end, color = start + n_wait, start + n_wait + n_control, control_color
            else:
                win_start, win_end, color = start, start + n_active, active_color
            win_start, win_end = max(win_start, 0), min(win_end, n_samples)
            if win_start < win_end:
                colors[win_start:win_end] = color
        return colors


    def extract_condition_window(recording, long_channels, condition,
                                task_duration_s=21, control_wait_s=5, control_duration_s=15,
                                margin_s=2, unit_scale=1e6):
        """
        condition : "Math" or "Hard_Math".

        Extracts events local to `recording` (correcting for recording.first_samp),
        isolates the Control/<condition> block (Math comes first in the recording,
        Hard_Math second -- we split on the last Math event), and trims the
        recording to:
        - start right after the *other* block ends (so Math samples never show
            up in the Hard_Math plot, and vice versa), and
        - end `margin_s` seconds after this block's last task window finishes.

        unit_scale : multiplier applied to the (Molar) HbO data, e.g. 1e6 for \u00b5M.

        Returns times, data (mean over long_channels, scaled to unit_scale), colors, boundaries.
        boundaries is expressed in the LOCAL (trimmed) sample frame, matching times/data/colors.
        """
        events, event_dict = mne.events_from_annotations(recording, verbose=False)
        events = events.copy()
        events[:, 0] = events[:, 0] - recording.first_samp

        control_id = event_dict["Control"]
        math_id = event_dict["Math"]
        condition_id = event_dict[condition]
        sfreq = recording.info["sfreq"]

        n_task = int(np.ceil(task_duration_s * sfreq))
        margin = int(np.ceil(margin_s * sfreq))

        # Math block comes first, Hard_Math block second -- split on the last Math event
        last_math_idx = np.where(events[:, 2] == math_id)[0][-1]
        if condition == "Math":
            candidate_events = events[: last_math_idx + 1]
        elif condition == "Hard_Math":
            candidate_events = events[last_math_idx + 1:]
        else:
            raise ValueError(f"Unknown condition: {condition!r} (expected 'Math' or 'Hard_Math')")

        relevant_events = candidate_events[np.isin(candidate_events[:, 2], [control_id, condition_id])]
        boundaries_abs = relevant_events[:, 0]

        if condition == "Math":
            start_sample = 0
        else:  # Hard_Math: only keep a few samples of lead-in before its own first boundary
            start_sample = max(boundaries_abs[0] - margin, 0)

        last_sample_abs = boundaries_abs[-1] + n_task + margin
        boundaries = boundaries_abs - start_sample  # shift to the local (trimmed) frame

        times = recording.times[start_sample:last_sample_abs]
        data = recording.copy().pick(good_long_channels).get_data().mean(axis=0)[start_sample:last_sample_abs] * unit_scale
        colors = build_condition_colors(
            len(times), boundaries, sfreq,
            active_duration_s=task_duration_s, control_wait_s=control_wait_s,
            control_duration_s=control_duration_s,
        )
        return times, data, colors, boundaries


    def plot_session(times_list, data_list, colors_list, active_label, gap_s=10, ax=None,
                    unit_label="\u0394[HbO] (\u00b5M)", recordings_names=None):
        """
        Stitches several already-extracted (times, data, colors) recordings into one
        figure: each recording's time axis is normalized to start at 0, then placed
        after the previous one with a `gap_s`-second visual gap and a dashed vertical
        line (labeled "Rec i -> Rec i+1") marking the recording boundary.

        The x-axis tick LABELS restart at 0 for each recording (even though the
        underlying plotted positions keep incrementing left-to-right with the gaps),
        so every recording reads as its own local timeline.

        Adds a Control/active legend.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 4))

        tick_positions = []
        tick_labels = []
        offset = 0.0

        for i, (times, data, colors) in enumerate(zip(times_list, data_list, colors_list)):
            local_times = times - times[0]
            plotted_times = local_times + offset
            ax.scatter(plotted_times, data, c=colors.tolist(), s=8, edgecolors="none")

            # "nice" local tick values (0, 50, 100, ...) for THIS recording, placed at
            # their corresponding global (offset) position but labeled with the local value
            local_ticks = MaxNLocator(nbins=6, steps=[1, 2, 5, 10]).tick_values(0, local_times[-1])
            local_ticks = local_ticks[(local_ticks >= 0) & (local_ticks <= local_times[-1])]
            tick_positions.extend(local_ticks + offset)
            tick_labels.extend(f"{int(t)}" for t in local_ticks)

            rec_end = plotted_times[-1]
            if i < len(times_list) - 1:
                line_x = rec_end + gap_s / 2
                ax.axvline(line_x, color="black", linestyle="--", linewidth=1)
                ax.text(
                    line_x, 1.01, f"{recording_name_dict.get(recordings_names[i])} \u2192 {recording_name_dict.get(recordings_names[i + 1])}",
                    transform=ax.get_xaxis_transform(),
                    rotation=90, ha="center", va="bottom", fontsize=8, color="#555555",
                )
            offset = rec_end + gap_s

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Time (s) \u2014 restarts at 0 for each recording; dashed lines mark recording boundaries")
        ax.set_ylabel(unit_label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(
            handles=[
                Patch(facecolor=CONTROL_COLOR, edgecolor="black", label="Rest"),
                Patch(facecolor=ACTIVE_COLOR, edgecolor="black", label=active_label),
            ],
            frameon=False, loc="upper right",
        )
        return ax


    def build_and_save_session_plot(condition, active_label, filename, recordings_names=None):
        times_list, data_list, colors_list = [], [], []
        for recording in name_session_map[ind]:
            times, data, colors, boundaries = extract_condition_window(recording, long_channels, condition)
            times_list.append(times)
            data_list.append(data)
            colors_list.append(colors)

        fig, ax = plt.subplots(figsize=(16, 4))
        plot_session(times_list, data_list, colors_list, active_label=active_label, ax=ax, recordings_names=recordings_names)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, filename), format="pdf", bbox_inches="tight")
        plt.close(fig)


    # --- Math ---
    build_and_save_session_plot("Math", "Math", f"math_session_scatter_{ind}.pdf", recordings_names=name_session_recording_map[ind])

    # --- Hard Math ---
    build_and_save_session_plot("Hard_Math", "Hard Math", f"hard_math_session_scatter_{ind}.pdf", recordings_names=name_session_recording_map[ind])

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

import numpy as np
import pandas as pd

# recording code -> human-readable label (already used elsewhere in your pipeline)
recording_name_dict = {"B": "T0", "P1": "T15", "P2": "T60"}

rows = []  # collect one dict per row, build the DataFrame ONCE at the end (see note below)

for patient, sessions in name_epoch_map.items():
    for name, epoch_index in sessions:
        session_id = name.split("_")[1]      # e.g. "S1"
        recording_code = name.split("_")[2]  # e.g. "B", "P1", "P2"
        recording_label = recording_name_dict.get(recording_code, recording_code)

        epochs = all_epochs[epoch_index].copy()
        good_long_channels = [ch for ch in long_channels if ch not in epochs.info["bads"]]
        epochs = epochs.pick(good_long_channels)

        math_HbO_mean = epochs.copy()["Math"].crop(math_t_start, math_t_end, True).get_data().mean(axis=2).mean(axis=1)
        Hard_math_HbO_mean = epochs.copy()["Hard_Math"].crop(math_t_start, math_t_end, True).get_data().mean(axis=2).mean(axis=1)
        Control_HbO_mean = epochs.copy()["Control"].crop(control_t_start, control_t_end, True).get_data().mean(axis=2).mean(axis=1)

        # Control epochs are laid out in two blocks within a single recording:
        # the first n_math are Math-paired, the next n_hardmath are Hard_Math-paired
        # (same structure established earlier for the session-level math_controls /
        # hard_math_controls split -- here n_blocks=1 since this is a single recording).
        n_math = len(math_HbO_mean)
        n_hardmath = len(Hard_math_HbO_mean)
        math_controls = np.arange(n_math)
        hard_math_controls = np.arange(n_math, n_math + n_hardmath)

        math_diff = math_HbO_mean - Control_HbO_mean[math_controls]
        hardmath_diff = Hard_math_HbO_mean - Control_HbO_mean[hard_math_controls]

        # for i in range(len(math_diff)):
        rows.append({
            "ID_prefix": patient,
            "Session": session_id,
            "Recording": recording_label,
            "Condition": "Math",
            "response_diff": math_diff.mean(),
            "n_pairs": len(math_diff),
        })

        rows.append({
            "ID_prefix": patient,
            "Session": session_id,
            "Recording": recording_label,
            "Condition": "Hard_Math",
            "response_diff": hardmath_diff.mean(),
            "n_pairs": len(hardmath_diff),
        })

rdf = pd.DataFrame(rows)
states = pd.read_excel(drug_path)
states["ID_prefix"] = "P" + states["Subject"].astype(str)
states["Session"] = "S" + states["Session_ID"].astype(str)

# Merge the state
rdf = rdf.merge(
    states[["ID_prefix", "Session", "Drug"]],
    on=["ID_prefix", "Session"],
    how="left"
)
rdf["response_diff"] = rdf["response_diff"] * 1e6
responding_ids = paired_mean_df.loc[
    (paired_mean_df["math_p_value"] < 0.05) | (paired_mean_df["hardmath_p_value"] < 0.05),
    "ID_prefix"
].unique()

rdf = rdf.merge(
    paired_mean_df[["ID_prefix", "Session", "Consciousness"]],
    on=["ID_prefix", "Session"],
    how="left"
)
rdf_subset = rdf[rdf["ID_prefix"].isin(responding_ids)]
rdf_subset.to_csv("lme_input_subset.csv", index=False)
rdf.to_csv("lme_input.csv", index=False)
paired_mean_df.to_csv("paired_test_results.csv", index=False)
print("debug")