import scipy.stats as stats
from load_data_function import data_loaders
import numpy as np
import matplotlib.pyplot as plt

def paired_t_test(sample_1, sample_2):
    results = stats.ttest_rel(sample_1, sample_2)
    return results

def cohens_d(sample_1, sample_2):
    # Calculate the differences between the paired measurements
    differences = sample_1 - sample_2

    # Mean of the differences
    mean_diff = np.mean(differences)

    # Standard deviation of the differences
    std_diff = np.std(differences, ddof=1)  # Using ddof=1 for sample standard deviation

    # Cohen's d
    cohen_d = mean_diff / std_diff
    return cohen_d

def cross_area_comparison(start_time=3, end_time=12, dataset1="fNIRS_Melika_hand_data_10Hz_load", 
                         dataset2="fNIRS_Melika_tongue_10Hz_data_load"):
    """
    Compare hand channels from hand dataset with tongue channels from tongue dataset.
    This analysis shows whether individuals with strong hand responses also have strong tongue responses.
    Includes standard deviation error bars for each individual.
    """
    
    # Define channels for each area
    hand_channels = ['S4_D3 hbo', 'S4_D5 hbo', 'S4_D6 hbo', 'S4_D7 hbo'] # We're only using the left side hand channels, as the participants are only using their right hand
    
    tongue_channels = ['S13_D11 hbo', 'S13_D13 hbo', 'S13_D15 hbo', 
                      'S5_D4 hbo', 'S5_D6 hbo', 'S5_D8 hbo']
    
    # Load data
    data_loader1 = data_loaders[dataset1](
        short_channel_correction=short_channel_correction, 
        negative_correlation_enhancement=negative_correlation_enhancement, 
        individuals=individuals, 
        interpolate_bad_channels=interpolate_bad_channels
    )

    all_epochs_hand, data_name_hand, all_data_hand, freq_hand, data_types_hand, all_individuals_hand = data_loader1.load_data()
    
    data_loader2 = data_loaders[dataset2](
    short_channel_correction=short_channel_correction, 
    negative_correlation_enhancement=negative_correlation_enhancement, 
    individuals=individuals, 
    interpolate_bad_channels=interpolate_bad_channels
    )

    all_epochs_tongue, data_name_tongue, all_data_tongue, freq_tongue, data_types_tongue, all_individuals_tongue = data_loader2.load_data()
    
    # Match channel indices
    reference_epochs_hand = all_individuals_hand[0].get_epochs()
    hand_channel_indices = [reference_epochs_hand.ch_names.index(ch) for ch in hand_channels]
    
    reference_epochs_tongue = all_individuals_tongue[0].get_epochs()
    tongue_channel_indices = [reference_epochs_tongue.ch_names.index(ch) for ch in tongue_channels]
    
    # Making sure we have the same participants in both datasets
    name_intersection = list(set([individual.get_name() for individual in all_individuals_hand]) & 
                           set([individual.get_name() for individual in all_individuals_tongue]))
    all_individuals_hand = [ind for ind in all_individuals_hand if ind.get_name() in name_intersection]
    all_individuals_tongue = [ind for ind in all_individuals_tongue if ind.get_name() in name_intersection]
    
    # Sort individuals by name to ensure they match up correctly
    all_individuals_hand.sort(key=lambda x: x.get_name())
    all_individuals_tongue.sort(key=lambda x: x.get_name())
    
    # Compute mean and std response per subject across hand channels from hand dataset
    hand_means = []
    hand_stds = []
    for individual in all_individuals_hand:
        epochs = individual.get_epochs()[data_types_hand[0]]
        cropped = epochs.copy().crop(tmin=start_time, tmax=end_time)
        data = cropped.get_data()  # shape: (n_epochs, n_channels, n_times)
        mean_over_time = data.mean(axis=2)  # (n_epochs, n_channels)
        
        # Average across hand channels per epoch
        hand_data = mean_over_time[:, hand_channel_indices].mean(axis=1)  # (n_epochs,)
        subject_mean = hand_data.mean()  # scalar value per subject
        subject_std = hand_data.std()    # standard deviation across epochs
        hand_means.append(subject_mean)
        hand_stds.append(subject_std)
    
    # Compute mean and std response per subject across tongue channels from tongue dataset
    tongue_means = []
    tongue_stds = []
    for individual in all_individuals_tongue:
        epochs = individual.get_epochs()[data_types_tongue[0]]
        cropped = epochs.copy().crop(tmin=start_time, tmax=end_time)
        data = cropped.get_data() 
        mean_over_time = data.mean(axis=2)
        
        # Average across tongue channels per epoch
        tongue_data = mean_over_time[:, tongue_channel_indices].mean(axis=1)
        subject_mean = tongue_data.mean()
        subject_std = tongue_data.std()
        tongue_means.append(subject_mean)
        tongue_stds.append(subject_std)
    
    hand_means = np.array(hand_means)
    hand_stds = np.array(hand_stds)
    tongue_means = np.array(tongue_means)
    tongue_stds = np.array(tongue_stds)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot points with error bars for both x and y directions
    ax.errorbar(hand_means, tongue_means, 
                xerr=hand_stds, yerr=tongue_stds,
                fmt='o', ecolor='gray', capsize=3, alpha=0.7)
    
    # Add labels for each participant
    for j in range(len(hand_means)):
        ax.annotate(f'P{j+1}', (hand_means[j], tongue_means[j]),
                    textcoords="offset points", xytext=(5, 5), ha='left', fontsize=8)
    
    # Calculate correlation
    corr, p_corr = stats.pearsonr(hand_means, tongue_means)
    
    # Add x=y line that spans from lowest point (mean-std) to highest point (mean+std)
    min_val = min(min(hand_means - hand_stds), min(tongue_means - tongue_stds))
    max_val = max(max(hand_means + hand_stds), max(tongue_means + tongue_stds))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label="y = x")
    
    ax.set_xlabel(f"Hand Channels (from {dataset1}) ± SD")
    ax.set_ylabel(f"Tongue Channels (from {dataset2}) ± SD")
    ax.set_title("Cross-Area Comparison: Hand vs. Tongue Motor Responses")
    
    # Calculate t-test to see if responses differ
    t_res = stats.ttest_rel(hand_means, tongue_means)
    d_val = cohens_d(np.array(hand_means), np.array(tongue_means))
    
    # Update legend to include the y = x line
    ax.legend(title=f"Correlation: r = {corr:.3f}, p = {p_corr:.3f}\n"
                   f"Difference: p = {t_res.pvalue:.3f}, d = {d_val:.3f}")
    ax.grid(True)
    
    plt.tight_layout()
    fig.savefig("Cross_area_comparison_hand_vs_tongue.pdf")
    
    return fig

# ---------- Parameters ----------

# Datasets
data_set_1 = "fNIRS_Melika_hand_data_10Hz_load"
data_set_2 = "fNIRS_Melika_tongue_10Hz_data_load"

# Flags
individuals = True
short_channel_correction = True
negative_correlation_enhancement = True
interpolate_bad_channels = False

# Analysis method
analysis_method = "both"  # Options: "area", "channel", "both"

# Time window (in seconds)
start_time = 3
end_time = 12

def statistical_analysis(Area_of_interest : str = "SMA", start_time = 3, end_time = 12, dataset1: str = "fNIRS_Melika_hand_data_10Hz_load", dataset2="fNIRS_Melika_tongue_10Hz_data_load", cross_comparison=True):
    if Area_of_interest not in ["SMA", "Tongue_all", "Tongue_right", "Tongue_left", "Hand_all", "Hand_right","Hand_left"]:
        raise ValueError("Area of interest needs to be one of: SMA, Tongue_all, Tongue_right, Tongue_left, Hand_all, Hand_right, Hand_left")
    
    figures = []  # List to store the figures

    # Channels to compare
    if Area_of_interest == "SMA":
        channels_of_interest = ['S2_D2 hbo', 'S2_D3 hbo', 'S2_D5 hbo', 'S10_D2 hbo', 'S10_D10 hbo', 'S10_D12 hbo']
    elif Area_of_interest == "Tongue_all":
        channels_of_interest = ['S13_D11 hbo', 'S13_D13 hbo', 'S13_D15 hbo', 'S5_D4 hbo', 'S5_D6 hbo', 'S5_D8 hbo']
    elif Area_of_interest == "Tongue_right":
        channels_of_interest = ['S13_D11 hbo', 'S13_D13 hbo', 'S13_D15 hbo']
    elif Area_of_interest == "Tongue_left":
        channels_of_interest = ['S5_D4 hbo', 'S5_D6 hbo', 'S5_D8 hbo']
    elif Area_of_interest == "Hand_all":
        channels_of_interest = ['S12_D10 hbo', 'S12_D12 hbo', 'S12_D13 hbo', 'S12_D14 hbo', 'S4_D3 hbo', 'S4_D5 hbo', 'S4_D6 hbo', 'S4_D7 hbo']
    elif Area_of_interest == "Hand_right":
        channels_of_interest = ['S12_D10 hbo', 'S12_D12 hbo', 'S12_D13 hbo', 'S12_D14 hbo']
    elif Area_of_interest == "Hand_left":
        channels_of_interest = ['S4_D3 hbo', 'S4_D5 hbo', 'S4_D6 hbo', 'S4_D7 hbo']

    # ---------- Load Data ----------
    all_epochs_1, data_name_1, all_data_1, freq_1, data_types_1, all_individuals_1 = load_data(data_set = dataset1, short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement, individuals = individuals, interpolate_bad_channels=interpolate_bad_channels)

    all_epochs_2, data_name_2, all_data_2, freq_2, data_types_2, all_individuals_2 = load_data(data_set = dataset2, short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement, individuals = individuals, interpolate_bad_channels=interpolate_bad_channels)

    # ---------- Helper Functions ----------
    # Process a group of individuals - for individual channel analysis
    def compute_mean_responses(individuals_group, epoch_type):
        subject_means = []
        for individual in individuals_group:
            epochs = individual.get_epochs()[epoch_type]
            cropped = epochs.copy().crop(tmin=start_time, tmax=end_time)
            data = cropped.get_data()  # shape: (n_epochs, n_channels, n_times)
            mean_over_time = data.mean(axis=2)  # (n_epochs, n_channels)
            subject_mean = mean_over_time.mean(axis=0)  # (n_channels,)
            subject_means.append(subject_mean)
        return np.array(subject_means)  # shape: (n_subjects, n_channels)

    # Process a group of individuals - for area analysis
    def compute_area_means(individuals_group, channel_indices, epoch_type):
        subject_means = []
        for individual in individuals_group:
            epochs = individual.get_epochs()[epoch_type]
            cropped = epochs.copy().crop(tmin=start_time, tmax=end_time)
            data = cropped.get_data()  # shape: (n_epochs, n_channels, n_times)
            mean_over_time = data.mean(axis=2)  # (n_epochs, n_channels)
            
            # Average across specified channels for this area
            area_data = mean_over_time[:, channel_indices].mean(axis=1)  # (n_epochs,)
            subject_mean = area_data.mean()  # scalar value per subject per area
            subject_means.append(subject_mean)
        return np.array(subject_means)  # shape: (n_subjects,)

    # ---------- Extract Data ----------
    # Match channel indices
    reference_epochs = all_individuals_1[0].get_epochs()
    channel_indices = [reference_epochs.ch_names.index(ch) for ch in channels_of_interest]

    # Making sure we have the same participants in both datasets
    name_intersection = list(set([individual.get_name() for individual in all_individuals_1]) & set([individual.get_name() for individual in all_individuals_2]))
    all_individuals_1 = [individual for individual in all_individuals_1 if individual.get_name() in name_intersection]
    all_individuals_2 = [individual for individual in all_individuals_2 if individual.get_name() in name_intersection]

    # For individual channel analysis
    if analysis_method in ["channel", "both"]:
        means_1 = compute_mean_responses(all_individuals_1, data_types_1[0])
        means_2 = compute_mean_responses(all_individuals_2, data_types_2[0])
        
        # Ensure same number of subjects
        n_subjects = min(means_1.shape[0], means_2.shape[0])
        means_1 = means_1[:n_subjects]
        means_2 = means_2[:n_subjects]

    # For area analysis
    if analysis_method in ["area", "both"]:
        # Compute mean response per subject across all channels in the area
        means_1_area = compute_area_means(all_individuals_1, channel_indices, data_types_1[0])
        means_2_area = compute_area_means(all_individuals_2, channel_indices, data_types_2[0])
        
        # Ensure same number of subjects
        n_subjects_area = min(len(means_1_area), len(means_2_area))
        means_1_area = means_1_area[:n_subjects_area]
        means_2_area = means_2_area[:n_subjects_area]

    # ---------- Analysis and Visualization ----------
    # Individual channel analysis
    if analysis_method in ["channel", "both"]:
        # Set up subplot grid
        num_channels = len(channel_indices)
        num_cols = min(3, num_channels)
        num_rows = (num_channels + num_cols - 1) // num_cols

        fig_channels, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows))

        if num_channels == 1:
            axes = np.array([[axes]])
        elif num_rows == 1:
            axes = np.array([axes])
        elif isinstance(axes, np.ndarray) and axes.ndim == 1:
            axes = axes.reshape((num_rows, num_cols))

        # Determine axis limits
        all_min = min(np.min(means_1[:, channel_indices]), np.min(means_2[:, channel_indices]))
        all_max = max(np.max(means_1[:, channel_indices]), np.max(means_2[:, channel_indices]))
        padding = (all_max - all_min) * 0.1
        all_min -= padding
        all_max += padding

        for i, ch_idx in enumerate(channel_indices):
            row = i // num_cols
            col = i % num_cols
            ax = axes[row, col]

            x = means_1[:, ch_idx]
            y = means_2[:, ch_idx]

            ax.scatter(x, y, color='blue', alpha=0.7)
            for j in range(len(x)):
                ax.annotate(f'P{j+1}', (x[j], y[j]), textcoords="offset points",
                            xytext=(5, 5), ha='left', fontsize=8)

            ax.plot([all_min, all_max], [all_min, all_max], 'k--')
            ax.set_xlim(all_min, all_max)
            ax.set_ylim(all_min, all_max)
            ax.set_xlabel(f"{data_set_1}", fontsize=8)
            ax.set_ylabel(f"{data_set_2}", fontsize=8)

            # Stats
            p_val = stats.ttest_rel(x, y).pvalue
            d_val = cohens_d(x, y)

            ax.set_title(f"{channels_of_interest[i]}\np = {p_val:.3f}\nCohen's d = {d_val:.2f}", fontsize=10)
            ax.grid(True)

        # Remove empty subplots
        for i in range(num_channels, num_rows * num_cols):
            fig_channels.delaxes(axes[i // num_cols, i % num_cols])

        fig_channels.suptitle(f"Area: {Area_of_interest} - Channel Comparisons", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        fig_channels.savefig(f"Channel_comparisons_{Area_of_interest}.pdf")
        figures.append(fig_channels)

    # Area analysis
    if analysis_method in ["area", "both"]:
        fig_area, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(means_1_area, means_2_area, color='blue', alpha=0.7)
        for j in range(len(means_1_area)):
            ax.annotate(f'P{j+1}', (means_1_area[j], means_2_area[j]),
                        textcoords="offset points", xytext=(5, 5), ha='left', fontsize=8)

        min_val = min(means_1_area.min(), means_2_area.min())
        max_val = max(means_1_area.max(), means_2_area.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', label="y = x")

        ax.set_xlabel(f"{data_set_1} (mean signal)")
        ax.set_ylabel(f"{data_set_2} (mean signal)")
        ax.set_title(f"Area: {Area_of_interest} - All Channels Combined")

        p_val = stats.ttest_rel(means_1_area, means_2_area).pvalue
        d_val = cohens_d(means_1_area, means_2_area)
        ax.legend(title=f"p = {p_val:.3f}\nCohen's d = {d_val:.3f}")
        ax.grid(True)

        plt.tight_layout()
        fig_area.savefig(f"Area_comparison_{Area_of_interest}.pdf")
        figures.append(fig_area)
    
    if cross_comparison:
        fig_cross = cross_area_comparison(start_time, end_time, dataset1, dataset2)
        figures.append(fig_cross)
    
    return figures