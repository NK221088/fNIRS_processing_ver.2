from datetime import datetime
from load_data_function import load_data
from epoch_plot import epoch_plot
from standard_fNIRS_response_plot import standard_fNIRS_response_plot
from paradigm_plot import paradigm_plot
import mne
import os
from collections import Counter
import numpy as np

############################
# Settings:
############################

# Data set:
data_set = "fNIRS_Melika_data" # "fNIRS_Alexandros_Healthy_data" # "fNirs_motor_full_data" # "AudioSpeechNoise" #     "fNIrs_motor" #      

epoch_type = "Control"
combine_strategy = "mean"
individuals = True

# Data processing:
bad_channels_strategy = "all"
short_channel_correction = True
negative_correlation_enhancement = True
threshold = 3
startTime = 7.5
stopTime = 12.5
K = 5
interpolate_bad_channels = False

# Plotting and saving:
plot_epochs = False
plot_std_fNIRS_response = False
plot_paradigms = True
plot_accuracy_across_k_folds = False

save_plot_epochs = False
save_plot_std_fNIRS_response = False
save_plot_accuracy_across_k_folds = False
save_results = False

############################

all_epochs, data_name, all_data, freq, data_types, all_individuals = load_data(data_set = data_set, short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement, individuals = individuals, interpolate_bad_channels=interpolate_bad_channels)

# Plot epochs and save results
if plot_epochs:
    epoch_plot(all_epochs, epoch_type=epoch_type, combine_strategy=combine_strategy, save=save_plot_epochs, bad_channels_strategy=bad_channels_strategy, threshold = threshold, data_set = data_name)

# Plot the standard fNIRS response plot
if plot_std_fNIRS_response:
    standard_fNIRS_response_plot(all_epochs, data_types, combine_strategy=combine_strategy, save=save_plot_std_fNIRS_response, bad_channels_strategy=bad_channels_strategy, threshold = threshold, data_set = data_name)

if plot_paradigms:
    paradigm_plot(all_individuals[0])