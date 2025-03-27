
import scipy.stats as stats
from load_data_function import load_data

def paired_t_test(sample_1, sample_2):
    results = stats.ttest_rel(sample_1, sample_2)
    return results

# Data set:
data_set_1 = "fNIRS_Melika_hand_data" #, "fNIRS_Melika_hand_data" # "fNIrs_motor" # "fNIRS_Melika_hand_data"    
data_set_2 = "fNIRS_Melika_tongue_data" # "fNIRS_Melika_tongue_data"

individuals = True

# Data processing:
short_channel_correction = True
negative_correlation_enhancement = True
interpolate_bad_channels = False

############################

# all_epochs_1, data_name_1, all_data_1, freq_1, data_types_1, all_individuals_1 = load_data(data_set = data_set_1, short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement, individuals = individuals, interpolate_bad_channels=interpolate_bad_channels)
all_epochs_2, data_name_2, all_data_2, freq_2, data_types_2, all_individuals_2 = load_data(data_set = data_set_2, short_channel_correction = short_channel_correction, negative_correlation_enhancement = negative_correlation_enhancement, individuals = individuals, interpolate_bad_channels=interpolate_bad_channels)
