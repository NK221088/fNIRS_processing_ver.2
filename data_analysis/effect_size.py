import numpy as np
import mne_nirs
from collections import defaultdict

def compute_effect_size(class_instance):
    
    # individual_data = {}
    # for individual in class_instance.individuals:
    raw_epochs = class_instance.all_raw_epochs
    preprocessed_epochs = class_instance.all_epochs
    tmin = 2.5
    tmax = 12.5
    channels = mne_nirs.channels.get_long_channels(class_instance.Individual_participants[0].raw_haemo).info["ch_names"]
    
    
    
    def _compute_effect_size(raw_epochs, epochs, tmin, tmax, channels):

        pause_id = class_instance.standard_event_ids["Pause"]
        
        participants_session_data = {}
        channel_means = defaultdict(lambda: defaultdict(list))
        for index, epoch in enumerate(epochs):
            pause_indices = np.where(raw_epochs[index].events[:, 2] == pause_id)
            pause_times = raw_epochs[index].events[pause_indices, 0]
            pause_times = np.append(pause_times, raw_epochs[index].events[-1][0])
            start_time = 0
            participant_data = {}
            session_data = []
            epoch = epoch[class_instance.data_types]
            for pause_time in pause_times:    
                event_indices = np.where((epoch.events[:,0] > start_time) & (epoch.events[:,0] < pause_time))
                session = epoch[event_indices[0]]
                session_data.append(session.crop(tmin, tmax) if len(session.event_id) == len(class_instance.data_types) else []) 
                start_time = pause_time

            
            session_differences = []
            for session in session_data:
                if len(session) > 0:
                    ch_differences = {}
                    for channel in channels:
                        means = []
                        for data_type in class_instance.data_types:
                            mean_each_epoch = np.mean(session[data_type].pick(channel).get_data(), axis=2)
                            mean_data_type = np.mean(mean_each_epoch)
                            channel_means[channel][data_type].extend(mean_each_epoch)
                            means.append(mean_data_type)  
                        session_difference = means[0] - means[1]
                        ch_differences[channel] = session_difference
                    session_differences.append(ch_differences)

            participant_data["session_differences"] = session_differences
        
            averages_over_sessions = {}
            for channel in channels:
                channel_sum = 0
                for session in session_differences:
                    channel_sum += session[channel]
                averages_over_sessions[channel] = channel_sum / len(session_differences)
            
            participant_data["averages_over_sessions"] = averages_over_sessions
            
            participants_session_data[f"participant_{index}"] = participant_data

        standard_deviation = defaultdict(int)
        kp_sum = 0
        for channel in channels:
            for participant in participants_session_data:
                participant_session_diferences = participants_session_data[participant]["session_differences"]
                participant_session_averages = participants_session_data[participant]['averages_over_sessions']
                kp_sum += len(participant_session_diferences) - 1
                for session in participant_session_diferences:
                    standard_deviation[channel] += (session[channel] - participant_session_averages[channel])**2
        
        for channel in channels:
            standard_deviation[channel] = np.sqrt(standard_deviation[channel]/kp_sum)
        
        grand_mean_participants = defaultdict(int)
        for participant in participants_session_data:
            participant_session_averages = participants_session_data[participant]['averages_over_sessions']
            for channel in channels:
                grand_mean_participants[channel] += participant_session_averages[channel]
        
        effect_sizes = {}
        for channel in grand_mean_participants:
            grand_mean_participants[channel] = grand_mean_participants[channel] / len(participants_session_data)
            effect_sizes[channel] = grand_mean_participants[channel] / standard_deviation[channel]
        
        channels_mean = defaultdict(lambda: defaultdict(int))
        channels_std_deviation = defaultdict(lambda: defaultdict(int))
        channels_median = defaultdict(lambda: defaultdict(int))
        channels_min = defaultdict(lambda: defaultdict(int))
        channels_max = defaultdict(lambda: defaultdict(int))

        for channel in channels:
            for data_type in class_instance.data_types:
                channel_means[channel][data_type] = [values for sublist in channel_means[channel][data_type] for values in sublist]
                channels_mean[channel][data_type] = np.mean(channel_means[channel][data_type])
                channels_std_deviation[channel][data_type] = np.std(channel_means[channel][data_type])
                channels_median[channel][data_type] = np.median(channel_means[channel][data_type])
                channels_min[channel][data_type] = np.min(channel_means[channel][data_type])
                channels_max[channel][data_type] = np.max(channel_means[channel][data_type])
                
        return (effect_sizes, channels_mean, channels_std_deviation, channels_median, channels_min, channels_max)

    _keys = ["Effect size", "Channels' means", "Channels' std. deviation", "Channels' medians", "Channels' min", "Channels' max"]
    raw_return_values = _compute_effect_size(raw_epochs, raw_epochs, tmin, tmax, channels)
    preprocessed_return_values = _compute_effect_size(raw_epochs, preprocessed_epochs, tmin, tmax, channels)
    raw_values = {key: val for key, val in zip(_keys, raw_return_values)}
    preprocessed_values = {key: val for key, val in zip(_keys, preprocessed_return_values)}
    return raw_values, preprocessed_values