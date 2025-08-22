import numpy as np
import mne_nirs

def compute_effect_size(class_instance):
    
    # individual_data = {}
    # for individual in class_instance.individuals:
    raw_epochs = class_instance.all_raw_epochs
    preprocessed_epochs = class_instance.all_epochs
    tmin = 2.5
    tmax = 12.5
    channels = mne_nirs.channels.get_long_channels(class_instance.Individual_participants[0].raw_haemo).info["ch_names"]
    
    pause_id = class_instance.standard_event_ids["Pause"]
    pause_indices = np.where(raw_epochs[0].events[:, 2] == pause_id)
    pause_times = raw_epochs[0].events[pause_indices, 0]
    pause_times = np.append(pause_times, raw_epochs[0].events[-1][0])
    number_of_pause_times = len(pause_times)
    
    start_time = 0
    # for epoch in preprocessed_epochs:
    participants_session_data = {}
    for index, epoch in enumerate(raw_epochs):
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
            ch_differences = {}
            for channel in channels:
                means = []
                for data_type in class_instance.data_types:
                    mean_each_epoch = np.mean(session[data_type].pick(channel).get_data(), axis=2)
                    mean_data_type = np.mean(mean_each_epoch)
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
        
        participants_session_data[f"participant_{index}: "] = participant_data
    standard_devation = 0
    for session in session_differences:
        for channel in channels:
            standard_devation += ((session[channel] -averages_over_sessions[channel])**2)
    standard_devation = np.sqrt(standard_devation/(len(session_differences)-1))
    
    effect_sizes = {}
    for channel in channels:
        effect_sizes[channel] = averages_over_sessions[channel] / standard_devation
    
    effect_size_preprocessed, effect_size_raw = 0
    return effect_size_raw, effect_size_preprocessed