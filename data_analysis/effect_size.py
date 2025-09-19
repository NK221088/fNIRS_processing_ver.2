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
        cond_session_means = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        for index, epoch in enumerate(epochs):
            pid = f"participant_{index}"
            pause_indices = np.where(raw_epochs[index].events[:, 2] == pause_id)[0]
            pause_times = raw_epochs[index].events[pause_indices, 0]
            pause_times = np.append(pause_times, raw_epochs[index].events[-1][0])
            start_time = 0
            participant_data = {}
            session_data = []
            epoch = epoch[class_instance.data_types]
            for pause_time in pause_times:    
                event_indices = np.where((epoch.events[:,0] > start_time) & (epoch.events[:,0] < pause_time))
                session = epoch[event_indices[0]]
                sess = session.crop(tmin, tmax)
                if len(sess.event_id) == len(class_instance.data_types):
                    session_data.append(sess)
                start_time = pause_time

            
            session_differences = []
            for session in session_data:
                if (len(session[class_instance.data_types[0]]) > 0) and (len(session[class_instance.data_types[1]]) > 0):
                    ch_differences = {}
                    for channel in channels:
                        means = {}
                        for data_type in class_instance.data_types:
                            mean_each_epoch = np.mean(session[data_type].pick(channel).get_data(), axis=2).ravel()
                            mean_data_type = np.mean(mean_each_epoch)
                            cond_session_means[data_type][channel][pid].append(mean_data_type)
                            means[data_type] = mean_data_type  
                        session_difference = means[class_instance.data_types[1 - np.where(np.array(class_instance.data_types) == "Control")[0][0]]] - means[class_instance.data_types[np.where(np.array(class_instance.data_types) == "Control")[0][0]]] # Always subtract the control from the active epochs
                        ch_differences[channel] = session_difference
                    session_differences.append(ch_differences)

            participant_data["session_differences"] = session_differences
        
            averages_over_sessions = {}
            for channel in channels:
                averages_over_sessions[channel] = np.mean([sd[channel] for sd in session_differences])
            
            participant_data["averages_over_sessions"] = averages_over_sessions
            
            participants_session_data[f"participant_{index}"] = participant_data

        condition_means = {cond: {} for cond in class_instance.data_types}
        for cond in class_instance.data_types:
            for ch in channels:
                per_person = []
                for pid, sess_vals in cond_session_means[cond][ch].items():
                    if len(sess_vals) > 0:
                        per_person.append(np.mean(sess_vals))  # equal-weight sessions within person
                condition_means[cond][ch] = float(np.mean(per_person)) if per_person else np.nan  # equal-weight across people

        standard_deviation = defaultdict(lambda: defaultdict())
        df_within = defaultdict(lambda: defaultdict())
       
        for channel in channels:
            for participant, pdata in participants_session_data.items():
                num = 0.0
                df = 0
                Dps = [sd[channel] for sd in pdata["session_differences"]]  # session diffs for this channel
                k_p = len(Dps)
                if k_p >= 2:
                    Dpbar = pdata["averages_over_sessions"][channel]
                    residuals = np.array(Dps, dtype=float) - float(Dpbar)
                    num += float((residuals**2).sum())
                    df  = (k_p - 1)
                # if k_p == 0 or 1: contribute nothing to num or df
                standard_deviation[participant][channel] = np.sqrt(num/df) if df > 0 else np.nan
                df_within[participant][channel] = df
        
        effect_sizes = defaultdict(lambda: defaultdict())
        
        for channel in channels:
            for participant, pdata in participants_session_data.items():
                Dpbar = pdata["averages_over_sessions"][channel]
                
                if not np.isnan(standard_deviation[participant][channel]):
                    effect_sizes[participant][channel] = Dpbar / standard_deviation[participant][channel]
                else:
                    effect_sizes[participant][channel] = np.nan

        grand_sum = {ch: 0.0 for ch in channels}
        P_ch = {ch: 0 for ch in channels}
        
        for participant, ch_effect_sizes in effect_sizes.items():
            for channel, value in ch_effect_sizes.items():
                grand_sum[channel] += value
                P_ch[channel] += 1
        
        # Per-channel grand mean across participants (equal person weight)
        grand_mean_participants = {
            ch: (grand_sum[ch] / P_ch[ch]) if P_ch[ch] > 0 else np.nan
            for ch in channels
        }   
                
        return (effect_sizes, participants_session_data, grand_mean_participants, standard_deviation, condition_means, df_within, P_ch)

    _keys = ["Effect size", "Channels' mean difference", "Channels' within-participant SD", "Conditions' mean", "DF within", "P Ch."]
    raw_return_values = _compute_effect_size(raw_epochs, raw_epochs, tmin, tmax, channels)
    preprocessed_return_values = _compute_effect_size(raw_epochs, preprocessed_epochs, tmin, tmax, channels)
    raw_values = {key: val for key, val in zip(_keys, raw_return_values)}
    preprocessed_values = {key: val for key, val in zip(_keys, preprocessed_return_values)}
    return raw_values, preprocessed_values