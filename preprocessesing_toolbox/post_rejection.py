import numpy as np

def reject_if_single_event_type(epochs, data_types):
    """
    Drop all epochs if:
    1) not all expected event types are present, OR
    2) fewer than 2 epochs exist for any event type.
    """
        
    # Get remaining event types
    remaining_events = epochs.events[:, 2]
    unique_event_types = np.unique(remaining_events)

    if len(unique_event_types) < (len(data_types)):
        # Drop all epochs by index
        epochs.drop(range(len(epochs)), reason=f"All epochs dropped: only {len(unique_event_types)} event type(s) remaining")
        print(f"All epochs dropped: only {len(unique_event_types)} event type(s) remaining")
        return epochs

    for event_type in data_types:
        if len(epochs[event_type]) < 2:
            # Drop all epochs by index
            epochs.drop(range(len(epochs)), reason=f"All epochs dropped: only {len(epochs[event_type])} event(s) of type {event_type} remaining")
            print(f"All epochs dropped: only {len(epochs[event_type])} event(s) of type {event_type} remaining")
            return epochs
    return epochs