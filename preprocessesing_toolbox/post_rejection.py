import numpy as np

def reject_if_single_event_type(epochs):
    """
    Drop all epochs if only one event type remains.
    """
        
    # Get remaining event types
    remaining_events = epochs.events[:, 2]
    unique_event_types = np.unique(remaining_events)
    
    if len(unique_event_types) <= 1:
        # Drop all epochs by index
        epochs.drop(range(len(epochs)))
        print(f"All epochs dropped: only {len(unique_event_types)} event type(s) remaining")
    
    return epochs