import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from collections import Counter

def plot_experiment_timeline(annotations, figsize=(16, 6)):
    """
    Plot experiment timeline from MNE annotations object.
    
    Parameters:
    -----------
    annotations : mne.Annotations
        MNE annotations object containing experiment events
    figsize : tuple, optional
        Figure size (width, height). Default is (16, 6)
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    
    # Extract data from annotations
    descriptions = annotations.description
    durations = annotations.duration
    onsets = annotations.onset
    
    # Generate color mapping dynamically for all unique event types
    unique_events = list(set(descriptions))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_events)))  # Use matplotlib colormap
    color_map = {event: colors[i] for i, event in enumerate(unique_events)}
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Plot timeline
    y_pos = 0
    for i, (desc, onset, duration) in enumerate(zip(descriptions, onsets, durations)):
        color = color_map.get(desc, '#CCCCCC')
        
        # Create rectangle for each event
        rect = patches.Rectangle((onset, y_pos), duration, 0.8, 
                               linewidth=1, edgecolor='black', facecolor=color, alpha=0.7)
        ax.add_patch(rect)
    
    # Set plot properties
    ax.set_xlim(min(onsets) - 5, max(onsets) + max(durations) + 10)
    ax.set_ylim(-0.2, 1)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_title('Experiment Timeline Overview', fontsize=14, fontweight='bold')
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [patches.Patch(facecolor=color_map[event], label=event, alpha=0.7) 
                      for event in unique_events]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    # Print experiment summary
    total_time = max(onsets) + durations[list(onsets).index(max(onsets))]
    print(f"\nExperiment Summary:")
    print(f"Total duration: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Number of events: {len(descriptions)}")
    
    # Count each event type
    event_counts = Counter(descriptions)
    print("\nEvent breakdown:")
    for event_type, count in event_counts.items():
        total_duration = sum(dur for desc, dur in zip(descriptions, durations) if desc == event_type)
        print(f"  {event_type}: {count} events, {total_duration:.1f}s total")
    
    plt.tight_layout()
    return fig, ax

# Example usage:
# fig, ax = plot_experiment_timeline(self.class_instance.Individual_participants[0].raw_haemo.annotations)
# plt.show()