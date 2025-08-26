import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Your data
descriptions = ['Introduction', 'Resting state', 'TongueMI', 'Control', 'TongueMI', 'Control', 
               'TongueMI', 'Control', 'TongueMI', 'Control', 'TongueMI', 'Control', 'TongueMI', 
               'Control', 'Pause', 'TongueMI', 'Control', 'TongueMI', 'Control', 'TongueMI', 
               'Control', 'TongueMI', 'Control', 'TongueMI', 'Control', 'TongueMI', 'Control', 
               'Pause', 'TongueMI', 'Control', 'TongueMI', 'Control', 'TongueMI', 'Control', 
               'TongueMI', 'Control', 'TongueMI', 'Control', 'TongueMI', 'Control', 'Outro']

durations = [80., 30., 21., 21., 21., 21., 21., 21., 21., 21., 21., 21., 21., 21., 30., 
             21., 21., 21., 21., 21., 21., 21., 21., 21., 21., 21., 21., 30., 21., 21., 
             21., 21., 21., 21., 21., 21., 21., 21., 21., 21., 10.]

onsets = [177.43872, 257.4315, 286.06464, 307.053, 329.515008, 350.5016, 372.965376, 
          393.9502, 416.612352, 437.5954, 460.161024, 481.1423, 503.611392, 524.5909, 
          545.5909, 577.732608, 598.7091, 621.28128, 642.256, 664.731648, 685.7046, 
          708.476928, 729.4481, 752.123904, 773.0933, 795.574272, 816.5419, 837.5419, 
          869.203968, 890.1686, 912.850944, 933.8138, 956.596224, 977.5573, 1000.046592, 
          1021.0059, 1043.595264, 1064.5528, 1087.045632, 1108.0014, 1129.0014]

# Color mapping for different event types
color_map = {
    'Introduction': '#FF6B6B',    # Red
    'Resting state': '#4ECDC4',  # Teal
    'TongueMI': '#45B7D1',       # Blue
    'Control': '#96CEB4',        # Green
    'Pause': '#FFEAA7',          # Yellow
    'Outro': '#DDA0DD'           # Plum
}

# Create the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

# Plot 1: Timeline view
y_pos = 0
for i, (desc, onset, duration) in enumerate(zip(descriptions, onsets, durations)):
    color = color_map.get(desc, '#CCCCCC')
    
    # Create rectangle for each event
    rect = patches.Rectangle((onset, y_pos), duration, 0.8, 
                           linewidth=1, edgecolor='black', facecolor=color, alpha=0.7)
    ax1.add_patch(rect)
    
    # Add text label for longer events
    if duration > 25:  # Only label longer events to avoid clutter
        ax1.text(onset + duration/2, y_pos + 0.4, desc, 
                ha='center', va='center', fontsize=10, fontweight='bold')
    elif desc in ['Introduction', 'Outro']:
        ax1.text(onset + duration/2, y_pos + 0.4, desc, 
                ha='center', va='center', fontsize=8, fontweight='bold')

ax1.set_xlim(170, max(onsets) + max(durations) + 10)
ax1.set_ylim(-0.2, 1)
ax1.set_xlabel('Time (seconds)', fontsize=12)
ax1.set_title('Experiment Timeline Overview', fontsize=14, fontweight='bold')
ax1.set_yticks([])
ax1.grid(True, alpha=0.3)

# Add legend
legend_elements = [patches.Patch(facecolor=color, label=event_type, alpha=0.7) 
                  for event_type, color in color_map.items()]
ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))

# Plot 2: Event sequence with alternating pattern highlight
event_counter = {'TongueMI': 0, 'Control': 0, 'Pause': 0}
y_positions = []
colors = []

for i, desc in enumerate(descriptions):
    if desc == 'TongueMI':
        event_counter['TongueMI'] += 1
        y_positions.append(2)
    elif desc == 'Control':
        event_counter['Control'] += 1
        y_positions.append(1)
    elif desc == 'Pause':
        event_counter['Pause'] += 1
        y_positions.append(1.5)
    elif desc == 'Introduction':
        y_positions.append(3)
    elif desc == 'Resting state':
        y_positions.append(2.5)
    elif desc == 'Outro':
        y_positions.append(0.5)
    
    colors.append(color_map.get(desc, '#CCCCCC'))

# Create scatter plot showing event sequence
for i, (onset, duration, y_pos, color, desc) in enumerate(zip(onsets, durations, y_positions, colors, descriptions)):
    ax2.barh(y_pos, duration, left=onset, height=0.3, color=color, alpha=0.7, edgecolor='black')
    
    # Label every few events to show the pattern
    if i % 4 == 0 or desc in ['Introduction', 'Outro', 'Pause', 'Resting state']:
        ax2.text(onset + duration/2, y_pos, f'{i+1}', ha='center', va='center', 
                fontsize=8, fontweight='bold')

ax2.set_xlim(170, max(onsets) + max(durations) + 10)
ax2.set_xlabel('Time (seconds)', fontsize=12)
ax2.set_ylabel('Event Type', fontsize=12)
ax2.set_title('Experiment Structure - Event Categories', fontsize=14, fontweight='bold')
ax2.set_yticks([0.5, 1, 1.5, 2, 2.5, 3])
ax2.set_yticklabels(['Outro', 'Control', 'Pause', 'TongueMI', 'Resting', 'Introduction'])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print experiment summary
total_time = max(onsets) + durations[onsets.index(max(onsets))]
print(f"\nExperiment Summary:")
print(f"Total duration: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
print(f"Number of events: {len(descriptions)}")

# Count each event type
from collections import Counter
event_counts = Counter(descriptions)
print("\nEvent breakdown:")
for event_type, count in event_counts.items():
    total_duration = sum(dur for desc, dur in zip(descriptions, durations) if desc == event_type)
    print(f"  {event_type}: {count} events, {total_duration:.1f}s total")