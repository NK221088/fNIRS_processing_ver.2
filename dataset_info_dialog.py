import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import numpy as np
import mne
from matplotlib.figure import Figure
from collections import Counter

def create_paradigm_plot(events_data, event_mapping=None, figure_size=(8, 6)):
    """
    Create a paradigm visualization plot from events data.
    
    Parameters:
    -----------
    events_data : array-like
        Event data in format [[timestamp, duration, event_id], ...]
        or MNE events format
    event_mapping : dict, optional
        Mapping from event_id to condition name {1: 'Control', 2: 'Task1', ...}
        If None, will use generic names
    figure_size : tuple
        Figure size (width, height)
    
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    
    # Handle different event data formats
    if hasattr(events_data, 'shape') and events_data.shape[1] >= 3:
        # MNE events format or similar
        events = np.array(events_data)
        timestamps = events[:, 0]
        event_ids = events[:, 2]
    elif isinstance(events_data, (list, tuple)) and len(events_data) > 0:
        # List format
        events = np.array(events_data)
        timestamps = events[:, 0]
        event_ids = events[:, 2]
    else:
        raise ValueError("Unsupported events data format")
    
    # Create event mapping if not provided
    if event_mapping is None:
        unique_ids = np.unique(event_ids)
        event_mapping = {int(id): f'Condition_{int(id)}' for id in unique_ids}
    
    # Convert to minutes for better readability
    timestamps_min = timestamps / 60
    
    # Create color mapping
    unique_conditions = list(event_mapping.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_conditions)))
    color_map = {condition: colors[i] for i, condition in enumerate(unique_conditions)}
    
    # Create the main figure
    fig = Figure(figsize=figure_size, dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5], hspace=0.4, wspace=0.3)
    
    # Calculate total duration
    total_duration = timestamps.max() / 60
    
    # 1. Condition distribution bar plot
    ax1 = fig.add_subplot(gs[0, 0])
    condition_counts = Counter([event_mapping[id] for id in event_ids])
    conditions = list(condition_counts.keys())
    counts = list(condition_counts.values())
    colors_bar = [color_map[condition] for condition in conditions]
    
    bars = ax1.bar(conditions, counts, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_title('Condition Distribution', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=9)
    
    # Add percentage labels inside bars
    total_events = len(events)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{count}\n({count/total_events*100:.1f}%)',
                ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    ax1.tick_params(axis='x', rotation=45, labelsize=8)
    ax1.tick_params(axis='y', labelsize=8)
    
    # 2. Inter-event intervals histogram
    ax2 = fig.add_subplot(gs[0, 1])
    inter_event_intervals = np.diff(timestamps)
    ax2.hist(inter_event_intervals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_title('Inter-Event Intervals', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Interval (seconds)', fontsize=9)
    ax2.set_ylabel('Frequency', fontsize=9)
    ax2.axvline(np.mean(inter_event_intervals), color='red', linestyle='--', 
               label=f'Mean: {np.mean(inter_event_intervals):.1f}s')
    ax2.legend(fontsize=8)
    ax2.tick_params(axis='both', labelsize=8)
    
    # 3. Event sequence timeline with scrollable view
    ax3 = fig.add_subplot(gs[1, :])
    
    # Use all events for the timeline
    event_sequence = event_ids
    time_sequence = timestamps
    
    # Create a timeline with colored blocks for all events
    for i, (time, event_id) in enumerate(zip(time_sequence, event_sequence)):
        condition = event_mapping[event_id]
        color = color_map[condition]
        
        # Create a rectangle for each event
        rect = plt.Rectangle((time/60, 0), 0.8, 1, facecolor=color, alpha=0.8, edgecolor='black')
        ax3.add_patch(rect)
    
    # Set up the plot limits and labels
    ax3.set_xlim(0, time_sequence[-1]/60 + 1)
    ax3.set_ylim(0, 1)
    ax3.set_xlabel('Time (minutes)', fontsize=9)
    ax3.set_title(f'Event Sequence Timeline (All {len(events)} Events)', fontsize=10, fontweight='bold')
    ax3.set_yticks([])
    ax3.tick_params(axis='x', labelsize=8)
    
    # Enable horizontal scrolling by setting initial view to first portion
    total_duration_min = time_sequence[-1]/60
    if total_duration_min > 20:  # If longer than 20 minutes, show first 20 minutes initially
        ax3.set_xlim(0, 20)
    
    # Add experiment info as text
    ax3.text(0.02, 0.85, f'Total Duration: {total_duration:.1f} min | Total Events: {len(events)}', 
             transform=ax3.transAxes, verticalalignment='top', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add legend for event sequence
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_map[condition], alpha=0.8, edgecolor='black') 
                      for condition in event_mapping.values()]
    ax3.legend(legend_elements, list(event_mapping.values()), 
              loc='upper right', bbox_to_anchor=(1, 1), fontsize=8)
    
    fig.tight_layout()
    return fig


class DatasetInfoDialog:
    def __init__(self, parent, all_epochs, data_name, all_data, freq, data_types, all_individuals=None):
        self.parent = parent
        self.all_epochs = all_epochs
        self.data_name = data_name
        self.all_data = all_data
        self.freq = freq
        self.data_types = data_types
        self.all_individuals = all_individuals

        self.dialog = tk.Toplevel(parent)
        self.dialog.title(f"Dataset Info - {data_name}")
        
        # Enhanced window management
        self.setup_window()
        self.create_widgets()
        self.populate_info()
        
        # Set focus and make window modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.focus_set()

    def setup_window(self):
        """Setup window with proper sizing and positioning"""
        # Get screen dimensions
        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()
        
        # Calculate window size (80% of screen, but with min/max limits)
        window_width = max(1000, min(int(screen_width * 0.8), 1600))
        window_height = max(700, min(int(screen_height * 0.8), 1200))
        
        # Center the window on screen
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.dialog.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Allow resizing
        self.dialog.resizable(True, True)
        
        # Set minimum size
        self.dialog.minsize(800, 600)
        
        # Configure window state options
        self.dialog.protocol("WM_DELETE_WINDOW", self.close_dialog)
        
        # Add maximize/minimize support
        self.dialog.state('normal')  # Can be 'normal', 'zoomed', 'iconic', 'withdrawn'
        
        # Bind resize event to handle dynamic content adjustment
        self.dialog.bind('<Configure>', self.on_window_resize)

    def on_window_resize(self, event):
        """Handle window resize events"""
        # Only respond to window resize events, not child widget events
        if event.widget == self.dialog:
            self.adjust_layout_for_size()

    def adjust_layout_for_size(self):
        """Adjust layout based on current window size"""
        current_width = self.dialog.winfo_width()
        current_height = self.dialog.winfo_height()
        
        # Adjust grid weights based on window size
        if current_width < 1000:
            # Smaller window - stack vertically
            self.main_frame.columnconfigure(0, weight=1)
            self.main_frame.columnconfigure(1, weight=1)
            self.main_frame.rowconfigure(0, weight=1)
            self.main_frame.rowconfigure(1, weight=1)
            self.main_frame.rowconfigure(2, weight=1)
        else:
            # Larger window - side by side layout
            self.main_frame.columnconfigure(0, weight=2)
            self.main_frame.columnconfigure(1, weight=3)
            self.main_frame.rowconfigure(0, weight=1)
            self.main_frame.rowconfigure(1, weight=1)

    def create_widgets(self):
        # Main container with padding
        container = ttk.Frame(self.dialog)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title bar with window controls
        title_frame = ttk.Frame(container)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = ttk.Label(title_frame, text=f"Dataset Information: {self.data_name}", 
                               font=("Arial", 14, "bold"))
        title_label.pack(side=tk.LEFT)
        
        # Window control buttons
        controls_frame = ttk.Frame(title_frame)
        controls_frame.pack(side=tk.RIGHT)
        
        # Minimize button
        minimize_btn = ttk.Button(controls_frame, text="🗕", width=3, 
                                 command=self.minimize_window)
        minimize_btn.pack(side=tk.LEFT, padx=2)
        
        # Maximize/Restore button
        self.maximize_btn = ttk.Button(controls_frame, text="🗖", width=3, 
                                      command=self.toggle_maximize)
        self.maximize_btn.pack(side=tk.LEFT, padx=2)
        
        # Close button
        close_btn = ttk.Button(controls_frame, text="✕", width=3, 
                              command=self.close_dialog)
        close_btn.pack(side=tk.LEFT, padx=2)
        
        # Main content area with notebook for better organization
        self.notebook = ttk.Notebook(container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Overview
        self.overview_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.overview_frame, text="Overview")
        
        # Tab 2: Detailed Info
        self.details_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.details_frame, text="Detailed Info")
        
        # Tab 3: Paradigm Analysis
        self.paradigm_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.paradigm_frame, text="Paradigm Analysis")
        
        # Tab 4: Channel Layout
        self.channel_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.channel_frame, text="Channel Layout")
        
        # Setup main frame for overview tab
        self.main_frame = ttk.Frame(self.overview_frame)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        self.main_frame.columnconfigure(0, weight=2)
        self.main_frame.columnconfigure(1, weight=3)
        self.main_frame.rowconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        
        # Info section (left side)
        self.info_frame = ttk.LabelFrame(self.main_frame, text="Dataset Summary")
        self.info_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10), pady=(0, 10))
        
        # Quick stats (top right)
        self.stats_frame = ttk.LabelFrame(self.main_frame, text="Quick Statistics")
        self.stats_frame.grid(row=0, column=1, sticky="nsew", pady=(0, 10))
        
        # Paradigm preview (bottom right)
        self.paradigm_preview_frame = ttk.LabelFrame(self.main_frame, text="Paradigm Preview")
        self.paradigm_preview_frame.grid(row=1, column=1, sticky="nsew")

    def minimize_window(self):
        """Minimize the window"""
        self.dialog.iconify()

    def toggle_maximize(self):
        """Toggle between maximized and normal window state"""
        if self.dialog.state() == 'zoomed':
            self.dialog.state('normal')
            self.maximize_btn.config(text="🗖")
        else:
            self.dialog.state('zoomed')
            self.maximize_btn.config(text="🗗")

    def populate_info(self):
        """Populate all tabs with information"""
        self.populate_overview()
        self.populate_detailed_info()
        self.populate_paradigm_analysis()
        self.populate_channel_layout()

    def populate_overview(self):
        """Populate the overview tab with summary information"""
        # Info section
        info_container = ttk.Frame(self.info_frame)
        info_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Basic info
        ttk.Label(info_container, text="Basic Information", font=("Arial", 11, "bold")).pack(anchor="w", pady=(0, 5))
        
        total_participants = len(self.all_epochs)
        ttk.Label(info_container, text=f"Participants: {total_participants}").pack(anchor="w", pady=2)
        
        if self.all_individuals:
            excluded_participants = len(self.all_individuals) - total_participants
            ttk.Label(info_container, text=f"Excluded: {excluded_participants}").pack(anchor="w", pady=2)
        
        ttk.Label(info_container, text=f"Sampling Rate: {self.freq:.2f} Hz").pack(anchor="w", pady=2)
        
        # Conditions
        ttk.Label(info_container, text="Conditions:", font=("Arial", 11, "bold")).pack(anchor="w", pady=(10, 5))
        for i, data_type in enumerate(self.data_types, 1):
            ttk.Label(info_container, text=f"{i}. {data_type}").pack(anchor="w", pady=1)
        
        # Quick stats
        self.populate_quick_stats()
        
        # Paradigm preview
        self.populate_paradigm_preview()

    def populate_quick_stats(self):
        """Populate quick statistics section"""
        stats_container = ttk.Frame(self.stats_frame)
        stats_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        if self.all_epochs:
            first_epoch = self.all_epochs[0]
            
            # Channel counts
            hbo_channels = [ch for ch in first_epoch.ch_names if ch.lower().endswith('hbo')]
            hbr_channels = [ch for ch in first_epoch.ch_names if ch.lower().endswith('hbr')]
            
            ttk.Label(stats_container, text="Channel Counts", font=("Arial", 11, "bold")).pack(anchor="w", pady=(0, 5))
            ttk.Label(stats_container, text=f"Total: {len(first_epoch.ch_names)}").pack(anchor="w", pady=2)
            ttk.Label(stats_container, text=f"HbO: {len(hbo_channels)}").pack(anchor="w", pady=2)
            ttk.Label(stats_container, text=f"HbR: {len(hbr_channels)}").pack(anchor="w", pady=2)
            
            # Epoch counts
            ttk.Label(stats_container, text="Epoch Counts", font=("Arial", 11, "bold")).pack(anchor="w", pady=(10, 5))
            total_epochs = sum(len(epoch) for epoch in self.all_epochs)
            ttk.Label(stats_container, text=f"Total: {total_epochs}").pack(anchor="w", pady=2)
            
            for data_type in self.data_types:
                if data_type in self.all_data:
                    n_epochs = self.all_data[data_type].shape[0]
                    ttk.Label(stats_container, text=f"{data_type}: {n_epochs}").pack(anchor="w", pady=2)

    def populate_paradigm_preview(self):
        """Populate paradigm preview section"""
        preview_container = ttk.Frame(self.paradigm_preview_frame)
        preview_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        try:
            if self.all_epochs:
                first_epoch = self.all_epochs[0]
                events = first_epoch.events
                event_mapping = {v: k for k, v in first_epoch.event_id.items()}
                
                # Create a small preview plot
                fig = Figure(figsize=(4, 2), dpi=100)
                ax = fig.add_subplot(111)
                
                # Simple event distribution
                condition_counts = Counter([event_mapping[id] for id in events[:, 2]])
                conditions = list(condition_counts.keys())
                counts = list(condition_counts.values())
                
                ax.bar(range(len(conditions)), counts)
                ax.set_xticks(range(len(conditions)))
                ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=8)
                ax.set_ylabel('Count', fontsize=8)
                ax.set_title('Event Distribution', fontsize=9)
                
                fig.tight_layout()
                
                canvas = FigureCanvasTkAgg(fig, preview_container)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
        except Exception as e:
            error_label = ttk.Label(preview_container, text=f"Preview unavailable: {str(e)}")
            error_label.pack(expand=True)

    def populate_detailed_info(self):
        """Populate detailed information tab"""
        # Create scrollable frame
        canvas = tk.Canvas(self.details_frame)
        scrollbar = ttk.Scrollbar(self.details_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Populate with detailed information
        ttk.Label(scrollable_frame, text="Detailed Dataset Information", font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 10))

        # All the detailed information from the original populate_general_info method
        total_participants = len(self.all_epochs)
        ttk.Label(scrollable_frame, text=f"Total Participants: {total_participants}").pack(anchor="w", pady=2)

        if self.all_individuals:
            excluded_participants = len(self.all_individuals) - total_participants
            ttk.Label(scrollable_frame, text=f"Excluded Participants: {excluded_participants}").pack(anchor="w", pady=2)

        ttk.Label(scrollable_frame, text=f"Sampling Frequency: {self.freq:.2f} Hz").pack(anchor="w", pady=2)

        # Continue with detailed channel information, bad channels, summary statistics etc.
        if self.all_epochs:
            first_epoch = self.all_epochs[0]
            
            ttk.Label(scrollable_frame, text="Channel Information:", font=("Arial", 12, "bold")).pack(anchor="w", pady=(10, 2))
            ttk.Label(scrollable_frame, text=f"  • Total Channels: {len(first_epoch.ch_names)}").pack(anchor="w", pady=1)

            hbo_channels = [ch for ch in first_epoch.ch_names if ch.lower().endswith('hbo')]
            hbr_channels = [ch for ch in first_epoch.ch_names if ch.lower().endswith('hbr')]
            ttk.Label(scrollable_frame, text=f"  • HbO Channels: {len(hbo_channels)}").pack(anchor="w", pady=1)
            ttk.Label(scrollable_frame, text=f"  • HbR Channels: {len(hbr_channels)}").pack(anchor="w", pady=1)

            if hasattr(first_epoch, 'info') and 'bads' in first_epoch.info:
                bad_channels = first_epoch.info['bads']
                ttk.Label(scrollable_frame, text=f"  • Bad Channels: {len(bad_channels)}").pack(anchor="w", pady=1)
                if bad_channels:
                    text = ", ".join(bad_channels[:5])
                    if len(bad_channels) > 5:
                        text += f" ... (+{len(bad_channels)-5} more)"
                    ttk.Label(scrollable_frame, text=f"    {text}").pack(anchor="w", pady=1)

        # Summary statistics
        ttk.Label(scrollable_frame, text="Summary Statistics:", font=("Arial", 12, "bold")).pack(anchor="w", pady=(10, 2))
        for data_type in self.data_types:
            if data_type in self.all_data:
                data = self.all_data[data_type]
                mean_val = np.mean(data)
                std_val = np.std(data)
                min_val = np.min(data)
                max_val = np.max(data)
                ttk.Label(scrollable_frame, text=f"  {data_type}:").pack(anchor="w", pady=1)
                ttk.Label(scrollable_frame, text=f"    Mean: {mean_val:.6f}").pack(anchor="w", pady=1)
                ttk.Label(scrollable_frame, text=f"    Std: {std_val:.6f}").pack(anchor="w", pady=1)
                ttk.Label(scrollable_frame, text=f"    Range: {min_val:.6f} to {max_val:.6f}").pack(anchor="w", pady=1)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def populate_paradigm_analysis(self):
        """Populate paradigm analysis tab with full paradigm plot"""
        try:
            if self.all_epochs:
                first_epoch = self.all_epochs[0]
                events = first_epoch.events
                event_mapping = {v: k for k, v in first_epoch.event_id.items()}
                
                # Create full paradigm plot
                fig = create_paradigm_plot(events, event_mapping, figure_size=(12, 8))
                
                canvas = FigureCanvasTkAgg(fig, self.paradigm_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                # Add navigation toolbar
                toolbar = NavigationToolbar2Tk(canvas, self.paradigm_frame)
                toolbar.update()
                
        except Exception as e:
            error_label = ttk.Label(self.paradigm_frame, text=f"Paradigm analysis error: {str(e)}")
            error_label.pack(expand=True)

    def populate_channel_layout(self):
        """Populate channel layout tab"""
        try:
            if self.all_epochs:
                fig = Figure(figsize=(8, 6), dpi=100)
                ax = fig.add_subplot(111)
                first_epoch = self.all_epochs[0]

                if hasattr(first_epoch, 'info'):
                    info = first_epoch.info
                    raw_for_plot = mne.io.RawArray(np.zeros((len(info['ch_names']), 1000)), info)
                    raw_for_plot.plot_sensors(kind="topomap", show_names=True, axes=ax)
                    ax.set_title("Sensor Layout", fontsize=14, fontweight='bold')

                canvas = FigureCanvasTkAgg(fig, self.channel_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                # Add navigation toolbar
                toolbar = NavigationToolbar2Tk(canvas, self.channel_frame)
                toolbar.update()

        except Exception as e:
            error_label = ttk.Label(self.channel_frame, text=f"Channel layout error: {str(e)}")
            error_label.pack(expand=True)

    def close_dialog(self):
        """Close the dialog"""
        self.dialog.destroy()


def show_dataset_info(parent, all_epochs, data_name, all_data, freq, data_types, all_individuals=None):
    """
    Show dataset information dialog with enhanced window management.
    
    Parameters:
    -----------
    parent : tkinter widget
        Parent window
    all_epochs : list
        List of MNE epochs objects
    data_name : str
        Name of the dataset
    all_data : dict
        Dictionary containing processed data
    freq : float
        Sampling frequency
    data_types : list
        List of data condition names
    all_individuals : list, optional
        List of all individual data (for exclusion info)
    """
    try:
        dialog = DatasetInfoDialog(parent, all_epochs, data_name, all_data, freq, data_types, all_individuals)
        return dialog
    except Exception as e:
        messagebox.showerror("Error", f"Failed to create dataset info dialog: {str(e)}")
        return None


# Example usage and testing
if __name__ == "__main__":
    # Test the enhanced dialog
    root = tk.Tk()
    root.title("Main Application")
    root.geometry("400x300")
    
    def test_dialog():
        # Create some sample data for testing
        sample_events = np.array([
            [61, 0, 1], [87, 0, 1], [117, 0, 3], [146, 0, 3], [212, 0, 2], [240, 0, 2],
            [275, 0, 1], [344, 0, 3], [373, 0, 2], [404, 0, 2], [474, 0, 3], [512, 0, 2]
        ])
        
        # Mock epoch object
        class MockEpoch:
            def __init__(self):
                self.events = sample_events
                self.event_id = {'Control': 1, 'Tapping/Left': 2, 'Tapping/Right': 3}
                self.ch_names = [f'S{i}_D{j} hbo' for i in range(1, 6) for j in range(1, 4)] + \
                               [f'S{i}_D{j} hbr' for i in range(1, 6) for j in range(1, 4)]
                self.info = {'bads': ['S1_D1 hbo', 'S2_D3 hbr']}
                
            def __len__(self):
                return len(self.events)
        
        mock_epochs = [MockEpoch()]
        mock_data = {
            'Control': np.random.randn(50, 30, 100),
            'Tapping/Left': np.random.randn(45, 30, 100),
            'Tapping/Right': np.random.randn(48, 30, 100)
        }
        
        show_dataset_info(root, mock_epochs, "Test Dataset", mock_data, 10.0, 
                         ['Control', 'Tapping/Left', 'Tapping/Right'])
    
    test_button = ttk.Button(root, text="Open Dataset Info", command=test_dialog)
    test_button.pack(pady=20)
    
    root.mainloop()