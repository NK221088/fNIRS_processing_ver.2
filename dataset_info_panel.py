import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import numpy as np
import mne
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")  # backend for figure creation; TkAgg is used only for embedding
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


def create_paradigm_plot(events_data, event_mapping=None, figure_size=(8, 6)):
    """
    Create a paradigm visualization plot from events data.

    Parameters
    ----------
    events_data : array-like
        Event data in format [[timestamp, duration, event_id], ...]
        or MNE events format
    event_mapping : dict, optional
        Mapping from event_id to condition name {1: 'Control', 2: 'Task1', ...}
        If None, will use generic names
    figure_size : tuple
        Figure size (width, height)

    Returns
    -------
    fig : matplotlib.figure.Figure
    timeline_ax : matplotlib.axes.Axes
    total_duration_min : float
    """
    if hasattr(events_data, 'shape') and events_data.shape[1] >= 3:
        events = np.array(events_data)
        timestamps = events[:, 0]
        event_ids = events[:, 2]
    elif isinstance(events_data, (list, tuple)) and len(events_data) > 0:
        events = np.array(events_data)
        timestamps = events[:, 0]
        event_ids = events[:, 2]
    else:
        raise ValueError("Unsupported events data format")

    if event_mapping is None:
        unique_ids = np.unique(event_ids)
        event_mapping = {int(_id): f'Condition_{int(_id)}' for _id in unique_ids}

    timestamps_min = timestamps / 60

    unique_conditions = list(event_mapping.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_conditions)))
    color_map = {condition: colors[i] for i, condition in enumerate(unique_conditions)}

    fig = Figure(figsize=figure_size, dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5], hspace=0.4, wspace=0.3)

    total_duration = timestamps.max() / 60

    # 1) Condition distribution
    ax1 = fig.add_subplot(gs[0, 0])
    condition_counts = Counter([event_mapping[_eid] for _eid in event_ids])
    conditions = list(condition_counts.keys())
    counts = list(condition_counts.values())
    colors_bar = [color_map[c] for c in conditions]

    bars = ax1.bar(conditions, counts, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_title('Condition Distribution', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=9)
    total_events = len(events)
    for bar, count in zip(bars, counts):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h/2,
                 f'{count}\n({count/total_events*100:.1f}%)',
                 ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    ax1.tick_params(axis='x', rotation=45, labelsize=8)
    ax1.tick_params(axis='y', labelsize=8)

    # 2) Inter-event intervals
    ax2 = fig.add_subplot(gs[0, 1])
    inter_event_intervals = np.diff(timestamps)
    if len(inter_event_intervals) > 0:
        ax2.hist(inter_event_intervals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(inter_event_intervals), color='red', linestyle='--',
                    label=f'Mean: {np.mean(inter_event_intervals):.1f}s')
        ax2.legend(fontsize=8)
    ax2.set_title('Inter-Event Intervals', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Interval (seconds)', fontsize=9)
    ax2.set_ylabel('Frequency', fontsize=9)
    ax2.tick_params(axis='both', labelsize=8)

    # 3) Timeline across all events
    ax3 = fig.add_subplot(gs[1, :])
    for time, event_id in zip(timestamps, event_ids):
        condition = event_mapping[event_id]
        color = color_map[condition]
        rect = matplotlib.patches.Rectangle((time/60, 0), 0.8, 1, facecolor=color, alpha=0.8, edgecolor='black')
        ax3.add_patch(rect)

    ax3.set_xlim(0, timestamps[-1]/60 + 1)
    ax3.set_ylim(0, 1)
    ax3.set_xlabel('Time (minutes)', fontsize=9)
    ax3.set_title(f'Event Sequence Timeline (All {len(events)} Events)', fontsize=10, fontweight='bold')
    ax3.set_yticks([])
    ax3.tick_params(axis='x', labelsize=8)

    total_duration_min = timestamps[-1]/60
    if total_duration_min > 20:
        ax3.set_xlim(0, 20)

    ax3.text(0.02, 0.85, f'Total Duration: {total_duration:.1f} min | Total Events: {len(events)}',
             transform=ax3.transAxes, va='top', fontsize=8,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    legend_elements = [matplotlib.patches.Rectangle((0, 0), 1, 1, facecolor=color_map[c], alpha=0.8, edgecolor='black')
                       for c in event_mapping.values()]
    ax3.legend(legend_elements, list(event_mapping.values()), loc='upper right', fontsize=8)

    fig.tight_layout()
    return fig, ax3, total_duration_min


class ScrollableTimelineFrame(ttk.Frame):
    """Canvas + optional horizontal scrollbar that updates the timeline view."""
    def __init__(self, parent, fig, timeline_ax, total_duration_min):
        super().__init__(parent)
        self.fig = fig
        self.timeline_ax = timeline_ax
        self.total_duration_min = total_duration_min
        self.view_window = 20

        if total_duration_min > self.view_window:
            self.scrollbar = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.on_scroll)
            self.scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        else:
            self.scrollbar = None

        self.canvas = FigureCanvasTkAgg(fig, self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        if self.scrollbar:
            self.update_scrollbar()

        self.canvas.get_tk_widget().bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.get_tk_widget().bind("<Button-4>", self.on_mousewheel)
        self.canvas.get_tk_widget().bind("<Button-5>", self.on_mousewheel)

        self.canvas.draw()

    def update_scrollbar(self):
        if self.scrollbar is None:
            return
        x0, x1 = self.timeline_ax.get_xlim()
        scroll_start = max(0, min(x0 / self.total_duration_min, 1))
        scroll_end = max(scroll_start, min(x1 / self.total_duration_min, 1))
        self.scrollbar.set(scroll_start, scroll_end)

    def on_scroll(self, *args):
        if args[0] == 'scroll':
            direction = int(args[1])
            units = args[2]
            x0, x1 = self.timeline_ax.get_xlim()
            width = x1 - x0
            shift = direction * width * (0.1 if units == 'units' else 0.8)
            new_start, new_end = x0 + shift, x1 + shift
        elif args[0] == 'moveto':
            position = float(args[1])
            x0, x1 = self.timeline_ax.get_xlim()
            width = x1 - x0
            new_start = position * self.total_duration_min
            new_end = new_start + width
        else:
            return

        if new_end > self.total_duration_min:
            new_end = self.total_duration_min
            new_start = new_end - (x1 - x0)
        if new_start < 0:
            new_start = 0
            new_end = new_start + (x1 - x0)

        self.timeline_ax.set_xlim(new_start, new_end)
        self.canvas.draw()
        self.update_scrollbar()

    def on_mousewheel(self, event):
        if self.scrollbar is None:
            return
        if event.delta:
            delta = -event.delta / 120
        elif event.num == 4:
            delta = -1
        elif event.num == 5:
            delta = 1
        else:
            return

        x0, x1 = self.timeline_ax.get_xlim()
        width = x1 - x0
        shift = delta * width * 0.1
        new_start, new_end = x0 + shift, x1 + shift

        if new_end > self.total_duration_min:
            new_end = self.total_duration_min
            new_start = new_end - width
        if new_start < 0:
            new_start = 0
            new_end = new_start + width

        self.timeline_ax.set_xlim(new_start, new_end)
        self.canvas.draw()
        self.update_scrollbar()


class _ZoomOverlay:
    """An in-window overlay (no OS pop-up) with a dimmed background and a large figure."""
    def __init__(self, root, build_figure_fn):
        self.root = root
        self.build_figure_fn = build_figure_fn

        self.root.update_idletasks()
        w, h = self.root.winfo_width(), self.root.winfo_height()

        # Full-window canvas overlay
        self.canvas = tk.Canvas(self.root, highlightthickness=0)
        self.canvas.place(x=0, y=0, relwidth=1, relheight=1)
        # Simulate translucency with stipple
        self.bg = self.canvas.create_rectangle(0, 0, w, h, fill="black", stipple="gray50", outline="")

        # Center container
        self.container = tk.Frame(self.canvas, bg="white", bd=2, relief="ridge")
        self.window_id = self.canvas.create_window(w // 2, h // 2, window=self.container, anchor="center")

        # Close button
        topbar = tk.Frame(self.container, bg="white")
        topbar.pack(fill="x")
        ttk.Button(topbar, text="Close ✕", command=self.destroy).pack(side="right", padx=6, pady=6)

        # Large figure
        max_w, max_h = int(w * 0.85), int(h * 0.85)
        fig = self.build_figure_fn(figsize=(max_w / 100.0, max_h / 100.0))
        self.fig = fig
        self.fig_canvas = FigureCanvasTkAgg(fig, master=self.container)
        self.fig_canvas.draw()
        self.fig_canvas.get_tk_widget().pack(fill="both", expand=True)

        # Interactions
        self.canvas.bind("<Button-1>", self._on_bg_click)
        # Prevent closing when clicking inside the content
        self.container.bind("<Button-1>", lambda e: "break")
        self.root.bind("<Escape>", self._on_escape)

        # --- IMPORTANT FIX ---
        # Ensure our content (window item) sits above the dim background.
        # Do NOT call self.canvas.lift(): on Canvas that maps to raising *items* and requires args.
        self.canvas.tag_raise(self.window_id)

    def _on_bg_click(self, event):
        # Click on dim background closes overlay
        if event.widget is self.canvas:
            self.destroy()

    def _on_escape(self, _):
        self.destroy()

    def destroy(self):
        try:
            self.root.unbind("<Escape>")
        except Exception:
            pass
        self.canvas.destroy()


class DatasetInfoPanel:
    """
    Panel version of Dataset Info that renders into a container (no separate window).
    - Shows General Info (scrollable)
    - Shows Sensor Setup (click to zoom overlay)
    - Shows Paradigm overview with a scrollable timeline
    - Shows Drop Log summary in its own tab
    """
    def __init__(self, class_instance, parent_container, all_epochs, data_name, all_data, freq, data_types, all_individuals=None):
        self.class_instance = class_instance
        self.parent_container = parent_container
        self.root = parent_container.winfo_toplevel()
        self.all_epochs = all_epochs or []
        self.data_name = data_name
        self.all_data = all_data or {}
        self.freq = freq
        self.data_types = data_types or []
        self.all_individuals = all_individuals

        # Main wrapper inside the container
        self.frame = ttk.Frame(parent_container)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self._build_ui()
        self._populate()

    def destroy(self):
        self.frame.destroy()

    # ---------- UI ----------
    def _build_ui(self):
        header = ttk.Frame(self.frame)
        header.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(
            header, text=f"Dataset Information: {self.data_name}",
            font=("Arial", 14, "bold")
        ).pack(side=tk.LEFT)

        ttk.Button(header, text="Clear", command=self.destroy).pack(side=tk.RIGHT)

        self.main = ttk.Frame(self.frame)
        self.main.pack(fill=tk.BOTH, expand=True)

        self.main.columnconfigure(0, weight=3)   # Info
        self.main.columnconfigure(1, weight=2)   # Plots
        self.main.rowconfigure(0, weight=1)
        self.main.rowconfigure(1, weight=1)

        # Left (scrollable info)
        self.info_frame = ttk.Frame(self.main)
        self.info_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))

        # Right top: sensor layout
        self.layout_frame = ttk.Frame(self.main)
        self.layout_frame.grid(row=0, column=1, sticky="nsew", pady=(0, 10))

        # Right bottom: tabs for paradigm and drop log only
        self.paradigm_frame = ttk.Frame(self.main)
        self.paradigm_frame.grid(row=1, column=1, sticky="nsew")

        # Notebook to hold multiple plots on the bottom-right
        self.plot_tabs = ttk.Notebook(self.paradigm_frame)
        self.paradigm_tab = ttk.Frame(self.plot_tabs)
        self.drop_tab = ttk.Frame(self.plot_tabs)
        self.bad_channels_tab = ttk.Frame(self.plot_tabs)
        self.plot_tabs.add(self.paradigm_tab, text="Paradigm & Timeline")
        self.plot_tabs.add(self.drop_tab, text="Drop Log")
        self.plot_tabs.add(self.bad_channels_tab, text="Bad Channels")
        self.plot_tabs.pack(fill=tk.BOTH, expand=True)

    # ---------- Data population ----------
    def _populate(self):
        self._populate_general_info()
        self._populate_sensor_layout()
        self._populate_paradigm()
        self._populate_drop_log()
        self._populate_bad_channels_visualization()

    def _analyze_bad_channels(self):
        """Analyze bad channels across all participants"""
        all_bads = []
        participant_bads = []
        bad_channel_participants = defaultdict(list)  # Track which participants have each bad channel
        
        for i, epochs in enumerate(self.all_epochs):
            if hasattr(epochs, 'info') and 'bads' in epochs.info:
                bads = epochs.info['bads']
                participant_bads.append(bads)
                all_bads.extend(bads)
                
                # Track which participants have each bad channel
                for bad_ch in bads:
                    bad_channel_participants[bad_ch].append(i)
            else:
                participant_bads.append([])
        
        bad_counts = Counter(all_bads)
        participants_with_bads = sum(1 for bads in participant_bads if bads)
        
        return {
            'total_unique_bads': len(bad_counts),
            'participants_with_bads': participants_with_bads,
            'avg_bads_per_participant': len(all_bads) / len(self.all_epochs) if self.all_epochs else 0,
            'most_common_bads': bad_counts.most_common(),
            'participant_bads': participant_bads,
            'bad_channel_participants': dict(bad_channel_participants)
        }

    def _populate_general_info(self):
        canvas = tk.Canvas(self.info_frame, highlightthickness=0)
        vbar = ttk.Scrollbar(self.info_frame, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)

        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vbar.set)
        
        total_participants = self.class_instance.number_of_participants
        ttk.Label(inner, text=f"Total Participants: {total_participants}").pack(anchor="w", pady=2)

        excluded = self.class_instance.number_of_participants - len(self.class_instance.all_epochs)
        ttk.Label(inner, text=f"Excluded Participants: {excluded}").pack(anchor="w", pady=2)

        ttk.Label(inner, text=f"Sampling Frequency: {self.freq:.2f} Hz").pack(anchor="w", pady=2)

        ttk.Label(inner, text="Conditions:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 2))
        for dt in self.data_types:
            ttk.Label(inner, text=f"  • {dt}").pack(anchor="w", pady=1)

        if self.all_epochs:
            first_epoch = self.all_epochs[0]
            ttk.Label(inner, text="Channel Information:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 2))
            ttk.Label(inner, text=f"  • Total Channels: {len(first_epoch.ch_names)}").pack(anchor="w", pady=1)

            # ENHANCED BAD CHANNELS ANALYSIS
            bad_channel_analysis = self._analyze_bad_channels()
            
            if bad_channel_analysis['total_unique_bads'] > 0:
                ttk.Label(inner, text=f"  • Bad Channels Summary:").pack(anchor="w", pady=1)
                ttk.Label(inner, text=f"    - Total unique bad channels: {bad_channel_analysis['total_unique_bads']}").pack(anchor="w", pady=1)
                ttk.Label(inner, text=f"    - Participants with bad channels: {bad_channel_analysis['participants_with_bads']}/{len(self.all_epochs)}").pack(anchor="w", pady=1)
                ttk.Label(inner, text=f"    - Average bad channels per participant: {bad_channel_analysis['avg_bads_per_participant']:.1f}").pack(anchor="w", pady=1)
                
                # Show most frequently bad channels
                if bad_channel_analysis['most_common_bads']:
                    ttk.Label(inner, text=f"    - Most frequently bad:").pack(anchor="w", pady=1)
                    for ch, count in bad_channel_analysis['most_common_bads'][:3]:
                        percentage = (count / len(self.all_epochs)) * 100
                        ttk.Label(inner, text=f"      {ch} ({count} participants, {percentage:.1f}%)").pack(anchor="w", pady=1)
                
                # Expandable detailed view
                self._add_expandable_bad_channels_section(inner, bad_channel_analysis)
            else:
                ttk.Label(inner, text=f"  • No bad channels marked across dataset").pack(anchor="w", pady=1)

        ttk.Label(inner, text="Epoch Information:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 2))
        total_epochs = sum(len(ep) for ep in self.all_epochs) if self.all_epochs else 0
        ttk.Label(inner, text=f"  • Total Epochs: {total_epochs}").pack(anchor="w", pady=1)

        for dt in self.data_types:
            if dt in self.all_data:
                n_epochs = self.all_data[dt].shape[0]
                ttk.Label(inner, text=f"  • {dt}: {n_epochs} epochs").pack(anchor="w", pady=1)

        ttk.Label(inner, text="Summary Statistics:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 2))
        for dt in self.data_types:
            if dt in self.all_data:
                data = self.all_data[dt]
                ttk.Label(inner, text=f"  {dt}:").pack(anchor="w", pady=1)
                ttk.Label(inner, text=f"    Mean: {np.mean(data):.6f}").pack(anchor="w", pady=1)
                ttk.Label(inner, text=f"    Std: {np.std(data):.6f}").pack(anchor="w", pady=1)
                ttk.Label(inner, text=f"    Range: {np.min(data):.6f} to {np.max(data):.6f}").pack(anchor="w", pady=1)

        canvas.pack(side="left", fill="both", expand=True)
        vbar.pack(side="right", fill="y")

    def _add_expandable_bad_channels_section(self, parent, analysis):
        """Add an expandable section for detailed bad channel information"""
        
        # Create expandable frame
        expand_frame = ttk.Frame(parent)
        expand_frame.pack(anchor="w", fill="x", pady=(5, 0))
        
        self.bad_channels_expanded = tk.BooleanVar(value=False)
        
        def toggle_bad_channels():
            if self.bad_channels_expanded.get():
                self._show_detailed_bad_channels(details_frame, analysis)
            else:
                # Clear the details frame
                for widget in details_frame.winfo_children():
                    widget.destroy()
        
        # Toggle button
        toggle_btn = ttk.Checkbutton(
            expand_frame,
            text="Show detailed bad channels breakdown",
            variable=self.bad_channels_expanded,
            command=toggle_bad_channels
        )
        toggle_btn.pack(anchor="w")
        
        # Details frame (initially empty)
        details_frame = ttk.Frame(expand_frame)
        details_frame.pack(anchor="w", fill="x", padx=(20, 0))

    def _show_detailed_bad_channels(self, parent, analysis):
        """Show detailed bad channel breakdown"""
        
        # Table-style view
        table_frame = ttk.Frame(parent)
        table_frame.pack(anchor="w", fill="x", pady=5)
        
        # Header
        ttk.Label(table_frame, text="Channel", font=("Arial", 9, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 10))
        ttk.Label(table_frame, text="Count", font=("Arial", 9, "bold")).grid(row=0, column=1, sticky="w", padx=(0, 10))
        ttk.Label(table_frame, text="Percentage", font=("Arial", 9, "bold")).grid(row=0, column=2, sticky="w", padx=(0, 10))
        ttk.Label(table_frame, text="Participants", font=("Arial", 9, "bold")).grid(row=0, column=3, sticky="w")
        
        # Separator
        ttk.Separator(table_frame, orient='horizontal').grid(row=1, column=0, columnspan=4, sticky="ew", pady=2)
        
        # Data rows
        for i, (channel, count) in enumerate(analysis['most_common_bads'][:10]):  # Show top 10
            row = i + 2
            percentage = (count / len(self.all_epochs)) * 100
            participant_list = analysis['bad_channel_participants'][channel]
            participant_str = ", ".join([f"P{p+1}" for p in participant_list[:5]])  # Show first 5
            if len(participant_list) > 5:
                participant_str += f" (+{len(participant_list)-5} more)"
            
            ttk.Label(table_frame, text=channel).grid(row=row, column=0, sticky="w", padx=(0, 10))
            ttk.Label(table_frame, text=str(count)).grid(row=row, column=1, sticky="w", padx=(0, 10))
            ttk.Label(table_frame, text=f"{percentage:.1f}%").grid(row=row, column=2, sticky="w", padx=(0, 10))
            ttk.Label(table_frame, text=participant_str, font=("Arial", 8)).grid(row=row, column=3, sticky="w")

    def _populate_bad_channels_visualization(self):
        """Create visualizations for bad channels"""
        
        analysis = self._analyze_bad_channels()
        
        if analysis['total_unique_bads'] == 0:
            ttk.Label(self.bad_channels_tab, text="No bad channels to visualize").pack(expand=True)
            return
        
        # Just show a simple text summary instead of plots
        summary_frame = ttk.Frame(self.bad_channels_tab)
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(summary_frame, 
                               text="Bad Channels Analysis", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Main statistics
        stats_text = f"""Dataset Bad Channels Summary:

• Total unique bad channels: {analysis['total_unique_bads']}
• Participants with bad channels: {analysis['participants_with_bads']}/{len(self.all_epochs)} ({analysis['participants_with_bads']/len(self.all_epochs)*100:.1f}%)
• Average bad channels per participant: {analysis['avg_bads_per_participant']:.1f}

Most problematic channels:"""
        
        stats_label = ttk.Label(summary_frame, text=stats_text, font=("Arial", 10))
        stats_label.pack(anchor="w", pady=(0, 10))
        
        # List of most common bad channels
        for i, (ch, count) in enumerate(analysis['most_common_bads'][:10]):
            percentage = (count / len(self.all_epochs)) * 100
            channel_text = f"  {i+1}. {ch}: {count} participants ({percentage:.1f}%)"
            ttk.Label(summary_frame, text=channel_text, font=("Arial", 9)).pack(anchor="w")
        
        # Additional statistics
        bad_counts_per_participant = [len(bads) for bads in analysis['participant_bads']]
        additional_stats = f"""

Additional Statistics:
• Participants with 0 bad channels: {sum(1 for count in bad_counts_per_participant if count == 0)} ({sum(1 for count in bad_counts_per_participant if count == 0)/len(self.all_epochs)*100:.1f}%)
• Participants with 1-3 bad channels: {sum(1 for count in bad_counts_per_participant if 1 <= count <= 3)}
• Participants with 4+ bad channels: {sum(1 for count in bad_counts_per_participant if count >= 4)}
• Maximum bad channels in single participant: {max(bad_counts_per_participant) if bad_counts_per_participant else 0}
• Range: {min(bad_counts_per_participant)} - {max(bad_counts_per_participant)} bad channels per participant"""
        
        additional_label = ttk.Label(summary_frame, text=additional_stats, font=("Arial", 10))
        additional_label.pack(anchor="w", pady=(20, 0))

    def _build_sensor_figure(self, figsize=(5, 4)):
        fig = Figure(figsize=figsize, dpi=100)
        ax = fig.add_subplot(111)
        first_epoch = self.all_epochs[0]
        info = first_epoch.info
        # Minimal dummy data with same channel montage
        raw_for_plot = mne.io.RawArray(np.zeros((len(info['ch_names']), 1000)), info, verbose='ERROR')
        raw_for_plot.plot_sensors(kind="topomap", show_names=True, axes=ax)
        ax.set_title("Sensor Setup")
        fig.tight_layout()
        return fig

    def _open_zoom_overlay(self):
        _ZoomOverlay(self.root, self._build_sensor_figure)

    def _populate_sensor_layout(self):
        try:
            if not self.all_epochs:
                ttk.Label(self.layout_frame, text="No epoch data available for sensor setup").pack()
                return
            fig = self._build_sensor_figure(figsize=(5, 4))
            canvas = FigureCanvasTkAgg(fig, self.layout_frame)
            canvas.draw()
            w = canvas.get_tk_widget()
            w.pack(fill=tk.BOTH, expand=True)

            # Click-to-zoom hint and handler
            hint = ttk.Label(self.layout_frame, text="Click the plot to zoom • Esc or click outside to close", foreground="gray")
            hint.pack(pady=(4, 0))
            w.bind("<Button-1>", lambda _e: self._open_zoom_overlay())
        except Exception as e:
            ttk.Label(self.layout_frame, text=f"Sensor setup error: {e}").pack(expand=True)

    def _populate_paradigm(self):
        try:
            if not self.all_epochs:
                ttk.Label(self.paradigm_tab, text="No epoch data available").pack(expand=True)
                return

            first_epoch = self.all_epochs[0]
            events = first_epoch.events
            event_mapping = {v: k for k, v in first_epoch.event_id.items()}

            fig, timeline_ax, total_duration_min = create_paradigm_plot(events, event_mapping, figure_size=(8, 6))
            timeline_frame = ScrollableTimelineFrame(self.paradigm_tab, fig, timeline_ax, total_duration_min)
            timeline_frame.pack(fill=tk.BOTH, expand=True)

            ttk.Label(
                self.paradigm_tab,
                text="Use the scrollbar or mouse wheel to navigate the timeline.",
                font=("Arial", 9), foreground="gray"
            ).pack(pady=(5, 0))
        except Exception as e:
            ttk.Label(self.paradigm_tab, text=f"Paradigm plot error: {e}").pack(expand=True)

    def _populate_drop_log(self):
        """Create a drop log figure like mne.viz.plot_drop_log for all participants and embed it."""
        container = self.drop_tab
        for w in container.winfo_children():
            w.destroy()

        if not self.all_epochs:
            ttk.Label(container, text="No epoch data available").pack(expand=True)
            return

        # Accumulate drop logs across all Epochs objects
        total_drop_log = ()
        try:
            for epochs in self.all_epochs:
                if hasattr(epochs, "drop_log") and epochs.drop_log is not None:
                    total_drop_log += epochs.drop_log
        except Exception as e:
            ttk.Label(container, text=f"Error collecting drop logs: {e}").pack(expand=True)
            return

        if len(total_drop_log) == 0:
            ttk.Label(container, text="No dropped epochs to display").pack(expand=True)
            return

        # Build figure using MNE's helper
        try:
            # Prefer not to open an external window; many MNE plotting functions accept show=False
            try:
                plot_ret = mne.viz.plot_drop_log(total_drop_log, show=False)
            except TypeError:
                plot_ret = mne.viz.plot_drop_log(total_drop_log)

            # Handle possible return signatures
            fig = plot_ret[0] if isinstance(plot_ret, (list, tuple)) and hasattr(plot_ret[0], "savefig") else plot_ret
            if not hasattr(fig, "savefig"):
                raise RuntimeError("mne.viz.plot_drop_log did not return a Figure as expected")

            # Simple toolbar with Save-as (pinned at top and always visible)
            toolbar = ttk.Frame(container)
            toolbar.pack(side=tk.TOP, fill=tk.X)

            def _save_as(ext):
                default_name = f"total_drop_log.{ext}"
                fpath = filedialog.asksaveasfilename(
                    title="Save Drop Log Figure",
                    defaultextension=f".{ext}",
                    initialfile=default_name,
                    filetypes=[(ext.upper(), f"*.{ext}"), ("PDF", "*.pdf"), ("PNG", "*.png"), ("All", "*.*")],
                )
                if fpath:
                    try:
                        fig.savefig(fpath, bbox_inches="tight")
                    except Exception as save_e:
                        messagebox.showerror("Save Error", f"Could not save figure: {save_e}")

            # Buttons first so they don't get hidden by the expanding canvas
            btn_pdf = ttk.Button(toolbar, text="Save as PDF", command=lambda: _save_as("pdf"))
            btn_pdf.pack(side=tk.LEFT, padx=4, pady=4)
            btn_png = ttk.Button(toolbar, text="Save as PNG", command=lambda: _save_as("png"))
            btn_png.pack(side=tk.LEFT, padx=4, pady=4)

            # Then the figure canvas, which can expand and won't obscure the toolbar
            canvas = FigureCanvasTkAgg(fig, master=container)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # Optional: compact textual summary of reasons
            # Count reasons across all dropped epochs
            reason_counts = Counter()
            for entry in total_drop_log:
                if isinstance(entry, (list, tuple)):
                    for r in entry:
                        if r:  # ignore empty strings
                            reason_counts[r] += 1

            if reason_counts:
                summary = ttk.Frame(container)
                summary.pack(fill=tk.X, pady=(2, 4))
                ttk.Label(
                    summary,
                    text="Top drop reasons:",
                    font=("Arial", 9, "bold"),
                    foreground="gray",
                ).pack(side=tk.LEFT, padx=6)
                txt = ", ".join([f"{k}: {v}" for k, v in reason_counts.most_common(5)])
                ttk.Label(summary, text=txt, foreground="gray").pack(side=tk.LEFT)

        except Exception as e:
            ttk.Label(container, text=f"Drop log plot error: {e}").pack(expand=True)


def show_dataset_info_in_container(class_instance, parent_container, all_epochs, data_name, all_data, freq, data_types, all_individuals=None):
    """
    Public helper — create and mount the info panel inside `parent_container`.
    Returns the DatasetInfoPanel instance (so you may call .destroy() later if needed).
    """
    return DatasetInfoPanel(class_instance, parent_container, all_epochs, data_name, all_data, freq, data_types, all_individuals)