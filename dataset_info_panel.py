import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import numpy as np
import mne
from collections import Counter, defaultdict
from data_analysis.effect_size import compute_effect_size
from plotting_functions.experiment_overview import plot_experiment_timeline

import matplotlib
matplotlib.use("Agg")  # backend for figure creation; TkAgg is used only for embedding
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


def scale_epochs_to_micro_molar(epochs):
    """
    Return a copy of epochs scaled to µM.
    Checks the current unit + multiplier and rescales if necessary.
    """
    scaled = epochs.copy()
    for ch_idx, ch in enumerate(scaled.info["chs"]):
        if ch["unit"] == mne.io.constants.FIFF.FIFF_UNIT_MOL:
            current_mul = 10 ** ch.get("unit_mul", 0)
            target_mul = 1e-6
            scale_factor = current_mul / target_mul
            scaled._data[:, ch_idx, :] *= scale_factor
            ch["unit"] = mne.io.constants.FIFF.FIFF_UNIT_MOL
            ch["unit_mul"] = -6  # explicitly mark µ (10^-6)
    return scaled


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
            return "break"
        if getattr(event, "delta", 0):
            delta = -event.delta / 120
        elif getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:
            return "break"

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
        return "break"


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
        self.container.bind("<Button-1>", lambda e: "break")
        self.root.bind("<Escape>", self._on_escape)

        # Ensure our content sits above the dim background
        self.canvas.tag_raise(self.window_id)

    def _on_bg_click(self, event):
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
    Uses compute_effect_size() outputs for channel-level summaries (no recomputation here).

    Visual refresh in this version:
      • A fixed (non-scrollable) summary band at the top with key figures.
      • Clear µM units for any concentration-like values.
      • Tighter, cleaner tables and spacing.
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

        self._effect_cache = None  # set by _get_effect_results()

        # Main wrapper inside the container
        self.frame = ttk.Frame(parent_container)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self._build_ui()
        self._populate()

    def destroy(self):
        self.frame.destroy()

    # ---------- UI ----------
    def _build_ui(self):
        # Overall header
        header = ttk.Frame(self.frame)
        header.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(
            header, text=f"Dataset Information: {self.data_name}",
            font=("Arial", 14, "bold")
        ).pack(side=tk.LEFT)

        ttk.Button(header, text="Clear", command=self.destroy).pack(side=tk.RIGHT)

        # Fixed summary band (non-scrollable)
        self.summary_band = ttk.Frame(self.frame)
        self.summary_band.pack(fill=tk.X, padx=4, pady=(0, 8))

        # A separator to visually distinguish the fixed band from the scrollable details
        ttk.Separator(self.frame, orient="horizontal").pack(fill=tk.X, pady=(0, 6))

        # Main 2x layout area
        self.main = ttk.Frame(self.frame)
        self.main.pack(fill=tk.BOTH, expand=True)

        self.main.columnconfigure(0, weight=3)   # Info (scrollable)
        self.main.columnconfigure(1, weight=2)   # Plots
        self.main.rowconfigure(0, weight=1)
        self.main.rowconfigure(1, weight=1)

        # Left (scrollable details)
        self.info_frame = ttk.Frame(self.main)
        self.info_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))

        # Right top: sensor layout
        self.layout_frame = ttk.Frame(self.main)
        self.layout_frame.grid(row=0, column=1, sticky="nsew", pady=(0, 10))

        # Right bottom: tabs for paradigm and drop log
        self.paradigm_frame = ttk.Frame(self.main)
        self.paradigm_frame.grid(row=1, column=1, sticky="nsew")

        self.plot_tabs = ttk.Notebook(self.paradigm_frame)
        self.paradigm_tab = ttk.Frame(self.plot_tabs)
        self.drop_tab = ttk.Frame(self.plot_tabs)
        self.plot_tabs.add(self.paradigm_tab, text="Paradigm & Timeline")
        self.plot_tabs.add(self.drop_tab, text="Drop Log")
        self.plot_tabs.pack(fill=tk.BOTH, expand=True)

        # Styling tweaks
        style = ttk.Style(self.root)
        style.configure("KeyValue.TLabel", font=("Arial", 10))
        style.configure("KeyValueBold.TLabel", font=("Arial", 10, "bold"))
        style.configure("Pill.TLabel", padding=(8, 2), relief="solid")

    # ---------- Data population ----------
    def _populate(self):
        self._populate_summary_band()  # fixed, non-scrollable
        self._populate_details_scrollable()
        self._populate_sensor_layout()
        self._populate_paradigm()
        self._populate_drop_log()

    # ===== Dataset summary (non-scrollable band) =====
    def _populate_summary_band(self):
        for w in self.summary_band.winfo_children():
            w.destroy()

        # 2-column grid of key figures
        grid = ttk.Frame(self.summary_band)
        grid.pack(fill=tk.X, padx=4)

        def kv(row, col, k, v):
            key = ttk.Label(grid, text=k, style="KeyValueBold.TLabel")
            val = ttk.Label(grid, text=v, style="KeyValue.TLabel")
            key.grid(row=row, column=col*2, sticky="w", padx=(0, 6), pady=2)
            val.grid(row=row, column=col*2+1, sticky="w", padx=(0, 16), pady=2)

        total_participants = getattr(self.class_instance, 'number_of_participants', len(self.all_epochs))
        excluded = total_participants - len(self.all_epochs)
        freq_txt = f"{self.freq:.2f} Hz"

        # Epoch counts (replicate logic from details so we keep info intact)
        try:
            events = self.class_instance.all_raw_epochs[0].events
            indices = []
            for j in range(len(self.class_instance.data_types)):
                indices.extend(np.where((events[:, 2] == self.class_instance.all_raw_epochs[0].event_id[self.class_instance.data_types[j]]))[0])
            total_epochs = len(indices) * total_participants if indices else 0
        except Exception:
            total_epochs = 0

        try:
            indices = []
            for i in range(len(self.class_instance.all_epochs)):
                events = self.class_instance.all_epochs[i].events
                for j in range(len(self.class_instance.data_types)):
                    indices.extend(np.where((events[:, 2] == self.class_instance.all_epochs[0].event_id[self.class_instance.data_types[j]]))[0])
            remaining_epochs = len(indices)
        except Exception:
            remaining_epochs = 0

        excluded_epochs = total_epochs - remaining_epochs if total_epochs is not None else 0

        # First row
        kv(0, 0, "Total Participants:", f"{total_participants}")
        kv(0, 1, "Excluded Participants:", f"{excluded}")
        kv(0, 2, "Sampling Frequency:", freq_txt)

        # Second row
        kv(1, 0, "Total Epochs (expected):", f"{total_epochs}")
        kv(1, 1, "Remaining Epochs:", f"{remaining_epochs}")
        kv(1, 2, "Excluded Epochs:", f"{excluded_epochs}")

        # Units pill (to make it explicit everywhere)
        pill = ttk.Label(self.summary_band, text="All concentration values are in µM (10⁻⁶)", foreground="gray")
        pill.pack(anchor="w", padx=4, pady=(2, 0))

    # ===== Scrollable details (everything else lives here) =====
    def _populate_details_scrollable(self):
        canvas = tk.Canvas(self.info_frame, highlightthickness=0)
        vbar = ttk.Scrollbar(self.info_frame, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)

        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vbar.set)

        # Conditions
        ttk.Label(inner, text="Conditions:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(6, 2))
        for dt in self.data_types:
            ttk.Label(inner, text=f"  • {dt}").pack(anchor="w", pady=1)

        # Channel information + bad channels
        if self.all_epochs:
            first_epoch = self.all_epochs[0]
            ttk.Label(inner, text="Channel Information:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 2))
            ttk.Label(inner, text=f"  • Total Channels: {len(first_epoch.ch_names)}").pack(anchor="w", pady=1)

            bad_channel_analysis = self._analyze_bad_channels()
            if bad_channel_analysis['total_unique_bads'] > 0:
                ttk.Label(inner, text=f"    - Total unique bad channels: {bad_channel_analysis['total_unique_bads']}").pack(anchor="w", pady=1)
                ttk.Label(inner, text=f"    - Participants with bad channels: {bad_channel_analysis['participants_with_bads']}/{len(self.all_epochs)}").pack(anchor="w", pady=1)
                ttk.Label(inner, text=f"    - Average bad channels per participant: {bad_channel_analysis['avg_bads_per_participant']:.1f}").pack(anchor="w", pady=1)
                if bad_channel_analysis['most_common_bads']:
                    ttk.Label(inner, text=f"    - Most frequently bad:").pack(anchor="w", pady=1)
                    for ch, count in bad_channel_analysis['most_common_bads'][:3]:
                        percentage = (count / len(self.all_epochs)) * 100
                        ttk.Label(inner, text=f"      {ch} ({count} participants, {percentage:.1f}%)").pack(anchor="w", pady=1)
                self._add_expandable_bad_channels_section(inner, bad_channel_analysis)
            else:
                ttk.Label(inner, text=f"  • No bad channels marked across dataset").pack(anchor="w", pady=1)

        # Channel Explorer (uses compute_effect_size outputs)
        _ = self._add_channel_explorer_section(inner, inline=True)

        canvas.pack(side="left", fill="both", expand=True)
        vbar.pack(side="right", fill="y")

    def _analyze_bad_channels(self):
        """Analyze bad channels across all participants"""
        all_bads = []
        participant_bads = []
        bad_channel_participants = defaultdict(list)

        for i, epochs in enumerate(self.all_epochs):
            if hasattr(epochs, 'info') and 'bads' in epochs.info:
                bads = epochs.info['bads']
                participant_bads.append(bads)
                all_bads.extend(bads)
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

    # ===== New helper: get effect-size outputs once and cache =====
    def _get_effect_results(self):
        if self._effect_cache is not None:
            return self._effect_cache
        try:
            raw_vals, pre_vals = compute_effect_size(self.class_instance)
        except Exception as e:
            raw_vals, pre_vals = ({}, {}), ({}, {})
            print(f"compute_effect_size error: {e}")
        self._effect_cache = {"raw": raw_vals, "pre": pre_vals}
        return self._effect_cache

    # ===== Channel Explorer: compact columns & styles, using compute_effect_size outputs =====
    def _add_channel_explorer_section(self, parent, inline=False):
        """
        Shows:
        1) Summary metrics per channel (Effect size, Mean diff, s_within, df_within, P)
            as a single Value column formatted 'Pre / Raw'.
        2) Person-weighted per-condition means formatted 'Pre / Raw'.
        3) Individual participant selection and comparison with grand means.
        4) ENHANCED: Detailed individual metrics showing the components that make up effect size
        Uses compact Treeview/LabelFrame styles to minimize whitespace.

        (Refined) Values for differences and means are shown with a µM suffix.
        """
        if not getattr(self, "_effect_cache", None):
            self._get_effect_results()

        # --- Compact styles ---
        style = ttk.Style(self.root)
        style.configure("Compact.Treeview", rowheight=18, font=("Arial", 9))
        style.configure("Compact.Treeview.Heading", padding=(4, 1), font=("Arial", 9, "bold"))
        style.configure("Compact.TLabelframe", padding=(6, 2, 6, 2))
        style.configure("Compact.TLabelframe.Label", padding=(4, 0, 4, 0))

        expand = ttk.Frame(parent)
        expand.pack(anchor="w", fill="x", pady=(6, 0))

        body = ttk.Frame(expand)
        body.pack(anchor="w", fill="x", expand=False)

        # Controls
        controls = ttk.Frame(body)
        controls.pack(anchor="w", fill="x", pady=(4, 2))

        ttk.Label(controls, text="Channel:").pack(side=tk.LEFT)

        # Try to get channel names from loaded epochs
        ch_names = []
        try:
            for epochs in self.all_epochs:
                if hasattr(epochs, 'ch_names') and epochs.ch_names:
                    ch_names = epochs.ch_names
                    break
        except Exception:
            pass
        
        sel = tk.StringVar(value=(ch_names[0] if ch_names else ""))
        combo = ttk.Combobox(controls, textvariable=sel, values=ch_names, state="readonly", width=24)
        combo.pack(side=tk.LEFT, padx=(6, 8))

        ttk.Label(controls, text="Values: Preprocessed / Raw", foreground="gray").pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(controls, text="(units: µM except d, df, P)", foreground="gray").pack(side=tk.LEFT, padx=(8, 0))

        # --- Summary metrics table (Metric | Value (Pre / Raw)) ---
        summary_frame = ttk.LabelFrame(body, text="Grand mean metrics (per channel)", style="Compact.TLabelframe")
        summary_frame.pack(anchor="w", fill="x", padx=0, pady=(4, 4))

        sum_cols = ["Metric", "Value (Pre / Raw)"]
        summary_tree = ttk.Treeview(
            summary_frame,
            columns=sum_cols,
            show="headings",
            height=6,
            style="Compact.Treeview"
        )
        # Keep total width tight; no stretching
        summary_tree.column("Metric", anchor="w", width=220, minwidth=160, stretch=False)
        summary_tree.column("Value (Pre / Raw)", anchor="w", width=260, minwidth=200, stretch=False)
        summary_tree.heading("Metric", text="Metric")
        summary_tree.heading("Value (Pre / Raw)", text="Value (Pre / Raw)")
        summary_tree.pack(anchor="w", padx=2, pady=2)

        # --- Per-condition means table (Condition | Mean (Pre / Raw)) ---
        cond_frame = ttk.LabelFrame(body, text="Person-weighted per-condition means (per channel)", style="Compact.TLabelframe")
        cond_frame.pack(anchor="w", fill="x", padx=0, pady=(0, 4))

        cond_cols = ["Condition", "Mean (Pre / Raw)"]
        cond_tree = ttk.Treeview(
            cond_frame,
            columns=cond_cols,
            show="headings",
            height=6,
            style="Compact.Treeview"
        )
        cond_tree.column("Condition", anchor="w", width=220, minwidth=160, stretch=False)
        cond_tree.column("Mean (Pre / Raw)", anchor="w", width=260, minwidth=200, stretch=False)
        cond_tree.heading("Condition", text="Condition")
        cond_tree.heading("Mean (Pre / Raw)", text="Mean (Pre / Raw)")
        cond_tree.pack(anchor="w", padx=2, pady=2)

        # --- ENHANCED: Individual participant section ---
        # Participant selection controls
        participant_controls = ttk.Frame(body)
        participant_controls.pack(anchor="w", fill="x", pady=(8, 2))

        ttk.Label(participant_controls, text="Individual:").pack(side=tk.LEFT)

        # Get participant names/IDs from effect size data
        participant_names = []
        try:
            pre = self._effect_cache.get("pre", {})
            if "Effect size" in pre:
                participant_names = list(pre["Effect size"].keys())
        except Exception:
            pass

        participant_sel = tk.StringVar(value=(participant_names[0] if participant_names else ""))
        participant_combo = ttk.Combobox(
            participant_controls, 
            textvariable=participant_sel, 
            values=participant_names, 
            state="readonly", 
            width=24
        )
        participant_combo.pack(side=tk.LEFT, padx=(6, 8))

        ttk.Label(participant_controls, text="Detailed effect size components", foreground="gray").pack(side=tk.LEFT, padx=(4, 0))

        # --- ENHANCED: Detailed individual participant metrics table ---
        individual_frame = ttk.LabelFrame(body, text="Individual participant effect size breakdown (per channel)", style="Compact.TLabelframe")
        individual_frame.pack(anchor="w", fill="x", padx=0, pady=(4, 4))

        # Updated columns to show more detailed breakdown
        ind_cols = ["Metric", "Individual (Pre / Raw)", "Grand Mean (Pre / Raw)", "Difference", "Notes"]
        individual_tree = ttk.Treeview(
            individual_frame,
            columns=ind_cols,
            show="headings",
            height=8,  # Increased height for more rows
            style="Compact.Treeview"
        )
        individual_tree.column("Metric", anchor="w", width=200, minwidth=150, stretch=False)
        individual_tree.column("Individual (Pre / Raw)", anchor="w", width=140, minwidth=110, stretch=False)
        individual_tree.column("Grand Mean (Pre / Raw)", anchor="w", width=140, minwidth=110, stretch=False)
        individual_tree.column("Difference", anchor="w", width=120, minwidth=90, stretch=False)
        individual_tree.column("Notes", anchor="w", width=150, minwidth=100, stretch=False)
        individual_tree.heading("Metric", text="Metric")
        individual_tree.heading("Individual (Pre / Raw)", text="Individual (Pre / Raw)")
        individual_tree.heading("Grand Mean (Pre / Raw)", text="Grand Mean (Pre / Raw)")
        individual_tree.heading("Difference", text="Difference")
        individual_tree.heading("Notes", text="Notes")
        individual_tree.pack(anchor="w", padx=2, pady=2)

        def _fmt_float(x):
            # Consistent, compact formatting for floats
            try:
                xf = float(x)
            except Exception:
                return str(x)
            ax = abs(xf)
            if ax >= 100:
                return f"{xf:,.1f}"
            if ax >= 1:
                return f"{xf:.3f}"
            if ax >= 1e-3:
                return f"{xf:.6f}"
            return f"{xf:.2e}"

        def _safe_isnan(x):
            """Safely check if value is NaN, handling various data types"""
            try:
                if x is None:
                    return True
                if isinstance(x, str):
                    return x.lower() in ['nan', 'none', '']
                if isinstance(x, (list, tuple, dict)):
                    return True
                return np.isnan(float(x))
            except (TypeError, ValueError, AttributeError):
                return True

        def _fmt_muM(x):
            if _safe_isnan(x):
                return "N/A"
            return f"{_fmt_float(x)} µM"

        def _fmt_plain(x):
            # for integers / df / P
            try:
                if _safe_isnan(x):
                    return "N/A"
                if isinstance(x, (int, np.integer)):
                    return str(int(x))
                # if it looks like an integer float (e.g., 2.0), show as int
                xf = float(x)
                if float(int(round(xf))) == xf:
                    return str(int(round(xf)))
                return _fmt_float(x)
            except Exception:
                return "N/A"

        def _get_from(cache, section, *keys, default=None):
            try:
                d = cache.get(section, {})
                for k in keys:
                    d = d[k]
                return d
            except Exception:
                return default

        def _calculate_difference(individual_val, grand_val):
            """Calculate difference between individual and grand mean values"""
            try:
                if _safe_isnan(individual_val) or _safe_isnan(grand_val):
                    return "N/A"
                ind_val = float(individual_val)
                grand_val_f = float(grand_val)
                return ind_val - grand_val_f
            except Exception:
                return "N/A"

        def refresh_tables():
            # clear old rows
            for row in summary_tree.get_children():
                summary_tree.delete(row)
            for row in cond_tree.get_children():
                cond_tree.delete(row)
            for row in individual_tree.get_children():
                individual_tree.delete(row)

            ch = sel.get()
            participant = participant_sel.get()
            if not ch:
                return

            pre = self._effect_cache.get("pre", {})
            raw = self._effect_cache.get("raw", {})

            # Grand mean values with/without units (corrected effect size calculation)
            grand_effect_pre = _get_from(pre, "Effect size", "grand_mean_participants", ch, default="N/A") if "grand_mean_participants" in pre.get("Effect size", {}) else _get_from(pre, "Channels' mean difference", ch, default="N/A")
            grand_effect_raw = _get_from(raw, "Effect size", "grand_mean_participants", ch, default="N/A") if "grand_mean_participants" in raw.get("Effect size", {}) else _get_from(raw, "Channels' mean difference", ch, default="N/A")
            
            metrics = [
                ("Effect size (d_within)",
                _fmt_plain(grand_effect_pre),
                _fmt_plain(grand_effect_raw)),
                ("Mean difference (\u0305D)",
                _fmt_muM(_get_from(pre, "Channels' mean difference", ch, default="N/A")),
                _fmt_muM(_get_from(raw, "Channels' mean difference", ch, default="N/A"))),
                ("Within-participant SD (s_within)",
                _fmt_muM(_get_from(pre, "Channels' within-participant SD", ch, default="N/A")),
                _fmt_muM(_get_from(raw, "Channels' within-participant SD", ch, default="N/A"))),
                ("df_within",
                _fmt_plain(_get_from(pre, "DF within", ch, default=0)),
                _fmt_plain(_get_from(raw, "DF within", ch, default=0))),
                ("Participants contributing (P)",
                _fmt_plain(_get_from(pre, "P Ch.", ch, default=0)),
                _fmt_plain(_get_from(raw, "P Ch.", ch, default=0))),
            ]
            for name, pre_v, raw_v in metrics:
                summary_tree.insert("", "end", values=[name, f"{pre_v} / {raw_v}"])

            # Per-condition "pre / raw" — with µM suffix
            conds = list(self.data_types) if self.data_types else []
            for cond in conds:
                pre_mean = _fmt_muM(_get_from(pre, "Conditions' mean", cond, ch, default="N/A"))
                raw_mean = _fmt_muM(_get_from(raw, "Conditions' mean", cond, ch, default="N/A"))
                cond_tree.insert("", "end", values=[cond, f"{pre_mean} / {raw_mean}"])

            # ENHANCED: Individual participant detailed breakdown
            if participant and ch:
                # Get individual effect size components
                ind_effect_pre = _get_from(pre, "Effect size", participant, ch, default="N/A")
                ind_effect_raw = _get_from(raw, "Effect size", participant, ch, default="N/A")
                
                # Get individual standard deviation (s_within)
                ind_sd_pre = _get_from(pre, "Channels' within-participant SD", participant, ch, default="N/A")
                ind_sd_raw = _get_from(raw, "Channels' within-participant SD", participant, ch, default="N/A")
                
                # Get individual mean difference (averages_over_sessions)
                ind_mean_diff_pre = _get_from(pre, "Channels' mean difference", participant, ch, default="N/A")
                ind_mean_diff_raw = _get_from(raw, "Channels' mean difference", participant, ch, default="N/A")
                
                # Get individual degrees of freedom
                ind_df_pre = _get_from(pre, "DF within", participant, ch, default="N/A")
                ind_df_raw = _get_from(raw, "DF within", participant, ch, default="N/A")

                # Get grand mean values for comparison
                grand_mean_diff_pre = _get_from(pre, "Channels' mean difference", ch, default="N/A")
                grand_mean_diff_raw = _get_from(raw, "Channels' mean difference", ch, default="N/A")
                grand_sd_pre = _get_from(pre, "Channels' within-participant SD", ch, default="N/A")
                grand_sd_raw = _get_from(raw, "Channels' within-participant SD", ch, default="N/A")

                # Calculate effect size manually to verify: d = mean_diff / sd
                calculated_effect_pre = "N/A"
                calculated_effect_raw = "N/A"
                try:
                    if not _safe_isnan(ind_mean_diff_pre) and not _safe_isnan(ind_sd_pre):
                        calculated_effect_pre = float(ind_mean_diff_pre) / float(ind_sd_pre)
                except:
                    pass
                try:
                    if not _safe_isnan(ind_mean_diff_raw) and not _safe_isnan(ind_sd_raw):
                        calculated_effect_raw = float(ind_mean_diff_raw) / float(ind_sd_raw)
                except:
                    pass

                # Enhanced individual metrics with more detail
                individual_metrics = [
                    ("Effect size (d_within)", ind_effect_pre, ind_effect_raw, grand_effect_pre, grand_effect_raw, _fmt_plain, True, "d = \u0305D / s_within"),
                    ("Mean difference (\u0305D)", ind_mean_diff_pre, ind_mean_diff_raw, grand_mean_diff_pre, grand_mean_diff_raw, _fmt_muM, True, "Session average"),
                    ("Within-participant SD (s_within)", ind_sd_pre, ind_sd_raw, grand_sd_pre, grand_sd_raw, _fmt_muM, True, "Variability measure"),
                    ("Degrees of freedom (df_within)", ind_df_pre, ind_df_raw, ind_df_pre, ind_df_raw, _fmt_plain, False, "Sessions - 1"),
                    ("Calculated d (verification)", calculated_effect_pre, calculated_effect_raw, "—", "—", _fmt_plain, False, "\u0305D / s_within"),
                ]

                for name, ind_pre, ind_raw, grand_pre, grand_raw, formatter, show_diff, notes in individual_metrics:
                    ind_pre_str = formatter(ind_pre)
                    ind_raw_str = formatter(ind_raw)
                    grand_pre_str = formatter(grand_pre) if grand_pre != "—" else "—"
                    grand_raw_str = formatter(grand_raw) if grand_raw != "—" else "—"
                    
                    if show_diff and grand_pre != "—" and grand_raw != "—":
                        diff_pre = _calculate_difference(ind_pre, grand_pre)
                        diff_raw = _calculate_difference(ind_raw, grand_raw)
                        if diff_pre == "N/A" or diff_raw == "N/A":
                            diff_str = "N/A"
                        else:
                            diff_pre_str = formatter(diff_pre) if hasattr(formatter, '__call__') else _fmt_float(diff_pre)
                            diff_raw_str = formatter(diff_raw) if hasattr(formatter, '__call__') else _fmt_float(diff_raw)
                            diff_str = f"{diff_pre_str} / {diff_raw_str}"
                    else:
                        diff_str = "—"
                    
                    individual_tree.insert("", "end", values=[
                        name, 
                        f"{ind_pre_str} / {ind_raw_str}",
                        f"{grand_pre_str} / {grand_raw_str}",
                        diff_str,
                        notes
                    ])

                # Add separator and session-level details if available
                try:
                    session_data_pre = _get_from(pre, "Channels' mean difference", participant, default={})
                    session_data_raw = _get_from(raw, "Channels' mean difference", participant, default={})
                    
                    if session_data_pre and ch in session_data_pre.get("session_differences", [{}]):
                        individual_tree.insert("", "end", values=["", "", "", "", ""])  # separator
                        individual_tree.insert("", "end", values=["Session Details:", "", "", "", ""])
                        
                        # Try to get session-level data
                        try:
                            sessions_pre = session_data_pre.get("session_differences", [])
                            sessions_raw = session_data_raw.get("session_differences", [])
                            
                            for i, (sess_pre, sess_raw) in enumerate(zip(sessions_pre, sessions_raw)):
                                if ch in sess_pre and ch in sess_raw:
                                    sess_diff_pre = _fmt_muM(sess_pre[ch])
                                    sess_diff_raw = _fmt_muM(sess_raw[ch])
                                    individual_tree.insert("", "end", values=[
                                        f"  Session {i+1} diff", 
                                        f"{sess_diff_pre} / {sess_diff_raw}",
                                        "—", "—", "Raw session data"
                                    ])
                        except Exception:
                            pass
                except Exception:
                    pass

        refresh_tables()
        combo.bind("<<ComboboxSelected>>", lambda _e=None: refresh_tables())
        participant_combo.bind("<<ComboboxSelected>>", lambda _e=None: refresh_tables())
        return True

    def _add_expandable_bad_channels_section(self, parent, analysis):
        """Add an expandable section for detailed bad channel information"""
        expand_frame = ttk.Frame(parent)
        expand_frame.pack(anchor="w", fill="x", pady=(5, 0))

        self.bad_channels_expanded = tk.BooleanVar(value=False)

        def toggle_bad_channels():
            if self.bad_channels_expanded.get():
                details_frame.pack(anchor="w", fill="x", padx=(20, 0))
                self._show_detailed_bad_channels(details_frame, analysis)
            else:
                for widget in details_frame.winfo_children():
                    widget.destroy()
                details_frame.pack_forget()
            try:
                canvas = parent.master
                if hasattr(canvas, 'bbox'):
                    canvas.update_idletasks()
                    canvas.configure(scrollregion=canvas.bbox("all"))
            except Exception:
                pass

        toggle_btn = ttk.Checkbutton(
            expand_frame,
            text="Show detailed bad channels breakdown",
            variable=self.bad_channels_expanded,
            command=toggle_bad_channels
        )
        toggle_btn.pack(anchor="w")

        details_frame = ttk.Frame(expand_frame)

    def _show_detailed_bad_channels(self, parent, analysis):
        """Show detailed bad channel breakdown"""
        table_frame = ttk.Frame(parent)
        table_frame.pack(anchor="w", fill="x", pady=5)

        ttk.Label(table_frame, text="Channel", font=("Arial", 9, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 10))
        ttk.Label(table_frame, text="Count", font=("Arial", 9, "bold")).grid(row=0, column=1, sticky="w", padx=(0, 10))
        ttk.Label(table_frame, text="Percentage", font=("Arial", 9, "bold")).grid(row=0,  column=2, sticky="w", padx=(0, 10))
        ttk.Label(table_frame, text="Participants", font=("Arial", 9, "bold")).grid(row=0, column=3, sticky="w")

        ttk.Separator(table_frame, orient='horizontal').grid(row=1, column=0, columnspan=4, sticky="ew", pady=2)

        for i, (channel, count) in enumerate(analysis['most_common_bads'][:10]):
            row = i + 2
            percentage = (count / len(self.all_epochs)) * 100
            participant_list = analysis['bad_channel_participants'][channel]
            participant_str = ", ".join([f"P{p+1}" for p in participant_list[:5]])
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

        summary_frame = ttk.Frame(self.bad_channels_tab)
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        title_label = ttk.Label(summary_frame,
                                text="Bad Channels Analysis",
                                font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))

        stats_text = f"""Dataset Bad Channels Summary:

    • Total unique bad channels: {analysis['total_unique_bads']}
    • Participants with bad channels: {analysis['participants_with_bads']}/{len(self.all_epochs)} ({analysis['participants_with_bads']/len(self.all_epochs)*100:.1f}%)
    • Average bad channels per participant: {analysis['avg_bads_per_participant']:.1f}

    Most problematic channels:"""

        stats_label = ttk.Label(summary_frame, text=stats_text, font=("Arial", 10))
        stats_label.pack(anchor="w", pady=(0, 10))

        for i, (ch, count) in enumerate(analysis['most_common_bads'][:10]):
            percentage = (count / len(self.all_epochs)) * 100
            channel_text = f"  {i+1}. {ch}: {count} participants ({percentage:.1f}%)"
            ttk.Label(summary_frame, text=channel_text, font=("Arial", 9)).pack(anchor="w")

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

            hint = ttk.Label(self.layout_frame, text="Click the plot to zoom • Esc or click outside to close", foreground="gray")
            hint.pack(pady=(4, 0))
            w.bind("<Button-1>", lambda _e: self._open_zoom_overlay())
        except Exception as e:
            ttk.Label(self.layout_frame, text=f"Sensor setup error: {e}").pack(expand=True)

    def _populate_paradigm(self):
        """
        Use the external plot_experiment_timeline(annotations, figure_size) to build the timeline figure,
        then embed it. If the experiment spans > 20 minutes, enable horizontal scrolling.
        """
        try:
            if not self.all_epochs:
                ttk.Label(self.paradigm_tab, text="No epoch data available").pack(expand=True)
                return

            annotations = self.class_instance.Individual_participants[0].raw_haemo.annotations
            fig, timeline_ax = plot_experiment_timeline(annotations, figsize=(8, 6))

            # Compute total duration (minutes) from annotations for deciding scroll behavior
            total_duration_min = None
            try:
                onsets = np.asarray(annotations.onset, dtype=float)
                durations = np.asarray(annotations.duration, dtype=float)
                if onsets.size and durations.size:
                    total_duration_min = float(np.max(onsets + durations)) / 60.0
            except Exception:
                total_duration_min = None

            # Embed with optional horizontal scroll
            if total_duration_min is not None and total_duration_min > 20.0:
                timeline_frame = ScrollableTimelineFrame(self.paradigm_tab, fig, timeline_ax, total_duration_min)
                timeline_frame.pack(fill=tk.BOTH, expand=True)

                ttk.Label(
                    self.paradigm_tab,
                    text="Use the scrollbar or mouse wheel to navigate the timeline.",
                    font=("Arial", 9), foreground="gray"
                ).pack(pady=(5, 0))
            else:
                canvas = FigureCanvasTkAgg(fig, self.paradigm_tab)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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

        incomplete_drop_log = self.class_instance.drop_log if hasattr(self.class_instance, 'drop_log') else []
        total_drop_log = ()
        events = self.class_instance.all_raw_epochs[0].events
        indices = []
        for j in range(len(self.class_instance.data_types)):
            indices.extend(np.where((events[:, 2] == self.class_instance.all_raw_epochs[0].event_id[self.class_instance.data_types[j]]))[0])
        for i in range(len(incomplete_drop_log)):
            for ind in indices:
                total_drop_log += (incomplete_drop_log[i][ind],)

        if len(total_drop_log) == 0:
            ttk.Label(container, text="No dropped epochs to display").pack(expand=True)
            return

        try:
            try:
                plot_ret = mne.viz.plot_drop_log(total_drop_log, show=False)
            except TypeError:
                plot_ret = mne.viz.plot_drop_log(total_drop_log)

            fig = plot_ret[0] if isinstance(plot_ret, (list, tuple)) and hasattr(plot_ret[0], "savefig") else plot_ret
            if not hasattr(fig, "savefig"):
                raise RuntimeError("mne.viz.plot_drop_log did not return a Figure as expected")

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

            btn_pdf = ttk.Button(toolbar, text="Save as PDF", command=lambda: _save_as("pdf"))
            btn_pdf.pack(side=tk.LEFT, padx=4, pady=4)
            btn_png = ttk.Button(toolbar, text="Save as PNG", command=lambda: _save_as("png"))
            btn_png.pack(side=tk.LEFT, padx=4, pady=4)

            canvas = FigureCanvasTkAgg(fig, master=container)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            reason_counts = Counter()
            for entry in total_drop_log:
                if isinstance(entry, (list, tuple)):
                    for r in entry:
                        if r:
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
