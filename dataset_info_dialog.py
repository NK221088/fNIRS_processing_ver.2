import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import mne
from matplotlib.figure import Figure

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
        self.dialog.geometry("1000x800")
        self.dialog.resizable(True, True)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self.create_widgets()
        self.populate_info()

    def create_widgets(self):
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        main_frame.columnconfigure(0, weight=3)  # Info
        main_frame.columnconfigure(1, weight=2)  # Plots
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Left info section
        self.info_frame = ttk.Frame(main_frame)
        self.info_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))

        # Top-right: Channel layout
        self.layout_frame = ttk.Frame(main_frame)
        self.layout_frame.grid(row=0, column=1, sticky="nsew", pady=(0, 10))

        # Bottom-right: Paradigm overview
        self.paradigm_frame = ttk.Frame(main_frame)
        self.paradigm_frame.grid(row=1, column=1, sticky="nsew")

        # Close button
        close_button = ttk.Button(self.dialog, text="Close", command=self.close_dialog)
        close_button.pack(pady=10)

    def populate_info(self):
        self.populate_general_info()
        self.populate_channel_layout()
        self.populate_paradigm_overview()

    def populate_general_info(self):
        canvas = tk.Canvas(self.info_frame)
        scrollbar = ttk.Scrollbar(self.info_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        ttk.Label(scrollable_frame, text=f"Dataset Name: {self.data_name}", font=("Arial", 12, "bold")).pack(anchor="w", pady=5)

        total_participants = len(self.all_epochs)
        ttk.Label(scrollable_frame, text=f"Total Participants: {total_participants}").pack(anchor="w", pady=2)

        if self.all_individuals:
            excluded_participants = len(self.all_individuals) - total_participants
            ttk.Label(scrollable_frame, text=f"Excluded Participants: {excluded_participants}").pack(anchor="w", pady=2)

        ttk.Label(scrollable_frame, text=f"Sampling Frequency: {self.freq:.2f} Hz").pack(anchor="w", pady=2)

        ttk.Label(scrollable_frame, text="Conditions:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 2))
        for data_type in self.data_types:
            ttk.Label(scrollable_frame, text=f"  • {data_type}").pack(anchor="w", pady=1)

        if self.all_epochs:
            first_epoch = self.all_epochs[0]
            ttk.Label(scrollable_frame, text="Channel Information:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 2))
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

        ttk.Label(scrollable_frame, text="Epoch Information:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 2))
        total_epochs = sum(len(epoch) for epoch in self.all_epochs)
        ttk.Label(scrollable_frame, text=f"  • Total Epochs: {total_epochs}").pack(anchor="w", pady=1)

        for data_type in self.data_types:
            if data_type in self.all_data:
                n_epochs = self.all_data[data_type].shape[0]
                ttk.Label(scrollable_frame, text=f"  • {data_type}: {n_epochs} epochs").pack(anchor="w", pady=1)

        ttk.Label(scrollable_frame, text="Summary Statistics:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(10, 2))
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

    def populate_channel_layout(self):
        try:
            if self.all_epochs:
                fig = Figure(figsize=(5, 4), dpi=100)
                ax = fig.add_subplot(111)
                first_epoch = self.all_epochs[0]

                if hasattr(first_epoch, 'info'):
                    info = first_epoch.info
                    raw_for_plot = mne.io.RawArray(np.zeros((len(info['ch_names']), 1000)), info)
                    raw_for_plot.plot_sensors(kind="topomap", show_names=True, axes=ax)
                    ax.set_title("Sensor Setup")

                canvas = FigureCanvasTkAgg(fig, self.layout_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            error_label = ttk.Label(self.layout_frame, text=f"Sensor setup error: {str(e)}")
            error_label.pack(expand=True)

    def populate_paradigm_overview(self):
        try:
            paradigm_plot(self.all_individuals[0], picks_ = self.all_individuals[0].epochs.info["ch_names"])
            if self.all_individuals and len(self.all_individuals) > 0:
                fig = Figure(figsize=(5, 4), dpi=100)
                ax = fig.add_subplot(111)
                individual = self.all_individuals[0]

                if hasattr(individual, 'raw_haemo'):
                    raw = individual.raw_haemo
                    events, event_dict = mne.events_from_annotations(raw)
                    times = events[:, 0] / raw.info['sfreq']
                    event_types = events[:, 2]

                    colors = plt.cm.tab10(np.linspace(0, 1, len(set(event_types))))
                    color_dict = {etype: colors[i] for i, etype in enumerate(set(event_types))}

                    for i, (time, etype) in enumerate(zip(times, event_types)):
                        ax.axvline(x=time, color=color_dict[etype], alpha=0.7, linewidth=2)
                        if i < 10:
                            ax.text(time, 0.5, f'E{etype}', rotation=90, ha='center', va='bottom')

                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Events')
                    ax.set_title('Paradigm Timeline')
                    ax.set_ylim(0, 1)
                    ax.grid(True, alpha=0.3)

                canvas = FigureCanvasTkAgg(fig, self.paradigm_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            error_label = ttk.Label(self.paradigm_frame, text=f"Paradigm error: {str(e)}")
            error_label.pack(expand=True)

    def close_dialog(self):
        self.dialog.destroy()


def show_dataset_info(parent, all_epochs, data_name, all_data, freq, data_types, all_individuals=None):
    try:
        dialog = DatasetInfoDialog(parent, all_epochs, data_name, all_data, freq, data_types, all_individuals)
        return dialog
    except Exception as e:
        messagebox.showerror("Error", f"Failed to create dataset info dialog: {str(e)}")
        return None
