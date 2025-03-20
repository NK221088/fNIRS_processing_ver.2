import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from load_data_function import load_data
from epoch_plot import epoch_plot
from standard_fNIRS_response_plot import standard_fNIRS_response_plot
from paradigm_plot import paradigm_plot
import mne

# Default settings
settings = {
    "data_set": "fNirs_motor_full_data",
    "epoch_type": "Tapping",
    "combine_strategy": "mean",
    "short_channel_correction": True,
    "negative_correlation_enhancement": True,
    "interpolate_bad_channels": False,
    "bad_channels_strategy": "all",
    "threshold": 3,
    "plot_type": "Epoch Plot",
    "individual": True  
}

# Function to update epoch types when dataset changes
def update_epoch_types(*args):
    """Load data and update epoch type dropdown based on dataset selection."""
    dataset = dataset_var.get()

    # Load data to extract available epoch types
    all_epochs, data_name, all_data, freq, data_types, all_individuals = load_data(
        data_set=dataset,
        short_channel_correction=settings["short_channel_correction"],
        negative_correlation_enhancement=settings["negative_correlation_enhancement"],
        interpolate_bad_channels=settings["interpolate_bad_channels"],
        individuals=settings["individual"]
    )
    
    # Update epoch type menu options
    epoch_type_menu["values"] = data_types
    epoch_type_var.set(data_types[0] if data_types else "N/A")  # Select first available type
    
    # Update individual selection dropdown
    individuals_menu["values"] = [individual.name for individual in all_individuals]
    if all_individuals:
        Individual_var.set(all_individuals[0].name)  # Select first individual by default

# Function to show/hide individual selection based on plot type
def toggle_individual_menu(*args):
    """Show or hide the individual selection dropdown based on plot type."""
    if plot_type_var.get() == "paradigm_plot":
        individual_label.pack(anchor="w")
        individuals_menu.pack(pady=5)
    else:
        individual_label.pack_forget()
        individuals_menu.pack_forget()

# Function to run the selected analysis
def run_analysis():
    """Run data processing and visualization based on selected plot type."""
    settings["data_set"] = dataset_var.get()
    settings["epoch_type"] = epoch_type_var.get()
    settings["combine_strategy"] = combine_strategy_var.get()
    settings["short_channel_correction"] = short_channel_correction_var.get()
    settings["negative_correlation_enhancement"] = negative_correlation_enhancement_var.get()
    settings["interpolate_bad_channels"] = interpolate_bad_channels_var.get()
    settings["bad_channels_strategy"] = bad_channels_strategy_var.get()
    settings["threshold"] = int(threshold_var.get())
    settings["plot_type"] = plot_type_var.get()

    # Load data
    all_epochs, data_name, all_data, freq, data_types, all_individuals = load_data(
        data_set=settings["data_set"],
        short_channel_correction=settings["short_channel_correction"],
        negative_correlation_enhancement=settings["negative_correlation_enhancement"],
        interpolate_bad_channels=settings["interpolate_bad_channels"],
        individuals=settings["individual"]
    )

    # Clear previous plots
    for widget in right_frame.winfo_children():
        widget.destroy()

    # Run the selected plot function
    if settings["plot_type"] == "Epoch Plot":
        figures = epoch_plot(all_epochs, epoch_type=settings["epoch_type"], combine_strategy=settings["combine_strategy"],
                             save=False, bad_channels_strategy=settings["bad_channels_strategy"],
                             threshold=settings["threshold"], data_set=data_name)
    elif settings["plot_type"] == "Standard fNIRS Response Plot":
        figures = [standard_fNIRS_response_plot(all_epochs, data_types, bad_channels_strategy=settings["bad_channels_strategy"],
                                                save=False, combine_strategy=settings["combine_strategy"],
                                                threshold=settings["threshold"], data_set=data_name)]
    elif settings["plot_type"] == "paradigm_plot":
        selected_individual = Individual_var.get()
        figures = [paradigm_plot(all_individuals[int(selected_individual.strip("Participant_"))-1])]
    else:
        figures = []

    # Ensure figures is always a list
    if figures:
        if not isinstance(figures, list):  
            figures = [figures]  
        else:
            flattened_figures = []
            for fig in figures:
                if isinstance(fig, list):
                    flattened_figures.extend(fig)  # Unpack nested lists
                else:
                    flattened_figures.append(fig)
            figures = flattened_figures

        # Display each figure in the right_frame
        for fig in figures:
            canvas = FigureCanvasTkAgg(fig, master=right_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, pady=5)

# Create GUI window
root = tk.Tk()
root.title("fNIRS Data Analysis")
root.geometry("800x600")

# Left panel for settings
left_frame = tk.Frame(root)
left_frame.pack(side="left", padx=20, pady=20, fill="y")

# Dataset selection
tk.Label(left_frame, text="Select Dataset:", font=("Arial", 12)).pack(anchor="w")
dataset_var = tk.StringVar(value=settings["data_set"])
dataset_menu = ttk.Combobox(left_frame, textvariable=dataset_var, values=[
    "fNIrs_motor", "AudioSpeechNoise", "fNirs_motor_full_data", "fNIRS_Alexandros_DoC_data",
    "fNIRS_Alexandros_Healthy_data", "fNIRS_CUH_patient_data", "fNIRS_Melika_data"
])
dataset_menu.pack(pady=5)
dataset_var.trace_add("write", update_epoch_types)

# Epoch type selection
tk.Label(left_frame, text="Epoch Type:", font=("Arial", 12)).pack(anchor="w")
epoch_type_var = tk.StringVar()
epoch_type_menu = ttk.Combobox(left_frame, textvariable=epoch_type_var)
epoch_type_menu.pack(pady=5)

# Combine strategy selection
tk.Label(left_frame, text="Combine Strategy:", font=("Arial", 12)).pack(anchor="w")
combine_strategy_var = tk.StringVar(value=settings["combine_strategy"])
combine_strategy_menu = ttk.Combobox(left_frame, textvariable=combine_strategy_var, values=["mean", "median", "std", "gfp"])
combine_strategy_menu.pack(pady=5)

# Bad Channels Strategy
tk.Label(left_frame, text="Bad Channels Strategy:", font=("Arial", 12)).pack(anchor="w")
bad_channels_strategy_var = tk.StringVar(value=settings["bad_channels_strategy"])
bad_channels_strategy_menu = ttk.Combobox(left_frame, textvariable=bad_channels_strategy_var, values=["all", "delete", "threshold"])
bad_channels_strategy_menu.pack(pady=5)

# Short channel correction
short_channel_correction_var = tk.BooleanVar(value=settings["short_channel_correction"])
tk.Checkbutton(left_frame, text="Short Channel Correction", variable=short_channel_correction_var).pack(anchor="w")

# Negative correlation enhancement
negative_correlation_enhancement_var = tk.BooleanVar(value=settings["negative_correlation_enhancement"])
tk.Checkbutton(left_frame, text="Negative Correlation Enhancement", variable=negative_correlation_enhancement_var).pack(anchor="w")

# Interpolate bad channels
interpolate_bad_channels_var = tk.BooleanVar(value=settings["interpolate_bad_channels"])
tk.Checkbutton(left_frame, text="Interpolate Bad Channels", variable=interpolate_bad_channels_var).pack(anchor="w")

# Threshold selection
tk.Label(left_frame, text="Threshold:", font=("Arial", 12)).pack(anchor="w")
threshold_var = tk.StringVar(value=str(settings["threshold"]))
threshold_entry = tk.Entry(left_frame, textvariable=threshold_var)
threshold_entry.pack(pady=5)

# Plot type selection
tk.Label(left_frame, text="Select Plot Type:", font=("Arial", 12)).pack(anchor="w")
plot_type_var = tk.StringVar(value=settings["plot_type"])
plot_type_menu = ttk.Combobox(left_frame, textvariable=plot_type_var, values=["Epoch Plot", "Standard fNIRS Response Plot", "paradigm_plot"])
plot_type_menu.pack(pady=5)
plot_type_var.trace_add("write", toggle_individual_menu)

# Individual selection (Initially hidden)
individual_label = tk.Label(left_frame, text="Select Individual:", font=("Arial", 12))
Individual_var = tk.StringVar()
individuals_menu = ttk.Combobox(left_frame, textvariable=Individual_var)
individual_label.pack_forget()
individuals_menu.pack_forget()

# Run Analysis button
run_button = tk.Button(left_frame, text="Run Analysis", command=run_analysis, bg="green", fg="white")
run_button.pack(pady=10)

# Right panel for displaying the plot
right_frame = tk.Frame(root)
right_frame.pack(side="right", padx=20, pady=20, expand=True, fill="both")

# Initialize GUI
update_epoch_types()
root.mainloop()
