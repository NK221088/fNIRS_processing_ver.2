import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from load_data_function import data_loaders
from epoch_plot import epoch_plot
from standard_fNIRS_response_plot import standard_fNIRS_response_plot
from paradigm_plot import paradigm_plot
from individual_frequency_plot import individual_frequency_plot
from statistical_analysis import statistical_analysis
from dataset_info_panel import show_dataset_info_in_container
from preprocessing_dialog import show_preprocessing_dialog
from plot_settings_dialog import show_plot_settings_dialog



dataSetList = list(data_loaders.keys())
plotTypesList = ["Epoch Plot",
                "Standard fNIRS Response Plot",
                "paradigm_plot",
                "individual frequency plot",
                "Statistical Analysis",
                ]

# Default settings (add hemoglobin type to settings)
settings = {
    "data_set": dataSetList[15],  # Default to first dataset
    "epoch_type": "HandMI",
    "individual": "All Individuals",
    "short_channel_correction": True,
    "negative_correlation_enhancement": False,
    "haemo_type": "hbo",
    "baseline_correction": "Previous rest period",
    "tmin": 0,
    "stimulus_duration": 5,
    "scalp_coupling_threshold": 0.8,
    "reject_criteria": dict(hbo=80e-6),
    "unwanted": ["15.0"],
    "filter_lower_value": 0.05,
    "filter_upper_value": 0.7,
    "h_trans_bandwidth": 0.2,           
    "l_trans_bandwidth": 0.02,
    "snr_rejection": "None",  # Default to None, can be set to "SNR" or "CV"
    "snr_threshold": 8,  # Default threshold for SNR
    "Apply_TDDR": False,  # Default to False, can be set to True for TDDR

    # Plot settings
    "save_plot": False,
    "plot_type": "Epoch Plot",
    "combine_strategy": "mean",
    "interpolate_bad_channels": False,
    "bad_channels_strategy": "all",
    "threshold": 3,
    "compare_with_raw": False, 
}

current_loader = None
first_data_load = True
all_individuals = []
start_up = True
# Track previous selections
previous_dataset = settings["data_set"]
previous_epoch_type = settings["epoch_type"]
previous_individual = settings["individual"]
previous_combine_strategy = settings["combine_strategy"]
previous_bad_channels_strategy = settings["bad_channels_strategy"]
previous_interpolate_bad_channels = settings["interpolate_bad_channels"]
previous_threshold = settings["threshold"]
previous_short_channel_correction = settings["short_channel_correction"]
previous_negative_correlation_enhancement = settings["negative_correlation_enhancement"]
previous_baseline_correction = settings["baseline_correction"]
previous_tmin = settings["tmin"]
previous_stimulus_duration = settings["stimulus_duration"]
previous_scalp_coupling_threshold = settings["scalp_coupling_threshold"]
previous_reject_criteria = settings["reject_criteria"]
previous_unwanted = settings["unwanted"]
previous_filter_lower_value = settings["filter_lower_value"]
previous_filter_upper_value = settings["filter_upper_value"]
previous_h_trans_bandwidth = settings["h_trans_bandwidth"]
previous_l_trans_bandwidth = settings["l_trans_bandwidth"]
previous_snr_rejection = settings["snr_rejection"]
previous_snr_threshold = settings["snr_threshold"]
previous_apply_tddr = settings["Apply_TDDR"]

def update_epoch_types(*args):
    """Load data and update epoch type dropdown based on dataset selection."""
    global previous_dataset, all_individuals, all_epochs, data_name, all_data, freq, data_types, start_up, first_data_load, current_loader
    dataset = dataset_var.get()
    
    # Only reload data if dataset is changed or first time
    if dataset != previous_dataset or start_up:
        try:
            current_loader = data_loaders[dataset_var.get()](
                data_name = dataset_var.get(),
                file_path = dataset_var.get(),
                short_channel_correction=settings["short_channel_correction"],
                negative_correlation_enhancement=settings["negative_correlation_enhancement"],
                interpolate_bad_channels=settings["interpolate_bad_channels"],
                baseline_correction=settings["baseline_correction"],
                tmin=settings["tmin"],
                filter_lower_value=settings["filter_lower_value"],
                filter_upper_value=settings["filter_upper_value"],
                l_trans_bandwidth=settings["l_trans_bandwidth"],
                h_trans_bandwidth=settings["h_trans_bandwidth"],
                scalp_coupling_threshold=settings["scalp_coupling_threshold"],
                reject_criteria=settings["reject_criteria"],
                snr_rejection=settings["snr_rejection"],
                snr_threshold=settings["snr_threshold"],
                apply_tddr=settings["Apply_TDDR"]
            )

            all_epochs, data_name, all_data, freq, data_types, all_individuals = current_loader.load_data()

            # Update dropdown options
            epoch_type_menu["values"] = data_types
            if data_types:
                epoch_type_var.set(data_types[0])  # Select first available type
                settings["epoch_type"] = data_types[0]  # Ensure internal setting is updated

            individual_names = [getattr(ind, "name", f"Participant_{i+1}") for i, ind in enumerate(all_individuals)]
            individuals_menu["values"] = ["All Individuals"] + individual_names if individual_names else ["All Individuals"]

            # Default to first actual individual if available, else fallback to "All Individuals"
            default_individual = individual_names[0] if individual_names else "All Individuals"
            Individual_var.set(default_individual)
            settings["individual"] = default_individual

            previous_dataset = dataset  # Update stored dataset
            start_up = False
            first_data_load = False
            
            # Force update of UI elements that depend on data
            toggle_individual_menu()
            
        except Exception as e:
            print(f"Error loading data: {e}")

def toggle_individual_menu(*args):
    """Show or hide settings based on plot type."""
    plot_type = plot_type_var.get()
    
    # First hide all specialized widgets
    for widget in [
        # Individual selection
        individual_label, individuals_menu,
        # Individual checkboxes
        individual_selection_label, individual_selection_frame,
        # Epoch type
        epoch_type_label, epoch_type_menu,
        # Channel selection
        channel_selection_label, channel_frame,
        # Hemoglobin type
        haemo_type_label, haemo_type_menu,
        # Statistical analysis
        area_of_interest_label, area_of_interest_menu, 
        time_window_label, time_window_frame,
        dataset1_label, dataset1_menu,
        dataset2_label, dataset2_menu,
        
    ]:
        widget.pack_forget()
    
    # Show only what's needed for each plot type
    if plot_type == "Statistical Analysis":
        # Show statistical analysis specific settings
        area_of_interest_label.pack(anchor="w")
        area_of_interest_menu.pack(pady=5)
        time_window_label.pack(anchor="w")
        time_window_frame.pack(pady=5)
        dataset1_label.pack(anchor="w")
        dataset1_menu.pack(pady=5)
        dataset2_label.pack(anchor="w")
        dataset2_menu.pack(pady=5)
        
    elif plot_type == "Standard fNIRS Response Plot":
        # Show individual selection checkboxes
        individual_selection_label.pack(anchor="w")
        individual_selection_frame.pack(fill="x", expand=False)
        # Show channel selection
        channel_selection_label.pack(anchor="w")
        channel_frame.pack(fill="x", expand=True)
        # Populate the individual checkboxes
        populate_individuals()
        # Populate channel checkboxes
        populate_channels()
        
    elif plot_type in ["individual frequency plot", "paradigm_plot"]:
        # Show hemoglobin type selection for paradigm_plot
        if plot_type == "paradigm_plot":
            haemo_type_label.pack(anchor="w")
            haemo_type_menu.pack(pady=5)
            
        # Show individual selection dropdown
        individual_label.pack(anchor="w") 
        individuals_menu.pack(pady=5)
        
        # Update to show only individual participants for paradigm_plot
        individuals_menu["values"] = [getattr(ind, "name", f"Participant_{i+1}") 
                                     for i, ind in enumerate(all_individuals)]
        # If "All Individuals" was previously selected, change to first individual
        if Individual_var.get() == "All Individuals" and individuals_menu["values"]:
            Individual_var.set(individuals_menu["values"][0])
        
        # Show channel selection for paradigm_plot
        if plot_type == "paradigm_plot":
            channel_selection_label.pack(anchor="w", pady=(10, 2))
            channel_frame.pack(fill="both", expand=False, pady=(0, 10))
            populate_channels()
        
    elif plot_type == "Epoch Plot":
        # Show epoch type selection
        epoch_type_label.pack(anchor="w")
        epoch_type_menu.pack(pady=5)
        
        # Show individual selection checkboxes
        individual_selection_label.pack(anchor="w")
        individual_selection_frame.pack(fill="x", expand=False)
        populate_individuals()
        
        # Show channel selection
        channel_selection_label.pack(anchor="w")
        channel_frame.pack(fill="x", expand=True)
        populate_channels()
        
    # Force the UI to update
    root.update_idletasks()
        
def show_dataset_info_view():
    """Render Dataset Info inside the main plot area (right_frame)."""
    try:
        if 'all_epochs' in globals() and all_epochs:
            # Clear the right-side plot area before showing info
            for widget in right_frame.winfo_children():
                widget.destroy()

            # Mount the info panel inside right_frame
            show_dataset_info_in_container(
                class_instance=current_loader,
                parent_container=right_frame,
                all_epochs=all_epochs,
                data_name=data_name,
                all_data=all_data,
                freq=freq,
                data_types=data_types,
                all_individuals=all_individuals
            )
        else:
            tk.messagebox.showwarning("No Data", "Please load a dataset first by selecting one from the dropdown.")
    except Exception as e:
        tk.messagebox.showerror("Error", f"Failed to show dataset info: {str(e)}")

# Updated run_analysis function
def run_analysis():
    """Run data processing and visualization based on selected plot type."""
    global previous_epoch_type, all_epochs, data_name, all_data, freq, data_types, all_individuals, first_data_load
    global previous_short_channel_correction, previous_negative_correlation_enhancement, previous_interpolate_bad_channels
    global previous_baseline_correction, previous_tmin, previous_individual, previous_combine_strategy
    global previous_bad_channels_strategy, previous_threshold, previous_scalp_coupling_threshold, previous_reject_criteria
    global previous_filter_lower_value, previous_filter_upper_value, previous_l_trans_bandwidth, previous_h_trans_bandwidth
    global previous_snr_rejection, previous_snr_threshold, previous_apply_tddr, current_loader

    settings["data_set"] = dataset_var.get()
    settings["plot_type"] = plot_type_var.get()
    settings["haemo_type"] = haemo_type_var.get()
    settings["individual"] = Individual_var.get()
    settings["epoch_type"] = epoch_type_var.get()
    # Determine if data needs to be reloaded
    reload_data = (
        (settings["plot_type"] not in ["individual frequency plot","paradigm_plot", "Epoch Plot", "Standard fNIRS Response Plot"])
        or settings["epoch_type"] != previous_epoch_type
        or first_data_load == True
        or settings["short_channel_correction"] != previous_short_channel_correction
        or settings["negative_correlation_enhancement"] != previous_negative_correlation_enhancement
        or settings["interpolate_bad_channels"] != previous_interpolate_bad_channels
        or settings["baseline_correction"] != previous_baseline_correction
        or settings["tmin"] != previous_tmin
        or settings["filter_lower_value"] != previous_filter_lower_value
        or settings["filter_upper_value"] != previous_filter_upper_value
        or settings["l_trans_bandwidth"] != previous_l_trans_bandwidth
        or settings["h_trans_bandwidth"] != previous_h_trans_bandwidth
        or settings["scalp_coupling_threshold"] != previous_scalp_coupling_threshold
        or settings["reject_criteria"] != previous_reject_criteria
        or settings["snr_rejection"] != previous_snr_rejection
        or settings["snr_threshold"] != previous_snr_threshold
        or settings["Apply_TDDR"] != previous_apply_tddr
    )
    if reload_data:
        current_loader = data_loaders[dataset_var.get()](
            data_name = dataset_var.get(),
            file_path = dataset_var.get(),
            short_channel_correction=settings["short_channel_correction"],
            negative_correlation_enhancement=settings["negative_correlation_enhancement"],
            interpolate_bad_channels=settings["interpolate_bad_channels"],
            baseline_correction=settings["baseline_correction"],
            tmin=settings["tmin"],
            filter_lower_value=settings["filter_lower_value"],
            filter_upper_value=settings["filter_upper_value"],
            l_trans_bandwidth=settings["l_trans_bandwidth"],
            h_trans_bandwidth=settings["h_trans_bandwidth"],
            scalp_coupling_threshold=settings["scalp_coupling_threshold"],
            reject_criteria=settings["reject_criteria"],
            snr_rejection=settings["snr_rejection"],
            snr_threshold=settings["snr_threshold"],
            apply_tddr=settings["Apply_TDDR"]
        )

        all_epochs, data_name, all_data, freq, data_types, all_individuals = current_loader.load_data()
            
        previous_epoch_type = settings["epoch_type"]
        previous_short_channel_correction = settings["short_channel_correction"]
        previous_negative_correlation_enhancement = settings["negative_correlation_enhancement"]
        previous_interpolate_bad_channels = settings["interpolate_bad_channels"]
        previous_baseline_correction = settings["baseline_correction"]
        previous_tmin = settings["tmin"]
        previous_filter_lower_value = settings["filter_lower_value"]  
        previous_filter_upper_value = settings["filter_upper_value"]  
        previous_h_trans_bandwidth = settings["h_trans_bandwidth"]  
        previous_l_trans_bandwidth = settings["l_trans_bandwidth"]
        previous_scalp_coupling_threshold = settings["scalp_coupling_threshold"]
        previous_reject_criteria = settings["reject_criteria"]
        previous_snr_rejection = settings["snr_rejection"]
        previous_snr_threshold = settings["snr_threshold"]
        previous_apply_tddr = settings["Apply_TDDR"]
        toggle_individual_menu()
    
    first_data_load = False
    # Clear previous plots
    for widget in right_frame.winfo_children():
        widget.destroy()

    # Get selected base channels (without hbo/hbr endings)
    selected_base_channels = [channel for channel, var in channel_vars.items() if var.get()]

    # Handle channel selection based on plot type
    if settings["plot_type"] == "paradigm_plot":
        # For paradigm plot, add the selected hemoglobin type to base channels
        selected_haemo_type = settings["haemo_type"]
        picks = [f"{channel} {selected_haemo_type}" for channel in selected_base_channels]
        
    elif settings["plot_type"] in ["Epoch Plot", "Standard fNIRS Response Plot"]:
        # For these plot types, expand base channels to include both hbo and hbr
        picks = []
        for base_channel in selected_base_channels:
            picks.extend([f"{base_channel} hbo", f"{base_channel} hbr"])
        
        # If no channels selected, use "all"
        if not picks:
            picks = "all"
            
    else:
        # For other plot types, use the original logic
        picks = selected_base_channels if len(selected_base_channels) < len(channel_vars) else "all"
    
    # Add hemoglobin type to settings
    settings["haemo_type"] = haemo_type_var.get()
    # Run the selected plot function
    if settings["plot_type"] == "Statistical Analysis":
        # Get statistical analysis parameters
        area_of_interest = area_of_interest_var.get()
        start_time = float(start_time_var.get())
        end_time = float(end_time_var.get())
        dataset1 = dataset1_var.get()
        dataset2 = dataset2_var.get()
        
        # Run the statistical analysis
        figures = statistical_analysis(
            Area_of_interest=area_of_interest,
            start_time=start_time,
            end_time=end_time,
            dataset1=dataset1,
            dataset2=dataset2,
        )
            
        # Create tab control for statistical analysis
        tab_control = ttk.Notebook(right_frame)
        tab_control.pack(expand=True, fill="both")
        
        # Ensure figures is always a list and flatten if needed
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
            # Create a tab for each figure
            for i, fig in enumerate(figures):
                # Create a new tab
                tab = ttk.Frame(tab_control)
                tab_control.add(tab, text=f"Plot {i+1}")
                
                # Display the figure in the tab
                canvas = FigureCanvasTkAgg(fig, master=tab)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True, pady=5)
                
        # If no figures were generated
        if not figures:
            # Create a single tab with message
            tab = ttk.Frame(tab_control)
            tab_control.add(tab, text="Info")
            tk.Label(tab, text="No statistical plots to display", font=("Arial", 14)).pack(pady=20)
            
    else:
        # For all other plot types, use the original display format
        figures = []
        
        if settings["plot_type"] == "Epoch Plot":
            selected_individuals = [name for name, var in individual_selection_vars.items() if var.get()]
            selected_all_epochs = []

            for name in selected_individuals:
                individual = next((ind for ind in all_individuals if getattr(ind, "name", "") == name), None)
                if individual is not None:
                    selected_all_epochs.append(individual.epochs)

            if selected_all_epochs:
                figures = [epoch_plot(
                    selected_all_epochs,
                    picks=picks,
                    epoch_type=settings["epoch_type"],
                    combine_strategy=settings["combine_strategy"],
                    save=settings["save_plot"],
                    bad_channels_strategy=settings["bad_channels_strategy"],
                    threshold=settings["threshold"],
                    data_set=data_name
                )]
            else:
                print("No individuals selected or found.")

        elif settings["plot_type"] == "Standard fNIRS Response Plot":
            selected_individuals = [ind_name for ind_name, var in individual_selection_vars.items() if var.get()]
            
            # Create the same filtered list as in populate_individuals()
            valid_individuals = []
            for individual in all_individuals:
                if hasattr(individual, 'epochs'):
                    has_all_data_types = True
                    for data_type in data_types:
                        # Match any event ID whose key contains the data_type substring
                        matching_ids = [
                            v for k, v in individual.epochs.event_id.items()
                            if data_type in k
                        ]

                        # Count events where the event ID matches one of the matching IDs
                        epoch_count = sum(
                            1 for event in individual.epochs.events
                            if event[2] in matching_ids
                        )

                        if epoch_count == 0:
                            has_all_data_types = False
                            break

                    if has_all_data_types:
                        valid_individuals.append(individual)
            
            if settings["compare_with_raw"]:
                # Create two separate lists for raw and processed epochs
                selected_raw_epochs = []
                selected_processed_epochs = []
                
                # Find the actual individual objects from their names and get both types of epochs
                for ind_name in selected_individuals:
                    # Find the individual object that matches this name FROM THE FILTERED LIST
                    individual = next((ind for i, ind in enumerate(valid_individuals) 
                                    if getattr(ind, "name", f"Participant_{i+1}") == ind_name), None)
                    
                    # If found, append both raw and processed epochs to respective lists
                    if individual is not None:
                        if hasattr(individual, 'raw_epochs'):
                            selected_raw_epochs.append(individual.raw_epochs)
                        selected_processed_epochs.append(individual.epochs)
                
                # Create both plots
                figures = []
                
                # Create raw epochs plot if raw epochs exist
                if selected_raw_epochs:
                    raw_figure = standard_fNIRS_response_plot(
                        selected_raw_epochs, 
                        data_types, 
                        bad_channels_strategy=settings["bad_channels_strategy"],
                        save=settings["save_plot"], 
                        combine_strategy=settings["combine_strategy"],
                        threshold=settings["threshold"], 
                        data_set=f"{data_name} (Raw/Non-processed)", 
                        picks_=picks
                    )
                    figures.append(raw_figure)
                
                # Create processed epochs plot
                if selected_processed_epochs:
                    processed_figure = standard_fNIRS_response_plot(
                        selected_processed_epochs, 
                        data_types, 
                        bad_channels_strategy=settings["bad_channels_strategy"],
                        save=settings["save_plot"], 
                        combine_strategy=settings["combine_strategy"],
                        threshold=settings["threshold"], 
                        data_set=f"{data_name} (Processed)", 
                        picks_=picks
                    )
                    figures.append(processed_figure)
                    
            else:
                # Original behavior - only processed epochs
                selected_processed_epochs = []
                
                # Find the actual individual objects from their names and get their processed epochs
                for ind_name in selected_individuals:
                    # Find the individual object that matches this name FROM THE FILTERED LIST
                    individual = next((ind for i, ind in enumerate(valid_individuals) 
                                    if getattr(ind, "name", f"Participant_{i+1}") == ind_name), None)
                    
                    # If found, append their processed epochs to our list
                    if individual is not None:
                        selected_processed_epochs.append(individual.epochs)
                # Create single plot with processed epochs
                figures = [standard_fNIRS_response_plot(
                    selected_processed_epochs, 
                    data_types, 
                    bad_channels_strategy=settings["bad_channels_strategy"],
                    save=settings["save_plot"], 
                    combine_strategy=settings["combine_strategy"],
                    threshold=settings["threshold"], 
                    data_set=data_name, 
                    picks_=picks
                )]
            
        elif settings["plot_type"] == "paradigm_plot":
            selected_individual = settings["individual"]
            selected_channels = [channel for channel, var in channel_vars.items() if var.get()]
            selected_haemo_type = settings["haemo_type"]
            picks = [f"{channel} {selected_haemo_type}" for channel in selected_channels]
            
            # Find the individual by name
            index = next((i for i, ind in enumerate(all_individuals) 
                        if getattr(ind, "name", f"Participant_{i+1}") == selected_individual), -1)
            if index >= 0:
                figures = [paradigm_plot(
                    all_individuals[index], 
                    picks_=picks, 
                    haemo_type=selected_haemo_type
                )]
        elif settings["plot_type"] == "individual frequency plot":
            selected_individual = settings["individual"]
            # Find the individual by name
            index = next((i for i, ind in enumerate(all_individuals) 
                        if getattr(ind, "name", f"Participant_{i+1}") == selected_individual), -1)
            if index >= 0:
                figures = [individual_frequency_plot(all_individuals[index])]
        
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
            # Display each figure in the right_frame (original method)
            for fig in figures:
                canvas = FigureCanvasTkAgg(fig, master=right_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True, pady=5)

# Function to update both channels and individual checkboxes
def update_ui_elements(*args):
    populate_channels()
    populate_individuals()

# Create GUI window
root = tk.Tk()
root.title("fNIRS Data Analysis")
root.geometry("800x600")

left_container = tk.Frame(root, width=300)  # Set desired width
left_container.pack(side="left", padx=20, pady=20, fill="y")
left_container.pack_propagate(False)  # Prevent resizing based on content

top_left_frame = tk.Frame(left_container)
top_left_frame.pack(side="top", fill="y", expand=True)

bottom_left_frame = tk.Frame(left_container)
bottom_left_frame.pack(side="bottom", fill="x")

# Helper function to create labels
def create_label(parent, text, pack_immediately=True):
    label = tk.Label(parent, text=text, font=("Arial", 12))
    if pack_immediately:
        label.pack(anchor="w")
    return label

# Helper function to create a combobox with a label
def create_combobox_with_label(parent, label_text, values=None, default_value="", width=None):
    label = create_label(parent, label_text)
    var = tk.StringVar(value=default_value)
    combo = ttk.Combobox(parent, textvariable=var, values=values or [], width=width)
    combo.pack(pady=5)
    return label, var, combo

# Helper function to adjust combobox width
def adjust_combobox_width(combobox):
    combobox["width"] = max(len(item) for item in combobox["values"])

# Dataset selection with info button
dataset_frame = tk.Frame(top_left_frame)
dataset_frame.pack(fill="x", pady=5)

dataset_label = tk.Label(dataset_frame, text="Select Dataset:", font=("Arial", 12))
dataset_label.pack(anchor="w")

dataset_selection_frame = tk.Frame(dataset_frame)
dataset_selection_frame.pack(fill="x", pady=5)

dataset_var = tk.StringVar(value=settings["data_set"])
dataset_menu = ttk.Combobox(dataset_selection_frame, textvariable=dataset_var, values=dataSetList, width=35)
dataset_menu.pack(side="left", padx=(0, 5))
dataset_menu["postcommand"] = lambda: adjust_combobox_width(dataset_menu)

# Dataset info button
info_button = tk.Button(
    dataset_selection_frame, text="Info",
    command=show_dataset_info_view,   # <-- changed
    bg="lightblue", fg="black", font=("Arial", 10), padx=10, pady=2
)
info_button.pack(side="left")

# Helper function to toggle widget visibility
def toggle_widgets(show, *widgets):
    for widget in widgets:
        if show:
            widget.pack(anchor="w")
        else:
            widget.pack_forget()

# Hemoglobin type selection (initially hidden)
haemo_type_label, haemo_type_var, haemo_type_menu = create_combobox_with_label(
    top_left_frame, "Hemoglobin Type:", ["hbo", "hbr"], settings["haemo_type"])
toggle_widgets(False, haemo_type_label, haemo_type_menu)

# Individual selection (initially hidden)
individual_label, Individual_var, individuals_menu = create_combobox_with_label(
    top_left_frame, "Select Individual:")
toggle_widgets(False, individual_label, individuals_menu)

preprocessing_frame = tk.Frame(top_left_frame)
preprocessing_frame.pack(fill="x", pady=10)

preprocessing_label = tk.Label(preprocessing_frame, text="Preprocessing Options:", font=("Arial", 12))
preprocessing_label.pack(anchor="w")

def open_preprocessing_dialog():
    """Open the preprocessing options dialog."""
    current_preprocessing_settings = {
        "short_channel_correction": settings["short_channel_correction"],
        "negative_correlation_enhancement": settings["negative_correlation_enhancement"],
        "interpolate_bad_channels": settings["interpolate_bad_channels"],
        "baseline_correction": settings["baseline_correction"],
        "tmin": settings["tmin"],
        "filter_lower_value": settings["filter_lower_value"],
        "filter_upper_value": settings["filter_upper_value"],
        "h_trans_bandwidth": settings["h_trans_bandwidth"],
        "l_trans_bandwidth": settings["l_trans_bandwidth"],
        "scalp_coupling_threshold": settings["scalp_coupling_threshold"],
        "reject_criteria": settings["reject_criteria"],
        "snr_rejection": settings["snr_rejection"],
        "snr_threshold": settings["snr_threshold"],
        "Apply_TDDR": settings["Apply_TDDR"]
    }
    
    result = show_preprocessing_dialog(root, current_preprocessing_settings)
    
    if result:  # If user clicked OK
        # Update the global settings
        settings["short_channel_correction"] = result["short_channel_correction"]
        settings["negative_correlation_enhancement"] = result["negative_correlation_enhancement"]
        settings["interpolate_bad_channels"] = result["interpolate_bad_channels"]
        settings["baseline_correction"] = result["baseline_correction"]
        settings["tmin"] = result["tmin"]
        settings["filter_lower_value"] = result["filter_lower_value"]
        settings["filter_upper_value"] = result["filter_upper_value"]
        settings["h_trans_bandwidth"] = result["h_trans_bandwidth"]
        settings["l_trans_bandwidth"] = result["l_trans_bandwidth"]
        settings["scalp_coupling_threshold"] = result["scalp_coupling_threshold"]
        settings["reject_criteria"] = result["reject_criteria"]
        settings["snr_rejection"] = result["snr_rejection"]
        settings["snr_threshold"] = result["snr_threshold"]
        settings["Apply_TDDR"] = result["Apply_TDDR"]

        # Mark that data needs to be reloaded
        global previous_short_channel_correction, previous_negative_correlation_enhancement, previous_interpolate_bad_channels,previous_tmin, previous_filter_lower_value, previous_filter_upper_value
        global previous_h_trans_bandwidth, previous_l_trans_bandwidth, previous_scalp_coupling_threshold, previous_reject_criteria, previous_snr_rejection, previous_snr_threshold, previous_apply_tddr, current_loader
        previous_short_channel_correction = None  # Force reload
        previous_negative_correlation_enhancement = None
        previous_baseline_correction = None
        previous_interpolate_bad_channels = None
        previous_tmin = None
        previous_filter_lower_value = None
        previous_filter_upper_value = None
        previous_h_trans_bandwidth = None
        previous_l_trans_bandwidth = None
        previous_scalp_coupling_threshold = None
        previous_reject_criteria = None
        previous_snr_rejection = None
        previous_snr_threshold = None
        previous_apply_tddr = None

preprocessing_button = tk.Button(
    preprocessing_frame, 
    text="Configure Preprocessing", 
    command=open_preprocessing_dialog,
    bg="lightblue", 
    fg="black", 
    font=("Arial", 11),
    padx=10, 
    pady=5
)
preprocessing_button.pack(pady=5)

# Plot type selection
tk.Label(top_left_frame, text="Select Plot Type:", font=("Arial", 12)).pack(anchor="w")
plot_type_var = tk.StringVar(value=settings["plot_type"])
plot_type_menu = ttk.Combobox(top_left_frame, textvariable=plot_type_var, values=plotTypesList)
# Call the function that updates the UI based on the selected plot type
plot_type_menu.pack(pady=5)

plot_settings_frame = tk.Frame(top_left_frame)
plot_settings_frame.pack(fill="x", pady=10)

plot_settings_label = tk.Label(plot_settings_frame, text="Plot Settings:", font=("Arial", 12))
plot_settings_label.pack(anchor="w")

def open_plot_settings_dialog():
    """Open the plot settings dialog."""
    current_plot_settings = {
        "plot_type": settings["plot_type"],
        "epoch_type": settings["epoch_type"],
        "combine_strategy": settings["combine_strategy"],
        "bad_channels_strategy": settings["bad_channels_strategy"],
        "threshold": settings["threshold"],
        "save_plot": settings["save_plot"]
    }
    
    result = show_plot_settings_dialog(
        parent=root,
        current_settings=current_plot_settings,
        )
    
    if result:  # If user clicked OK
        # Update the global settings
        settings["combine_strategy"] = result["combine_strategy"]
        settings["bad_channels_strategy"] = result["bad_channels_strategy"]
        settings["threshold"] = result["threshold"]
        settings["save_plot"] = result["save_plot"]
        if "compare_with_raw" in result.keys():
            settings["compare_with_raw"] = result["compare_with_raw"]
        
        # Mark that data needs to be reloaded if certain settings changed
        global previous_combine_strategy, previous_bad_channels_strategy, previous_threshold, current_loader

        
        # Update other tracking variables
        previous_combine_strategy = settings["combine_strategy"]
        previous_bad_channels_strategy = settings["bad_channels_strategy"]
        previous_threshold = settings["threshold"]

plot_settings_button = tk.Button(
    plot_settings_frame,
    text="Configure Plot Settings",
    command=open_plot_settings_dialog,
    bg="lightgreen",
    fg="black",
    font=("Arial", 11),
    padx=10,
    pady=5
)
plot_settings_button.pack(pady=5)

# Epoch type selection
epoch_type_label, epoch_type_var, epoch_type_menu = create_combobox_with_label(
    top_left_frame, "Epoch Type:")

# Helper function to create checkbox with label
def create_checkbox_with_label(parent, label_text, default_value):
    label = create_label(parent, label_text)
    var = tk.BooleanVar(value=default_value)
    checkbox = tk.Checkbutton(parent, text="Enable", variable=var)
    checkbox.pack(anchor="w")
    return label, var, checkbox

# Channel selection
channel_selection_label = tk.Label(top_left_frame, text="Select Channels:", font=("Arial", 12))
channel_selection_label.pack(anchor="w")

# Create a frame to hold the channel checkboxes with a scrollbar
channel_frame = tk.Frame(top_left_frame)
channel_frame.pack(fill="both", expand=False)
channel_canvas = tk.Canvas(channel_frame, height=150)  # Set as needed
channel_scrollbar = tk.Scrollbar(channel_frame, orient="vertical", command=channel_canvas.yview)
channel_scrollbar_horizontal = tk.Scrollbar(channel_frame, orient="horizontal", command=channel_canvas.xview)
channel_container = tk.Frame(channel_canvas)
channel_canvas.create_window((0, 0), window=channel_container, anchor="nw")
channel_canvas.configure(yscrollcommand=channel_scrollbar.set, xscrollcommand=channel_scrollbar_horizontal.set)

# Variable to track channel selections
channel_vars = {}

# Individual selection checkboxes
individual_selection_label = tk.Label(top_left_frame, text="Select Individuals:", font=("Arial", 12))
individual_selection_label.pack(anchor="w")

# Create a frame to hold the individual checkboxes with a scrollbar
individual_selection_frame = tk.Frame(top_left_frame)
individual_selection_frame.pack(fill="x", expand=False)
individual_canvas = tk.Canvas(individual_selection_frame, height=100)
individual_scrollbar = tk.Scrollbar(individual_selection_frame, orient="vertical", command=individual_canvas.yview)
individual_scrollbar_horizontal = tk.Scrollbar(individual_selection_frame, orient="horizontal", command=individual_canvas.xview)
individual_container = tk.Frame(individual_canvas)
individual_canvas.create_window((0, 0), window=individual_container, anchor="nw")
individual_canvas.configure(yscrollcommand=individual_scrollbar.set, xscrollcommand=individual_scrollbar_horizontal.set)

# Variable to track individual selections
individual_selection_vars = {}

# Helper function to clear container widgets
def clear_container(container):
    for widget in container.winfo_children():
        widget.destroy()

# Helper function to update the scroll region of the canvas
def update_canvas_scroll(container, canvas):
    container.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

def populate_channels():
    # Clear existing checkboxes
    clear_container(channel_container)
    # Reset channel variables
    channel_vars.clear()
    # Get the plot type
    current_plot_type = plot_type_var.get()
    
    # For paradigm_plot, use selected individual from dropdown
    if current_plot_type == "paradigm_plot":
        # Get selected individual from dropdown
        selected_individual_name = Individual_var.get()
        
        # Skip if "All Individuals" is selected
        if selected_individual_name == "All Individuals":
            channel_selection_label.pack_forget()
            channel_frame.pack_forget()
            return
        
        # Show channel selection
        channel_selection_label.pack(anchor="w")
        channel_frame.pack(fill="x", expand=True)
        
        # Find the selected individual object by name
        selected_individual = None
        for ind_index, ind in enumerate(all_individuals):
            if getattr(ind, "name", f"Participant_{ind_index+1}") == selected_individual_name:
                selected_individual = ind
                break
        
        if selected_individual and hasattr(selected_individual, 'epochs'):
            try:
                # Get bad channels for this individual
                bad_channels = selected_individual.epochs.info.get("bads", [])
                
                # Get all channels and extract unique base channel names
                all_channels = selected_individual.epochs.ch_names
                unique_base_channels = set()
                
                for channel in all_channels:
                    if channel not in bad_channels:
                        # Remove the hemoglobin type suffix (" hbo" or " hbr")
                        base_channel = channel.rsplit(' ', 1)[0] if ' ' in channel else channel
                        unique_base_channels.add(base_channel)
                
                # Create checkboxes for unique base channels
                for i, base_channel in enumerate(sorted(unique_base_channels)):
                    is_checked = (i == 0)  # Default: first one checked
                    channel_vars[base_channel] = tk.BooleanVar(value=is_checked)
                    cb = tk.Checkbutton(channel_container, text=base_channel, variable=channel_vars[base_channel])
                    cb.grid(row=i // 3, column=i % 3, sticky="w")
                    
                # Update the canvas scroll region after adding new widgets
                update_canvas_scroll(channel_container, channel_canvas)
            except Exception as e:
                print(f"Error accessing channels for {selected_individual_name}: {e}")
    
    elif current_plot_type == "Epoch Plot":
        # Show channel selection
        channel_selection_label.pack(anchor="w")
        channel_frame.pack(fill="x", expand=False)

        # Clear previous checkboxes
        clear_container(channel_container)
        channel_vars.clear()

        # Get selected individuals from checkboxes
        selected_names = [name for name, var in individual_selection_vars.items() if var.get()]
        
        if selected_names:
            # Get channels that are common to ALL selected individuals
            common_channels = None
            
            for name in selected_names:
                individual = next((ind for ind in all_individuals if getattr(ind, "name", "") == name), None)
                if individual and hasattr(individual, "epochs"):
                    individual_channels = set(individual.epochs.ch_names)
                    if common_channels is None:
                        common_channels = individual_channels
                    else:
                        common_channels = common_channels.intersection(individual_channels)
            
            # If we found common channels, extract unique base channel names
            if common_channels:
                # Extract unique base channel names (without hbo/hbr endings)
                unique_base_channels = set()
                for channel in common_channels:
                    base_channel = channel.rsplit(' ', 1)[0] if ' ' in channel else channel
                    unique_base_channels.add(base_channel)
                
                # Force update the container to ensure all widgets are destroyed
                channel_container.update_idletasks()
                
                for i, base_channel in enumerate(sorted(unique_base_channels)):
                    is_checked = (i == 0)  # Optional: first channel pre-checked
                    channel_vars[base_channel] = tk.BooleanVar(value=is_checked)
                    cb = tk.Checkbutton(channel_container, text=base_channel, variable=channel_vars[base_channel])
                    cb.grid(row=i // 3, column=i % 3, sticky="w")
                
                # Update the canvas scroll region after adding new widgets
                update_canvas_scroll(channel_container, channel_canvas)
            else:
                # If no common channels, ensure container is empty
                update_canvas_scroll(channel_container, channel_canvas)
    
    elif current_plot_type == "Standard fNIRS Response Plot":
        # Show channel selection
        channel_selection_label.pack(anchor="w")
        channel_frame.pack(fill="x", expand=False)

        # Clear previous checkboxes
        clear_container(channel_container)
        channel_vars.clear()

        # Get selected individuals
        selected_names = [name for name, var in individual_selection_vars.items() if var.get()]
        
        if selected_names:
            # Get channels that are common to ALL selected individuals
            common_channels = None
            
            for name in selected_names:
                individual = next((ind for ind in all_individuals if getattr(ind, "name", "") == name), None)
                if individual and hasattr(individual, "epochs"):
                    individual_channels = set(individual.epochs.ch_names)
                    if common_channels is None:
                        common_channels = individual_channels
                    else:
                        common_channels = common_channels.intersection(individual_channels)
            
            # If we found common channels, extract unique base channel names
            if common_channels:
                # Extract unique base channel names (without hbo/hbr endings)
                unique_base_channels = set()
                for channel in common_channels:
                    base_channel = channel.rsplit(' ', 1)[0] if ' ' in channel else channel
                    unique_base_channels.add(base_channel)
                
                # Force update the container to ensure all widgets are destroyed
                channel_container.update_idletasks()
                
                for i, base_channel in enumerate(sorted(unique_base_channels)):
                    is_checked = (i == 0)  # Optional: first channel pre-checked
                    channel_vars[base_channel] = tk.BooleanVar(value=is_checked)
                    cb = tk.Checkbutton(channel_container, text=base_channel, variable=channel_vars[base_channel])
                    cb.grid(row=i // 3, column=i % 3, sticky="w")
                
                # Update the canvas scroll region after adding new widgets
                update_canvas_scroll(channel_container, channel_canvas)
            else:
                # If no common channels, ensure container is empty
                update_canvas_scroll(channel_container, channel_canvas)
    
    # For other plot types that don't need channel selection, hide the channel frame
    else:
        channel_selection_label.pack_forget()
        channel_frame.pack_forget()

def update_channels_on_haemo_type_change(*args):
    """Update channel selections when hemoglobin type changes."""
    if plot_type_var.get() == "paradigm_plot":
        # Store the current selected base channel names (without hemoglobin type)
        current_selected = [ch for ch, var in channel_vars.items() if var.get()]
        
        # Clear and repopulate the channels
        populate_channels()
        
        # Try to reselect the previously selected channels if they exist in the new list
        for channel, var in channel_vars.items():
            if channel in current_selected:
                var.set(True)

# Replace the existing trace for haemo_type_var with this improved version
haemo_type_var.trace_add("write", update_channels_on_haemo_type_change)

def attach_checkbox_callbacks():
    for var in individual_selection_vars.values():
        var.trace_add("write", lambda *args: populate_channels())
        
def populate_individuals():
    """Populate the individual selection checkboxes regardless of dropdown selection."""
    # Store current selections before clearing
    current_selections = {name: var.get() for name, var in individual_selection_vars.items()}
    
    # Clear existing checkboxes
    clear_container(individual_container)
    
    # Get the current plot type to determine filtering logic
    current_plot_type = plot_type_var.get()
    
    # Filter individuals based on plot type
    if current_plot_type == "Standard fNIRS Response Plot":
        valid_individuals = []
        for individual in all_individuals:
            if hasattr(individual, 'epochs'):
                has_all_data_types = True
                for data_type in data_types:
                    # Match any event ID whose key contains the data_type substring
                    matching_ids = [
                        v for k, v in individual.epochs.event_id.items()
                        if data_type in k
                    ]

                    # Count events where the event ID matches one of the matching IDs
                    epoch_count = sum(
                        1 for event in individual.epochs.events
                        if event[2] in matching_ids
                    )

                    if epoch_count == 0:
                        has_all_data_types = False
                        break

                if has_all_data_types:
                    valid_individuals.append(individual)

            
        # Use filtered list for Standard fNIRS Response Plot
        individuals_to_display = valid_individuals
    else:
        # For all other plot types, use all individuals
        individuals_to_display = all_individuals
    
    # Add checkboxes for each valid individual
    if individuals_to_display:
        for i, individual in enumerate(individuals_to_display):
            individual_name = getattr(individual, "name", f"Participant_{i+1}")
            
            # Use existing selection if available, otherwise default to first only
            if individual_name in current_selections:
                is_checked = current_selections[individual_name]
            else:
                is_checked = (i == 0)  # Default: only first checked
                
            individual_selection_vars[individual_name] = tk.BooleanVar(value=is_checked)
            cb = tk.Checkbutton(individual_container, text=individual_name, variable=individual_selection_vars[individual_name])
            cb.grid(row=i // 3, column=i % 3, sticky="w")
    
    # Update scroll region
    individual_container.update_idletasks()
    individual_canvas.config(scrollregion=individual_canvas.bbox("all"))
    
    attach_checkbox_callbacks()

# Helper function for trace removal:
def clear_traces(*variables):
    for var in variables:
        try:
            var.trace_remove('write', None)
        except:
            pass
        
# Define setup_ui_callbacks function
def setup_ui_callbacks():
    # Clear any existing traces
    clear_traces(dataset_var, plot_type_var, Individual_var, haemo_type_var)
    
    # Set up the main callbacks in the correct order
    dataset_var.trace_add("write", lambda *args: (update_epoch_types(), toggle_individual_menu()))
    plot_type_var.trace_add("write",  lambda *args: (toggle_individual_menu(), populate_channels()))
    Individual_var.trace_add("write", lambda *args: populate_channels())  # Only update channels based on selection
    haemo_type_var.trace_add("write", update_channels_on_haemo_type_change)

# Pack the scrollable frame
def setup_scrollable_frame(canvas, v_scrollbar, h_scrollbar):
    canvas.pack(side="left", fill="both", expand=True)
    v_scrollbar.pack(side="right", fill="y")
    h_scrollbar.pack(side="bottom", fill="x")

setup_scrollable_frame(channel_canvas, channel_scrollbar, channel_scrollbar_horizontal)
setup_scrollable_frame(individual_canvas, individual_scrollbar, individual_scrollbar_horizontal)

# Area of Interest selection for Statistical Analysis
area_of_interest_label = tk.Label(top_left_frame, text="Area of Interest:", font=("Arial", 12))
area_of_interest_var = tk.StringVar(value="SMA")
area_of_interest_menu = ttk.Combobox(top_left_frame, textvariable=area_of_interest_var, 
                                     values=["SMA", "Tongue_all", "Tongue_right", "Tongue_left", 
                                             "Hand_all", "Hand_right", "Hand_left"])
# Time window for Statistical Analysis
time_window_label = tk.Label(top_left_frame, text="Time Window (seconds):", font=("Arial", 12))
time_window_frame = tk.Frame(top_left_frame)
start_time_label = tk.Label(time_window_frame, text="Start:")
start_time_var = tk.StringVar(value="3")
start_time_entry = tk.Entry(time_window_frame, textvariable=start_time_var, width=5)
end_time_label = tk.Label(time_window_frame, text="End:")
end_time_var = tk.StringVar(value="12")
end_time_entry = tk.Entry(time_window_frame, textvariable=end_time_var, width=5)

start_time_label.pack(side="left", padx=2)
start_time_entry.pack(side="left", padx=2)
end_time_label.pack(side="left", padx=2)
end_time_entry.pack(side="left", padx=2)

time_window_label.pack_forget()  # Initially hidden
time_window_frame.pack_forget()  # Initially hidden

# Dataset comparison for Statistical Analysis
dataset1_label = tk.Label(top_left_frame, text="Dataset 1:", font=("Arial", 12))
dataset1_var = tk.StringVar(value=settings["data_set"])
dataset1_menu = ttk.Combobox(top_left_frame, textvariable=dataset1_var, values=dataSetList, width=40)
dataset1_label.pack_forget()  # Initially hidden
dataset1_menu.pack_forget()   # Initially hidden
dataset1_menu["postcommand"] = lambda: adjust_combobox_width(dataset1_menu)

dataset2_label = tk.Label(top_left_frame, text="Dataset 2:", font=("Arial", 12))
dataset2_var = tk.StringVar(value=dataSetList[0])
dataset2_menu = ttk.Combobox(top_left_frame, textvariable=dataset2_var, values=dataSetList, width=40)
dataset2_label.pack_forget()  # Initially hidden
dataset2_menu.pack_forget()   # Initially hidden
    
def _on_mousewheel(event):
    channel_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

channel_canvas.bind_all("<MouseWheel>", _on_mousewheel)  # For Windows and Mac

dataset2_menu["postcommand"] = lambda: adjust_combobox_width(dataset2_menu)

run_button = tk.Button(bottom_left_frame, text="Run Analysis", command=run_analysis, bg="green", fg="white", 
                       font=("Arial", 12, "bold"), padx=20, pady=10)
run_button.pack(pady=20, padx=10, fill="x")


# Right panel for displaying the plot
right_frame = tk.Frame(root)
right_frame.pack(side="right", padx=20, pady=20, expand=True, fill="both")

# Add trace to Individual_var to update individual checkboxes when selection changes
Individual_var.trace_add("write", lambda *args: populate_individuals())
# Initial population of individuals
populate_individuals()

# Call this function at the end of your initialization code
setup_ui_callbacks()

# Initialize GUI
update_epoch_types()
toggle_individual_menu()
root.mainloop()