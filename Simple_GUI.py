import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from load_data_function import load_data
from epoch_plot import epoch_plot
from standard_fNIRS_response_plot import standard_fNIRS_response_plot
from paradigm_plot import paradigm_plot
from individual_frequency_plot import individual_frequency_plot
from statistical_analysis import statistical_analysis

# Default settings (add hemoglobin type to settings)
settings = {
    "data_set": "fNIrs_motor",
    "epoch_type": "Tapping",
    "combine_strategy": "mean",
    "short_channel_correction": True,
    "negative_correlation_enhancement": False,
    "interpolate_bad_channels": False,
    "bad_channels_strategy": "all",
    "threshold": 3,
    "plot_type": "Epoch Plot",
    "individual": True,
    "haemo_type": "hbo"  # New setting for hemoglobin type
}

first_data_load = True
all_individuals = []
start_up = True
# Track previous selections
previous_dataset = settings["data_set"]
previous_epoch_type = settings["epoch_type"]
previous_short_channel_correction = settings["short_channel_correction"]
previous_negative_correlation_enhancement = settings["negative_correlation_enhancement"]
previous_interpolate_bad_channels = settings["interpolate_bad_channels"]

def update_epoch_types(*args):
    """Load data and update epoch type dropdown based on dataset selection."""
    global previous_dataset, all_individuals, all_epochs, data_name, all_data, freq, data_types, start_up, first_data_load
    dataset = dataset_var.get()
    
    # Only reload data if dataset is changed or first time
    if dataset != previous_dataset or start_up:
        try:
            all_epochs, data_name, all_data, freq, data_types, all_individuals = load_data(
                data_set=dataset,
                short_channel_correction=settings["short_channel_correction"],
                negative_correlation_enhancement=settings["negative_correlation_enhancement"],
                interpolate_bad_channels=settings["interpolate_bad_channels"],
                individuals=settings["individual"]
            )
            # Update dropdown options
            epoch_type_menu["values"] = data_types
            if data_types:
                epoch_type_var.set(data_types[0])  # Select first available type
            # Update individuals dropdown
            # This is where we'll modify for paradigm_plot
            individuals_menu["values"] = ["All Individuals"] + [getattr(ind, "name", f"Participant_{i+1}") 
                                                           for i, ind in enumerate(all_individuals)]
            Individual_var.set("All Individuals")  # Default to "All Individuals"
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
        # Channel selection
        channel_selection_label, channel_frame,
        # Hemoglobin type
        haemo_type_label, haemo_type_menu,
        # Statistical analysis
        area_of_interest_label, area_of_interest_menu, 
        time_window_label, time_window_frame,
        dataset1_label, dataset1_menu,
        dataset2_label, dataset2_menu,
        # Data processing settings
        combine_strategy_label, combine_strategy_menu,
        bad_channels_strategy_label, bad_channels_strategy_menu,
        short_channel_correction_label, short_channel_correction_checkbox,
        negative_correlation_label, negative_correlation_checkbox,
        interpolate_bad_channels_label, interpolate_bad_channels_checkbox,
        threshold_label, threshold_entry,
        # Epoch selection
        epoch_type_label, epoch_type_menu
    ]:
        widget.pack_forget()
    
    # Then show only what's needed for each plot type
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
        # Show data processing settings
        combine_strategy_label.pack(anchor="w")
        combine_strategy_menu.pack(pady=5)
        bad_channels_strategy_label.pack(anchor="w")
        bad_channels_strategy_menu.pack(pady=5)
        short_channel_correction_label.pack(anchor="w")
        short_channel_correction_checkbox.pack(anchor="w")
        negative_correlation_label.pack(anchor="w")
        negative_correlation_checkbox.pack(anchor="w")
        interpolate_bad_channels_label.pack(anchor="w")
        interpolate_bad_channels_checkbox.pack(anchor="w")
        threshold_label.pack(anchor="w")
        threshold_entry.pack(pady=5)
        # Show individual selection
        individual_selection_label.pack(anchor="w")
        individual_selection_frame.pack(fill="x", expand=False)
        # Show channel selection
        channel_selection_label.pack(anchor="w")
        channel_frame.pack(fill="x", expand=True)
        # Populate the individual checkboxes
        populate_individuals()
        # Populate channel checkboxes
        populate_channels()
        
    elif plot_type == "individual frequency plot":
        # Show epoch type selection
        epoch_type_label.pack(anchor="w")
        epoch_type_menu.pack(pady=5)
        # Show individual selection dropdown
        individual_label.pack(anchor="w")
        individuals_menu.pack(pady=5)
        # Update to show only individual participants for paradigm_plot
        individuals_menu["values"] = [getattr(ind, "name", f"Participant_{i+1}") 
                                     for i, ind in enumerate(all_individuals)]
        # If "All Individuals" was previously selected, change to first individual
        if Individual_var.get() == "All Individuals" and individuals_menu["values"]:
            Individual_var.set(individuals_menu["values"][0])
        
    elif plot_type == "paradigm_plot":
        # Show data processing settings
        short_channel_correction_label.pack(anchor="w")
        short_channel_correction_checkbox.pack(anchor="w")
        negative_correlation_label.pack(anchor="w")
        negative_correlation_checkbox.pack(anchor="w")
        interpolate_bad_channels_label.pack(anchor="w")
        interpolate_bad_channels_checkbox.pack(anchor="w")
        # Show individual selection dropdown
        individual_label.pack(anchor="w") 
        individuals_menu.pack(pady=5)
        # Show hemoglobin type selection for paradigm_plot
        haemo_type_label.pack(anchor="w")
        haemo_type_menu.pack(pady=5)
        
        # Update to show only individual participants for paradigm_plot
        individuals_menu["values"] = [getattr(ind, "name", f"Participant_{i+1}") 
                                     for i, ind in enumerate(all_individuals)]
        # If "All Individuals" was previously selected, change to first individual
        if Individual_var.get() == "All Individuals" and individuals_menu["values"]:
            Individual_var.set(individuals_menu["values"][0])
        
        
        # Show channel selection
        channel_selection_label.pack(anchor="w", pady=(10, 2))
        channel_frame.pack(fill="both", expand=False, pady=(0, 10))

        # Populate the channel checkboxes
        populate_channels()
        
    elif plot_type == "Epoch Plot":
        # Show epoch type selection
        epoch_type_label.pack(anchor="w")
        epoch_type_menu.pack(pady=5)
        # Show individual selection dropdown with "All Individuals" option
        individual_label.pack(anchor="w")
        individuals_menu["values"] = ["All Individuals"] + [getattr(ind, "name", f"Participant_{i+1}") 
                                                       for i, ind in enumerate(all_individuals)]
        individuals_menu.pack(pady=5)
        # Show data processing settings
        combine_strategy_label.pack(anchor="w")
        combine_strategy_menu.pack(pady=5)
        bad_channels_strategy_label.pack(anchor="w")
        bad_channels_strategy_menu.pack(pady=5)
        short_channel_correction_label.pack(anchor="w")
        short_channel_correction_checkbox.pack(anchor="w")
        negative_correlation_label.pack(anchor="w")
        negative_correlation_checkbox.pack(anchor="w")
        interpolate_bad_channels_label.pack(anchor="w")
        interpolate_bad_channels_checkbox.pack(anchor="w")
        threshold_label.pack(anchor="w")
        threshold_entry.pack(pady=5)
        
    # Force the UI to update
    root.update_idletasks()

# Updated run_analysis function
def run_analysis():
    """Run data processing and visualization based on selected plot type."""
    global previous_epoch_type, all_epochs, data_name, all_data, freq, data_types, all_individuals, first_data_load, previous_short_channel_correction, previous_negative_correlation_enhancement, previous_interpolate_bad_channels
    settings["data_set"] = dataset_var.get()
    settings["epoch_type"] = epoch_type_var.get()
    settings["combine_strategy"] = combine_strategy_var.get()
    settings["short_channel_correction"] = short_channel_correction_var.get()
    settings["negative_correlation_enhancement"] = negative_correlation_enhancement_var.get()
    settings["interpolate_bad_channels"] = interpolate_bad_channels_var.get()
    settings["bad_channels_strategy"] = bad_channels_strategy_var.get()
    settings["threshold"] = int(threshold_var.get())
    settings["plot_type"] = plot_type_var.get()
    # Determine if data needs to be reloaded
    reload_data = (
        (settings["plot_type"] not in ["individual frequency plot","paradigm_plot", "Epoch Plot", "Standard fNIRS Response Plot"])
        or settings["epoch_type"] != previous_epoch_type # Reload only if epoch type changed
        or first_data_load == True
        or settings["short_channel_correction"] != previous_short_channel_correction
        or settings["negative_correlation_enhancement"] != previous_negative_correlation_enhancement
        or settings["interpolate_bad_channels"] != previous_interpolate_bad_channels
    )
    if reload_data:
        all_epochs, data_name, all_data, freq, data_types, all_individuals = load_data(
            data_set=settings["data_set"],
            short_channel_correction=settings["short_channel_correction"],
            negative_correlation_enhancement=settings["negative_correlation_enhancement"],
            interpolate_bad_channels=settings["interpolate_bad_channels"],
            individuals=settings["individual"]
        )
        previous_epoch_type = settings["epoch_type"]  # Update stored epoch type
        previous_short_channel_correction = settings["short_channel_correction"] # Update chosen short_channel_correction_setting
        previous_negative_correlation_enhancement = settings["negative_correlation_enhancement"]
        previous_interpolate_bad_channels = settings["interpolate_bad_channels"]
        toggle_individual_menu()
    
    first_data_load = False
    # Clear previous plots
    for widget in right_frame.winfo_children():
        widget.destroy()
    
    # If paradigm_plot, we need to handle picks differently
    if settings["plot_type"] == "paradigm_plot":
        # For paradigm_plot, we need to select channels with appropriate suffix based on haemo_type
        selected_channels = [channel for channel, var in channel_vars.items() if var.get()]
        selected_haemo_type = settings["haemo_type"]
        
        # For paradigm_plot, add hemoglobin type suffix to channel names
        # Make sure the base channel names don't already have a hemoglobin type suffix
        picks = [f"{channel} {selected_haemo_type}" for channel in selected_channels]
    else:
        # Modify channel selection for plotting (original logic for other plot types)
        selected_channels = [channel for channel, var in channel_vars.items() if var.get()]
        # If ALL channels are selected, set picks to None to let MNE handle channel types
        picks = selected_channels if len(selected_channels) < len(channel_vars) else "all"
    
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
            selected_individual = Individual_var.get()
            
            if selected_individual == "All Individuals":
                figures = [epoch_plot(
                    all_epochs, picks=picks, epoch_type=settings["epoch_type"], 
                    combine_strategy=settings["combine_strategy"],
                    save=False, bad_channels_strategy=settings["bad_channels_strategy"],
                    threshold=settings["threshold"], data_set=data_name
                )]
            else:
                individual_index = next((i for i, ind in enumerate(all_individuals) if ind.name == selected_individual), None)
                if individual_index is not None:
                    individual_data = all_individuals[individual_index]
                    figures = [epoch_plot(
                        [individual_data.epochs], picks=picks, epoch_type=settings["epoch_type"], 
                        combine_strategy=settings["combine_strategy"],
                        save=False, bad_channels_strategy=settings["bad_channels_strategy"],
                        threshold=settings["threshold"], data_set=data_name
                    )]
        elif settings["plot_type"] == "Standard fNIRS Response Plot":
            selected_individuals = [ind_name for ind_name, var in individual_selection_vars.items() if var.get()]
            selected_all_epochs = []
            # Find the actual individual objects from their names and get their epochs
            for ind_name in selected_individuals:
                # Find the individual object that matches this name
                individual = next((ind for ind in all_individuals if ind.name == ind_name), None)
                
                # If found, append their epochs to our list
                if individual is not None:
                    # Make sure to call the get_epochs method with parentheses
                    selected_all_epochs.append(individual.epochs)
            figures = [standard_fNIRS_response_plot(selected_all_epochs, data_types, bad_channels_strategy=settings["bad_channels_strategy"],
                                                    save=False, combine_strategy=settings["combine_strategy"],
                                                    threshold=settings["threshold"], data_set=data_name, picks_=picks)]
        elif settings["plot_type"] == "paradigm_plot":
            selected_individual = Individual_var.get()
            
            # For paradigm_plot, we need to select channels with appropriate suffix based on haemo_type
            selected_channels = [channel for channel, var in channel_vars.items() if var.get()]
            selected_haemo_type = haemo_type_var.get()  # Get the current hemoglobin type
            
            # Create picks with the correct hemoglobin type suffix
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
            selected_individual = Individual_var.get()
            # Find the individual by name
            index = next((i for i, ind in enumerate(all_individuals) if getattr(ind, "name", f"Participant_{i+1}") == selected_individual), -1)
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

left_container = tk.Frame(root)
left_container.pack(side="left", padx=20, pady=20, fill="y")

top_left_frame = tk.Frame(left_container)
top_left_frame.pack(side="top", fill="y", expand=True)

bottom_left_frame = tk.Frame(left_container)
bottom_left_frame.pack(side="bottom", fill="x")

# Dataset selection
tk.Label(top_left_frame, text="Select Dataset:", font=("Arial", 12)).pack(anchor="w")
dataset_var = tk.StringVar(value=settings["data_set"])
dataset_menu = ttk.Combobox(top_left_frame, textvariable=dataset_var, values=[
    "fNIrs_motor",
    "AudioSpeechNoise",
    "fNirs_motor_full_data",
    "fNIRS_Alexandros_DoC_data",
    "fNIRS_Alexandros_Healthy_data",
    "fNIRS_CUH_patient_data",
    "fNIRS_Melika_hand_data_5Hz_load",
    "fNIRS_Melika_tongue_5Hz_data_load",
    "fNIRS_Melika_old_data",
    "fNIRS_Melika_hand_data_10Hz_load",
    "fNIRS_Melika_tongue_10Hz_data_load",
    "fNIRS_Melika_hand_data_long_load",
    "fNIRS_Melika_tongue_long_data_load",
    "fNIRS_Pardis_DOC_data",
    "fNIRS_Pardis_HC_data",], width=40)

# Dynamically adjust dropdown width
def adjust_menu_width():
    dataset_menu["width"] = max(len(item) for item in dataset_menu["values"])  # Adjust as needed
    
dataset_menu["postcommand"] = adjust_menu_width
dataset_menu.pack(pady=5)

# Epoch type selection
epoch_type_label = tk.Label(top_left_frame, text="Epoch Type:", font=("Arial", 12))  # Define the label as a variable
epoch_type_label.pack(anchor="w")  # Pack the label
epoch_type_var = tk.StringVar()  # Define the StringVar for epoch type
epoch_type_menu = ttk.Combobox(top_left_frame, textvariable=epoch_type_var)  # Create the combobox for epoch type
epoch_type_menu.pack(pady=5)  # Pack the combobox

# Plot type selection
tk.Label(top_left_frame, text="Select Plot Type:", font=("Arial", 12)).pack(anchor="w")
plot_type_var = tk.StringVar(value=settings["plot_type"])
plot_type_menu = ttk.Combobox(top_left_frame, textvariable=plot_type_var, values=[
                                                                              "Epoch Plot",
                                                                              "Standard fNIRS Response Plot",
                                                                              "paradigm_plot",
                                                                              "individual frequency plot",
                                                                              "Statistical Analysis"
                                                                              ])
# Call the function that updates the UI based on the selected plot type
plot_type_menu.pack(pady=5)

# Add hemoglobin type selection (similar to other dropdowns)
haemo_type_label = tk.Label(top_left_frame, text="Hemoglobin Type:", font=("Arial", 12))
haemo_type_var = tk.StringVar(value=settings["haemo_type"])
haemo_type_menu = ttk.Combobox(top_left_frame, textvariable=haemo_type_var, values=["hbo", "hbr"])
haemo_type_label.pack_forget()
haemo_type_menu.pack_forget()

# Individual selection (Initially hidden)
individual_label = tk.Label(top_left_frame, text="Select Individual:", font=("Arial", 12))
Individual_var = tk.StringVar()
individuals_menu = ttk.Combobox(top_left_frame, textvariable=Individual_var)
individual_label.pack_forget()
individuals_menu.pack_forget()

# Combine strategy selection
combine_strategy_label = tk.Label(top_left_frame, text="Combine Strategy:", font=("Arial", 12))
combine_strategy_label.pack(anchor="w")
combine_strategy_var = tk.StringVar(value=settings["combine_strategy"])
combine_strategy_menu = ttk.Combobox(top_left_frame, textvariable=combine_strategy_var, values=["mean", "median", "std", "gfp"])
combine_strategy_menu.pack(pady=5)

# Bad Channels Strategy
bad_channels_strategy_label = tk.Label(top_left_frame, text="Bad Channels Strategy:", font=("Arial", 12))
bad_channels_strategy_label.pack(anchor="w")
bad_channels_strategy_var = tk.StringVar(value=settings["bad_channels_strategy"])
bad_channels_strategy_menu = ttk.Combobox(top_left_frame, textvariable=bad_channels_strategy_var, values=["all", "delete", "threshold"])
bad_channels_strategy_menu.pack(pady=5)

# Short channel correction
short_channel_correction_label = tk.Label(top_left_frame, text="Short Channel Correction:", font=("Arial", 12))
short_channel_correction_label.pack(anchor="w")
short_channel_correction_var = tk.BooleanVar(value=settings["short_channel_correction"])
short_channel_correction_checkbox = tk.Checkbutton(top_left_frame, text="Enable", variable=short_channel_correction_var)
short_channel_correction_checkbox.pack(anchor="w")

# Negative correlation enhancement
negative_correlation_label = tk.Label(top_left_frame, text="Negative Correlation Enhancement:", font=("Arial", 12))
negative_correlation_label.pack(anchor="w")
negative_correlation_enhancement_var = tk.BooleanVar(value=settings["negative_correlation_enhancement"])
negative_correlation_checkbox = tk.Checkbutton(top_left_frame, text="Enable", variable=negative_correlation_enhancement_var)
negative_correlation_checkbox.pack(anchor="w")

# Interpolate bad channels
interpolate_bad_channels_label = tk.Label(top_left_frame, text="Interpolate Bad Channels:", font=("Arial", 12))
interpolate_bad_channels_label.pack(anchor="w")
interpolate_bad_channels_var = tk.BooleanVar(value=settings["interpolate_bad_channels"])
interpolate_bad_channels_checkbox = tk.Checkbutton(top_left_frame, text="Enable", variable=interpolate_bad_channels_var)
interpolate_bad_channels_checkbox.pack(anchor="w")

# Threshold selection
threshold_label = tk.Label(top_left_frame, text="Threshold:", font=("Arial", 12))
threshold_label.pack(anchor="w")
threshold_var = tk.StringVar(value=str(settings["threshold"]))
threshold_entry = tk.Entry(top_left_frame, textvariable=threshold_var)
threshold_entry.pack(pady=5)

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

def populate_channels():
    # Clear existing checkboxes
    for widget in channel_container.winfo_children():
        widget.destroy()
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
            # Get channels for this individual
            current_haemo_type = haemo_type_var.get()
            
            try:
                # Get bad channels for this individual
                bad_channels = selected_individual.epochs.info.get("bads", [])
                
                # Filter channels by hemoglobin type and exclude bad channels
                filtered_channels = [
                    channel for channel in selected_individual.epochs.ch_names 
                    if current_haemo_type in channel.lower() and channel not in bad_channels
                ]
                
                # Create a list of unique channel names without the hemoglobin type suffix
                unique_channels = []
                for channel in filtered_channels:
                    # Remove the hemoglobin type suffix (" hbo" or " hbr")
                    base_channel = channel.rsplit(' ', 1)[0] if ' ' in channel else channel
                    if base_channel not in unique_channels:
                        unique_channels.append(base_channel)
                
                # Create checkboxes for unique channels
                for i, channel in enumerate(unique_channels):
                    is_checked = (i == 0)  # Default: first one checked
                    channel_vars[channel] = tk.BooleanVar(value=is_checked)
                    cb = tk.Checkbutton(channel_container, text=channel, variable=channel_vars[channel])
                    cb.grid(row=i // 3, column=i % 3, sticky="w")
            except Exception as e:
                print(f"Error accessing channels for {selected_individual_name}: {e}")
    elif current_plot_type == "Standard fNIRS Response Plot":
        # Show channel selection
        channel_selection_label.pack(anchor="w")
        channel_frame.pack(fill="x", expand=False)

        # Clear previous checkboxes
        for widget in channel_container.winfo_children():
            widget.destroy()
        channel_vars.clear()

        # Get selected individuals
        selected_names = [name for name, var in individual_selection_vars.items() if var.get()]
        selected_channels = set()

        for name in selected_names:
            individual = next((ind for ind in all_individuals if getattr(ind, "name", "") == name), None)
            if individual and hasattr(individual, "epochs"):
                selected_channels.update(individual.epochs.ch_names)  # Includes bads

        # Populate checkboxes for all found channels
        for i, ch in enumerate(sorted(selected_channels)):
            is_checked = (i == 0)  # Optional: first channel pre-checked
            channel_vars[ch] = tk.BooleanVar(value=is_checked)
            cb = tk.Checkbutton(channel_container, text=ch, variable=channel_vars[ch])
            cb.grid(row=i // 3, column=i % 3, sticky="w")
    
    channel_container.update_idletasks()
    channel_canvas.config(scrollregion=channel_canvas.bbox("all"))




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
    for widget in individual_container.winfo_children():
        widget.destroy()
    
    # Add checkboxes for each individual
    if all_individuals:
        for i, individual in enumerate(all_individuals):
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

# Define setup_ui_callbacks function
def setup_ui_callbacks():
    # Clear any existing traces
    for trace_name in ['write']:
        try:
            dataset_var.trace_remove(trace_name, None)
            plot_type_var.trace_remove(trace_name, None)
            Individual_var.trace_remove(trace_name, None)
            haemo_type_var.trace_remove(trace_name, None)
        except:
            pass
    
    # Set up the main callbacks in the correct order
    dataset_var.trace_add("write", lambda *args: (update_epoch_types(), toggle_individual_menu()))
    plot_type_var.trace_add("write",  lambda *args: (toggle_individual_menu(), populate_channels()))
    Individual_var.trace_add("write", lambda *args: populate_channels())  # Only update channels based on selection
    haemo_type_var.trace_add("write", update_channels_on_haemo_type_change)

# Pack the scrollable frame
channel_canvas.pack(side="left", fill="both", expand=True)
channel_scrollbar.pack(side="right", fill="y")
channel_scrollbar_horizontal.pack(side="bottom", fill="x")

# Pack the individual scrollable frame
individual_canvas.pack(side="left", fill="both", expand=True)
individual_scrollbar.pack(side="right", fill="y")
individual_scrollbar_horizontal.pack(side="bottom", fill="x")

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
dataset1_menu = ttk.Combobox(top_left_frame, textvariable=dataset1_var, values=[
    "fNIrs_motor",
    "AudioSpeechNoise",
    "fNirs_motor_full_data",
    "fNIRS_Alexandros_DoC_data",
    "fNIRS_Alexandros_Healthy_data",
    "fNIRS_CUH_patient_data",
    "fNIRS_Melika_hand_data_5Hz_load",
    "fNIRS_Melika_tongue_5Hz_data_load",
    "fNIRS_Melika_old_data",
    "fNIRS_Melika_hand_data_10Hz_load",
    "fNIRS_Melika_tongue_10Hz_data_load",
    "fNIRS_Melika_hand_data_long_load",
    "fNIRS_Melika_tongue_long_data_load"], width=40)
dataset1_label.pack_forget()  # Initially hidden
dataset1_menu.pack_forget()   # Initially hidden

# Apply the same width adjustment function
def adjust_dataset1_menu_width():
    dataset1_menu["width"] = max(len(item) for item in dataset1_menu["values"])
    
dataset1_menu["postcommand"] = adjust_dataset1_menu_width

dataset2_label = tk.Label(top_left_frame, text="Dataset 2:", font=("Arial", 12))
dataset2_var = tk.StringVar(value="fNIrs_motor")
dataset2_menu = ttk.Combobox(top_left_frame, textvariable=dataset2_var, values=[
    "fNIrs_motor",
    "AudioSpeechNoise",
    "fNirs_motor_full_data",
    "fNIRS_Alexandros_DoC_data",
    "fNIRS_Alexandros_Healthy_data",
    "fNIRS_CUH_patient_data",
    "fNIRS_Melika_hand_data_5Hz_load",
    "fNIRS_Melika_tongue_5Hz_data_load",
    "fNIRS_Melika_old_data",
    "fNIRS_Melika_hand_data_10Hz_load",
    "fNIRS_Melika_tongue_10Hz_data_load",
    "fNIRS_Melika_hand_data_long_load",
    "fNIRS_Melika_tongue_long_data_load"], width=40)
dataset2_label.pack_forget()  # Initially hidden
dataset2_menu.pack_forget()   # Initially hidden

# Apply the same width adjustment function
def adjust_dataset2_menu_width():
    dataset2_menu["width"] = max(len(item) for item in dataset2_menu["values"])
    
def _on_mousewheel(event):
    channel_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

channel_canvas.bind_all("<MouseWheel>", _on_mousewheel)  # For Windows and Mac

dataset2_menu["postcommand"] = adjust_dataset2_menu_width

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