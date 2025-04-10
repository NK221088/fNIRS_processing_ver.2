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
    "negative_correlation_enhancement": True,
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

# Replace your toggle_individual_menu function with this improved version
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
        # Show epoch type selection
        epoch_type_label.pack(anchor="w")
        epoch_type_menu.pack(pady=5)
        # Show individual checkboxes
        individual_selection_label.pack(anchor="w")
        individual_selection_frame.pack(fill="x", expand=True)
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
        
    elif plot_type == "paradigm_plot":
        # Show hemoglobin type selection
        haemo_type_label.pack(anchor="w")
        haemo_type_menu.pack(pady=5)
        # Show individual selection dropdown
        individual_label.pack(anchor="w")
        individuals_menu.pack(pady=5)
        # Show channel selection
        channel_selection_label.pack(anchor="w")
        channel_frame.pack(fill="x", expand=True)
        # Populate the channel checkboxes
        populate_channels()
        
    elif plot_type == "Epoch Plot":
        # Show epoch type selection
        epoch_type_label.pack(anchor="w")
        epoch_type_menu.pack(pady=5)
        # Show individual selection dropdown
        individual_label.pack(anchor="w")
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
    


import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Updated run_analysis function
def run_analysis():
    """Run data processing and visualization based on selected plot type."""

    global previous_epoch_type, all_epochs, data_name, all_data, freq, data_types, all_individuals, first_data_load

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
        (settings["plot_type"] not in ["individual frequency plot", "paradigm_plot", "Epoch Plot", "Standard fNIRS Response Plot"])
        or settings["epoch_type"] != previous_epoch_type  or first_data_load == True # Reload only if epoch type changed
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
        toggle_individual_menu()
    
    first_data_load = False

    # Clear previous plots
    for widget in right_frame.winfo_children():
        widget.destroy()

    # Modify channel selection for plotting
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
            dataset2=dataset2
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
            
            # Use the global all_individuals list
            individual_index = int(selected_individual.strip("Participant_")) - 1
            
            # Pass the hemoglobin type to the paradigm_plot function
            index = next((i for i, ind in enumerate(all_individuals) if ind.get_name() == selected_individual), -1)

            figures = [paradigm_plot(
                all_individuals[index], 
                picks_=picks, 
                haemo_type=settings["haemo_type"]
            )]

        elif settings["plot_type"] == "individual frequency plot":
            selected_individual = Individual_var.get()
            figures = [individual_frequency_plot(all_individuals[int(selected_individual.strip("Participant_"))-1])]
        
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

# Dynamically adjust dropdown width
def adjust_menu_width():
    dataset_menu["width"] = max(len(item) for item in dataset_menu["values"])  # Adjust as needed
    
dataset_menu["postcommand"] = adjust_menu_width


dataset_menu.pack(pady=5)
dataset_var.trace_add("write", lambda *args: (update_epoch_types(), toggle_individual_menu()))# dataset_var.trace_add("write", lambda *args: (update_epoch_types(), toggle_individual_menu()))

# Epoch type selection
epoch_type_label = tk.Label(left_frame, text="Epoch Type:", font=("Arial", 12))  # Define the label as a variable
epoch_type_label.pack(anchor="w")  # Pack the label
epoch_type_var = tk.StringVar()  # Define the StringVar for epoch type
epoch_type_menu = ttk.Combobox(left_frame, textvariable=epoch_type_var)  # Create the combobox for epoch type
epoch_type_menu.pack(pady=5)  # Pack the combobox

# Plot type selection
tk.Label(left_frame, text="Select Plot Type:", font=("Arial", 12)).pack(anchor="w")
plot_type_var = tk.StringVar(value=settings["plot_type"])
plot_type_menu = ttk.Combobox(left_frame, textvariable=plot_type_var, values=[
                                                                                "Epoch Plot",
                                                                                "Standard fNIRS Response Plot",
                                                                                "paradigm_plot",
                                                                                "individual frequency plot",
                                                                                "Statistical Analysis"
                                                                                ])
# Call the function that updates the UI based on the selected plot type
plot_type_menu.pack(pady=5)
plot_type_var.trace_add("write", toggle_individual_menu)

# Add hemoglobin type selection (similar to other dropdowns)
haemo_type_label = tk.Label(left_frame, text="Hemoglobin Type:", font=("Arial", 12))
haemo_type_var = tk.StringVar(value=settings["haemo_type"])
haemo_type_menu = ttk.Combobox(left_frame, textvariable=haemo_type_var, values=["hbo", "hbr"])
haemo_type_label.pack_forget()
haemo_type_menu.pack_forget()

# Individual selection (Initially hidden)
individual_label = tk.Label(left_frame, text="Select Individual:", font=("Arial", 12))
Individual_var = tk.StringVar()
individuals_menu = ttk.Combobox(left_frame, textvariable=Individual_var)
individual_label.pack_forget()
individuals_menu.pack_forget()

# Combine strategy selection
combine_strategy_label = tk.Label(left_frame, text="Combine Strategy:", font=("Arial", 12))
combine_strategy_label.pack(anchor="w")
combine_strategy_var = tk.StringVar(value=settings["combine_strategy"])
combine_strategy_menu = ttk.Combobox(left_frame, textvariable=combine_strategy_var, values=["mean", "median", "std", "gfp"])
combine_strategy_menu.pack(pady=5)

# Bad Channels Strategy
bad_channels_strategy_label = tk.Label(left_frame, text="Bad Channels Strategy:", font=("Arial", 12))
bad_channels_strategy_label.pack(anchor="w")
bad_channels_strategy_var = tk.StringVar(value=settings["bad_channels_strategy"])
bad_channels_strategy_menu = ttk.Combobox(left_frame, textvariable=bad_channels_strategy_var, values=["all", "delete", "threshold"])
bad_channels_strategy_menu.pack(pady=5)

# Short channel correction
short_channel_correction_label = tk.Label(left_frame, text="Short Channel Correction:", font=("Arial", 12))
short_channel_correction_label.pack(anchor="w")
short_channel_correction_var = tk.BooleanVar(value=settings["short_channel_correction"])
short_channel_correction_checkbox = tk.Checkbutton(left_frame, text="Enable", variable=short_channel_correction_var)
short_channel_correction_checkbox.pack(anchor="w")

# Negative correlation enhancement
negative_correlation_label = tk.Label(left_frame, text="Negative Correlation Enhancement:", font=("Arial", 12))
negative_correlation_label.pack(anchor="w")
negative_correlation_enhancement_var = tk.BooleanVar(value=settings["negative_correlation_enhancement"])
negative_correlation_checkbox = tk.Checkbutton(left_frame, text="Enable", variable=negative_correlation_enhancement_var)
negative_correlation_checkbox.pack(anchor="w")

# Interpolate bad channels
interpolate_bad_channels_label = tk.Label(left_frame, text="Interpolate Bad Channels:", font=("Arial", 12))
interpolate_bad_channels_label.pack(anchor="w")
interpolate_bad_channels_var = tk.BooleanVar(value=settings["interpolate_bad_channels"])
interpolate_bad_channels_checkbox = tk.Checkbutton(left_frame, text="Enable", variable=interpolate_bad_channels_var)
interpolate_bad_channels_checkbox.pack(anchor="w")

# Threshold selection
threshold_label = tk.Label(left_frame, text="Threshold:", font=("Arial", 12))
threshold_label.pack(anchor="w")
threshold_var = tk.StringVar(value=str(settings["threshold"]))
threshold_entry = tk.Entry(left_frame, textvariable=threshold_var)
threshold_entry.pack(pady=5)

# Channel selection
channel_selection_label = tk.Label(left_frame, text="Select Channels:", font=("Arial", 12))
channel_selection_label.pack(anchor="w")

# Create a frame to hold the channel checkboxes with a scrollbar
channel_frame = tk.Frame(left_frame)
channel_frame.pack(fill="x", expand=True)

channel_canvas = tk.Canvas(channel_frame)
channel_scrollbar = tk.Scrollbar(channel_frame, orient="vertical", command=channel_canvas.yview)
channel_scrollbar_horizontal = tk.Scrollbar(channel_frame, orient="horizontal", command=channel_canvas.xview)

channel_container = tk.Frame(channel_canvas)

channel_canvas.create_window((0, 0), window=channel_container, anchor="nw")
channel_canvas.configure(yscrollcommand=channel_scrollbar.set, xscrollcommand=channel_scrollbar_horizontal.set)


# Variable to track channel selections
channel_vars = {}

# Individual selection checkboxes
individual_selection_label = tk.Label(left_frame, text="Select Individuals:", font=("Arial", 12))
individual_selection_label.pack(anchor="w")

# Create a frame to hold the individual checkboxes with a scrollbar
individual_selection_frame = tk.Frame(left_frame)
individual_selection_frame.pack(fill="x", expand=True)

individual_canvas = tk.Canvas(individual_selection_frame)
individual_scrollbar = tk.Scrollbar(individual_selection_frame, orient="vertical", command=individual_canvas.yview)
individual_scrollbar_horizontal = tk.Scrollbar(individual_selection_frame, orient="horizontal", command=individual_canvas.xview)

individual_container = tk.Frame(individual_canvas)

individual_canvas.create_window((0, 0), window=individual_container, anchor="nw")
individual_canvas.configure(yscrollcommand=individual_scrollbar.set, xscrollcommand=individual_scrollbar_horizontal.set)

# Variable to track individual selections
individual_selection_vars = {}

def populate_channels():
    # Don't show channel selection for certain plot types
    if plot_type_var.get() not in ["paradigm_plot", "Standard fNIRS Response Plot"]:
        channel_selection_label.pack_forget()
        channel_frame.pack_forget()
        return
        
    # Clear existing checkboxes
    for widget in channel_container.winfo_children():
        widget.destroy()

    # Reset channel variables
    channel_vars.clear()
    
    # Show channel selection
    channel_selection_label.pack(anchor="w")
    channel_frame.pack(fill="x", expand=True)

    # Get available channels based on plot type and selections
    selected_individual_name = Individual_var.get()
    
    # Find the selected individual
    if not all_individuals:
        return  # Exit if no data is loaded yet
        
    if selected_individual_name == "All Individuals" and all_individuals:
        # Use the first individual as reference for channels
        selected_individual = all_individuals[0]
    else:
        selected_individual = next((ind for ind in all_individuals if getattr(ind, "name", "") == selected_individual_name), 
                                  all_individuals[0] if all_individuals else None)

    # Populate channel checkboxes if an individual is found
    if selected_individual and hasattr(selected_individual, 'epochs'):
        try:
            # Get channels for the selected epoch type
            current_epoch_type = epoch_type_var.get()
            if current_epoch_type in selected_individual.epochs:
                epochs = selected_individual.epochs[current_epoch_type]
                
                # Filter channels based on plot type
                if plot_type_var.get() == "paradigm_plot":
                    current_haemo_type = haemo_type_var.get()
                    channels = [ch for ch in epochs.ch_names if current_haemo_type in ch.lower()]
                elif plot_type_var.get() == "Standard fNIRS Response Plot":
                    channels = list(set([s.removesuffix(" hbo").removesuffix(" hbr") for s in epochs.ch_names]))
                else:
                    channels = epochs.ch_names
                    
                # Create checkboxes for filtered channels
                for i, channel in enumerate(channels):
                    is_checked = (i == 0)
                    channel_vars[channel] = tk.BooleanVar(value=is_checked)
                    cb = tk.Checkbutton(channel_container, text=channel, variable=channel_vars[channel])
                    cb.grid(row=i // 3, column=i % 3, sticky="w")
        except Exception as e:
            print(f"Error populating channels: {e}")

    # Update scroll region
    channel_container.update_idletasks()
    channel_canvas.config(scrollregion=channel_canvas.bbox("all"))

# Add a trace to hemoglobin type to update channels when it changes
def update_channels_on_haemo_type_change(*args):
    if plot_type_var.get() == "paradigm_plot":
        populate_channels()

haemo_type_var.trace_add("write", update_channels_on_haemo_type_change)

def populate_individuals():
    # Store current selections before clearing
    current_selections = {name: var.get() for name, var in individual_selection_vars.items()}
    
    # Clear existing checkboxes
    for widget in individual_container.winfo_children():
        widget.destroy()
    
    # Show individual selection
    individual_selection_label.pack(anchor="w")
    individual_selection_frame.pack(fill="x", expand=True)

    # Create checkboxes for individuals
    for i, individual in enumerate(all_individuals):
        individual_name = getattr(individual, "name", f"Participant {i+1}")
        
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

# Modify the existing traces to prevent multiple updates
def combined_update(*args):
    # Remove existing traces to prevent multiple calls
    Individual_var.trace_remove("write", update_traces[0])
    
    # Perform updates
    update_epoch_types()
    toggle_individual_menu()
    populate_channels()

    # Add hemoglobin type visibility check
    if plot_type_var.get() != "paradigm_plot":
        # Hide hemoglobin type selection
        haemo_type_label.pack_forget()
        haemo_type_menu.pack_forget()

# Store the trace ID to allow removal
update_traces = []
trace_id = Individual_var.trace_add("write", combined_update)
update_traces.append(trace_id)

# Add trace to Individual_var to update channels when individual changes
Individual_var.trace_add("write", lambda *args: populate_channels())

# Pack the scrollable frame
channel_canvas.pack(side="left", fill="both", expand=True)
channel_scrollbar.pack(side="right", fill="y")
channel_scrollbar_horizontal.pack(side="bottom", fill="x")

# Add this to update channels when dataset changes
def update_channel_selection(*args):
    populate_channels()

# Trace the dataset variable to update channel selection
dataset_var.trace_add("write", update_channel_selection)

# Initial population
populate_channels()

# Pack the individual scrollable frame
individual_canvas.pack(side="left", fill="both", expand=True)
individual_scrollbar.pack(side="right", fill="y")
individual_scrollbar_horizontal.pack(side="bottom", fill="x")

# Area of Interest selection for Statistical Analysis
area_of_interest_label = tk.Label(left_frame, text="Area of Interest:", font=("Arial", 12))
area_of_interest_var = tk.StringVar(value="SMA")
area_of_interest_menu = ttk.Combobox(left_frame, textvariable=area_of_interest_var, 
                                     values=["SMA", "Tongue_all", "Tongue_right", "Tongue_left", 
                                             "Hand_all", "Hand_right", "Hand_left"])
area_of_interest_label.pack_forget()  # Initially hidden
area_of_interest_menu.pack_forget()   # Initially hidden

# Time window for Statistical Analysis
time_window_label = tk.Label(left_frame, text="Time Window (seconds):", font=("Arial", 12))
time_window_frame = tk.Frame(left_frame)
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
dataset1_label = tk.Label(left_frame, text="Dataset 1:", font=("Arial", 12))
dataset1_var = tk.StringVar(value=settings["data_set"])
dataset1_menu = ttk.Combobox(left_frame, textvariable=dataset1_var, values=[
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

dataset2_label = tk.Label(left_frame, text="Dataset 2:", font=("Arial", 12))
dataset2_var = tk.StringVar(value="fNIrs_motor")
dataset2_menu = ttk.Combobox(left_frame, textvariable=dataset2_var, values=[
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
    
dataset2_menu["postcommand"] = adjust_dataset2_menu_width

# Run Analysis button
run_button = tk.Button(left_frame, text="Run Analysis", command=run_analysis, bg="green", fg="white")
run_button.pack(pady=10)

# Right panel for displaying the plot
right_frame = tk.Frame(root)
right_frame.pack(side="right", padx=20, pady=20, expand=True, fill="both")

# Add trace to Individual_var to update individual checkboxes when selection changes
Individual_var.trace_add("write", lambda *args: populate_individuals())
# Initial population of individuals
populate_individuals()

# Replace your combined_update function and update_traces approach with this:
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
    dataset_var.trace_add("write", update_epoch_types)
    plot_type_var.trace_add("write", toggle_individual_menu)
    Individual_var.trace_add("write", populate_channels)
    haemo_type_var.trace_add("write", populate_channels)

# Call this function at the end of your initialization code
setup_ui_callbacks()

# Initialize GUI
update_epoch_types()
toggle_individual_menu()
root.mainloop()
