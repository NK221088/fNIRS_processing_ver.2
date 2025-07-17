import tkinter as tk
from tkinter import ttk
import tkinter.messagebox


class PlotSettingsDialog:
    def __init__(self, parent, current_settings, data_types=None, all_individuals=None):
        self.parent = parent
        self.result = None
        self.settings = current_settings.copy()
        self.data_types = data_types or []
        self.all_individuals = all_individuals or []
        
        # Create the dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Plot Settings")
        self.dialog.geometry("450x600")
        self.dialog.resizable(True, True)  # Allow resizing
        
        # Make dialog modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog on parent window
        self.center_dialog()
        
        # Create the UI
        self.create_widgets()
        
        # Wait for dialog to close
        self.dialog.wait_window()
    
    def center_dialog(self):
        """Center the dialog on the parent window."""
        self.dialog.update_idletasks()
        x = self.parent.winfo_x() + (self.parent.winfo_width() // 2) - (self.dialog.winfo_width() // 2)
        y = self.parent.winfo_y() + (self.parent.winfo_height() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f"+{x}+{y}")
    
    def create_widgets(self):
        """Create all the widgets for the dialog."""
        # Create buttons frame first (at bottom)
        self.create_buttons()
        
        # Create scrollable area frame
        scroll_frame = tk.Frame(self.dialog)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create canvas and scrollbar
        canvas = tk.Canvas(scroll_frame)
        scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        # Configure scrolling
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Main content frame (inside scrollable frame)
        main_frame = tk.Frame(scrollable_frame)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(main_frame, text="Plot Settings", font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Create sections
        self.create_epoch_section(main_frame)
        self.create_individual_section(main_frame)
        self.create_processing_section(main_frame)
        self.create_channel_section(main_frame)
        
        # Bind mousewheel to canvas for scrolling
        self.bind_mousewheel(canvas)
    
    def bind_mousewheel(self, canvas):
        """Bind mousewheel events to canvas for scrolling."""
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        canvas.bind('<Enter>', _bind_to_mousewheel)
        canvas.bind('<Leave>', _unbind_from_mousewheel)
    
    def create_epoch_section(self, parent):
        """Create epoch type selection section."""
        # Epoch Type Frame
        epoch_frame = tk.LabelFrame(parent, text="Epoch Settings", font=("Arial", 12, "bold"))
        epoch_frame.pack(fill="x", pady=(0, 15))
        
        # Epoch Type Selection
        tk.Label(epoch_frame, text="Epoch Type:", font=("Arial", 11)).pack(anchor="w", padx=10, pady=(10, 5))
        
        self.epoch_type_var = tk.StringVar(value=self.settings.get("epoch_type", ""))
        self.epoch_type_combo = ttk.Combobox(
            epoch_frame, 
            textvariable=self.epoch_type_var, 
            values=self.data_types,
            state="readonly"
        )
        self.epoch_type_combo.pack(fill="x", padx=10, pady=(0, 10))
        
        # Update the values when dialog opens
        self.refresh_epoch_types()
    
    def create_individual_section(self, parent):
        """Create individual selection section."""
        # Individual Selection Frame
        individual_frame = tk.LabelFrame(parent, text="Individual Selection", font=("Arial", 12, "bold"))
        individual_frame.pack(fill="x", pady=(0, 15))
        
        # Individual Selection
        tk.Label(individual_frame, text="Select Individual:", font=("Arial", 11)).pack(anchor="w", padx=10, pady=(10, 5))
        
        # Create individual options list
        individual_options = ["All Individuals"]
        individual_options.extend([
            getattr(ind, "name", f"Participant_{i+1}") 
            for i, ind in enumerate(self.all_individuals)
        ])
        
        self.individual_var = tk.StringVar(value=self.settings.get("individual", "All Individuals"))
        self.individual_combo = ttk.Combobox(
            individual_frame, 
            textvariable=self.individual_var, 
            values=individual_options, 
            state="readonly"
        )
        self.individual_combo.pack(fill="x", padx=10, pady=(0, 10))
    
    def create_processing_section(self, parent):
        """Create data processing settings section."""
        # Processing Settings Frame
        processing_frame = tk.LabelFrame(parent, text="Data Processing", font=("Arial", 12, "bold"))
        processing_frame.pack(fill="x", pady=(0, 15))
        
        # Combine Strategy
        tk.Label(processing_frame, text="Combine Strategy:", font=("Arial", 11)).pack(anchor="w", padx=10, pady=(10, 5))
        
        self.combine_strategy_var = tk.StringVar(value=self.settings.get("combine_strategy", "mean"))
        self.combine_strategy_combo = ttk.Combobox(
            processing_frame, 
            textvariable=self.combine_strategy_var, 
            values=["mean", "median", "gfp"], 
            state="readonly"
        )
        self.combine_strategy_combo.pack(fill="x", padx=10, pady=(0, 10))
        
        # Bad Channels Strategy
        tk.Label(processing_frame, text="Bad Channels Strategy:", font=("Arial", 11)).pack(anchor="w", padx=10, pady=(5, 5))
        
        self.bad_channels_strategy_var = tk.StringVar(value=self.settings.get("bad_channels_strategy", "all"))
        self.bad_channels_strategy_combo = ttk.Combobox(
            processing_frame, 
            textvariable=self.bad_channels_strategy_var, 
            values=["all", "delete", "threshold"], 
            state="readonly"
        )
        self.bad_channels_strategy_combo.pack(fill="x", padx=10, pady=(0, 10))
        
        # Interpolate Bad Channels
        self.interpolate_bad_channels_var = tk.BooleanVar(value=self.settings.get("interpolate_bad_channels", False))
        self.interpolate_checkbox = tk.Checkbutton(
            processing_frame,
            text="Interpolate Bad Channels",
            variable=self.interpolate_bad_channels_var,
            font=("Arial", 11)
        )
        self.interpolate_checkbox.pack(anchor="w", padx=10, pady=(5, 10))
        
        # Threshold
        threshold_frame = tk.Frame(processing_frame)
        threshold_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        tk.Label(threshold_frame, text="Threshold:", font=("Arial", 11)).pack(side="left")
        
        self.threshold_var = tk.StringVar(value=str(self.settings.get("threshold", 3)))
        self.threshold_entry = tk.Entry(threshold_frame, textvariable=self.threshold_var, width=10)
        self.threshold_entry.pack(side="left", padx=(10, 0))
        
        # Add validation for threshold entry
        self.threshold_entry.bind('<FocusOut>', self.validate_threshold)
        self.threshold_entry.bind('<Return>', self.validate_threshold)
    
    def create_channel_section(self, parent):
        """Create channel selection section placeholder."""
        # Channel Selection Frame (placeholder for future expansion)
        channel_frame = tk.LabelFrame(parent, text="Channel Selection", font=("Arial", 12, "bold"))
        channel_frame.pack(fill="x", pady=(0, 15))
        
        info_label = tk.Label(
            channel_frame, 
            text="Channel selection is handled in the main interface",
            font=("Arial", 10),
            fg="gray"
        )
        info_label.pack(padx=10, pady=10)
    
    def create_buttons(self):
        """Create OK and Cancel buttons at the bottom of the dialog."""
        # Create a fixed frame at the bottom for buttons
        button_frame = tk.Frame(self.dialog, bg="lightgray", relief="raised", bd=1)
        button_frame.pack(side="bottom", fill="x", padx=0, pady=0)
        
        # Inner frame for proper padding
        inner_frame = tk.Frame(button_frame, bg="lightgray")
        inner_frame.pack(fill="x", padx=20, pady=10)
        
        # Cancel button
        cancel_button = tk.Button(
            inner_frame,
            text="Cancel",
            command=self.cancel,
            font=("Arial", 11),
            padx=20,
            pady=5
        )
        cancel_button.pack(side="right", padx=(10, 0))
        
        # OK button
        ok_button = tk.Button(
            inner_frame,
            text="OK",
            command=self.ok,
            font=("Arial", 11, "bold"),
            bg="lightblue",
            padx=20,
            pady=5
        )
        ok_button.pack(side="right")
        
        # Make OK button the default
        self.dialog.bind('<Return>', lambda e: self.ok())
        self.dialog.bind('<Escape>', lambda e: self.cancel())
    
    def validate_threshold(self, event=None):
        """Validate threshold input."""
        try:
            threshold_value = float(self.threshold_var.get())
            if threshold_value < 0:
                tk.messagebox.showerror("Invalid Input", "Threshold must be a positive number.")
                self.threshold_var.set("3")  # Reset to default
                return False
        except ValueError:
            tk.messagebox.showerror("Invalid Input", "Threshold must be a valid number.")
            self.threshold_var.set("3")  # Reset to default
            return False
        return True
    
    def ok(self):
        """Handle OK button click."""
        # Validate inputs
        if not self.validate_threshold():
            return
        
        # Prepare result
        self.result = {
            "epoch_type": self.epoch_type_var.get(),
            "individual": self.individual_var.get(),
            "combine_strategy": self.combine_strategy_var.get(),
            "bad_channels_strategy": self.bad_channels_strategy_var.get(),
            "interpolate_bad_channels": self.interpolate_bad_channels_var.get(),
            "threshold": int(float(self.threshold_var.get()))
        }
        
        self.dialog.destroy()
    
    def cancel(self):
        """Handle Cancel button click."""
        self.result = None
        self.dialog.destroy()
    
    def refresh_epoch_types(self):
        """Refresh the epoch type dropdown with current data types."""
        if self.data_types:
            self.epoch_type_combo['values'] = self.data_types
            # If current selection is not in new data types, reset to first available
            current_selection = self.epoch_type_var.get()
            if current_selection not in self.data_types and self.data_types:
                self.epoch_type_var.set(self.data_types[0])


def show_plot_settings_dialog(parent, current_settings, data_types=None, all_individuals=None):
    """
    Show the plot settings dialog.
    
    Args:
        parent: Parent window
        current_settings: Dictionary with current plot settings
        data_types: List of available epoch types
        all_individuals: List of all individuals for selection
        
    Returns:
        Dictionary with updated settings or None if cancelled
    """
    dialog = PlotSettingsDialog(parent, current_settings, data_types, all_individuals)
    return dialog.result

