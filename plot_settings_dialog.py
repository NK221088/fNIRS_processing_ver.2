import tkinter as tk
from tkinter import ttk
import tkinter.messagebox
from preprocessesing_toolbox.plot_settings_info import PlotSettingsInfo


class PlotSettingsDialog:
    def __init__(self, parent, current_settings):
        self.parent = parent
        self.result = None
        self.settings = current_settings.copy()
        
        # Create the dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Plot Settings")
        self.dialog.geometry("450x450")  # Reduced height since we removed file format section
        self.dialog.resizable(False, False)
        
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
        # Update parent to get current geometry
        self.parent.update_idletasks()
        
        # Get parent window position and size
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        # Calculate center position
        dialog_width = 450
        dialog_height = 450  # Updated height
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        self.dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
    
    def validate_threshold(self, value):
        """Validate threshold input."""
        if value == "":
            return True  # Allow empty string during typing
        try:
            num = float(value)
            return num >= 0
        except ValueError:
            return False
    
    def create_widgets(self):
        """Create the dialog widgets."""
        # Main frame with scrollbar
        main_frame = tk.Frame(self.dialog)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Create a canvas and scrollbar for scrolling
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Title
        title_label = tk.Label(scrollable_frame, text="Plot Settings", 
                              font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Create sections
        self.create_processing_section(scrollable_frame)
        self.create_save_section(scrollable_frame)
        
        # Separator
        separator = ttk.Separator(scrollable_frame, orient="horizontal")
        separator.pack(fill="x", pady=20)
        
        # Buttons frame
        buttons_frame = tk.Frame(scrollable_frame)
        buttons_frame.pack(fill="x", pady=10)
        
        # Cancel button
        cancel_btn = tk.Button(
            buttons_frame, 
            text="Cancel",
            command=self.cancel,
            width=10,
            font=("Arial", 11)
        )
        cancel_btn.pack(side="left", padx=(0, 10))
        
        # Reset to defaults button
        reset_btn = tk.Button(
            buttons_frame, 
            text="Reset to Defaults",
            command=self.reset_to_defaults,
            width=15,
            font=("Arial", 11)
        )
        reset_btn.pack(side="left", padx=(0, 10))
        
        # OK button
        ok_btn = tk.Button(
            buttons_frame, 
            text="OK",
            command=self.ok,
            width=10,
            bg="green",
            fg="white",
            font=("Arial", 11, "bold")
        )
        ok_btn.pack(side="right")
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind Enter and Escape keys
        self.dialog.bind('<Return>', lambda e: self.ok())
        self.dialog.bind('<Escape>', lambda e: self.cancel())
        
        # Set focus to OK button
        ok_btn.focus_set()
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            if canvas.winfo_exists():  # Check if canvas still exists
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        self._on_mousewheel = _on_mousewheel  # Store reference for later cleanup
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    def create_processing_section(self, parent):
        """Create data processing settings section."""
        # Processing Settings Frame
        processing_frame = tk.LabelFrame(parent, text="Data Processing", font=("Arial", 12, "bold"))
        processing_frame.pack(fill="x", pady=(0, 15))
        
        # Combine Strategy
        combine_frame = tk.Frame(processing_frame)
        combine_frame.pack(fill="x", padx=10, pady=5)
        
        combine_label = tk.Label(combine_frame, text="Combine Strategy:", font=("Arial", 11))
        combine_label.pack(side="left")
        
        info_button1 = tk.Button(
            combine_frame, 
            text="?", 
            command=PlotSettingsInfo.show_combine_strategy_info,
            width=2,
            height=1,
            bg="lightblue",
            font=("Arial", 8)
        )
        info_button1.pack(side="right")
        
        self.combine_strategy_var = tk.StringVar(value=self.settings.get("combine_strategy", "mean"))
        self.combine_strategy_combo = ttk.Combobox(
            processing_frame, 
            textvariable=self.combine_strategy_var, 
            values=["mean", "median", "gfp"], 
            state="readonly",
            font=("Arial", 10)
        )
        self.combine_strategy_combo.pack(fill="x", padx=10, pady=(0, 10))
        
        # Bad Channels Strategy
        bad_channels_frame = tk.Frame(processing_frame)
        bad_channels_frame.pack(fill="x", padx=10, pady=5)
        
        bad_channels_label = tk.Label(bad_channels_frame, text="Bad Channels Strategy:", font=("Arial", 11))
        bad_channels_label.pack(side="left")
        
        info_button2 = tk.Button(
            bad_channels_frame, 
            text="?", 
            command=PlotSettingsInfo.show_bad_channels_info,
            width=2,
            height=1,
            bg="lightblue",
            font=("Arial", 8)
        )
        info_button2.pack(side="right")
        
        self.bad_channels_strategy_var = tk.StringVar(value=self.settings.get("bad_channels_strategy", "all"))
        self.bad_channels_strategy_combo = ttk.Combobox(
            processing_frame, 
            textvariable=self.bad_channels_strategy_var, 
            values=["all", "delete", "threshold"], 
            state="readonly",
            font=("Arial", 10)
        )
        self.bad_channels_strategy_combo.pack(fill="x", padx=10, pady=(0, 10))
        
        # Interpolate Bad Channels
        interpolate_frame = tk.Frame(processing_frame)
        interpolate_frame.pack(fill="x", padx=10, pady=5)
        
        self.interpolate_bad_channels_var = tk.BooleanVar(value=self.settings.get("interpolate_bad_channels", False))
        self.interpolate_checkbox = tk.Checkbutton(
            interpolate_frame,
            text="Interpolate Bad Channels",
            variable=self.interpolate_bad_channels_var,
            font=("Arial", 11)
        )
        self.interpolate_checkbox.pack(side="left")
        
        info_button3 = tk.Button(
            interpolate_frame, 
            text="?", 
            command=PlotSettingsInfo.show_interpolate_info,
            width=2,
            height=1,
            bg="lightblue",
            font=("Arial", 8)
        )
        info_button3.pack(side="right")
        
        # Threshold
        threshold_frame = tk.Frame(processing_frame)
        threshold_frame.pack(fill="x", padx=10, pady=5)
        
        threshold_label = tk.Label(threshold_frame, text="Threshold:", font=("Arial", 11))
        threshold_label.pack(side="left")
        
        info_button4 = tk.Button(
            threshold_frame, 
            text="?", 
            command=PlotSettingsInfo.show_threshold_info,
            width=2,
            height=1,
            bg="lightblue",
            font=("Arial", 8)
        )
        info_button4.pack(side="right")
        
        # Register validation function
        vcmd_threshold = (self.dialog.register(self.validate_threshold), '%P')
        
        self.threshold_var = tk.StringVar(value=str(self.settings.get("threshold", 3)))
        self.threshold_entry = tk.Entry(
            threshold_frame, 
            textvariable=self.threshold_var, 
            width=10,
            font=("Arial", 10),
            validate='key',
            validatecommand=vcmd_threshold
        )
        self.threshold_entry.pack(side="left", padx=(10, 0))
    
    def create_save_section(self, parent):
        """Create save plot settings section."""
        # Save Settings Frame
        save_frame = tk.LabelFrame(parent, text="Save Options", font=("Arial", 12, "bold"))
        save_frame.pack(fill="x", pady=(0, 15))
        
        # Save Plot Checkbox
        save_plot_frame = tk.Frame(save_frame)
        save_plot_frame.pack(fill="x", padx=10, pady=10)
        
        self.save_plot_var = tk.BooleanVar(value=self.settings.get("save_plot", False))
        self.save_plot_checkbox = tk.Checkbutton(
            save_plot_frame,
            text="Save Plot to File",
            variable=self.save_plot_var,
            font=("Arial", 11)
        )
        self.save_plot_checkbox.pack(side="left")
        
        save_info_btn = tk.Button(
            save_plot_frame, 
            text="?", 
            command=PlotSettingsInfo.show_save_plot_info,
            width=2,
            height=1,
            bg="lightblue",
            font=("Arial", 8)
        )
        save_info_btn.pack(side="right")
    
    def reset_to_defaults(self):
        """Reset all settings to their default values."""
        self.combine_strategy_var.set("mean")
        self.bad_channels_strategy_var.set("all")
        self.interpolate_bad_channels_var.set(False)
        self.threshold_var.set("3")
        self.save_plot_var.set(False)
    
    def validate_inputs(self):
        """Validate all input fields before accepting."""
        # Validate threshold
        threshold_str = self.threshold_var.get().strip()
        if not threshold_str:
            tkinter.messagebox.showerror("Invalid Input", "Please enter a value for threshold.")
            return False
        try:
            threshold_value = float(threshold_str)
            if threshold_value < 0:
                tkinter.messagebox.showerror("Invalid Input", "Threshold must be a positive number.")
                return False
        except ValueError:
            tkinter.messagebox.showerror("Invalid Input", "Threshold must be a valid number.")
            return False
        
        return True
    
    def ok(self):
        """Handle OK button click."""
        # Validate inputs
        if not self.validate_inputs():
            return
        
        # Prepare result
        self.result = {
            "combine_strategy": self.combine_strategy_var.get(),
            "bad_channels_strategy": self.bad_channels_strategy_var.get(),
            "interpolate_bad_channels": self.interpolate_bad_channels_var.get(),
            "threshold": float(self.threshold_var.get().strip()),
            "save_plot": self.save_plot_var.get()
        }
        
        self.dialog.unbind_all("<MouseWheel>")  # Cleanup mousewheel binding
        self.dialog.destroy()
    
    def cancel(self):
        """Handle Cancel button click."""
        self.result = None
        self.dialog.unbind_all("<MouseWheel>")  # Cleanup mousewheel binding
        self.dialog.destroy()
    
    def get_result(self):
        """Return the result of the dialog."""
        return self.result


def show_plot_settings_dialog(parent, current_settings):
    """
    Show the plot settings dialog and return the selected settings.
    
    Args:
        parent: The parent window
        current_settings: Dictionary with current plot settings
        
    Returns:
        Dictionary with updated settings or None if cancelled
    """
    dialog = PlotSettingsDialog(parent, current_settings)
    return dialog.get_result()