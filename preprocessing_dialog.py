import tkinter as tk
from tkinter import ttk
import tkinter.messagebox
from preprocessesing_toolbox.baselineCorrection import baselineCorrection

class PreprocessingDialog:
    def __init__(self, parent, current_settings):
        self.parent = parent
        self.result = None
        self.settings = current_settings.copy()
        
        # Create the dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Preprocessing Options")
        self.dialog.geometry("400x400")  # Increased height for additional fields
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
    
    def get_baseline_correction_methods(self):
        """Get available baseline correction methods from the baselineCorrection class."""
        # Create a temporary instance to get the available methods
        temp_corrector = baselineCorrection("temp")
        return temp_corrector.get_available_methods()
    
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
        dialog_width = 400
        dialog_height = 400  # Updated height
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        self.dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
    
    def validate_tmin_input(self, value):
        """Validate that tmin input is a valid integer (positive or negative)."""
        if value == "" or value == "-":
            return True  # Allow empty string and lone minus sign during typing
        try:
            int(value)
            return True
        except ValueError:
            return False
    
    def on_baseline_method_change(self, event=None):
        """Handle changes in baseline correction method selection."""
        selected_method = self.baseline_correction_var.get()
        
        if selected_method == "xSecondsBefore":
            # Show tmin input field
            self.tmin_frame.pack(fill="x", pady=5, after=self.baseline_correction_frame)
        else:
            # Hide tmin input field
            self.tmin_frame.pack_forget()
    
    def create_widgets(self):
        """Create the dialog widgets."""
        # Main frame
        main_frame = tk.Frame(self.dialog, padx=20, pady=20)
        main_frame.pack(fill="both", expand=True)
        
        # Title
        title_label = tk.Label(main_frame, text="Preprocessing Options", 
                              font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Preprocessing options frame
        options_frame = tk.Frame(main_frame)
        options_frame.pack(fill="x", pady=10)
        
        # Short Channel Correction
        self.short_channel_var = tk.BooleanVar(value=self.settings.get("short_channel_correction", True))
        short_channel_frame = tk.Frame(options_frame)
        short_channel_frame.pack(fill="x", pady=5)
        
        self.short_channel_cb = tk.Checkbutton(
            short_channel_frame, 
            text="Short Channel Correction",
            variable=self.short_channel_var,
            font=("Arial", 11)
        )
        self.short_channel_cb.pack(side="left")
        
        # Info button for short channel correction
        short_info_btn = tk.Button(
            short_channel_frame, 
            text="?", 
            command=self.show_short_channel_info,
            width=2, 
            height=1,
            bg="lightblue",
            font=("Arial", 8)
        )
        short_info_btn.pack(side="right")
        
        # Negative Correlation Enhancement
        self.negative_corr_var = tk.BooleanVar(value=self.settings.get("negative_correlation_enhancement", False))
        negative_corr_frame = tk.Frame(options_frame)
        negative_corr_frame.pack(fill="x", pady=5)
        
        self.negative_corr_cb = tk.Checkbutton(
            negative_corr_frame, 
            text="Negative Correlation Enhancement",
            variable=self.negative_corr_var,
            font=("Arial", 11)
        )
        self.negative_corr_cb.pack(side="left")
        
        # Info button for negative correlation enhancement
        negative_info_btn = tk.Button(
            negative_corr_frame, 
            text="?", 
            command=self.show_negative_corr_info,
            width=2, 
            height=1,
            bg="lightblue",
            font=("Arial", 8)
        )
        negative_info_btn.pack(side="right")
        
        # Baseline Correction
        self.baseline_correction_frame = tk.Frame(options_frame)
        self.baseline_correction_frame.pack(fill="x", pady=5)
        
        # Label for baseline correction
        baseline_label = tk.Label(
            self.baseline_correction_frame,
            text="Baseline Correction:",
            font=("Arial", 11)
        )
        baseline_label.pack(side="left")
        
        # Get available methods dynamically
        baseline_options = self.get_baseline_correction_methods()
        self.baseline_correction_var = tk.StringVar(value=self.settings.get("baseline_correction"))
        
        self.baseline_correction_dropdown = ttk.Combobox(
            self.baseline_correction_frame,
            textvariable=self.baseline_correction_var,
            values=baseline_options,
            state="readonly",
            width=15,
            font=("Arial", 10)
        )
        self.baseline_correction_dropdown.pack(side="left", padx=(10, 0))
        
        # Bind the change event to the dropdown
        self.baseline_correction_dropdown.bind('<<ComboboxSelected>>', self.on_baseline_method_change)

        # Info button for baseline correction
        baseline_info_btn = tk.Button(
            self.baseline_correction_frame, 
            text="?", 
            command=self.show_baseline_correction_info,
            width=2, 
            height=1,
            bg="lightblue",
            font=("Arial", 8)
        )
        baseline_info_btn.pack(side="right")
        
        # tmin input frame (initially hidden)
        self.tmin_frame = tk.Frame(options_frame)
        
        # Label for tmin
        tmin_label = tk.Label(
            self.tmin_frame,
            text="tmin (seconds):",
            font=("Arial", 11)
        )
        tmin_label.pack(side="left")
        
        # Register validation function
        vcmd = (self.dialog.register(self.validate_tmin_input), '%P')
        
        # tmin input field
        self.tmin_var = tk.StringVar(value=str(self.settings.get("tmin", -5)))
        self.tmin_entry = tk.Entry(
            self.tmin_frame,
            textvariable=self.tmin_var,
            width=10,
            font=("Arial", 10),
            validate='key',
            validatecommand=vcmd
        )
        self.tmin_entry.pack(side="left", padx=(10, 0))
        
        # Info button for tmin
        tmin_info_btn = tk.Button(
            self.tmin_frame, 
            text="?", 
            command=self.show_tmin_info,
            width=2, 
            height=1,
            bg="lightblue",
            font=("Arial", 8)
        )
        tmin_info_btn.pack(side="right")
        
        # Show/hide tmin field based on initial selection
        self.on_baseline_method_change()
        
        # Separator
        separator = ttk.Separator(main_frame, orient="horizontal")
        separator.pack(fill="x", pady=20)
        
        # Buttons frame
        buttons_frame = tk.Frame(main_frame)
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
        
        # Bind Enter and Escape keys
        self.dialog.bind('<Return>', lambda e: self.ok())
        self.dialog.bind('<Escape>', lambda e: self.cancel())
        
        # Set focus to OK button
        ok_btn.focus_set()
    
    def show_short_channel_info(self):
        """Show information about short channel correction."""
        info_text = (
            "Short Channel Correction:\n\n"
            "Short channels are typically placed close to the scalp surface "
            "and are used to measure systemic physiological noise (like heartbeat, "
            "respiration, and blood pressure changes) that affects the fNIRS signal.\n\n"
            "When enabled, this correction removes this systemic noise from the "
            "longer channels that measure brain activity, improving the signal-to-noise "
            "ratio and the reliability of the brain activation measurements."
        )
        
        tk.messagebox.showinfo("Short Channel Correction", info_text)
    
    def show_negative_corr_info(self):
        """Show information about negative correlation enhancement."""
        info_text = (
            "Negative Correlation Enhancement:\n\n"
            "This technique enhances the anticorrelation between oxyhemoglobin (HbO) "
            "and deoxyhemoglobin (HbR) signals.\n\n"
            "In healthy brain tissue, these signals typically show opposite patterns "
            "during activation. This enhancement can improve the detection of brain "
            "activation by emphasizing this natural anticorrelation pattern.\n\n"
            "Note: Use with caution as it may also amplify noise in some cases."
        )
        
        tk.messagebox.showinfo("Negative Correlation Enhancement", info_text)
    
    def show_baseline_correction_info(self):
        """Show information about baseline correction."""
        available_methods = self.get_baseline_correction_methods()
        methods_text = "\n".join([f"• {method}" for method in available_methods])
        
        info_text = (
            "Baseline Correction:\n\n"
            "Baseline correction removes drift and systematic changes in the signal "
            "that are not related to brain activation.\n\n"
            "Available methods:\n"
            f"{methods_text}\n\n"
            "Choose based on the type of baseline drift in your data."
        )
        
        tk.messagebox.showinfo("Baseline Correction", info_text)
    
    def show_tmin_info(self):
        """Show information about tmin parameter."""
        info_text = (
            "tmin (Time Minimum):\n\n"
            "This parameter defines the start time (in seconds) relative to the "
            "event onset for baseline correction when using 'xSecondsBefore' method.\n\n"
            "• Negative values: Time before the event (e.g., -5 means 5 seconds before)\n"
            "• Positive values: Time after the event (e.g., 2 means 2 seconds after)\n"
            "• Zero: Exactly at the event onset\n\n"
            "Common usage: -5 to -2 seconds before stimulus onset for pre-stimulus baseline."
        )
        
        tk.messagebox.showinfo("tmin Parameter", info_text)

    def reset_to_defaults(self):
        """Reset all settings to their default values."""
        self.short_channel_var.set(True)
        self.negative_corr_var.set(False)
        self.baseline_correction_var.set("usePreviousRest")
        self.tmin_var.set("-5")
        # Update UI based on default baseline method
        self.on_baseline_method_change()
    
    def ok(self):
        """Accept the settings and close the dialog."""
        # Validate tmin if xSecondsBefore is selected
        if self.baseline_correction_var.get() == "xSecondsBefore":
            tmin_str = self.tmin_var.get().strip()
            if not tmin_str:
                tk.messagebox.showerror("Invalid Input", "Please enter a value for tmin.")
                return
            try:
                tmin_value = int(tmin_str)
            except ValueError:
                tk.messagebox.showerror("Invalid Input", "tmin must be a valid integer.")
                return
        else:
            tmin_value = self.settings.get("tmin", -5)  # Use existing or default value
        
        self.result = {
            "short_channel_correction": self.short_channel_var.get(),
            "negative_correlation_enhancement": self.negative_corr_var.get(),
            "baseline_correction": self.baseline_correction_var.get(),
            "tmin": tmin_value
        }
        self.dialog.destroy()
    
    def cancel(self):
        """Cancel and close the dialog without saving changes."""
        self.result = None
        self.dialog.destroy()
    
    def get_result(self):
        """Return the result of the dialog."""
        return self.result


def show_preprocessing_dialog(parent, current_settings):
    """
    Show the preprocessing dialog and return the selected settings.
    
    Args:
        parent: The parent window
        current_settings: Dictionary with current preprocessing settings
    
    Returns:
        Dictionary with selected settings or None if cancelled
    """
    dialog = PreprocessingDialog(parent, current_settings)
    return dialog.get_result()