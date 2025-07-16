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
        self.dialog.geometry("450x550")  # Increased height for additional fields
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
        dialog_width = 450
        dialog_height = 550  # Updated height
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        self.dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
    
    def validate_numeric_input(self, value, allow_negative=False):
        """Validate that input is a valid number (int or float)."""
        if value == "" or (allow_negative and value == "-"):
            return True  # Allow empty string and lone minus sign during typing
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def validate_tmin_input(self, value):
        """Validate that tmin input is a valid integer (positive or negative)."""
        return self.validate_numeric_input(value, allow_negative=True)
    
    def validate_positive_numeric_input(self, value):
        """Validate that input is a positive number."""
        if value == "":
            return True
        try:
            num = float(value)
            return num > 0
        except ValueError:
            return False
    
    def validate_coupling_threshold_input(self, value):
        """Validate that coupling threshold is between 0 and 1."""
        if value == "":
            return True
        try:
            num = float(value)
            return 0 <= num <= 1
        except ValueError:
            return False
    
    def validate_unwanted_labels_input(self, value):
        """Validate that unwanted labels input is properly formatted."""
        if value.strip() == "":
            return True  # Allow empty input
        
        # Check if it looks like a valid format (numbers separated by commas)
        try:
            # Remove brackets if present
            clean_value = value.strip().strip('[]')
            if clean_value == "":
                return True
            
            # Split by comma and check each part
            parts = [part.strip().strip('"\'') for part in clean_value.split(',')]
            for part in parts:
                if part:  # Skip empty parts
                    float(part)  # Check if it can be converted to float
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
        title_label = tk.Label(scrollable_frame, text="Preprocessing Options", 
                              font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Preprocessing options frame
        options_frame = tk.Frame(scrollable_frame)
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
        vcmd_tmin = (self.dialog.register(self.validate_tmin_input), '%P')
        
        # tmin input field
        self.tmin_var = tk.StringVar(value=str(self.settings.get("tmin", -5)))
        self.tmin_entry = tk.Entry(
            self.tmin_frame,
            textvariable=self.tmin_var,
            width=10,
            font=("Arial", 10),
            validate='key',
            validatecommand=vcmd_tmin
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
        separator1 = ttk.Separator(options_frame, orient="horizontal")
        separator1.pack(fill="x", pady=15)
        
        # Scalp Coupling Threshold
        coupling_threshold_frame = tk.Frame(options_frame)
        coupling_threshold_frame.pack(fill="x", pady=5)
        
        coupling_threshold_label = tk.Label(
            coupling_threshold_frame,
            text="Scalp Coupling Threshold:",
            font=("Arial", 11)
        )
        coupling_threshold_label.pack(side="left")
        
        # Register validation function for coupling threshold (0-1)
        vcmd_coupling = (self.dialog.register(self.validate_coupling_threshold_input), '%P')
        
        self.coupling_threshold_var = tk.StringVar(value=str(self.settings.get("scalp_coupling_threshold", 0.8)))
        self.coupling_threshold_entry = tk.Entry(
            coupling_threshold_frame,
            textvariable=self.coupling_threshold_var,
            width=10,
            font=("Arial", 10),
            validate='key',
            validatecommand=vcmd_coupling
        )
        self.coupling_threshold_entry.pack(side="left", padx=(10, 0))
        
        # Info button for coupling threshold
        coupling_threshold_info_btn = tk.Button(
            coupling_threshold_frame, 
            text="?", 
            command=self.show_coupling_threshold_info,
            width=2, 
            height=1,
            bg="lightblue",
            font=("Arial", 8)
        )
        coupling_threshold_info_btn.pack(side="right")
        
        # Reject Criteria (HbO threshold)
        reject_criteria_frame = tk.Frame(options_frame)
        reject_criteria_frame.pack(fill="x", pady=5)
        
        reject_criteria_label = tk.Label(
            reject_criteria_frame,
            text="HbO Rejection Threshold (×10⁻⁶):",
            font=("Arial", 11)
        )
        reject_criteria_label.pack(side="left")
        
        # Convert from scientific notation to user-friendly format
        current_hbo = self.settings.get("reject_criteria", {}).get("hbo", 80e-6)
        hbo_display_value = current_hbo * 1e6  # Convert to ×10⁻⁶ units
        
        # Register validation function for positive numeric input
        vcmd_positive = (self.dialog.register(self.validate_positive_numeric_input), '%P')
        
        self.reject_criteria_var = tk.StringVar(value=str(hbo_display_value))
        self.reject_criteria_entry = tk.Entry(
            reject_criteria_frame,
            textvariable=self.reject_criteria_var,
            width=10,
            font=("Arial", 10),
            validate='key',
            validatecommand=vcmd_positive
        )
        self.reject_criteria_entry.pack(side="left", padx=(10, 0))
        
        # Info button for reject criteria
        reject_criteria_info_btn = tk.Button(
            reject_criteria_frame, 
            text="?", 
            command=self.show_reject_criteria_info,
            width=2, 
            height=1,
            bg="lightblue",
            font=("Arial", 8)
        )
        reject_criteria_info_btn.pack(side="right")
        
        # Unwanted Labels (formerly Unwanted Frequency)
        unwanted_labels_frame = tk.Frame(options_frame)
        unwanted_labels_frame.pack(fill="x", pady=5)
        
        unwanted_labels_label = tk.Label(
            unwanted_labels_frame,
            text="Unwanted Labels:",
            font=("Arial", 11)
        )
        unwanted_labels_label.pack(side="left")
        
        # Register validation function for unwanted labels
        vcmd_unwanted_labels = (self.dialog.register(self.validate_unwanted_labels_input), '%P')
        
        # Format the current unwanted labels for display
        current_unwanted = self.settings.get("unwanted", ["15.0"])
        if isinstance(current_unwanted, list):
            display_value = ", ".join(current_unwanted)
        else:
            display_value = str(current_unwanted)
        
        self.unwanted_labels_var = tk.StringVar(value=display_value)
        self.unwanted_labels_entry = tk.Entry(
            unwanted_labels_frame,
            textvariable=self.unwanted_labels_var,
            width=20,
            font=("Arial", 10),
            validate='key',
            validatecommand=vcmd_unwanted_labels
        )
        self.unwanted_labels_entry.pack(side="left", padx=(10, 0))
        
        # Info button for unwanted labels
        unwanted_labels_info_btn = tk.Button(
            unwanted_labels_frame, 
            text="?", 
            command=self.show_unwanted_labels_info,
            width=2, 
            height=1,
            bg="lightblue",
            font=("Arial", 8)
        )
        unwanted_labels_info_btn.pack(side="right")
        
        # Separator
        separator2 = ttk.Separator(options_frame, orient="horizontal")
        separator2.pack(fill="x", pady=20)
        
        # Buttons frame
        buttons_frame = tk.Frame(options_frame)
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
    
    def show_coupling_threshold_info(self):
        """Show information about scalp coupling threshold."""
        info_text = (
            "Scalp Coupling Threshold:\n\n"
            "This parameter determines the minimum required coupling quality "
            "between the optodes and the scalp surface.\n\n"
            "Values range from 0.0 to 1.0:\n"
            "• 0.0: No coupling requirement (accept all channels)\n"
            "• 0.5: Moderate coupling requirement\n"
            "• 0.8: High coupling requirement (recommended)\n"
            "• 1.0: Perfect coupling required\n\n"
            "Channels below this threshold will be excluded from analysis "
            "due to poor signal quality."
        )
        
        tk.messagebox.showinfo("Scalp Coupling Threshold", info_text)
    
    def show_reject_criteria_info(self):
        """Show information about rejection criteria."""
        info_text = (
            "HbO Rejection Threshold:\n\n"
            "This parameter sets the maximum allowed oxyhemoglobin (HbO) "
            "concentration change for accepting data epochs.\n\n"
            "Values are specified in ×10⁻⁶ units (micromolar):\n"
            "• Typical range: 50-100 ×10⁻⁶\n"
            "• Lower values: More stringent artifact rejection\n"
            "• Higher values: More lenient artifact rejection\n\n"
            "Epochs with HbO changes exceeding this threshold will be "
            "rejected as artifacts (likely due to motion or other noise)."
        )
        
        tk.messagebox.showinfo("HbO Rejection Threshold", info_text)
    
    def show_unwanted_labels_info(self):
        """Show information about unwanted labels."""
        info_text = (
            "Unwanted Labels:\n\n"
            "This parameter specifies data labels that should be excluded "
            "from analysis.\n\n"
            "Format: Enter labels separated by commas\n"
            "• Example: 15.0, 20.5, 30.0\n"
            "• Labels are typically numeric values encoded as strings\n"
            "• Empty field means no labels will be excluded\n\n"
            "Data epochs/trials with these labels will be filtered out "
            "during preprocessing to focus on the conditions of interest."
        )
        
        tk.messagebox.showinfo("Unwanted Labels", info_text)

    def reset_to_defaults(self):
        """Reset all settings to their default values."""
        self.short_channel_var.set(True)
        self.negative_corr_var.set(False)
        self.baseline_correction_var.set("usePreviousRest")
        self.tmin_var.set("-5")
        self.coupling_threshold_var.set("0.8")
        self.reject_criteria_var.set("80")
        self.unwanted_labels_var.set("15.0")
        # Update UI based on default baseline method
        self.on_baseline_method_change()
    
    def parse_unwanted_labels(self, input_string):
        """Parse the unwanted labels input string into a list."""
        if not input_string.strip():
            return []
        
        # Remove brackets if present
        clean_input = input_string.strip().strip('[]')
        if not clean_input:
            return []
        
        # Split by comma and clean each label
        labels = []
        for label in clean_input.split(','):
            label = label.strip().strip('"\'')
            if label:
                labels.append(label)
        
        return labels
    
    def validate_inputs(self):
        """Validate all input fields before accepting."""
        # Validate tmin if xSecondsBefore is selected
        if self.baseline_correction_var.get() == "xSecondsBefore":
            tmin_str = self.tmin_var.get().strip()
            if not tmin_str:
                tk.messagebox.showerror("Invalid Input", "Please enter a value for tmin.")
                return False
            try:
                int(tmin_str)
            except ValueError:
                tk.messagebox.showerror("Invalid Input", "tmin must be a valid integer.")
                return False
        
        # Validate coupling threshold
        coupling_threshold_str = self.coupling_threshold_var.get().strip()
        if not coupling_threshold_str:
            tk.messagebox.showerror("Invalid Input", "Please enter a value for scalp coupling threshold.")
            return False
        try:
            coupling_threshold = float(coupling_threshold_str)
            if not (0 <= coupling_threshold <= 1):
                tk.messagebox.showerror("Invalid Input", "Scalp coupling threshold must be between 0 and 1.")
                return False
        except ValueError:
            tk.messagebox.showerror("Invalid Input", "Scalp coupling threshold must be a valid number.")
            return False
        
        # Validate reject criteria
        reject_criteria_str = self.reject_criteria_var.get().strip()
        if not reject_criteria_str:
            tk.messagebox.showerror("Invalid Input", "Please enter a value for HbO rejection threshold.")
            return False
        try:
            reject_criteria = float(reject_criteria_str)
            if reject_criteria <= 0:
                tk.messagebox.showerror("Invalid Input", "HbO rejection threshold must be positive.")
                return False
        except ValueError:
            tk.messagebox.showerror("Invalid Input", "HbO rejection threshold must be a valid number.")
            return False
        
        # Validate unwanted labels
        unwanted_labels_str = self.unwanted_labels_var.get().strip()
        if unwanted_labels_str:
            try:
                self.parse_unwanted_labels(unwanted_labels_str)
            except:
                tk.messagebox.showerror("Invalid Input", 
                                      "Unwanted labels format is invalid. Please use format: 15.0, 20.5, 30.0")
                return False
        
        return True
    
    def ok(self):
        """Accept the settings and close the dialog."""
        if not self.validate_inputs():
            return
        
        # Get tmin value
        if self.baseline_correction_var.get() == "xSecondsBefore":
            tmin_value = int(self.tmin_var.get().strip())
        else:
            tmin_value = self.settings.get("tmin", -5)  # Use existing or default value
        
        # Convert reject criteria back to scientific notation
        reject_criteria_value = float(self.reject_criteria_var.get().strip()) * 1e-6
        
        # Parse unwanted labels
        unwanted_labels = self.parse_unwanted_labels(self.unwanted_labels_var.get().strip())
        
        self.result = {
            "short_channel_correction": self.short_channel_var.get(),
            "negative_correlation_enhancement": self.negative_corr_var.get(),
            "baseline_correction": self.baseline_correction_var.get(),
            "tmin": tmin_value,
            "scalp_coupling_threshold": float(self.coupling_threshold_var.get().strip()),
            "reject_criteria": {"hbo": reject_criteria_value},
            "unwanted": unwanted_labels
        }
        
        self.dialog.unbind_all("<MouseWheel>") # Cleanup mousewheel binding
        self.dialog.destroy()
    
    def cancel(self):
        """Cancel and close the dialog without saving changes."""
        self.result = None
        self.dialog.unbind_all("<MouseWheel>") # Cleanup mousewheel binding
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