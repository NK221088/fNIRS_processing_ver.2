import tkinter as tk
from tkinter import ttk
from preprocessesing_toolbox.baselineCorrection import baselineCorrection
from preprocessesing_toolbox.preprocessing_info import PreprocessingInfo

class PreprocessingDialog:
    def __init__(self, parent, current_settings):
        self.parent = parent
        self.result = None
        self.settings = current_settings.copy()
        
        # Create the dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Preprocessing Options")
        self.dialog.geometry("450x600")  # Increased height for additional fields including interpolate bad channels
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
        dialog_height = 600  # Updated height
        x = parent_x + (parent_width - dialog_height) // 2
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
        """Validate that tmin input is a positive number greater than 0 for xSecondsBefore method."""
        if value == "":
            return True  # Allow empty string during typing
        try:
            num = float(value)
            return num > 0  # Only allow positive values greater than 0
        except ValueError:
            return False
    
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

        # Try to process a comma-separated list, optionally in brackets
        clean_value = value.strip().strip('[]')
        if clean_value == "":
            return True

        # Split the input into parts
        parts = [part.strip().strip('"\'') for part in clean_value.split(',')]

        # Validate each part - for general strings, just check non-emptiness
        for part in parts:
            if not part:  # Empty after stripping
                return False 
        return True
    
    def get_default_tmin_value(self, baseline_method):
        """Get the default tmin value based on baseline method."""
        if baseline_method == "xSecondsBefore":
            return 5  # Default 5 seconds before for xSecondsBefore
        else:
            return 0  # Default 0 for other methods
    
    def on_baseline_method_change(self, event=None):
        """Handle changes in baseline correction method selection."""
        selected_method = self.baseline_correction_var.get()
        
        if selected_method == "xSecondsBefore":
            # Show tmin input field and set appropriate default if currently 0
            self.tmin_frame.pack(fill="x", pady=5, after=self.baseline_correction_frame)
            
            # If current value is 0 (default for other methods), set to 5
            current_value = self.tmin_var.get().strip()
            if current_value == "" or current_value == "0":
                self.tmin_var.set("5")
        else:
            # Hide tmin input field and set to 0
            self.tmin_frame.pack_forget()
            self.tmin_var.set("0")
    
    def on_snr_rejection_change(self, event=None):
        """Handle changes in SNR rejection method selection."""
        selected_method = self.snr_rejection_var.get()
        
        if selected_method in ["SNR", "CV"]:
            # Show threshold input field and update default value based on method
            self.snr_threshold_frame.pack(fill="x", pady=5, after=self.snr_rejection_frame)
            
            # Set appropriate default value based on method
            current_value = self.snr_threshold_var.get()
            if selected_method == "SNR" and current_value == "0.15":
                # Switching from CV to SNR
                self.snr_threshold_var.set("8")
            elif selected_method == "CV" and current_value == "8":
                # Switching from SNR to CV
                self.snr_threshold_var.set("0.15")
        else:
            # Hide threshold input field
            self.snr_threshold_frame.pack_forget()
            
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
        
        # SNR rejection method
        snr_options = ["None", "SNR", "CV"]
        self.snr_rejection_var = tk.StringVar(value=self.settings.get("snr_rejection", "None"))
        
        # Create frame for the feature
        self.snr_rejection_frame = tk.Frame(options_frame)
        self.snr_rejection_frame.pack(fill="x", pady=5)
        
        snr_rejection_label = tk.Label(
            self.snr_rejection_frame,
            text="SNR Rejection Method:",
            font=("Arial", 11)
        )
        snr_rejection_label.pack(side="left")
        
        # Add the dropdown widget
        self.snr_rejection_dropdown = ttk.Combobox(
            self.snr_rejection_frame,
            textvariable=self.snr_rejection_var,
            values=snr_options,
            state="readonly",
            width=15,
            font=("Arial", 10)
        )
        self.snr_rejection_dropdown.pack(side="left", padx=(10, 0))
        self.snr_rejection_dropdown.set(self.snr_rejection_var.get())

        # Bind the change event to the dropdown
        self.snr_rejection_dropdown.bind('<<ComboboxSelected>>', self.on_snr_rejection_change)
        
        # Add info button
        snr_rejection_info_btn = tk.Button(
            self.snr_rejection_frame,
            text="?",
            command=self.show_snr_rejection_info,
            width=2,
            height=1,
            bg="lightblue",
            font=("Arial", 8)
        )
        snr_rejection_info_btn.pack(side="right")
        
        # SNR/CV threshold input frame (initially hidden)
        self.snr_threshold_frame = tk.Frame(options_frame)
        
        # Label for threshold
        snr_threshold_label = tk.Label(
            self.snr_threshold_frame,
            text="Threshold Value:",
            font=("Arial", 11)
        )
        snr_threshold_label.pack(side="left")
        
        # Register validation function for positive numeric input
        vcmd_snr_threshold = (self.dialog.register(self.validate_positive_numeric_input), '%P')
        
        # Threshold input field - use single threshold value
        self.snr_threshold_var = tk.StringVar(value=str(self.settings.get("snr_threshold", 8)))
        self.snr_threshold_entry = tk.Entry(
            self.snr_threshold_frame,
            textvariable=self.snr_threshold_var,
            width=10,
            font=("Arial", 10),
            validate='key',
            validatecommand=vcmd_snr_threshold
        )
        self.snr_threshold_entry.pack(side="left", padx=(10, 0))
        
        # Info button for threshold
        snr_threshold_info_btn = tk.Button(
            self.snr_threshold_frame,
            text="?",
            command=self.show_snr_threshold_info,
            width=2,
            height=1,
            bg="lightblue",
            font=("Arial", 8)
        )
        snr_threshold_info_btn.pack(side="right")
        
        # Show/hide threshold field based on initial selection
        self.on_snr_rejection_change()

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
        
        # Apply Temporal Derivative Distribution Removal (TDDR)
        tddr_frame = tk.Frame(options_frame)
        tddr_frame.pack(fill="x", pady=5)

        self.apply_tddr_var = tk.BooleanVar(value=self.settings.get("Apply_TDDR", False))
        self.tddr_checkbox = tk.Checkbutton(
            tddr_frame,
            text="Apply TDDR",
            variable=self.apply_tddr_var,
            font=("Arial", 11)
        )
        self.tddr_checkbox.pack(side="left")

        info_button_tddr = tk.Button(
            tddr_frame,
            text="?",
            command=self.show_tddr_info,
            width=2,
            height=1,
            bg="lightblue",
            font=("Arial", 8)
        )
        info_button_tddr.pack(side="right")
        
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
        
        # Interpolate Bad Channels
        interpolate_frame = tk.Frame(options_frame)
        interpolate_frame.pack(fill="x", pady=5)
        
        self.interpolate_bad_channels_var = tk.BooleanVar(value=self.settings.get("interpolate_bad_channels", False))
        self.interpolate_checkbox = tk.Checkbutton(
            interpolate_frame,
            text="Interpolate Bad Channels",
            variable=self.interpolate_bad_channels_var,
            font=("Arial", 11)
        )
        self.interpolate_checkbox.pack(side="left")
        
        info_button_interpolate = tk.Button(
            interpolate_frame, 
            text="?", 
            command=self.show_interpolate_info,
            width=2,
            height=1,
            bg="lightblue",
            font=("Arial", 8)
        )
        info_button_interpolate.pack(side="right")
        
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
        
        # Set default baseline correction method if not in settings or invalid
        default_baseline = self.settings.get("baseline_correction", "Previous rest period")
        if default_baseline not in baseline_options:
            default_baseline = "Previous rest period"
        
        self.baseline_correction_var = tk.StringVar(value=default_baseline)
        
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
        
        # Label for tmin - updated to reflect positive input requirement
        tmin_label = tk.Label(
            self.tmin_frame,
            text="Seconds Before Event Onset:",
            font=("Arial", 11)
        )
        tmin_label.pack(side="left")
        
        # Register validation function for positive tmin input
        vcmd_tmin = (self.dialog.register(self.validate_tmin_input), '%P')
        
        # Initialize tmin value based on baseline method
        current_baseline = self.baseline_correction_var.get()
        if current_baseline == "xSecondsBefore":
            # Convert stored negative value to positive for display, or use default
            stored_tmin = self.settings.get("tmin", -5)
            if stored_tmin < 0:
                display_tmin = abs(stored_tmin)
            elif stored_tmin == 0:
                display_tmin = 5  # Default for xSecondsBefore
            else:
                display_tmin = stored_tmin
        else:
            # For other methods, default to 0 (which will be hidden anyway)
            display_tmin = 0
            
        self.tmin_var = tk.StringVar(value=str(display_tmin))
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
        
        # === FILTER SETTINGS SECTION ===
        # Filter settings title
        filter_title = tk.Label(options_frame, text="Filter Settings", 
                               font=("Arial", 12, "bold"))
        filter_title.pack(pady=(0, 10))
        
        # Filter Lower Value
        filter_lower_frame = tk.Frame(options_frame)
        filter_lower_frame.pack(fill="x", pady=5)
        
        filter_lower_label = tk.Label(
            filter_lower_frame,
            text="Filter Lower Value (Hz):",
            font=("Arial", 11)
        )
        filter_lower_label.pack(side="left")
        
        # Register validation function for positive numeric input
        vcmd_positive = (self.dialog.register(self.validate_positive_numeric_input), '%P')
        
        self.filter_lower_var = tk.StringVar(value=str(self.settings.get("filter_lower_value", 0.01)))
        self.filter_lower_entry = tk.Entry(
            filter_lower_frame,
            textvariable=self.filter_lower_var,
            width=10,
            font=("Arial", 10),
            validate='key',
            validatecommand=vcmd_positive
        )
        self.filter_lower_entry.pack(side="left", padx=(10, 0))
        
        # Info button for filter lower value
        filter_lower_info_btn = tk.Button(
            filter_lower_frame, 
            text="?", 
            command=self.show_filter_lower_info,
            width=2, 
            height=1,
            bg="lightblue",
            font=("Arial", 8)
        )
        filter_lower_info_btn.pack(side="right")
        
        # Filter Upper Value
        filter_upper_frame = tk.Frame(options_frame)
        filter_upper_frame.pack(fill="x", pady=5)
        
        filter_upper_label = tk.Label(
            filter_upper_frame,
            text="Filter Upper Value (Hz):",
            font=("Arial", 11)
        )
        filter_upper_label.pack(side="left")
        
        self.filter_upper_var = tk.StringVar(value=str(self.settings.get("filter_upper_value", 0.5)))
        self.filter_upper_entry = tk.Entry(
            filter_upper_frame,
            textvariable=self.filter_upper_var,
            width=10,
            font=("Arial", 10),
            validate='key',
            validatecommand=vcmd_positive
        )
        self.filter_upper_entry.pack(side="left", padx=(10, 0))
        
        # Info button for filter upper value
        filter_upper_info_btn = tk.Button(
            filter_upper_frame, 
            text="?", 
            command=self.show_filter_upper_info,
            width=2, 
            height=1,
            bg="lightblue",
            font=("Arial", 8)
        )
        filter_upper_info_btn.pack(side="right")
        
        # High Transition Bandwidth
        h_trans_frame = tk.Frame(options_frame)
        h_trans_frame.pack(fill="x", pady=5)
        
        h_trans_label = tk.Label(
            h_trans_frame,
            text="High Transition Bandwidth (Hz):",
            font=("Arial", 11)
        )
        h_trans_label.pack(side="left")
        
        self.h_trans_var = tk.StringVar(value=str(self.settings.get("h_trans_bandwidth", 0.2)))
        self.h_trans_entry = tk.Entry(
            h_trans_frame,
            textvariable=self.h_trans_var,
            width=10,
            font=("Arial", 10),
            validate='key',
            validatecommand=vcmd_positive
        )
        self.h_trans_entry.pack(side="left", padx=(10, 0))
        
        # Info button for high transition bandwidth
        h_trans_info_btn = tk.Button(
            h_trans_frame, 
            text="?", 
            command=self.show_h_trans_info,
            width=2, 
            height=1,
            bg="lightblue",
            font=("Arial", 8)
        )
        h_trans_info_btn.pack(side="right")
        
        # Low Transition Bandwidth
        l_trans_frame = tk.Frame(options_frame)
        l_trans_frame.pack(fill="x", pady=5)
        
        l_trans_label = tk.Label(
            l_trans_frame,
            text="Low Transition Bandwidth (Hz):",
            font=("Arial", 11)
        )
        l_trans_label.pack(side="left")
        
        self.l_trans_var = tk.StringVar(value=str(self.settings.get("l_trans_bandwidth", 0.01)))
        self.l_trans_entry = tk.Entry(
            l_trans_frame,
            textvariable=self.l_trans_var,
            width=10,
            font=("Arial", 10),
            validate='key',
            validatecommand=vcmd_positive
        )
        self.l_trans_entry.pack(side="left", padx=(10, 0))
        
        # Info button for low transition bandwidth
        l_trans_info_btn = tk.Button(
            l_trans_frame, 
            text="?", 
            command=self.show_l_trans_info,
            width=2, 
            height=1,
            bg="lightblue",
            font=("Arial", 8)
        )
        l_trans_info_btn.pack(side="right")
        
        # Separator
        separator2 = ttk.Separator(options_frame, orient="horizontal")
        separator2.pack(fill="x", pady=15)
        
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
        separator3 = ttk.Separator(options_frame, orient="horizontal")
        separator3.pack(fill="x", pady=20)
        
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
        PreprocessingInfo.show_short_channel_info()
    
    def show_tddr_info(self):
        """Show information about TDDR."""
        PreprocessingInfo.show_tddr_info()
    
    def show_negative_corr_info(self):
        """Show information about negative correlation enhancement."""
        PreprocessingInfo.show_negative_corr_info()
    
    def show_interpolate_info(self):
        """Show information about interpolate bad channels."""
        PreprocessingInfo.show_interpolate_info()
    
    def show_baseline_correction_info(self):
        """Show information about baseline correction."""
        available_methods = self.get_baseline_correction_methods()
        PreprocessingInfo.show_baseline_correction_info(available_methods)
    
    def show_tmin_info(self):
        """Show information about tmin parameter."""
        PreprocessingInfo.show_tmin_info()
    
    def show_coupling_threshold_info(self):
        """Show information about scalp coupling threshold."""
        PreprocessingInfo.show_coupling_threshold_info()

    def show_reject_criteria_info(self):
        """Show information about rejection criteria."""
        PreprocessingInfo.show_reject_criteria_info()

    def show_unwanted_labels_info(self):
        """Show information about unwanted labels."""
        PreprocessingInfo.show_unwanted_labels_info()
    
    def show_filter_lower_info(self):
        """Show information about filter lower value."""
        PreprocessingInfo.show_filter_lower_info()

    def show_filter_upper_info(self):
        """Show information about filter upper value."""
        PreprocessingInfo.show_filter_upper_info()

    def show_h_trans_info(self):
        """Show information about high transition bandwidth."""
        PreprocessingInfo.show_h_trans_info()

    def show_l_trans_info(self):
        """Show information about low transition bandwidth."""
        PreprocessingInfo.show_l_trans_info()
    
    def show_snr_rejection_info(self):
        """Show information about SNR rejection methods."""
        PreprocessingInfo.show_snr_rejection_info()
    
    def show_snr_threshold_info(self):
        """Show information about SNR/CV threshold values."""
        PreprocessingInfo.show_snr_threshold_info()

    def reset_to_defaults(self):
        """Reset all settings to their default values."""
        self.short_channel_var.set(True)
        self.negative_corr_var.set(False)
        self.interpolate_bad_channels_var.set(False)
        self.baseline_correction_var.set("Previous rest period")
        self.tmin_var.set("0")  # Will be updated by on_baseline_method_change
        self.filter_lower_var.set("0.05")  
        self.filter_upper_var.set("0.7")   
        self.h_trans_var.set("0.2")        
        self.l_trans_var.set("0.02")
        self.coupling_threshold_var.set("0.8")
        self.reject_criteria_var.set("80")
        self.unwanted_labels_var.set("15.0")
        self.snr_rejection_var.set("None")
        self.snr_threshold_var.set("8")  # Default for SNR method
        self.apply_tddr_var.set(False)
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
        # Validate SNR/CV threshold if SNR or CV method is selected
        if self.snr_rejection_var.get() in ["SNR", "CV"]:
            threshold_str = self.snr_threshold_var.get().strip()
            if not threshold_str:
                method_name = self.snr_rejection_var.get()
                tk.messagebox.showerror("Invalid Input", f"Please enter a threshold value for {method_name} rejection.")
                return False
            try:
                threshold_value = float(threshold_str)
                if threshold_value <= 0:
                    tk.messagebox.showerror("Invalid Input", "Threshold value must be positive.")
                    return False
            except ValueError:
                tk.messagebox.showerror("Invalid Input", "Threshold value must be a valid number.")
                return False
            
        # Validate tmin if xSecondsBefore is selected
        if self.baseline_correction_var.get() == "xSecondsBefore":
            tmin_str = self.tmin_var.get().strip()
            if not tmin_str:
                tk.messagebox.showerror("Invalid Input", "Please enter a value for seconds before event onset.")
                return False
            try:
                tmin_value = float(tmin_str)
                if tmin_value <= 0:
                    tk.messagebox.showerror("Invalid Input", "Seconds before event onset must be a positive value greater than zero.")
                    return False
            except ValueError:
                tk.messagebox.showerror("Invalid Input", "Seconds before event onset must be a valid positive number.")
                return False
        
        # Validate filter lower value
        filter_lower_str = self.filter_lower_var.get().strip()
        if not filter_lower_str:
            tk.messagebox.showerror("Invalid Input", "Please enter a value for filter lower value.")
            return False
        try:
            filter_lower = float(filter_lower_str)
            if filter_lower <= 0:
                tk.messagebox.showerror("Invalid Input", "Filter lower value must be positive.")
                return False
        except ValueError:
            tk.messagebox.showerror("Invalid Input", "Filter lower value must be a valid number.")
            return False
        
        # Validate filter upper value
        filter_upper_str = self.filter_upper_var.get().strip()
        if not filter_upper_str:
            tk.messagebox.showerror("Invalid Input", "Please enter a value for filter upper value.")
            return False
        try:
            filter_upper = float(filter_upper_str)
            if filter_upper <= 0:
                tk.messagebox.showerror("Invalid Input", "Filter upper value must be positive.")
                return False
        except ValueError:
            tk.messagebox.showerror("Invalid Input", "Filter upper value must be a valid number.")
            return False
        
        # Validate high transition bandwidth
        h_trans_str = self.h_trans_var.get().strip()
        if not h_trans_str:
            tk.messagebox.showerror("Invalid Input", "Please enter a value for high transition bandwidth.")
            return False
        try:
            h_trans = float(h_trans_str)
            if h_trans <= 0:
                tk.messagebox.showerror("Invalid Input", "High transition bandwidth must be positive.")
                return False
        except ValueError:
            tk.messagebox.showerror("Invalid Input", "High transition bandwidth must be a valid number.")
            return False
        
        # Validate low transition bandwidth
        l_trans_str = self.l_trans_var.get().strip()
        if not l_trans_str:
            tk.messagebox.showerror("Invalid Input", "Please enter a value for low transition bandwidth.")
            return False
        try:
            l_trans = float(l_trans_str)
            if l_trans <= 0:
                tk.messagebox.showerror("Invalid Input", "Low transition bandwidth must be positive.")
                return False
        except ValueError:
            tk.messagebox.showerror("Invalid Input", "Low transition bandwidth must be a valid number.")
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
        
        # Get threshold value
        snr_threshold = float(self.snr_threshold_var.get().strip()) if self.snr_rejection_var.get() in ["SNR", "CV"] else self.settings.get("snr_threshold", 8)

        # Get tmin value - handle based on baseline method
        if self.baseline_correction_var.get() == "xSecondsBefore":
            tmin_input = float(self.tmin_var.get().strip())
            tmin_value = -abs(tmin_input)  # Convert positive input to negative value
        else:
            tmin_value = 0  # Set to 0 for other methods
        
        # Convert reject criteria back to scientific notation
        reject_criteria_value = float(self.reject_criteria_var.get().strip()) * 1e-6
        
        # Parse unwanted labels
        unwanted_labels = self.parse_unwanted_labels(self.unwanted_labels_var.get().strip())
        
        self.result = {
        "short_channel_correction": self.short_channel_var.get(),
        "negative_correlation_enhancement": self.negative_corr_var.get(),
        "interpolate_bad_channels": self.interpolate_bad_channels_var.get(),
        "baseline_correction": self.baseline_correction_var.get(),
        "tmin": tmin_value,
        "filter_lower_value": float(self.filter_lower_var.get().strip()),      
        "filter_upper_value": float(self.filter_upper_var.get().strip()),     
        "h_trans_bandwidth": float(self.h_trans_var.get().strip()),           
        "l_trans_bandwidth": float(self.l_trans_var.get().strip()),           
        "scalp_coupling_threshold": float(self.coupling_threshold_var.get().strip()),
        "reject_criteria": {"hbo": reject_criteria_value},
        "unwanted": unwanted_labels,
        "snr_rejection": self.snr_rejection_var.get(),
        "snr_threshold": snr_threshold,
        "Apply_TDDR": self.apply_tddr_var.get()
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