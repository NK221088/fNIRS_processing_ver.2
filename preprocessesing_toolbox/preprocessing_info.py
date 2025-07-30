"""
Information dialogs for preprocessing options.
Contains all help text and info functions for the preprocessing dialog.
"""

import tkinter.messagebox as messagebox


class PreprocessingInfo:
    """Container for all preprocessing-related information dialogs."""
    
    @staticmethod
    def show_short_channel_info():
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
        messagebox.showinfo("Short Channel Correction", info_text)
    
    @staticmethod
    def show_negative_corr_info():
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
        messagebox.showinfo("Negative Correlation Enhancement", info_text)
    
    @staticmethod
    def show_baseline_correction_info(available_methods):
        """Show information about baseline correction."""
        methods_text = "\n".join([f"• {method}" for method in available_methods])
        
        info_text = (
            "Baseline Correction:\n\n"
            "Baseline correction removes drift and systematic changes in the signal "
            "that are not related to brain activation.\n\n"
            "Available methods:\n"
            f"{methods_text}\n\n"
            "Choose based on the type of baseline drift in your data."
        )
        messagebox.showinfo("Baseline Correction", info_text)
    
    @staticmethod
    def show_tmin_info():
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
        messagebox.showinfo("tmin Parameter", info_text)
    
    @staticmethod
    def show_filter_lower_info():
        """Show information about filter lower value."""
        info_text = (
            "Filter Lower Value (Hz):\n\n"
            "This sets the lower cutoff frequency for the bandpass filter.\n\n"
            "• Typical range: 0.01 - 0.1 Hz\n"
            "• Lower values: Preserve more slow variations\n"
            "• Higher values: Remove more slow drift\n\n"
            "This filter removes very slow changes and DC offset from the signal."
        )
        messagebox.showinfo("Filter Lower Value", info_text)
    
    @staticmethod
    def show_filter_upper_info():
        """Show information about filter upper value."""
        info_text = (
            "Filter Upper Value (Hz):\n\n"
            "This sets the upper cutoff frequency for the bandpass filter.\n\n"
            "• Typical range: 0.5 - 2.0 Hz\n"
            "• Lower values: More aggressive noise removal\n"
            "• Higher values: Preserve more signal details\n\n"
            "This filter removes high-frequency noise while preserving "
            "the hemodynamic response signal."
        )
        messagebox.showinfo("Filter Upper Value", info_text)
    
    @staticmethod
    def show_h_trans_info():
        """Show information about high transition bandwidth."""
        info_text = (
            "High Transition Bandwidth (Hz):\n\n"
            "This parameter controls the steepness of the filter's transition "
            "at the high-frequency cutoff.\n\n"
            "• Smaller values: Steeper transition (more selective)\n"
            "• Larger values: Gradual transition (less selective)\n\n"
            "Typical range: 0.1 - 0.5 Hz\n"
            "A steeper transition provides better frequency selectivity "
            "but may introduce ringing artifacts."
        )
        messagebox.showinfo("High Transition Bandwidth", info_text)
    
    @staticmethod
    def show_l_trans_info():
        """Show information about low transition bandwidth."""
        info_text = (
            "Low Transition Bandwidth (Hz):\n\n"
            "This parameter controls the steepness of the filter's transition "
            "at the low-frequency cutoff.\n\n"
            "• Smaller values: Steeper transition (more selective)\n"
            "• Larger values: Gradual transition (less selective)\n\n"
            "Typical range: 0.01 - 0.05 Hz\n"
            "A steeper transition provides better frequency selectivity "
            "but may introduce ringing artifacts."
        )
        messagebox.showinfo("Low Transition Bandwidth", info_text)
    
    @staticmethod
    def show_coupling_threshold_info():
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
        messagebox.showinfo("Scalp Coupling Threshold", info_text)
    
    @staticmethod
    def show_reject_criteria_info():
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
        messagebox.showinfo("HbO Rejection Threshold", info_text)
    
    @staticmethod
    def show_unwanted_labels_info():
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
        messagebox.showinfo("Unwanted Labels", info_text)
    
    @staticmethod
    def show_snr_rejection_info():
        """Display information about SNR rejection methods."""
        info_text = """
        SNR Rejection Method Information:
        
        Signal quality-based rejection helps remove channels with poor signal characteristics.
        
        Available Methods:
        • None: No signal quality rejection
        • SNR: Signal-to-Noise Ratio based rejection
          - Removes channels with SNR below threshold
          - Default threshold: 8
          - Higher values = stricter rejection
        • CV: Coefficient of Variation based rejection
          - Removes channels with CV above threshold
          - Default threshold: 0.15 (but uses same setting as SNR)
          - Lower values = stricter rejection
        
        Default: None
        
        Note: Quality rejection is typically applied after preprocessing 
        but before statistical analysis.
        """
        
        messagebox.showinfo("SNR Rejection Method", info_text)
    
    @staticmethod
    def show_snr_threshold_info():
        """Display information about SNR/CV threshold values."""
        info_text = """
        Threshold Value Information:
        
        The threshold value determines the sensitivity of quality rejection:
        
        For SNR Method:
        • Range: Typically 3-15
        • Default: 8
        • Higher values = stricter rejection (fewer channels kept)
        • Lower values = more lenient (more channels kept)
        
        For CV Method:
        • Range: Typically 0.05-0.3
        • Default: 0.15
        • Lower values = stricter rejection (fewer channels kept)
        • Higher values = more lenient (more channels kept)
        
        Note: The optimal threshold depends on your data quality 
        and experimental requirements.
        """

        messagebox.showinfo("Threshold Value", info_text)
    
    @staticmethod
    def show_interpolate_info():
        """Display information about interpolation of bad channels."""
        info_text = """
        Interpolate Bad Channels:
        
        Interpolation is used to estimate and replace data in channels 
        that have been marked as 'bad' due to excessive noise or signal loss.
        
        Key Points:
        • Only channels marked as bad (raw.info['bads']) are interpolated.
        • Uses neighboring good channels to estimate the missing signal.
        • Helps preserve spatial consistency in the data.
        
        When to Use:
        • After identifying bad channels (e.g., from visual inspection or metrics)
        • Before further analysis like averaging or source localization
        
        MNE Method Used:
        • raw.interpolate_bads()

        Note:
        • Interpolation is not a substitute for good data quality.
        • Should be used with care, especially if many channels are bad.
        """
        messagebox.showinfo("Interpolate Bad Channels", info_text)
