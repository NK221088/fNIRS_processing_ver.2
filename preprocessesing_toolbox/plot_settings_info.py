import tkinter.messagebox


class PlotSettingsInfo:
    """Container for all plot settings-related information dialogs."""

    @staticmethod
    def show_combine_strategy_info():
        """Show information about combine strategy options."""
        info_text = (
            "Combine Strategy:\n\n"
            "• Mean: Averages all epochs for each channel\n"
            "• Median: Uses the median value across epochs (more robust to outliers)\n"
            "• Sum: Sums the values across epochs\n"
            "• GFP: Global Field Power – calculates the standard deviation across channels\n\n"
            "Choose 'mean' for standard averaging, 'median' for outlier-resistant averaging, "
            "'sum' for cumulative signal strength, or 'gfp' to analyze global activity."
        )
        tkinter.messagebox.showinfo("Combine Strategy", info_text)

    @staticmethod
    def show_bad_channels_info():
        """Show information about bad channels strategy."""
        info_text = (
            "Bad Channels Strategy:\n\n"
            "• All: Combines all bad channels across all subjects and applies to each\n"
            "• Delete: Removes all channels marked as bad from each subject individually\n"
            "• Threshold: Removes channels marked bad in more than a certain number of epochs\n\n"
            "This affects which channels are retained for evoked response plots."
        )
        tkinter.messagebox.showinfo("Bad Channels Strategy", info_text)

    @staticmethod
    def show_interpolate_info():
        """Show information about channel interpolation."""
        info_text = (
            "Interpolate Bad Channels:\n\n"
            "When enabled, bad channels are reconstructed using data from neighboring channels "
            "rather than being removed.\n\n"
            "This can preserve spatial consistency, but should be used cautiously depending on analysis goals."
        )
        tkinter.messagebox.showinfo("Interpolate Bad Channels", info_text)

    @staticmethod
    def show_threshold_info():
        """Show information about threshold setting."""
        info_text = (
            "Threshold:\n\n"
            "Used only with the 'threshold' bad channel strategy.\n"
            "Channels that appear as bad more than the specified number of times "
            "across subjects will be excluded.\n\n"
            "Higher thresholds are more lenient; lower thresholds are stricter.\n"
            "Example: threshold = 3 means a channel must be marked bad in more than 3 epochs to be removed."
        )
        tkinter.messagebox.showinfo("Threshold", info_text)

    @staticmethod
    def show_save_plot_info():
        """Show information about saving plots."""
        info_text = (
            "Save Plot to File:\n\n"
            "When enabled, each generated plot is saved as a PDF file in the 'Plots' directory.\n\n"
            "For standard fNIRS response plots, the filename will look like:\n"
            "  'standard_fNIRS_response_plot_<timestamp>.pdf'\n\n"
            "For other plots, the filename includes:\n"
            "  - Epoch type\n"
            "  - Plot type (e.g., 'topomap', 'power')\n"
            "  - Bad channel strategy\n"
            "  - Dataset name\n"
            "  - Timestamp\n\n"
            "Example:\n"
            "  'stimulus_epochs_plot_power_threshold_datasetX_2025-07-18_15-30-00.pdf'"
        )
        tkinter.messagebox.showinfo("Save Plot", info_text)
