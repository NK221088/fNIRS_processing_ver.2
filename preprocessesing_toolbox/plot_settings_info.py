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
            "• GFP: Global Field Power - calculates the standard deviation across channels\n\n"
            "Choose 'mean' for standard averaging, 'median' for outlier-resistant averaging, "
            "or 'gfp' for analyzing overall signal strength patterns."
        )
        tkinter.messagebox.showinfo("Combine Strategy", info_text)
    
    @staticmethod
    def show_bad_channels_info():
        """Show information about bad channels strategy."""
        info_text = (
            "Bad Channels Strategy:\n\n"
            "• All: Include all channels regardless of quality\n"
            "• Delete: Remove channels marked as bad from analysis\n"
            "• Threshold: Remove channels based on signal quality threshold\n\n"
            "Bad channels can contain artifacts or poor signal quality that may "
            "affect your analysis results."
        )
        tkinter.messagebox.showinfo("Bad Channels Strategy", info_text)
    
    @staticmethod
    def show_interpolate_info():
        """Show information about channel interpolation."""
        info_text = (
            "Interpolate Bad Channels:\n\n"
            "When enabled, bad channels are reconstructed using interpolation "
            "from neighboring good channels rather than being removed entirely.\n\n"
            "This can help maintain spatial continuity in your data while "
            "still addressing problematic channels."
        )
        tkinter.messagebox.showinfo("Interpolate Bad Channels", info_text)
    
    @staticmethod
    def show_threshold_info():
        """Show information about threshold setting."""
        info_text = (
            "Threshold:\n\n"
            "Sets the quality threshold for determining bad channels when "
            "using 'threshold' strategy.\n\n"
            "Higher values are more permissive (fewer channels marked as bad), "
            "lower values are more strict (more channels marked as bad).\n\n"
            "Typical range: 1-5, default is 3."
        )
        tkinter.messagebox.showinfo("Threshold", info_text)

    @staticmethod
    def show_save_plot_info():
        """Show information about saving plots."""
        info_text = (
            "Save Plot to File:\n\n"
            "When enabled, each generated plot will be automatically saved as a PDF file.\n\n"
            "Saved plots are stored in the 'Plots' directory. Filenames are automatically generated using:\n"
            "- The epoch type\n"
            "- The plot type (e.g., 'power', 'topomap')\n"
            "- The bad channel handling strategy\n"
            "- The dataset name\n"
            "- The current timestamp\n\n"
            "Example filename:\n"
            "'stimulus_epochs_plot_power_interpolate_datasetA_20250718_153000.pdf'\n\n"
            "This makes it easy to track plots generated under different analysis conditions."
        )
        tkinter.messagebox.showinfo("Save Plot", info_text)
