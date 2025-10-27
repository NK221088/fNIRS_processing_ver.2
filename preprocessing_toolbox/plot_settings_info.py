"""
Information dialogs for plot settings options.
Contains all help text and info functions for the plot settings dialog.
"""

import tkinter.messagebox as messagebox


class PlotSettingsInfo:
    """Container for all plot settings-related information dialogs."""
    
    @staticmethod
    def show_combine_strategy_info():
        """Show information about combine strategy."""
        info_text = (
            "Combine Strategy:\n\n"
            "Determines how multiple data channels are combined for visualization:\n\n"
            "• Mean: Average all selected channels\n"
            "• Median: Use median value of selected channels\n"
            "• GFP: Global Field Power - root mean square across channels\n\n"
            "The mean strategy is most commonly used for standard fNIRS analysis."
        )
        messagebox.showinfo("Combine Strategy", info_text)
    
    @staticmethod
    def show_bad_channels_info():
        """Show information about bad channels strategy."""
        info_text = (
            "Bad Channels Strategy:\n\n"
            "Defines how channels with poor signal quality are handled:\n\n"
            "• All: Include all channels regardless of quality\n"
            "• Delete: Remove bad channels from analysis entirely\n"
            "• Threshold: Apply quality threshold to determine inclusion\n\n"
            "Bad channels are typically identified based on signal-to-noise ratio "
            "or other quality metrics during preprocessing."
        )
        messagebox.showinfo("Bad Channels Strategy", info_text)
    
    @staticmethod
    def show_threshold_info():
        """Show information about threshold parameter."""
        info_text = (
            "Threshold:\n\n"
            "Quality threshold value used when 'threshold' bad channels strategy "
            "is selected.\n\n"
            "• Higher values: More stringent quality requirements\n"
            "• Lower values: More lenient quality requirements\n"
            "• Typical range: 1-10\n\n"
            "Channels with quality metrics below this threshold will be "
            "excluded from the analysis."
        )
        messagebox.showinfo("Threshold", info_text)
    
    @staticmethod
    def show_compare_with_raw_info():
        """Show information about comparing with raw data."""
        info_text = (
            "Compare with Raw Data:\n\n"
            "When enabled, this option displays both the preprocessed fNIRS data "
            "and the original raw data side by side in the same plot.\n\n"
            "Benefits:\n"
            "• Visual inspection of preprocessing effects\n"
            "• Quality assessment of signal cleaning\n"
            "• Identification of artifact removal effectiveness\n"
            "• Validation of preprocessing pipeline performance\n\n"
            "The raw data appears in a lighter color or different line style, "
            "while the preprocessed data is highlighted as the primary signal.\n\n"
            "This comparison is particularly useful for:\n"
            "• Motion artifact correction evaluation\n"
            "• Filter effects visualization\n"
            "• Baseline correction assessment\n"
            "• Overall preprocessing validation"
        )
        messagebox.showinfo("Compare with Raw Data", info_text)
    
    @staticmethod
    def show_save_plot_info():
        """Show information about saving plots."""
        info_text = (
            "Save Plot to File:\n\n"
            "When enabled, the generated plot will be automatically saved "
            "to a file in addition to being displayed.\n\n"
            "File Details:\n"
            "• Format: High-resolution PNG or PDF\n"
            "• Location: Same directory as the data file\n"
            "• Naming: Based on plot type and timestamp\n\n"
            "Useful for:\n"
            "• Documentation and reporting\n"
            "• Publication preparation\n"
            "• Batch processing workflows\n"
            "• Creating analysis records"
        )
        messagebox.showinfo("Save Plot", info_text)