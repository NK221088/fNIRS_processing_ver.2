# plot_config.py
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
from enum import Enum

class PlotType(Enum):
    EPOCH_PLOT = "Epoch Plot"
    STANDARD_RESPONSE = "Standard fNIRS Response Plot"
    PARADIGM_PLOT = "paradigm_plot"
    FREQUENCY_PLOT = "individual frequency plot"
    STATISTICAL_ANALYSIS = "Statistical Analysis"

class WidgetType(Enum):
    COMBOBOX = "combobox"
    CHECKBOX = "checkbox"
    ENTRY = "entry"
    SPINBOX = "spinbox"
    MULTI_SELECT = "multi_select"
    CHANNEL_SELECTOR = "channel_selector"
    INDIVIDUAL_SELECTOR = "individual_selector"

@dataclass
class WidgetConfig:
    """Configuration for individual widgets"""
    name: str
    widget_type: WidgetType
    label: str
    required: bool = True
    visible: bool = True
    default_value: Any = None
    values: Optional[List[str]] = None
    min_value: Optional[int] = None
    max_value: Optional[int] = None
    validator: Optional[Callable] = None
    help_text: Optional[str] = None

@dataclass
class PlotTypeConfig:
    """Configuration for each plot type"""
    name: str
    plot_type: PlotType
    required_widgets: List[str]
    optional_widgets: List[str]
    supports_multiple_individuals: bool = False
    supports_all_individuals: bool = False
    requires_channel_selection: bool = False
    requires_hemoglobin_type: bool = False
    uses_tabs: bool = False
    description: Optional[str] = None

# Define plot configurations
PLOT_CONFIGS = {
    PlotType.EPOCH_PLOT: PlotTypeConfig(
        name="Epoch Plot",
        plot_type=PlotType.EPOCH_PLOT,
        required_widgets=["dataset", "epoch_type", "individual", "channels"],
        optional_widgets=["combine_strategy", "bad_channels_strategy", "short_channel", 
                         "negative_corr", "interpolate", "threshold"],
        supports_multiple_individuals=False,
        supports_all_individuals=True,
        requires_channel_selection=True,
        requires_hemoglobin_type=False,
        uses_tabs=False,
        description="Plot epochs for individual participants or all participants combined"
    ),
    
    PlotType.STANDARD_RESPONSE: PlotTypeConfig(
        name="Standard fNIRS Response Plot",
        plot_type=PlotType.STANDARD_RESPONSE,
        required_widgets=["dataset", "individual_selection", "channels"],
        optional_widgets=["combine_strategy", "bad_channels_strategy", "short_channel",
                         "negative_corr", "interpolate", "threshold"],
        supports_multiple_individuals=True,
        supports_all_individuals=False,
        requires_channel_selection=True,
        requires_hemoglobin_type=False,
        uses_tabs=False,
        description="Standard fNIRS response plot with multiple individual selection"
    ),
    
    PlotType.PARADIGM_PLOT: PlotTypeConfig(
        name="Paradigm Plot",
        plot_type=PlotType.PARADIGM_PLOT,
        required_widgets=["dataset", "individual", "haemo_type", "channels"],
        optional_widgets=["short_channel", "negative_corr", "interpolate"],
        supports_multiple_individuals=False,
        supports_all_individuals=False,
        requires_channel_selection=True,
        requires_hemoglobin_type=True,
        uses_tabs=False,
        description="Plot paradigm data for a specific individual and hemoglobin type"
    ),
    
    PlotType.FREQUENCY_PLOT: PlotTypeConfig(
        name="Individual Frequency Plot",
        plot_type=PlotType.FREQUENCY_PLOT,
        required_widgets=["dataset", "individual"],
        optional_widgets=["channels"],
        supports_multiple_individuals=False,
        supports_all_individuals=False,
        requires_channel_selection=False,
        requires_hemoglobin_type=False,
        uses_tabs=False,
        description="Frequency analysis plot for individual participants"
    ),
    
    PlotType.STATISTICAL_ANALYSIS: PlotTypeConfig(
        name="Statistical Analysis",
        plot_type=PlotType.STATISTICAL_ANALYSIS,
        required_widgets=["area_of_interest", "time_window_start", "time_window_end", "dataset1", "dataset2"],
        optional_widgets=["channels"],
        supports_multiple_individuals=False,
        supports_all_individuals=False,
        requires_channel_selection=False,
        requires_hemoglobin_type=False,
        uses_tabs=True,
        description="Statistical analysis comparing two datasets"
    )
}

# Widget definitions
WIDGET_CONFIGS = {
    "dataset": WidgetConfig(
        name="dataset",
        widget_type=WidgetType.COMBOBOX,
        label="Select Dataset:",
        required=True,
        help_text="Choose the dataset for analysis"
    ),
    
    "epoch_type": WidgetConfig(
        name="epoch_type",
        widget_type=WidgetType.COMBOBOX,
        label="Epoch Type:",
        required=False,
        default_value="Tapping",
        help_text="Select the type of epochs to analyze"
    ),
    
    "plot_type": WidgetConfig(
        name="plot_type",
        widget_type=WidgetType.COMBOBOX,
        label="Select Plot Type:",
        required=True,
        values=[config.name for config in PLOT_CONFIGS.values()],
        default_value="Epoch Plot",
        help_text="Choose the type of plot to generate"
    ),
    
    "individual": WidgetConfig(
        name="individual",
        widget_type=WidgetType.INDIVIDUAL_SELECTOR,
        label="Select Individual:",
        required=False,
        help_text="Choose individual participant or all participants"
    ),
    
    "individual_selection": WidgetConfig(
        name="individual_selection",
        widget_type=WidgetType.MULTI_SELECT,
        label="Select Individuals:",
        required=True,
        help_text="Select multiple individuals for comparison"
    ),
    
    "channels": WidgetConfig(
        name="channels",
        widget_type=WidgetType.CHANNEL_SELECTOR,
        label="Select Channels:",
        required=False,
        help_text="Select the channels to include in the analysis"
    ),
    
    "combine_strategy": WidgetConfig(
        name="combine_strategy",
        widget_type=WidgetType.COMBOBOX,
        label="Combine Strategy:",
        required=False,
        values=["mean", "median"],
        default_value="mean",
        help_text="Strategy for combining multiple epochs"
    ),
    
    "bad_channels_strategy": WidgetConfig(
        name="bad_channels_strategy",
        widget_type=WidgetType.COMBOBOX,
        label="Bad Channels Strategy:",
        required=False,
        values=["all", "interpolate", "exclude"],
        default_value="all",
        help_text="How to handle bad channels"
    ),
    
    "short_channel": WidgetConfig(
        name="short_channel",
        widget_type=WidgetType.CHECKBOX,
        label="Short Channel Correction:",
        required=False,
        default_value=True,
        help_text="Apply short channel correction"
    ),
    
    "negative_corr": WidgetConfig(
        name="negative_corr",
        widget_type=WidgetType.CHECKBOX,
        label="Negative Correlation Enhancement:",
        required=False,
        default_value=False,
        help_text="Apply negative correlation enhancement"
    ),
    
    "interpolate": WidgetConfig(
        name="interpolate",
        widget_type=WidgetType.CHECKBOX,
        label="Interpolate Bad Channels:",
        required=False,
        default_value=False,
        help_text="Interpolate bad channels"
    ),
    
    "threshold": WidgetConfig(
        name="threshold",
        widget_type=WidgetType.SPINBOX,
        label="Threshold:",
        required=False,
        default_value=3,
        min_value=1,
        max_value=10,
        help_text="Threshold value for analysis"
    ),
    
    "haemo_type": WidgetConfig(
        name="haemo_type",
        widget_type=WidgetType.COMBOBOX,
        label="Hemoglobin Type:",
        required=False,
        values=["hbo", "hbr", "hbt"],
        default_value="hbo",
        help_text="Type of hemoglobin to analyze"
    ),
    
    "area_of_interest": WidgetConfig(
        name="area_of_interest",
        widget_type=WidgetType.COMBOBOX,
        label="Area of Interest:",
        required=True,
        values=["SMA", "M1", "PMC", "S1"],
        default_value="SMA",
        help_text="Brain area of interest for statistical analysis"
    ),
    
    "time_window_start": WidgetConfig(
        name="time_window_start",
        widget_type=WidgetType.ENTRY,
        label="Start Time (s):",
        required=True,
        default_value="3.0",
        help_text="Start time for analysis window"
    ),
    
    "time_window_end": WidgetConfig(
        name="time_window_end",
        widget_type=WidgetType.ENTRY,
        label="End Time (s):",
        required=True,
        default_value="12.0",
        help_text="End time for analysis window"
    ),
    
    "dataset1": WidgetConfig(
        name="dataset1",
        widget_type=WidgetType.COMBOBOX,
        label="Dataset 1:",
        required=True,
        help_text="First dataset for comparison"
    ),
    
    "dataset2": WidgetConfig(
        name="dataset2",
        widget_type=WidgetType.COMBOBOX,
        label="Dataset 2:",
        required=True,
        default_value="fNIrs_motor",
        help_text="Second dataset for comparison"
    )
}

def get_plot_config(plot_type: PlotType) -> PlotTypeConfig:
    """Get configuration for a specific plot type"""
    return PLOT_CONFIGS.get(plot_type)

def get_widget_config(widget_name: str) -> WidgetConfig:
    """Get configuration for a specific widget"""
    return WIDGET_CONFIGS.get(widget_name)

def get_required_widgets(plot_type: PlotType) -> List[str]:
    """Get list of required widgets for a plot type"""
    config = get_plot_config(plot_type)
    return config.required_widgets if config else []

def get_optional_widgets(plot_type: PlotType) -> List[str]:
    """Get list of optional widgets for a plot type"""
    config = get_plot_config(plot_type)
    return config.optional_widgets if config else []

def get_all_widgets_for_plot(plot_type: PlotType) -> List[str]:
    """Get all widgets (required + optional) for a plot type"""
    config = get_plot_config(plot_type)
    if not config:
        return []
    return config.required_widgets + config.optional_widgets

def validate_plot_configuration(plot_type: PlotType, widget_values: Dict[str, Any]) -> List[str]:
    """Validate that all required widgets have values"""
    errors = []
    required_widgets = get_required_widgets(plot_type)
    
    for widget_name in required_widgets:
        if widget_name not in widget_values or widget_values[widget_name] is None:
            widget_config = get_widget_config(widget_name)
            label = widget_config.label if widget_config else widget_name
            errors.append(f"Required field '{label}' is missing")
    
    return errors