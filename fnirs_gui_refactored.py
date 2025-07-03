# main_gui_refactored.py
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from load_data_function import data_loaders, load_data
from epoch_plot import epoch_plot
from standard_fNIRS_response_plot import standard_fNIRS_response_plot
from paradigm_plot import paradigm_plot
from individual_frequency_plot import individual_frequency_plot
from statistical_analysis import statistical_analysis

from plot_config import (
    PlotType, PLOT_CONFIGS, get_plot_config, get_widget_config,
    get_required_widgets, get_optional_widgets, get_all_widgets_for_plot,
    validate_plot_configuration
)
from widget_factory import WidgetFactory, BaseWidget


@dataclass
class AppSettings:
    """Centralized settings configuration"""
    data_set: str = ""
    epoch_type: str = "Tapping"
    combine_strategy: str = "mean"
    short_channel_correction: bool = True
    negative_correlation_enhancement: bool = False
    interpolate_bad_channels: bool = False
    bad_channels_strategy: str = "all"
    threshold: int = 3
    plot_type: PlotType = PlotType.EPOCH_PLOT
    individual: bool = True
    haemo_type: str = "hbo"
    area_of_interest: str = "SMA"
    time_window_start: float = 3.0
    time_window_end: float = 12.0
    dataset1: str = ""
    dataset2: str = "fNIrs_motor"
    
    def __post_init__(self):
        if not self.data_set:
            self.data_set = list(data_loaders.keys())[0]
        if not self.dataset1:
            self.dataset1 = self.data_set


class DataManager:
    """Handles data loading and caching"""
    
    def __init__(self):
        self.all_epochs = []
        self.data_name = ""
        self.all_data = None
        self.freq = None
        self.data_types = []
        self.all_individuals = []
        self.bad_channels = []  # Add bad channels storage
        self._cache = {}
        self._previous_settings = {}
    
    def needs_reload(self, settings: AppSettings) -> bool:
        """Check if data needs to be reloaded based on settings changes"""
        key_settings = {
            'data_set': settings.data_set,
            'epoch_type': settings.epoch_type,
            'short_channel_correction': settings.short_channel_correction,
            'negative_correlation_enhancement': settings.negative_correlation_enhancement,
            'interpolate_bad_channels': settings.interpolate_bad_channels,
        }
        
        if key_settings != self._previous_settings:
            self._previous_settings = key_settings.copy()
            return True
        return False
    
    def load_data(self, settings: AppSettings) -> bool:
        """Load data based on settings"""
        try:
            self.all_epochs, self.data_name, self.all_data, self.freq, self.data_types, self.all_individuals = load_data(
                data_set=settings.data_set,
                short_channel_correction=settings.short_channel_correction,
                negative_correlation_enhancement=settings.negative_correlation_enhancement,
                interpolate_bad_channels=settings.interpolate_bad_channels,
                individuals=settings.individual
            )
            
            # Extract bad channels if available
            if self.all_individuals and hasattr(self.all_individuals[0], 'epochs'):
                # Get bad channels from first individual's epochs
                epochs = self.all_individuals[0].epochs
                if hasattr(epochs, 'info') and hasattr(epochs.info, 'bads'):
                    self.bad_channels = epochs.info.bads
                else:
                    self.bad_channels = []
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False


class PlotDisplayManager:
    """Manages plot display in the GUI"""
    
    def __init__(self, parent):
        self.parent = parent
        self.tab_control = None
    
    def clear_display(self):
        """Clear all existing plots"""
        for widget in self.parent.winfo_children():
            widget.destroy()
        self.tab_control = None
    
    def display_figures(self, figures: List[Any], use_tabs: bool = False):
        """Display matplotlib figures"""
        if not figures:
            tk.Label(self.parent, text="No plots to display", font=("Arial", 14)).pack(pady=20)
            return
        
        # Ensure figures is a list
        if not isinstance(figures, list):
            figures = [figures]
        
        # Flatten nested lists
        flattened_figures = []
        for fig in figures:
            if isinstance(fig, list):
                flattened_figures.extend(fig)
            else:
                flattened_figures.append(fig)
        figures = flattened_figures
        
        if use_tabs and len(figures) > 1:
            self._display_with_tabs(figures)
        else:
            self._display_without_tabs(figures)
    
    def _display_with_tabs(self, figures: List[Any]):
        """Display figures in tabs"""
        self.tab_control = ttk.Notebook(self.parent)
        self.tab_control.pack(expand=True, fill="both")
        
        for i, fig in enumerate(figures):
            tab = ttk.Frame(self.tab_control)
            self.tab_control.add(tab, text=f"Plot {i+1}")
            
            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, pady=5)
    
    def _display_without_tabs(self, figures: List[Any]):
        """Display figures without tabs"""
        for fig in figures:
            canvas = FigureCanvasTkAgg(fig, master=self.parent)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True, pady=5)


class PlotGenerator:
    """Handles plot generation based on settings"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
    
    def generate_plot(self, plot_type: PlotType, widget_values: Dict[str, Any]) -> List[Any]:
        """Generate plot based on plot type and widget values"""
        
        if plot_type == PlotType.EPOCH_PLOT:
            return self._generate_epoch_plot(widget_values)
        elif plot_type == PlotType.STANDARD_RESPONSE:
            return self._generate_standard_response_plot(widget_values)
        elif plot_type == PlotType.PARADIGM_PLOT:
            return self._generate_paradigm_plot(widget_values)
        elif plot_type == PlotType.FREQUENCY_PLOT:
            return self._generate_frequency_plot(widget_values)
        elif plot_type == PlotType.STATISTICAL_ANALYSIS:
            return self._generate_statistical_analysis(widget_values)
        else:
            return []
    
    def _generate_epoch_plot(self, values: Dict[str, Any]) -> List[Any]:
        """Generate epoch plot"""
        selected_channels = values.get('channels', [])
        selected_individual = values.get('individual', 'All Individuals')
        epoch_type = values.get('epoch_type', 'Tapping')
        combine_strategy = values.get('combine_strategy', 'mean')
        bad_channels_strategy = values.get('bad_channels_strategy', 'all')
        threshold = values.get('threshold', 3)
        
        picks = selected_channels if selected_channels and len(selected_channels) < len(self.data_manager.all_individuals[0].epochs.ch_names) else "all"
        
        if selected_individual == "All Individuals":
            return [epoch_plot(
                self.data_manager.all_epochs, picks=picks, epoch_type=epoch_type,
                combine_strategy=combine_strategy, save=False,
                bad_channels_strategy=bad_channels_strategy,
                threshold=threshold, data_set=self.data_manager.data_name
            )]
        else:
            individual_index = self._find_individual_index(selected_individual)
            if individual_index is not None:
                individual_data = self.data_manager.all_individuals[individual_index]
                return [epoch_plot(
                    [individual_data.epochs], picks=picks, epoch_type=epoch_type,
                    combine_strategy=combine_strategy, save=False,
                    bad_channels_strategy=bad_channels_strategy,
                    threshold=threshold, data_set=self.data_manager.data_name
                )]
        return []
    
    def _generate_standard_response_plot(self, values: Dict[str, Any]) -> List[Any]:
        """Generate standard fNIRS response plot"""
        selected_individuals = values.get('individual_selection', [])
        selected_channels = values.get('channels', [])
        combine_strategy = values.get('combine_strategy', 'mean')
        bad_channels_strategy = values.get('bad_channels_strategy', 'all')
        threshold = values.get('threshold', 3)
        
        if not selected_individuals:
            return []
        
        selected_all_epochs = []
        for ind_name in selected_individuals:
            individual = next((ind for ind in self.data_manager.all_individuals 
                             if getattr(ind, "name", "") == ind_name), None)
            if individual:
                selected_all_epochs.append(individual.epochs)
        
        picks = selected_channels if selected_channels and len(selected_channels) < len(self.data_manager.all_individuals[0].epochs.ch_names) else "all"
        
        return [standard_fNIRS_response_plot(
            selected_all_epochs, self.data_manager.data_types,
            bad_channels_strategy=bad_channels_strategy,
            save=False, combine_strategy=combine_strategy,
            threshold=threshold, data_set=self.data_manager.data_name,
            picks_=picks
        )]
    
    def _generate_paradigm_plot(self, values: Dict[str, Any]) -> List[Any]:
        """Generate paradigm plot"""
        selected_channels = values.get('channels', [])
        selected_individual = values.get('individual', '')
        haemo_type = values.get('haemo_type', 'hbo')
        
        picks = [f"{channel} {haemo_type}" for channel in selected_channels]
        individual_index = self._find_individual_index(selected_individual)
        
        if individual_index is not None:
            return [paradigm_plot(
                self.data_manager.all_individuals[individual_index],
                picks_=picks,
                haemo_type=haemo_type
            )]
        return []
    
    def _generate_frequency_plot(self, values: Dict[str, Any]) -> List[Any]:
        """Generate frequency plot"""
        selected_individual = values.get('individual', '')
        individual_index = self._find_individual_index(selected_individual)
        
        if individual_index is not None:
            return [individual_frequency_plot(self.data_manager.all_individuals[individual_index])]
        return []
    
    def _generate_statistical_analysis(self, values: Dict[str, Any]) -> List[Any]:
        """Generate statistical analysis"""
        area_of_interest = values.get('area_of_interest', 'SMA')
        start_time = float(values.get('time_window_start', 3.0))
        end_time = float(values.get('time_window_end', 12.0))
        dataset1 = values.get('dataset1', '')
        dataset2 = values.get('dataset2', 'fNIrs_motor')
        
        return statistical_analysis(
            Area_of_interest=area_of_interest,
            start_time=start_time,
            end_time=end_time,
            dataset1=dataset1,
            dataset2=dataset2
        )
    
    def _find_individual_index(self, individual_name: str) -> Optional[int]:
        """Find individual index by name"""
        for i, ind in enumerate(self.data_manager.all_individuals):
            if getattr(ind, "name", f"Participant_{i+1}") == individual_name:
                return i
        return None


class fNIRSGUI:
    """Main GUI application class"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("fNIRS Data Analysis")
        self.root.geometry("1200x800")
        
        self.settings = AppSettings()
        self.data_manager = DataManager()
        self.plot_generator = PlotGenerator(self.data_manager)
        
        self.widgets: Dict[str, BaseWidget] = {}
        self.current_plot_type = PlotType.EPOCH_PLOT
        
        self._setup_ui()
        self._setup_callbacks()
        
        # Initial data load
        self._load_initial_data()
    
    def _setup_ui(self):
        """Setup the main UI layout"""
        # Main container
        main_container = tk.Frame(self.root)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel (controls)
        left_panel = tk.Frame(main_container)
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        
        # Controls frame
        self.controls_frame = tk.Frame(left_panel)
        self.controls_frame.pack(side="top", fill="both", expand=True)
        
        # Button frame
        button_frame = tk.Frame(left_panel)
        button_frame.pack(side="bottom", fill="x", pady=(10, 0))
        
        # Right panel (plots)
        self.plot_frame = tk.Frame(main_container)
        self.plot_frame.pack(side="right", fill="both", expand=True)
        
        # Create initial widgets
        self._create_initial_widgets()
        self._create_run_button(button_frame)
        
        # Create plot display manager
        self.plot_display = PlotDisplayManager(self.plot_frame)
    
    def _create_initial_widgets(self):
        """Create initial widgets that are always visible"""
        # Always show dataset and plot_type widgets
        self.widgets['dataset'] = WidgetFactory.create_widget(self.controls_frame, 'dataset')
        self.widgets['plot_type'] = WidgetFactory.create_widget(self.controls_frame, 'plot_type')
        
        # Populate dataset widget
        if hasattr(self.widgets['dataset'], 'update_values'):
            self.widgets['dataset'].update_values(list(data_loaders.keys()))
        
        # Show initial widgets
        self.widgets['dataset'].show()
        self.widgets['plot_type'].show()
    
    def _create_run_button(self, parent):
        """Create the run analysis button"""
        self.run_button = tk.Button(parent, text="Run Analysis", command=self._run_analysis,
                                   bg="green", fg="white", font=("Arial", 12, "bold"),
                                   padx=20, pady=10)
        self.run_button.pack(fill="x")
    
    def _setup_callbacks(self):
        """Setup widget callbacks"""
        self.widgets['dataset'].bind_callback(self._on_dataset_change)
        self.widgets['plot_type'].bind_callback(self._on_plot_type_change)
    
    def _on_dataset_change(self, *args):
        """Handle dataset change"""
        self.settings.data_set = self.widgets['dataset'].get_value()
        if self.data_manager.load_data(self.settings):
            self._update_dynamic_widgets()
    
    def _on_plot_type_change(self, *args):
        """Handle plot type change"""
        plot_type_name = self.widgets['plot_type'].get_value()
        self.current_plot_type = self._get_plot_type_from_name(plot_type_name)
        self._update_widgets_for_plot_type()
    
    def _on_individual_change(self, *args):
        """Handle individual selection change"""
        self._update_channels()
    
    def _on_haemo_type_change(self, *args):
        """Handle hemoglobin type change"""
        if self.current_plot_type == PlotType.PARADIGM_PLOT:
            # Preserve current channel selections
            current_selected = []
            if 'channels' in self.widgets:
                current_selected = self.widgets['channels'].get_value()
            
            # Update channels
            self._update_channels()
            
            # Restore selections if possible
            if current_selected and 'channels' in self.widgets:
                self.widgets['channels'].set_value(current_selected)
    
    def _on_individual_selection_change(self, *args):
        """Handle individual selection change for multi-select"""
        self._update_channels()
    
    def _get_plot_type_from_name(self, name: str) -> PlotType:
        """Convert plot type name to enum"""
        for plot_type, config in PLOT_CONFIGS.items():
            if config.name == name:
                return plot_type
        return PlotType.EPOCH_PLOT
    
    def _update_widgets_for_plot_type(self):
        """Update widgets based on selected plot type"""
        # Hide all widgets except dataset and plot_type
        for name, widget in self.widgets.items():
            if name not in ['dataset', 'plot_type']:
                widget.hide()
        
        # Get required widgets for current plot type
        required_widgets = get_required_widgets(self.current_plot_type)
        optional_widgets = get_optional_widgets(self.current_plot_type)
        all_widgets = required_widgets + optional_widgets
        
        # Create and show widgets for current plot type
        for widget_name in all_widgets:
            if widget_name not in self.widgets:
                try:
                    self.widgets[widget_name] = WidgetFactory.create_widget(self.controls_frame, widget_name)
                except Exception as e:
                    print(f"Error creating widget {widget_name}: {e}")
                    continue
            
            if widget_name in self.widgets:
                # Setup callbacks for specific widgets
                if widget_name == 'individual':
                    self.widgets[widget_name].bind_callback(self._on_individual_change)
                elif widget_name == 'haemo_type':
                    self.widgets[widget_name].bind_callback(self._on_haemo_type_change)
                elif widget_name == 'individual_selection':
                    self.widgets[widget_name].bind_callback(self._on_individual_selection_change)
                
                self.widgets[widget_name].show()
        
        # Update dynamic content
        self._update_dynamic_widgets()
    
    def _update_dynamic_widgets(self):
        """Update dynamic content of widgets"""
        # Update epoch types
        if 'epoch_type' in self.widgets and hasattr(self.widgets['epoch_type'], 'update_values'):
            self.widgets['epoch_type'].update_values(self.data_manager.data_types)
        
        # Update individuals
        if 'individual' in self.widgets and hasattr(self.widgets['individual'], 'update_individuals'):
            plot_config = get_plot_config(self.current_plot_type)
            include_all = plot_config.supports_all_individuals if plot_config else False
            self.widgets['individual'].update_individuals(self.data_manager.all_individuals, include_all)
        
        # Update individual selection (multi-select)
        if 'individual_selection' in self.widgets and hasattr(self.widgets['individual_selection'], 'update_values'):
            individual_names = [getattr(ind, "name", f"Participant_{i+1}") 
                              for i, ind in enumerate(self.data_manager.all_individuals)]
            self.widgets['individual_selection'].update_values(individual_names)
        
        # Update dataset widgets
        datasets = list(data_loaders.keys())
        for dataset_widget in ['dataset1', 'dataset2']:
            if dataset_widget in self.widgets and hasattr(self.widgets[dataset_widget], 'update_values'):
                self.widgets[dataset_widget].update_values(datasets)
        
        # Update channels
        self._update_channels()
    
    def _update_channels(self):
        """Update channel options based on current selection"""
        if 'channels' not in self.widgets or not self.data_manager.all_individuals:
            return
        
        plot_config = get_plot_config(self.current_plot_type)
        if not plot_config or not hasattr(plot_config, 'requires_channel_selection') or not plot_config.requires_channel_selection:
            return
        
        # Get current selections
        selected_individual = None
        if 'individual' in self.widgets:
            selected_individual = self.widgets['individual'].get_value()
        
        selected_individuals = []
        if 'individual_selection' in self.widgets:
            selected_individuals = self.widgets['individual_selection'].get_value()
        
        # Get bad channels
        bad_channels = getattr(self.data_manager, 'bad_channels', [])
        
        if self.current_plot_type == PlotType.PARADIGM_PLOT:
            # For paradigm plot, use individual selection
            if selected_individual and selected_individual != "All Individuals":
                individual_index = self.plot_generator._find_individual_index(selected_individual)
                if individual_index is not None:
                    individual = self.data_manager.all_individuals[individual_index]
                    if hasattr(individual, 'epochs'):
                        channels = individual.epochs.ch_names
                        haemo_type = self.widgets.get('haemo_type', {}).get_value() if 'haemo_type' in self.widgets else 'hbo'
                        if hasattr(self.widgets['channels'], 'populate_channels'):
                            self.widgets['channels'].populate_channels(channels, haemo_type, bad_channels)
            else:
                # Hide channels for "All Individuals"
                self.widgets['channels'].hide()
                return
        
        elif self.current_plot_type == PlotType.STANDARD_RESPONSE:
            # For standard response plot, find common channels across selected individuals
            if selected_individuals:
                common_channels = None
                for name in selected_individuals:
                    individual = next((ind for ind in self.data_manager.all_individuals 
                                     if getattr(ind, "name", "") == name), None)
                    if individual and hasattr(individual, 'epochs'):
                        individual_channels = set(individual.epochs.ch_names)
                        if common_channels is None:
                            common_channels = individual_channels
                        else:
                            common_channels = common_channels.intersection(individual_channels)
                
                if common_channels and hasattr(self.widgets['channels'], 'populate_channels'):
                    self.widgets['channels'].populate_channels(sorted(common_channels), None, bad_channels)
                elif hasattr(self.widgets['channels'], 'populate_channels'):
                    self.widgets['channels'].populate_channels([], None, bad_channels)
            elif hasattr(self.widgets['channels'], 'populate_channels'):
                self.widgets['channels'].populate_channels([], None, bad_channels)
        
        else:
            # For other plot types, use first individual's channels
            if hasattr(self.data_manager.all_individuals[0], 'epochs'):
                channels = self.data_manager.all_individuals[0].epochs.ch_names
                if hasattr(self.widgets['channels'], 'populate_channels'):
                    self.widgets['channels'].populate_channels(channels, None, bad_channels)
    
    def _load_initial_data(self):
        """Load initial data"""
        if self.data_manager.load_data(self.settings):
            self._update_dynamic_widgets()
    
    def _collect_widget_values(self) -> Dict[str, Any]:
        """Collect values from all widgets"""
        values = {}
        for name, widget in self.widgets.items():
            try:
                values[name] = widget.get_value()
            except Exception as e:
                print(f"Error getting value from widget {name}: {e}")
                values[name] = None
        return values
    
    def _run_analysis(self):
        """Run the analysis and display results"""
        # Collect widget values
        widget_values = self._collect_widget_values()
        
        # Validate required fields
        validation_errors = validate_plot_configuration(self.current_plot_type, widget_values)
        if validation_errors:
            error_msg = "\n".join(validation_errors)
            messagebox.showerror("Validation Error", error_msg)
            return
        
        # Update settings from widgets
        self._update_settings_from_widgets(widget_values)
        
        # Reload data if needed
        if self.data_manager.needs_reload(self.settings):
            if not self.data_manager.load_data(self.settings):
                messagebox.showerror("Error", "Failed to load data")
                return
        
        # Generate plots
        try:
            figures = self.plot_generator.generate_plot(self.current_plot_type, widget_values)
            
            # Display plots
            self.plot_display.clear_display()
            plot_config = get_plot_config(self.current_plot_type)
            use_tabs = plot_config.uses_tabs if plot_config else False
            self.plot_display.display_figures(figures, use_tabs)
            
        except Exception as e:
            print(f"Error generating plots: {e}")
            self.plot_display.clear_display()
            tk.Label(self.plot_frame, text=f"Error: {str(e)}", 
                    font=("Arial", 14), fg="red").pack(pady=20)
    
    def _update_settings_from_widgets(self, widget_values: Dict[str, Any]):
        """Update settings from widget values"""
        for key, value in widget_values.items():
            if hasattr(self.settings, key) and value is not None:
                setattr(self.settings, key, value)
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()


if __name__ == "__main__":
    app = fNIRSGUI()
    app.run()