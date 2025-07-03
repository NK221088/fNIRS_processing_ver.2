# widget_factory.py
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod
from plot_config import WidgetConfig, WidgetType, get_widget_config


class BaseWidget(ABC):
    """Base class for all widgets"""
    
    def __init__(self, parent: tk.Widget, config: WidgetConfig):
        self.parent = parent
        self.config = config
        self.widget = None
        self.var = None
        self._setup_widget()
    
    @abstractmethod
    def _setup_widget(self):
        """Setup the widget"""
        pass
    
    def show(self):
        """Show the widget"""
        if self.widget:
            self.widget.pack(anchor="w", pady=2)
    
    def hide(self):
        """Hide the widget"""
        if self.widget:
            self.widget.pack_forget()
    
    @abstractmethod
    def get_value(self):
        """Get the current value of the widget"""
        pass
    
    def set_value(self, value):
        """Set the value of the widget"""
        if self.var:
            self.var.set(value)
    
    def enable(self):
        """Enable the widget"""
        if hasattr(self.widget, 'configure'):
            self.widget.configure(state='normal')
    
    def disable(self):
        """Disable the widget"""
        if hasattr(self.widget, 'configure'):
            self.widget.configure(state='disabled')
    
    def bind_callback(self, callback: Callable):
        """Bind a callback to the widget"""
        if self.var:
            self.var.trace_add("write", callback)


class ComboboxWidget(BaseWidget):
    """Combobox widget implementation"""
    
    def _setup_widget(self):
        frame = tk.Frame(self.parent)
        self.widget = frame
        
        # Label
        label = tk.Label(frame, text=self.config.label, font=("Arial", 12))
        label.pack(anchor="w")
        
        # Combobox
        self.var = tk.StringVar(value=self.config.default_value or "")
        self.combo = ttk.Combobox(frame, textvariable=self.var, width=40)
        
        if self.config.values:
            self.combo['values'] = self.config.values
        
        self.combo.pack(pady=5)
        
        # Help text
        if self.config.help_text:
            help_label = tk.Label(frame, text=self.config.help_text, 
                                font=("Arial", 8), fg="gray")
            help_label.pack(anchor="w")
    
    def get_value(self):
        return self.var.get()
    
    def update_values(self, values: List[str]):
        """Update combobox values"""
        self.combo['values'] = values
        if values and not self.var.get():
            self.var.set(values[0])


class CheckboxWidget(BaseWidget):
    """Checkbox widget implementation"""
    
    def _setup_widget(self):
        frame = tk.Frame(self.parent)
        self.widget = frame
        
        # Label
        label = tk.Label(frame, text=self.config.label, font=("Arial", 12))
        label.pack(anchor="w")
        
        # Checkbox
        self.var = tk.BooleanVar(value=self.config.default_value or False)
        self.checkbox = tk.Checkbutton(frame, text="Enable", variable=self.var)
        self.checkbox.pack(anchor="w")
        
        # Help text
        if self.config.help_text:
            help_label = tk.Label(frame, text=self.config.help_text, 
                                font=("Arial", 8), fg="gray")
            help_label.pack(anchor="w")
    
    def get_value(self):
        return self.var.get()


class EntryWidget(BaseWidget):
    """Entry widget implementation"""
    
    def _setup_widget(self):
        frame = tk.Frame(self.parent)
        self.widget = frame
        
        # Label
        label = tk.Label(frame, text=self.config.label, font=("Arial", 12))
        label.pack(anchor="w")
        
        # Entry
        self.var = tk.StringVar(value=str(self.config.default_value or ""))
        self.entry = tk.Entry(frame, textvariable=self.var, width=40)
        self.entry.pack(pady=5)
        
        # Help text
        if self.config.help_text:
            help_label = tk.Label(frame, text=self.config.help_text, 
                                font=("Arial", 8), fg="gray")
            help_label.pack(anchor="w")
    
    def get_value(self):
        return self.var.get()


class SpinboxWidget(BaseWidget):
    """Spinbox widget implementation"""
    
    def _setup_widget(self):
        frame = tk.Frame(self.parent)
        self.widget = frame
        
        # Label
        label = tk.Label(frame, text=self.config.label, font=("Arial", 12))
        label.pack(anchor="w")
        
        # Spinbox
        self.var = tk.IntVar(value=self.config.default_value or 0)
        self.spinbox = tk.Spinbox(
            frame, 
            textvariable=self.var,
            from_=self.config.min_value or 0,
            to=self.config.max_value or 100,
            width=10
        )
        self.spinbox.pack(pady=5)
        
        # Help text
        if self.config.help_text:
            help_label = tk.Label(frame, text=self.config.help_text, 
                                font=("Arial", 8), fg="gray")
            help_label.pack(anchor="w")
    
    def get_value(self):
        return self.var.get()


class MultiSelectWidget(BaseWidget):
    """Multi-select listbox widget implementation"""
    
    def _setup_widget(self):
        frame = tk.Frame(self.parent)
        self.widget = frame
        
        # Label
        label = tk.Label(frame, text=self.config.label, font=("Arial", 12))
        label.pack(anchor="w")
        
        # Listbox with scrollbar
        list_frame = tk.Frame(frame)
        list_frame.pack(fill="both", expand=True, pady=5)
        
        self.listbox = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, height=6)
        scrollbar = tk.Scrollbar(list_frame, orient="vertical")
        
        self.listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.listbox.yview)
        
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Help text
        if self.config.help_text:
            help_label = tk.Label(frame, text=self.config.help_text, 
                                font=("Arial", 8), fg="gray")
            help_label.pack(anchor="w")
    
    def get_value(self):
        """Get selected items"""
        selected_indices = self.listbox.curselection()
        return [self.listbox.get(i) for i in selected_indices]
    
    def update_values(self, values: List[str]):
        """Update listbox values"""
        self.listbox.delete(0, tk.END)
        for value in values:
            self.listbox.insert(tk.END, value)


class ChannelSelectorWidget(BaseWidget):
    """Channel selection widget with scrollable checkboxes"""
    
    def _setup_widget(self):
        """Initialize the channel selector widget"""
        frame = tk.Frame(self.parent)
        self.widget = frame
        
        # Label
        label = tk.Label(frame, text=self.config.label, font=("Arial", 12))
        label.pack(anchor="w")
        
        # Create scrollable frame
        canvas = tk.Canvas(frame, height=150, bg="white")
        scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        self.container = tk.Frame(canvas)
        
        # Configure scrolling
        canvas.create_window((0, 0), window=self.container, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Store references for later use
        self.canvas = canvas
        self.scrollbar = scrollbar
        self.channel_vars = {}
        
        # Add mouse wheel support
        self._setup_mouse_wheel()
        
        # Help text
        if self.config.help_text:
            help_label = tk.Label(frame, text=self.config.help_text, 
                                font=("Arial", 8), fg="gray")
            help_label.pack(anchor="w")
    
    def _setup_mouse_wheel(self):
        """Setup mouse wheel scrolling support"""
        def _on_mousewheel(event):
            """Handle mouse wheel scrolling"""
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        # Bind mouse wheel to canvas
        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # For Linux systems
        def _on_mouse_button(event):
            self.canvas.yview_scroll(-1, "units")
        
        def _on_mouse_button_down(event):
            self.canvas.yview_scroll(1, "units")
        
        self.canvas.bind_all("<Button-4>", _on_mouse_button)
        self.canvas.bind_all("<Button-5>", _on_mouse_button_down)
    
    def get_value(self) -> List[str]:
        """Get selected channels"""
        return [channel for channel, var in self.channel_vars.items() if var.get()]
    
    def set_value(self, selected_channels: List[str]):
        """Set selected channels"""
        for channel, var in self.channel_vars.items():
            var.set(channel in selected_channels)
    
    def populate_channels(self, channels: List[str], haemo_type: Optional[str] = None, 
                         bad_channels: Optional[List[str]] = None):
        """Populate channel checkboxes with optional filtering"""
        # Clear existing widgets
        for widget in self.container.winfo_children():
            widget.destroy()
        self.channel_vars.clear()
        
        if bad_channels is None:
            bad_channels = []
        
        # Filter channels if needed
        if haemo_type:
            filtered_channels = [
                channel for channel in channels 
                if haemo_type.lower() in channel.lower() and channel not in bad_channels
            ]
            # Remove duplicate channel names (keep unique base names)
            unique_channels = []
            seen = set()
            for channel in filtered_channels:
                # Extract base name by removing hemoglobin type suffix
                base_name = channel.rsplit(' ', 1)[0] if ' ' in channel else channel
                if base_name not in seen:
                    unique_channels.append(base_name)
                    seen.add(base_name)
            channels = unique_channels
        else:
            channels = [ch for ch in channels if ch not in bad_channels]
        
        # Sort channels for consistent ordering
        channels = sorted(channels)
        
        # Add new checkboxes
        for i, channel in enumerate(channels):
            is_checked = (i == 0)  # Default: first one checked
            self.channel_vars[channel] = tk.BooleanVar(value=is_checked)
            
            # Create checkbox with improved styling
            cb = tk.Checkbutton(
                self.container, 
                text=channel, 
                variable=self.channel_vars[channel],
                anchor="w",
                justify="left"
            )
            cb.grid(row=i // 3, column=i % 3, sticky="w", padx=5, pady=2)
        
        # Update scroll region
        self.container.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
    
    def clear_channels(self):
        """Clear all channel checkboxes"""
        for widget in self.container.winfo_children():
            widget.destroy()
        self.channel_vars.clear()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
    
    def select_all(self):
        """Select all available channels"""
        for var in self.channel_vars.values():
            var.set(True)
    
    def deselect_all(self):
        """Deselect all channels"""
        for var in self.channel_vars.values():
            var.set(False)
    
    def get_channel_count(self) -> int:
        """Get total number of available channels"""
        return len(self.channel_vars)
    
    def get_selected_count(self) -> int:
        """Get number of selected channels"""
        return len(self.get_value())
    
    def has_channels(self) -> bool:
        """Check if any channels are available"""
        return len(self.channel_vars) > 0
    
    def bind_callback(self, callback):
        """Bind callback to channel selection changes"""
        self.callback = callback
        # Bind to all existing checkboxes
        for var in self.channel_vars.values():
            var.trace_add("write", callback)
    
    def _update_callback_bindings(self):
        """Update callback bindings for newly created checkboxes"""
        if hasattr(self, 'callback'):
            for var in self.channel_vars.values():
                # Remove any existing traces first
                for trace_id in var.trace_info():
                    var.trace_remove(trace_id[0], trace_id[1])
                # Add new trace
                var.trace_add("write", self.callback)
    
    def show(self):
        """Show the widget"""
        if hasattr(self, 'widget') and self.widget:
            self.widget.pack(fill="x", padx=10, pady=5)
    
    def hide(self):
        """Hide the widget"""
        if hasattr(self, 'widget') and self.widget:
            self.widget.pack_forget()
    
    def is_visible(self) -> bool:
        """Check if widget is currently visible"""
        return hasattr(self, 'widget') and self.widget and bool(self.widget.winfo_viewable())
    
    def destroy(self):
        """Clean up the widget"""
        if hasattr(self, 'canvas'):
            # Unbind mouse wheel events
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")
        
        if hasattr(self, 'widget') and self.widget:
            self.widget.destroy()
        
        self.channel_vars.clear()


class IndividualSelectorWidget(BaseWidget):
    """Individual selection widget"""
    
    def _setup_widget(self):
        frame = tk.Frame(self.parent)
        self.widget = frame
        
        # Label
        label = tk.Label(frame, text=self.config.label, font=("Arial", 12))
        label.pack(anchor="w")
        
        # Combobox
        self.var = tk.StringVar()
        self.combo = ttk.Combobox(frame, textvariable=self.var, width=40)
        self.combo.pack(pady=5)
        
        # Help text
        if self.config.help_text:
            help_label = tk.Label(frame, text=self.config.help_text, 
                                font=("Arial", 8), fg="gray")
            help_label.pack(anchor="w")
    
    def get_value(self):
        return self.var.get()
    
    def update_individuals(self, individuals: List[Any], include_all: bool = True):
        """Update available individuals"""
        values = []
        if include_all:
            values.append("All Individuals")
        
        for i, ind in enumerate(individuals):
            name = getattr(ind, "name", f"Participant_{i+1}")
            values.append(name)
        
        self.combo['values'] = values
        if values:
            self.var.set(values[0])


class WidgetFactory:
    """Factory for creating widgets based on configuration"""
    
    _widget_classes = {
        WidgetType.COMBOBOX: ComboboxWidget,
        WidgetType.CHECKBOX: CheckboxWidget,
        WidgetType.ENTRY: EntryWidget,
        WidgetType.SPINBOX: SpinboxWidget,
        WidgetType.MULTI_SELECT: MultiSelectWidget,
        WidgetType.CHANNEL_SELECTOR: ChannelSelectorWidget,
        WidgetType.INDIVIDUAL_SELECTOR: IndividualSelectorWidget,
    }
    
    @classmethod
    def create_widget(cls, parent: tk.Widget, widget_name: str) -> BaseWidget:
        """Create a widget by name"""
        config = get_widget_config(widget_name)
        if not config:
            raise ValueError(f"Unknown widget: {widget_name}")
        
        widget_class = cls._widget_classes.get(config.widget_type)
        if not widget_class:
            raise ValueError(f"Unknown widget type: {config.widget_type}")
        
        return widget_class(parent, config)
    
    @classmethod
    def create_widgets(cls, parent: tk.Widget, widget_names: List[str]) -> Dict[str, BaseWidget]:
        """Create multiple widgets"""
        widgets = {}
        for name in widget_names:
            try:
                widgets[name] = cls.create_widget(parent, name)
            except ValueError as e:
                print(f"Warning: Could not create widget {name}: {e}")
        return widgets