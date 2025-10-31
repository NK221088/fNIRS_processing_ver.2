from shared_GUI_functions import *
import data_analysis.GLM_class as GLM_class
from collections import defaultdict
from preprocessing_toolbox.load_data_function import data_loaders
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

"""
GLM Analysis Tab Module
This module contains the UI and functionality for the GLM Analysis tab.
"""
def create_glm_tab(parent_notebook, dataset_var, dataSetList):
    """
    Create and configure the GLM Analysis tab.
    
    Parameters:
    -----------
    parent_notebook : ttk.Notebook
        The notebook widget to add the GLM tab to
        
    Returns:
    --------
    tuple : (glm_tab, glm_left_container, glm_right_frame)
        References to the main components for external access if needed
    """
    
    ######################################################################################
    # Initialization
    ######################################################################################
    
    import tkinter as tk
    from tkinter import ttk
    # Create GLM Analysis tab
    glm_tab = tk.Frame(parent_notebook)
    parent_notebook.add(glm_tab, text="GLM Analysis")
    
    # Create left and right containers
    glm_left_container = tk.Frame(glm_tab)
    glm_left_container.pack(side="left", padx=20, pady=20, fill="both", expand=False)
    
    glm_right_frame = tk.Frame(glm_tab)
    glm_right_frame.pack(side="right", padx=20, pady=20, expand=True, fill="both")
    
    # Make the editor area occupy more space
    glm_right_left_frame = tk.Frame(glm_right_frame)
    glm_right_left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

    # Make the plotting area slightly narrower and reduce inner padding
    glm_right_right_frame = tk.Frame(glm_right_frame, bg="lightgray", width=500)
    glm_right_right_frame.pack(side="right", fill="y", expand=False, padx=(5, 0))

    # Plot container with tighter padding
    plot_container = tk.Frame(glm_right_right_frame, bg="white")
    plot_container.pack(fill="both", expand=True, padx=5, pady=5)

    
    # Dropdown frame at the top
    dropdown_frame = tk.Frame(plot_container, bg="white")
    dropdown_frame.pack(fill="x", pady=(0, 10))
    
    tk.Label(dropdown_frame, text="Select Figure:", font=("Arial", 10), bg="white").pack(side="left", padx=(0, 10))
    
    figure_var = tk.StringVar()
    figure_dropdown = ttk.Combobox(dropdown_frame, textvariable=figure_var, state="readonly", width=30)
    figure_dropdown.pack(side="left", padx=(0, 10))
    
    # Canvas frame for matplotlib figure
    canvas_frame = tk.Frame(plot_container, bg="white")
    canvas_frame.pack(fill="both", expand=True)
    
    # Placeholder label (shown when no figure is available)
    placeholder_label = tk.Label(canvas_frame, text="No figure to display\n\nRun a GLM analysis to generate figures", 
                                  font=("Arial", 12), bg="white", fg="gray")
    placeholder_label.pack(expand=True)
    
    # Variable to store current canvas
    current_canvas = {'canvas': None, 'toolbar': None}
    current_figures = {'figures': None}
    
    def display_figure(figure_key):
        # Hide placeholder
        placeholder_label.pack_forget()
        
        # Get the figure
        fig = current_figures['figures'][figure_key]

        # ✅ Handle Seaborn FacetGrid objects
        if isinstance(fig, sns.axisgrid.FacetGrid):
            fig = fig.fig

        # Create new canvas
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)

        # Add toolbar
        toolbar = NavigationToolbar2Tk(canvas, canvas_frame)
        toolbar.update()

        # Store references
        current_canvas['canvas'] = canvas
        current_canvas['toolbar'] = toolbar

    
    def update_figure_dropdown(figures_dict):
        """Update the dropdown with available figures"""
        if figures_dict is None or len(figures_dict) == 0:
            figure_dropdown['values'] = []
            figure_var.set('')
            current_figures['figures'] = None
            
            # Clear any existing canvas
            if current_canvas['canvas'] is not None:
                current_canvas['canvas'].get_tk_widget().destroy()
                if current_canvas['toolbar'] is not None:
                    current_canvas['toolbar'].destroy()
                current_canvas['canvas'] = None
                current_canvas['toolbar'] = None
            
            # Show placeholder
            placeholder_label.pack(expand=True)
            return
        
        # Store figures
        current_figures['figures'] = figures_dict
        
        # Update dropdown values
        figure_names = list(figures_dict.keys())
        figure_dropdown['values'] = figure_names
        
        # Select first figure by default
        if len(figure_names) > 0:
            figure_var.set(figure_names[0])
            display_figure(figure_names[0])
    
    # Bind dropdown selection to display function
    figure_dropdown.bind('<<ComboboxSelected>>', lambda e: display_figure(figure_var.get()))
    
    # Subdivide the left container into top and bottom sections
    glm_top_left_frame = tk.Frame(glm_left_container)
    glm_top_left_frame.pack(side="top", fill="both", expand=False)
    
    glm_bottom_left_frame = tk.Frame(glm_left_container)
    glm_bottom_left_frame.pack(side="bottom", fill="both", expand=True)
    ######################################################################################
    # Settings and helper functions
    ######################################################################################
    
    # Default settings
    settings = {
        "data_set": dataSetList[12],
    }
        
    ######################################################################################
    # Dataset(s) selection — dynamic list version
    ######################################################################################
    dataset_frame = tk.Frame(glm_top_left_frame)
    dataset_frame.pack(fill="x", pady=5)
    dataset_label = tk.Label(dataset_frame, text="Select Dataset(s):", font=("Arial", 12))
    dataset_label.pack(anchor="w")
    dataset_selection_container = tk.Frame(dataset_frame)
    dataset_selection_container.pack(fill="x", pady=5)
    # Keep track of dataset selectors
    dataset_vars = []
    def add_dataset_selector(initial_value=""):
        """Add a new dataset selector row"""
        row_frame = tk.Frame(dataset_selection_container)
        row_frame.pack(fill="x", pady=2)
        var = tk.StringVar(value=initial_value or settings["data_set"])
        combo = ttk.Combobox(row_frame, textvariable=var, values=dataSetList, width=35)
        combo.pack(side="left", padx=(0, 5))
        combo["postcommand"] = lambda: adjust_combobox_width(combo)
        # Remove button for this specific row
        remove_btn = tk.Button(
            row_frame,
            text="-",
            fg="black",
            font=("Arial", 10, "bold"),
            command=lambda: remove_dataset_selector(row_frame, var)
        )
        remove_btn.pack(side="left")
        dataset_vars.append(var)
    def remove_dataset_selector(frame, var):
        """Remove a dataset selector row"""
        dataset_vars.remove(var)
        frame.destroy()
    def get_selected_datasets():
        """Return all selected dataset names"""
        return [v.get() for v in dataset_vars if v.get()]
    # Add button to add new dataset rows
    add_button = tk.Button(
        dataset_frame,
        text="+ Add dataset",
        command=lambda: add_dataset_selector(),
        fg="black",
        font=("Arial", 10, "bold"),
        padx=8,
        pady=2
    )
    add_button.pack(anchor="w", pady=5)
    # Initialize with one dataset selector
    add_dataset_selector()
    ######################################################################################
    # GLM Analysis
    ######################################################################################
    
    
    # At the top of your GUI file
    SPACING_SMALL = 5
    SPACING_MEDIUM = 10
    SPACING_LARGE = 20
    # Dataset selection with info button
    GLM_tools_frame = tk.Frame(glm_bottom_left_frame)
    GLM_tools_frame.pack(fill="x", pady=5)
    GLM_tools_label = tk.Label(GLM_tools_frame, text="GLM analysis:", font=("Arial", 12))
    GLM_tools_label.pack(anchor="w")
    import tkinter as tk
    from tkinter import ttk
    class GLMParameterDialog:
        """Dialog for adding a new GLM"""
        
        def __init__(self, parent):
            """Initialize the parameter dialog for adding new GLM"""
            self.result = None
            
            # Create the popup window
            self.dialog = tk.Toplevel(parent)
            self.dialog.title("Add GLM - Set Parameters")
            self.dialog.geometry("400x500")
            
            # Make it modal
            self.dialog.transient(parent)
            self.dialog.grab_set()
            
            # Center the dialog on screen
            self.center_window()
            
            # Create parameter variables
            self.param1_var = tk.StringVar(value="default_value")
            self.param2_var = tk.StringVar(value="spm")
            self.param3_var = tk.StringVar(value="cosine")
            
            # Create the UI
            self.create_widgets()
            
        def center_window(self):
            """Center the dialog window on screen"""
            self.dialog.update_idletasks()
            width = self.dialog.winfo_width()
            height = self.dialog.winfo_height()
            x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
            y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
            self.dialog.geometry(f'{width}x{height}+{x}+{y}')
            
        def create_widgets(self):
            """Create all the widgets in the dialog"""
            main_frame = ttk.Frame(self.dialog, padding="20")
            main_frame.pack(fill="both", expand=True)
            
            title_label = ttk.Label(main_frame, text="Configure GLM Parameters", font=("Arial", 14, "bold"))
            title_label.pack(pady=(0, 20))
            
            params_frame = ttk.Frame(main_frame)
            params_frame.pack(fill="both", expand=True)
            
            ttk.Label(params_frame, text="Name of GLM:").grid(row=0, column=0, sticky="w", pady=5)
            ttk.Entry(params_frame, textvariable=self.param1_var, width=30).grid(row=0, column=1, pady=5, padx=(10, 0))
            
            ttk.Label(params_frame, text="HRF Model:").grid(row=1, column=0, sticky="w", pady=5)
            hrf_options = ["spm", "glover"]
            hrf_combo = ttk.Combobox(params_frame, textvariable=self.param2_var, values=hrf_options, width=28, state="readonly")
            hrf_combo.grid(row=1, column=1, pady=5, padx=(10, 0))
            ttk.Label(params_frame, text="Drift model:").grid(row=2, column=0, sticky="w", pady=5)
            drift_options = ["cosine"]
            drift_combo = ttk.Combobox(params_frame, textvariable=self.param3_var, values=drift_options, width=28, state="readonly")
            drift_combo.grid(row=2, column=1, pady=5, padx=(10, 0))
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(side="bottom", fill="x", pady=(20, 0))
            
            cancel_btn = ttk.Button(button_frame, text="Cancel", command=self.cancel)
            cancel_btn.pack(side="right", padx=(5, 0))
            
            ok_btn = ttk.Button(button_frame, text="Add GLM", command=self.ok)
            ok_btn.pack(side="right")
            
        def validate_inputs(self):
            """Validate the user inputs before accepting"""
            if not self.param1_var.get().strip():
                messagebox.showerror("Error", "GLM name cannot be empty")
                return False
            return True
            
        def ok(self):
            """Handle OK button click"""
            if self.validate_inputs():
                self.result = {
                    'GLMName': self.param1_var.get(),
                    'HRF_model': self.param2_var.get(),
                    'drift_model': self.param3_var.get(),
                }
                self.dialog.destroy()
        
        def cancel(self):
            """Handle Cancel button click"""
            self.result = None
            self.dialog.destroy()
            
        def show(self):
            """Show the dialog and wait for it to close"""
            self.dialog.wait_window()
            return self.result
    class GLMParameterEditor:
        """Editor for GLM parameters in the right pane (for editing existing GLMs only)"""
        
        def __init__(self, parent_frame):
            """
            Initialize the parameter editor
            
            Args:
                parent_frame: The frame to place the editor in
            """
            self.parent_frame = parent_frame
            self.current_glm = None
            
            # Create the main container with scrollbar
            self.main_container = ttk.Frame(parent_frame)
            self.main_container.pack(fill="both", expand=True, padx=0, pady=10)
            
            # Canvas and scrollbar for scrolling
            self.canvas = tk.Canvas(self.main_container, highlightthickness=0)
            self.scrollbar = ttk.Scrollbar(self.main_container, orient="vertical", command=self.canvas.yview)
            self.main_frame = ttk.Frame(self.canvas)
            
            self.main_frame.bind(
                "<Configure>",
                lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            )
            
            self.canvas.create_window((0, 0), window=self.main_frame, anchor="nw")
            self.canvas.configure(yscrollcommand=self.scrollbar.set)
            
            self.canvas.pack(side="left", fill="both", expand=True)
            self.scrollbar.pack(side="right", fill="y")
            
            # Create parameter variables
            self.param1_var = tk.StringVar()
            self.param2_var = tk.StringVar()
            self.param3_var = tk.StringVar()
            
            # Create the UI
            self.create_widgets()
            self.show_empty_state()
            
        def create_widgets(self):
            """Create all the widgets in the editor"""
            
            # --- Title ---
            self.title_label = ttk.Label(self.main_frame, text="GLM Configuration", 
                                         font=("Arial", 14, "bold"))
            self.title_label.pack(pady=(0, 20))
            
            # --- Empty state frame ---
            self.empty_frame = ttk.Frame(self.main_frame)
            empty_label = ttk.Label(self.empty_frame, 
                                   text="Select 'Edit' on a GLM\nto modify parameters",
                                   font=("Arial", 11), justify="center", foreground="gray")
            empty_label.pack(pady=50)
            
            # --- Parameters frame ---
            self.params_frame = ttk.Frame(self.main_frame)
            
            # ---- Name of GLM ----
            ttk.Label(self.params_frame, text="Name of GLM:").grid(row=0, column=0, sticky="w", pady=5, padx=(0, 10))
            self.name_entry = ttk.Entry(self.params_frame, textvariable=self.param1_var, width=30)
            self.name_entry.grid(row=0, column=1, pady=5, sticky="ew")
            
            # ---- HRF model ----
            ttk.Label(self.params_frame, text="HRF Model:").grid(row=1, column=0, sticky="w", pady=5, padx=(0, 10))
            hrf_options = ["spm", "glover"]
            self.hrf_combo = ttk.Combobox(self.params_frame, textvariable=self.param2_var, 
                                         values=hrf_options, width=28, state="readonly")
            self.hrf_combo.grid(row=1, column=1, pady=5, sticky="ew")
            # ---- Drift model ----
            ttk.Label(self.params_frame, text="Drift model:").grid(row=2, column=0, sticky="w", pady=5, padx=(0, 10))
            drift_options = ["cosine"]
            self.drift_combo = ttk.Combobox(self.params_frame, textvariable=self.param3_var, 
                                           values=drift_options, width=28, state="readonly")
            self.drift_combo.grid(row=2, column=1, pady=5, sticky="ew")
            
            # Configure grid weights
            self.params_frame.columnconfigure(1, weight=1)
            
            # --- getResults frame (initially hidden) ---
            self.results_frame = ttk.Frame(self.main_frame)
            # Results title
            self.results_title = ttk.Label(self.results_frame, text="GLM Results", 
                                        font=("Arial", 12, "bold"))
            self.results_title.pack(pady=(20, 10))
            # Create a notebook for tabbed results view
            self.results_notebook = ttk.Notebook(self.results_frame)
            self.results_notebook.pack(fill="both", expand=True, pady=(0, 10))
            # Full Results tab
            self.full_results_tab = ttk.Frame(self.results_notebook)
            self.results_notebook.add(self.full_results_tab, text="Full Table")
            
            # --- Bottom buttons ---
            self.button_frame = ttk.Frame(self.main_frame)
            
            self.cancel_btn = ttk.Button(self.button_frame, text="Cancel", command=self.cancel)
            self.cancel_btn.pack(side="right", padx=(5, 0))
            
            self.save_btn = ttk.Button(self.button_frame, text="Save GLM", command=self.save)
            self.save_btn.pack(side="right")
            
        def show_empty_state(self):
            """Show the empty state when no GLM is being edited"""
            self.params_frame.pack_forget()
            self.results_frame.pack_forget()
            self.button_frame.pack_forget()
            self.empty_frame.pack(fill="both", expand=True)
            self.current_glm = None
            
        def show_editor(self, glm_instance):
            """Show the editor for editing an existing GLM"""
            self.empty_frame.pack_forget()
            self.params_frame.pack(fill="x", pady=20)
            
            self.current_glm = glm_instance
            
            # Load existing GLM parameters
            self.title_label.config(text="Edit GLM Configuration")
            self.param1_var.set(glm_instance.getName())
            self.param2_var.set(glm_instance.getHRFModel())
            self.param3_var.set(glm_instance.getDriftModel())
            self.save_btn.config(text="Update GLM")
            
            # Show results if GLM has been run
            if glm_instance.getHasRun():
                self.display_results(glm_instance)
                self.results_frame.pack(fill="both", expand=True, pady=(10, 20))
            else:
                self.results_frame.pack_forget()
            
            self.button_frame.pack(fill="x", pady=(20, 0))
                    
        def display_results(self, glm_instance):
            """Display the GLM results in a simple table view"""
            # Clear previous content
            for widget in self.full_results_tab.winfo_children():
                widget.destroy()
            
            try:
                results = glm_instance.getResults()
                
                if results and len(results) > 0:
                    anova_df = results[0]
                    
                    # === FULL TABLE TAB: Simple view ===
                    if not anova_df.empty:
                        self._create_simple_table(self.full_results_tab, anova_df)
                    else:
                        ttk.Label(self.full_results_tab, text="No results available", 
                                font=("Arial", 11), foreground="gray").pack(pady=50)
                else:
                    ttk.Label(self.full_results_tab, text="No results available", 
                            font=("Arial", 11), foreground="gray").pack(pady=50)
                        
            except Exception as e:
                error_msg = f"Error displaying results: {str(e)}"
                ttk.Label(self.full_results_tab, text=error_msg, 
                        font=("Arial", 11), foreground="red").pack(pady=50)
        
        def _create_simple_table(self, parent, df):
            """Create a clean, simple table view that shows all results"""
            
            # Main container
            container = ttk.Frame(parent)
            container.pack(fill="both", expand=True, padx=20, pady=20)
            
            # Title section
            title_frame = ttk.Frame(container)
            title_frame.pack(fill="x", pady=(0, 15))
            
            ttk.Label(title_frame, text="GLM Analysis Results", 
                    font=("Arial", 14, "bold")).pack(side="left")
            
            # Info label
            total_rows = len(df)
            ttk.Label(title_frame, text=f"{total_rows} contrast{'s' if total_rows != 1 else ''}", 
                    font=("Arial", 9), foreground="gray").pack(side="right", padx=5)
            
            # Table container with border
            table_container = tk.Frame(container, bg="#e0e0e0", bd=1, relief="solid")
            table_container.pack(fill="both", expand=True)
            
            # Inner frame for table content
            table_frame = tk.Frame(table_container, bg="white")
            table_frame.pack(fill="both", expand=True, padx=1, pady=1)
            
            # Create header with styling
            header_frame = tk.Frame(table_frame, bg="#4a4a4a", height=40)
            header_frame.pack(fill="x", pady=0)
            header_frame.pack_propagate(False)
            
            # Index column header
            index_header = tk.Label(header_frame, text="Contrast", 
                                font=("Arial", 10, "bold"), 
                                bg="#4a4a4a", fg="white",
                                anchor="w", padx=15)
            index_header.pack(side="left", fill="both", expand=True)
            
            # Data column headers
            for col in df.columns:
                col_header = tk.Label(header_frame, text=col, 
                                    font=("Arial", 10, "bold"), 
                                    bg="#4a4a4a", fg="white",
                                    anchor="center", padx=10)
                col_header.pack(side="left", fill="both", expand=True)
            
            # Create data rows with alternating colors
            for idx, (index, row) in enumerate(df.iterrows()):
                # Alternate row colors
                bg_color = "#f8f9fa" if idx % 2 == 0 else "white"
                
                row_frame = tk.Frame(table_frame, bg=bg_color, height=35)
                row_frame.pack(fill="x", pady=0)
                row_frame.pack_propagate(False)
                
                # Index/Contrast name
                index_label = tk.Label(row_frame, text=str(index), 
                                    font=("Arial", 9), 
                                    bg=bg_color, fg="#2c3e50",
                                    anchor="w", padx=15)
                index_label.pack(side="left", fill="both", expand=True)
                
                # Data values
                for col_name, value in row.items():
                    # Format numeric values
                    if isinstance(value, (int, float)):
                        if isinstance(value, float):
                            # Format based on magnitude
                            if abs(value) < 0.0001 and value != 0:
                                formatted_value = f"{value:.2e}"
                            else:
                                formatted_value = f"{value:.4f}"
                        else:
                            formatted_value = str(value)
                    else:
                        formatted_value = str(value)
                    
                    # Color-code p-values if this is a p-value column
                    fg_color = "#2c3e50"
                    font_weight = "normal"
                    if 'p' in col_name.lower() and isinstance(value, float):
                        if value < 0.001:
                            fg_color = "#27ae60"  # Green for highly significant
                            font_weight = "bold"
                        elif value < 0.05:
                            fg_color = "#f39c12"  # Orange for significant
                            font_weight = "bold"
                        else:
                            fg_color = "#95a5a6"  # Gray for non-significant
                    
                    value_label = tk.Label(row_frame, text=formatted_value, 
                                        font=("Arial", 9, font_weight), 
                                        bg=bg_color, fg=fg_color,
                                        anchor="center", padx=10)
                    value_label.pack(side="left", fill="both", expand=True)
        def validate_inputs(self):
            """
            Validate the user inputs before accepting
            
            Returns:
                bool: True if inputs are valid, False otherwise
            """
            if not self.param1_var.get().strip():
                messagebox.showerror("Error", "GLM name cannot be empty")
                return False
            
            return True
        
        def save(self):
            """Handle Save button click"""
            if self.validate_inputs():
                params = {
                    'GLMName': self.param1_var.get(),
                    'HRF_model': self.param2_var.get(),
                    'drift_model': self.param3_var.get(),
                }
                
                # Update existing GLM
                self.current_glm.update_parameters(params)
                update_glm_table()
                self.show_empty_state()

        def cancel(self):
            """Handle Cancel button click"""
            self.show_empty_state()
    
    # Create the parameter editor in the right-left frame
    glm_editor = GLMParameterEditor(glm_right_left_frame)
    
    glm_instances = dict()
    class GLMTable(ttk.Frame):
        def __init__(self, parent):
            super().__init__(parent)
            
            # --- Scrollable frame setup ---
            canvas = tk.Canvas(self, highlightthickness=0)
            scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
            self.scrollable_frame = ttk.Frame(canvas)
            
            self.scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Pack canvas and scrollbar to fill available space
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            self.rows = []
            # Header with adjusted widths
            header = ttk.Frame(self.scrollable_frame)
            header.pack(fill="x", pady=(0, 5))
            ttk.Label(header, text="GLM Name", width=20).pack(side="left", padx=5)
            ttk.Label(header, text="Status", width=12).pack(side="left", padx=5)
            ttk.Label(header, text="Actions", width=15).pack(side="left", padx=5)
        def clear_table(self):
            for row in self.rows:
                row.destroy()
            self.rows.clear()
        def add_row(self, name, status, edit_callback, run_callback):
            row = ttk.Frame(self.scrollable_frame)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=name, width=20).pack(side="left", padx=5)
            ttk.Label(row, text=status, width=12).pack(side="left", padx=5)
            # Action buttons in a frame
            action_frame = ttk.Frame(row)
            action_frame.pack(side="left", padx=5)
            ttk.Button(action_frame, text="⚙️ Edit", command=edit_callback, width=8).pack(side="left", padx=2)
            ttk.Button(action_frame, text="▶ Run", command=run_callback, width=8).pack(side="left", padx=2)
            self.rows.append(row)
    def add_GLM():
        """Show popup dialog for adding a new GLM"""
        dialog = GLMParameterDialog(glm_tab)
        params = dialog.show()
        
        if params is not None:
            GLM_instance = GLM_class.GLM_class(*params.values())
            glm_instances[GLM_instance.getName()] = GLM_instance
            update_glm_table()
    def update_glm_table():
        """Refresh the GLM table with current GLM instances"""
        glm_table.clear_table()
        for glm_name, glm_instance in glm_instances.items():
            # Determine status based on whether GLM has been run
            status = "Complete" if glm_instance.getHasRun() else "Ready"
            
            # --- Define callbacks INSIDE the loop so glm_instance is captured ---
            def edit_callback(g=glm_instance):
                """Edit this specific GLM in the right pane editor"""
                glm_editor.show_editor(glm_instance=g)
            def run_callback(g=glm_instance):
                """Run this specific GLM"""
                dataLoaders = get_selected_datasets()
                datasets = defaultdict(dict)
                for data_loader in dataLoaders:
                    settings = {
                        "data_set": data_loader,
                        "epoch_type": "HandMI",
                        "individual": "All Individuals",
                        "short_channel_correction": True,
                        "negative_correlation_enhancement": False,
                        "haemo_type": "hbo",
                        "baseline_correction": "Previous rest period",
                        "tmin": 0,
                        "stimulus_duration": 5,
                        "scalp_coupling_threshold": 0.8,
                        "reject_criteria": dict(hbo=80e-6),
                        "unwanted": ["15.0"],
                        "filter_lower_value": 0.01,
                        "filter_upper_value": 0.5,
                        "h_trans_bandwidth": 0.2,
                        "l_trans_bandwidth": 0.01,
                        "snr_rejection": "None",
                        "snr_threshold": 8,
                        "Apply_TDDR": True,
                        "interpolate_bad_channels": True,
                    }
                    current_loader = data_loaders[data_loader](
                        data_name=data_loader,
                        file_path=data_loader,
                        short_channel_correction=settings["short_channel_correction"],
                        negative_correlation_enhancement=settings["negative_correlation_enhancement"],
                        interpolate_bad_channels=settings["interpolate_bad_channels"],
                        baseline_correction=settings["baseline_correction"],
                        tmin=settings["tmin"],
                        filter_lower_value=settings["filter_lower_value"],
                        filter_upper_value=settings["filter_upper_value"],
                        l_trans_bandwidth=settings["l_trans_bandwidth"],
                        h_trans_bandwidth=settings["h_trans_bandwidth"],
                        scalp_coupling_threshold=settings["scalp_coupling_threshold"],
                        reject_criteria=settings["reject_criteria"],
                        snr_rejection=settings["snr_rejection"],
                        snr_threshold=settings["snr_threshold"],
                        apply_tddr=settings["Apply_TDDR"]
                    )
                    data = current_loader.load_data()
                    variables = ("all_epochs", "data_name", "all_data", "freq", "data_types", "all_individuals")
                    datasets[data_loader] = {key: value for key, value in zip(variables, data)}
                all_participants = []
                number_of_subjects = []
                for ds_name, ds_data in datasets.items():
                    all_participants += ds_data["all_individuals"]
                    number_of_subjects.append(len(ds_data["all_individuals"]))
                g.runGLM(all_participants, current_loader, number_of_subjects)
                results = g.getResults()
                # Store the results in the GLM instance
                g.setHasRun(True)
                
                # Update the figure dropdown with new figures
                if results and len(results) >= 4:
                    figures_dict = results[3]  # The 4th element contains the figures dictionary
                    update_figure_dropdown(figures_dict)
                
                update_glm_table()
            # Add a row for this GLM with its callbacks
            glm_table.add_row(glm_name, status, edit_callback, run_callback)
    # --- Add GLM button (top of bottom-left section) ---
    add_GLM_button = tk.Button(
        glm_bottom_left_frame,
        text="Add GLM",
        command=add_GLM,
        bg="green",
        fg="white",
        font=("Arial", 12, "bold"),
        padx=20,
        pady=10
    )
    add_GLM_button.pack(
        pady=(SPACING_MEDIUM, SPACING_SMALL),
        padx=10,
        fill="x"
    )
    # --- GLM Table (directly under Add button) ---
    glm_table = GLMTable(glm_bottom_left_frame)
    glm_table.pack(
        fill="both",
        expand=True,
        padx=5,
        pady=(SPACING_SMALL, 0)
    )
                
    return glm_tab, glm_left_container, glm_right_frame