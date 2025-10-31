from shared_GUI_functions import *
import data_analysis.GLM_class as GLM_class
from collections import defaultdict
from preprocessing_toolbox.load_data_function import data_loaders

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
    
    # Split right frame into left (editor) and right (plotting) sections
    glm_right_left_frame = tk.Frame(glm_right_frame)
    glm_right_left_frame.pack(side="left", padx=0, pady=0, fill="both", expand=False)
    
    glm_right_right_frame = tk.Frame(glm_right_frame, bg="lightgray")
    glm_right_right_frame.pack(side="right", padx=0, pady=0, expand=True, fill="both")
    
    # Placeholder for plotting area
    plot_label = tk.Label(glm_right_right_frame, text="Plotting Area\n(To be implemented)", 
                          font=("Arial", 14), bg="lightgray", fg="gray")
    plot_label.pack(expand=True)
    
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

            # Summary tab
            self.summary_tab = ttk.Frame(self.results_notebook)
            self.results_notebook.add(self.summary_tab, text="Summary")

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
            """Display the GLM results in a modern card-based layout with no scrolling"""
            # Clear previous content
            for widget in self.summary_tab.winfo_children():
                widget.destroy()
            for widget in self.full_results_tab.winfo_children():
                widget.destroy()
            
            try:
                results = glm_instance.getResults()
                
                if results and len(results) > 0:
                    anova_df = results[0]
                    
                    # === SUMMARY TAB: Card-based layout ===
                    if not anova_df.empty:
                        cards_container = ttk.Frame(self.summary_tab)
                        cards_container.pack(fill="both", expand=True, padx=10, pady=10)
                        
                        # Create metric cards in a grid
                        row = 0
                        col = 0
                        max_cols = 2
                        
                        for idx, (index, row_data) in enumerate(anova_df.iterrows()):
                            card = self._create_metric_card(cards_container, index, row_data)
                            card.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
                            
                            col += 1
                            if col >= max_cols:
                                col = 0
                                row += 1
                        
                        # Configure grid weights for responsive layout
                        for i in range(max_cols):
                            cards_container.columnconfigure(i, weight=1)
                        for i in range(row + 1):
                            cards_container.rowconfigure(i, weight=1)
                    
                    # === FULL TABLE TAB: Paginated view ===
                    self._create_paginated_table(self.full_results_tab, anova_df)
                else:
                    ttk.Label(self.summary_tab, text="No results available", 
                            font=("Arial", 11), foreground="gray").pack(pady=50)
                    ttk.Label(self.full_results_tab, text="No results available", 
                            font=("Arial", 11), foreground="gray").pack(pady=50)
                    
            except Exception as e:
                error_msg = f"Error displaying results: {str(e)}"
                ttk.Label(self.summary_tab, text=error_msg, 
                        font=("Arial", 11), foreground="red").pack(pady=50)
                ttk.Label(self.full_results_tab, text=error_msg, 
                        font=("Arial", 11), foreground="red").pack(pady=50)

        def _create_metric_card(self, parent, title, data):
            """Create a modern card widget for displaying a metric"""
            card = ttk.Frame(parent, relief="solid", borderwidth=1)
            card_inner = ttk.Frame(card, padding=15)
            card_inner.pack(fill="both", expand=True)
            
            # Title
            title_label = ttk.Label(card_inner, text=str(title), 
                                font=("Arial", 11, "bold"))
            title_label.pack(anchor="w", pady=(0, 10))
            
            # Display each column value in the row
            for col_name, value in data.items():
                metric_frame = ttk.Frame(card_inner)
                metric_frame.pack(fill="x", pady=2)
                
                ttk.Label(metric_frame, text=f"{col_name}:", 
                        font=("Arial", 9)).pack(side="left")
                
                # Format numeric values
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                else:
                    formatted_value = str(value)
                
                ttk.Label(metric_frame, text=formatted_value, 
                        font=("Arial", 9, "bold")).pack(side="right")
            
            return card

        def _create_paginated_table(self, parent, df, rows_per_page=10):
            """Create a paginated table view with no scrolling"""
            self.current_page = 0
            self.rows_per_page = rows_per_page
            self.total_pages = (len(df) + rows_per_page - 1) // rows_per_page if len(df) > 0 else 1
            self.df_to_display = df
            
            # Container for the entire paginated view
            container = ttk.Frame(parent)
            container.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Navigation bar at top
            nav_frame = ttk.Frame(container)
            nav_frame.pack(fill="x", pady=(0, 10))
            
            self.prev_btn = ttk.Button(nav_frame, text="← Previous", 
                                        command=lambda: self._change_page(-1))
            self.prev_btn.pack(side="left", padx=5)
            
            self.page_label = ttk.Label(nav_frame, text="", font=("Arial", 10))
            self.page_label.pack(side="left", padx=20)
            
            self.next_btn = ttk.Button(nav_frame, text="Next →", 
                                        command=lambda: self._change_page(1))
            self.next_btn.pack(side="left", padx=5)
            
            # Table frame (will be refreshed on page change)
            self.table_frame = ttk.Frame(container)
            self.table_frame.pack(fill="both", expand=True)
            
            # Initial display
            self._refresh_table_page()

        def _refresh_table_page(self):
            """Refresh the table to show the current page"""
            # Clear existing table
            for widget in self.table_frame.winfo_children():
                widget.destroy()
            
            df = self.df_to_display
            start_idx = self.current_page * self.rows_per_page
            end_idx = min(start_idx + self.rows_per_page, len(df))
            page_df = df.iloc[start_idx:end_idx]
            
            # Create header
            header_frame = ttk.Frame(self.table_frame)
            header_frame.pack(fill="x", pady=(0, 5))
            
            ttk.Label(header_frame, text="Index", font=("Arial", 9, "bold"), 
                    width=15).pack(side="left", padx=2)
            for col in df.columns:
                ttk.Label(header_frame, text=col, font=("Arial", 9, "bold"), 
                        width=15).pack(side="left", padx=2)
            
            # Create rows
            for idx, (index, row) in enumerate(page_df.iterrows()):
                row_frame = ttk.Frame(self.table_frame)
                row_frame.pack(fill="x", pady=1)
                
                # Alternate row colors
                bg_color = "#f0f0f0" if idx % 2 == 0 else "white"
                row_frame.configure(style="TableRow.TFrame")
                
                ttk.Label(row_frame, text=str(index), width=15, 
                        background=bg_color).pack(side="left", padx=2)
                
                for value in row:
                    if isinstance(value, (int, float)):
                        formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                    else:
                        formatted_value = str(value)
                    
                    ttk.Label(row_frame, text=formatted_value, width=15, 
                            background=bg_color).pack(side="left", padx=2)
            
            # Update navigation
            self.page_label.config(text=f"Page {self.current_page + 1} of {self.total_pages}")
            self.prev_btn.config(state="normal" if self.current_page > 0 else "disabled")
            self.next_btn.config(state="normal" if self.current_page < self.total_pages - 1 else "disabled")

        def _change_page(self, delta):
            """Change the current page by delta (-1 or +1)"""
            new_page = self.current_page + delta
            if 0 <= new_page < self.total_pages:
                self.current_page = new_page
                self._refresh_table_page()
        
            
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

                results = g.runGLM(all_participants, current_loader, number_of_subjects)
                
                # Store the results in the GLM instance
                g.setHasRun(True)
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