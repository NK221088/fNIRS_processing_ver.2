from shared_GUI_functions import *
import GLM_class
from collections import defaultdict


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
    glm_left_container = tk.Frame(glm_tab, width=300)
    glm_left_container.pack(side="left", padx=20, pady=20, fill="y")
    glm_left_container.pack_propagate(True)
    
    glm_right_frame = tk.Frame(glm_tab)
    glm_right_frame.pack(side="right", padx=20, pady=20, expand=True, fill="both")
    
    # Subdivide the left container into top and bottom sections
    glm_top_left_frame = tk.Frame(glm_left_container)
    glm_top_left_frame.pack(side="top", fill="y", expand=True)
    
    glm_bottom_left_frame = tk.Frame(glm_left_container)
    glm_bottom_left_frame.pack(side="bottom", fill="x")

    ######################################################################################
    # Settings and helper functions
    ######################################################################################
    
    # Default settings (add hemoglobin type to settings)
    settings = {
        "data_set": dataSetList[12],
    }
        
    ######################################################################################
    # Dataset(s) selection
    ######################################################################################
    
    # Dataset selection with info button
    dataset_frame = tk.Frame(glm_top_left_frame)
    dataset_frame.pack(fill="x", pady=5)

    dataset_label = tk.Label(dataset_frame, text="Select Dataset:", font=("Arial", 12))
    dataset_label.pack(anchor="w")

    dataset_selection_frame = tk.Frame(dataset_frame)
    dataset_selection_frame.pack(fill="x", pady=5)
    dataset_var = tk.StringVar(value=settings["data_set"])
    dataset_menu = ttk.Combobox(dataset_selection_frame, textvariable=dataset_var, values=dataSetList, width=35)
    dataset_menu.pack(side="left", padx=(0, 5))
    dataset_menu["postcommand"] = lambda: adjust_combobox_width(dataset_menu)
        
    # Second dataset selection frame (initially hidden)
    dataset2_selection_frame = tk.Frame(dataset_frame)
    dataset2_var = tk.StringVar(value=settings["data_set"])
    dataset2_menu = ttk.Combobox(dataset2_selection_frame, textvariable=dataset2_var, values=dataSetList, width=35)
    dataset2_menu.pack(side="left", padx=(0, 5))
    dataset2_menu["postcommand"] = lambda: adjust_combobox_width(dataset2_menu)
    
    # Remove button for second dataset
    remove_button = tk.Button(dataset2_selection_frame,
                              text="-",
                              command=lambda: toggle_second_dataset(False),
                              fg="black",
                              font=("Arial", 10, "bold"))
    remove_button.pack(side="left")
    
    # Plus button frame
    add_dataset_frame = tk.Frame(dataset_frame)
    add_dataset_frame.pack(fill="x", pady=2)
    
    add_button = tk.Button(
        add_dataset_frame,
        text="+",
        command=lambda: toggle_second_dataset(True),
        fg="black",
        font=("Arial", 12, "bold"),
        padx=10,
        pady=2
    )
    add_button.pack(anchor="w")
    
    def toggle_second_dataset(show):
        """Show or hide the second dataset selection"""
        if show:
            dataset2_selection_frame.pack(fill="x", pady=5, after=dataset_selection_frame)
            add_dataset_frame.pack_forget()
        else:
            dataset2_selection_frame.pack_forget()
            dataset2_var.set("") #Clear the selection
            add_dataset_frame.pack(fill="x", pady=2)
    
    # Initially hide the second dataset
    toggle_second_dataset(False)

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
    
    def add_GLM():
        return
    
    import tkinter as tk
    from tkinter import ttk

    class GLMParameterDialog:
        """Dialog for setting GLM parameters"""
        
        def __init__(self, parent):
            """
            Initialize the parameter dialog
            
            Args:
                parent: The parent tkinter window
            """
            self.result = None  # Will store the parameters dict when user confirms
            
            # Create the popup window
            self.dialog = tk.Toplevel(parent)
            self.dialog.title("Add GLM - Set Parameters")
            self.dialog.geometry("400x500")  # Adjust size as needed
            
            # Make it modal (user must interact with this before returning to main window)
            self.dialog.transient(parent)
            self.dialog.grab_set()
            
            # Center the dialog on screen (optional)
            self.center_window()
            
            # Store parameter variables - add your parameters here
            self.param1_var = tk.StringVar(value="default_value")
            self.param2_var = tk.IntVar(value=0)
            self.param3_var = tk.DoubleVar(value=1.0)
            # Add more parameter variables as needed
            
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
            
            # Main container with padding
            main_frame = ttk.Frame(self.dialog, padding="20")
            main_frame.pack(fill="both", expand=True)
            
            # Title/Instructions
            title_label = ttk.Label(main_frame, text="Configure GLM Parameters", 
                                font=("Arial", 14, "bold"))
            title_label.pack(pady=(0, 20))
            
            # Parameters frame - add your parameter inputs here
            params_frame = ttk.Frame(main_frame)
            params_frame.pack(fill="both", expand=True)
            
            # Example Parameter 1
            ttk.Label(params_frame, text="Name of GLM:").grid(row=0, column=0, sticky="w", pady=5)
            ttk.Entry(params_frame, textvariable=self.param1_var, width=30).grid(row=0, column=1, pady=5, padx=(10, 0))
            
            # Example Parameter 2
            ttk.Label(params_frame, text="Parameter 2:").grid(row=1, column=0, sticky="w", pady=5)
            ttk.Spinbox(params_frame, from_=0, to=100, textvariable=self.param2_var, width=28).grid(row=1, column=1, pady=5, padx=(10, 0))
            
            # Example Parameter 3
            ttk.Label(params_frame, text="Parameter 3:").grid(row=2, column=0, sticky="w", pady=5)
            ttk.Entry(params_frame, textvariable=self.param3_var, width=30).grid(row=2, column=1, pady=5, padx=(10, 0))
            
            # Add more parameter widgets here following the same pattern
            
            # Buttons frame at the bottom
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(side="bottom", fill="x", pady=(20, 0))
            
            # Cancel button
            cancel_btn = ttk.Button(button_frame, text="Cancel", command=self.cancel)
            cancel_btn.pack(side="right", padx=(5, 0))
            
            # OK button
            ok_btn = ttk.Button(button_frame, text="Add GLM analysis", command=self.ok)
            ok_btn.pack(side="right")
            
        def validate_inputs(self):
            """
            Validate the user inputs before accepting
            
            Returns:
                bool: True if inputs are valid, False otherwise
            """
            
            # Example          
            if type(self.param1_var.get()) != str:
                messagebox.showerror("Error", "Parameter should be a string")
                return False
            
            return True
            
        def ok(self):
            """Handle OK button click"""
            if self.validate_inputs():
                # Store the parameters in a dictionary
                self.result = {
                    'param1': self.param1_var.get(),
                    'param2': self.param2_var.get(),
                    'param3': self.param3_var.get(),
                    # Add more parameters here
                }
                self.dialog.destroy()
        
        def cancel(self):
            """Handle Cancel button click"""
            self.result = None
            self.dialog.destroy()
            
        def show(self):
            """
            Show the dialog and wait for it to close
            
            Returns:
                dict or None: Dictionary of parameters if OK was clicked, None if cancelled
            """
            self.dialog.wait_window()  # Wait for dialog to close
            return self.result
    
    glm_instances = dict()
    
    def add_GLM():
        """
        Function called when Add GLM button is clicked
        This is what you set as command=add_GLM in your button
        """
        # Show the parameter dialog
        dialog = GLMParameterDialog(glm_tab)
        params = dialog.show()
        
        # Check if user clicked OK (params will be None if cancelled)
        if params is not None:
            GLM_instance = GLM_class.GLM_class(params["param1"])
            glm_instances[GLM_instance.getName()] : GLM_instance

    add_GLM_button = tk.Button(glm_bottom_left_frame, text="Add GLM", command=add_GLM, bg="green", fg="white", 
                       font=("Arial", 12, "bold"), padx=20, pady=10)
    add_GLM_button.pack(pady=(SPACING_MEDIUM, SPACING_SMALL), padx=10, fill="x")

    
    GLMs_tabel_label = tk.Label(GLM_tools_frame, text="GLMs:", font=("Arial", 12))
    GLMs_tabel_label.pack(anchor="w", pady=(SPACING_LARGE, SPACING_SMALL))

    
    def run_GLM_analysis():
        return
    
    run_button = tk.Button(glm_bottom_left_frame, text="Run Analysis", command=run_GLM_analysis, bg="green", fg="white", 
                       font=("Arial", 12, "bold"), padx=20, pady=10)
    run_button.pack(pady=(SPACING_SMALL, SPACING_MEDIUM), padx=10, fill="x")

                
    return glm_tab, glm_left_container, glm_right_frame