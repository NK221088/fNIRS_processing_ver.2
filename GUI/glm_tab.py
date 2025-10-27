"""
GLM Analysis Tab Module
This module contains the UI and functionality for the GLM Analysis tab.
"""
import tkinter as tk
from tkinter import ttk

def create_glm_tab(parent_notebook):
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
    # Create GLM Analysis tab
    glm_tab = tk.Frame(parent_notebook)
    parent_notebook.add(glm_tab, text="GLM Analysis")
    
    # Create left and right containers
    glm_left_container = tk.Frame(glm_tab, width=300)
    glm_left_container.pack(side="left", padx=20, pady=20, fill="y")
    glm_left_container.pack_propagate(True)
    
    glm_right_frame = tk.Frame(glm_tab)
    glm_right_frame.pack(side="right", padx=20, pady=20, expand=True, fill="both")
    
    # Add your GLM interface here
    
    return glm_tab, glm_left_container, glm_right_frame