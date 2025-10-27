# GUI/shared_functions.py
import tkinter as tk
from tkinter import messagebox
from GUI.dataset_info_panel import show_dataset_info_in_container

def show_dataset_info_view(right_frame, current_loader, all_epochs, data_name, 
                           all_data, freq, data_types, all_individuals):
    """Render Dataset Info inside the main plot area (right_frame)."""
    try:
        if all_epochs:
            # Clear the right-side plot area before showing info
            for widget in right_frame.winfo_children():
                widget.destroy()
            # Mount the info panel inside right_frame
            show_dataset_info_in_container(
                class_instance=current_loader,
                parent_container=right_frame,
                all_epochs=all_epochs,
                data_name=data_name,
                all_data=all_data,
                freq=freq,
                data_types=data_types,
                all_individuals=all_individuals
            )
        else:
            messagebox.showwarning("No Data", "Please load a dataset first by selecting one from the dropdown.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to show dataset info: {str(e)}")
        
# Helper function to adjust combobox width
def adjust_combobox_width(combobox):
    combobox["width"] = max(len(item) for item in combobox["values"])