import tkinter as tk
from tkinter import Frame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Sample datasets
data_sets = {
    "Dataset 1": [1, 2, 3, 4, 6],
    "Dataset 2": [1, 1, 2, 4, 6],
    "Dataset 3": [3, 3, 3, 3, 3]
}

# Function to update the plot
def select_option(dataset_name):
    label.config(text=f"Selected: {dataset_name}")
    update_plot(data_sets[dataset_name])

def update_plot(data):
    ax.clear()
    ax.plot(data, marker='o', linestyle='-')  # Line plot with markers
    ax.set_title("Dataset Visualization")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    canvas.draw()

# Create main window
root = tk.Tk()
root.title("Dataset Selector")
root.geometry("600x400")

# Create a left frame for buttons
left_frame = Frame(root)
left_frame.pack(side="left", padx=20, pady=20)

# Label to display selection
label = tk.Label(left_frame, text="Choose a dataset", font=("Arial", 14))
label.pack(pady=10)

# Create buttons for selecting datasets
for dataset_name in data_sets.keys():
    button = tk.Button(left_frame, text=dataset_name, command=lambda name=dataset_name: select_option(name))
    button.pack(pady=5)

# Create a right frame for the plot
right_frame = Frame(root)
right_frame.pack(side="right", padx=20, pady=20)

# Matplotlib figure and canvas
fig, ax = plt.subplots(figsize=(4, 3))
canvas = FigureCanvasTkAgg(fig, master=right_frame)
canvas.get_tk_widget().pack()

# Start the GUI event loop
root.mainloop()
