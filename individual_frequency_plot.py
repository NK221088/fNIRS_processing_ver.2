from Participant_class import individual_participant_class
import mne
import matplotlib.pyplot as plt
from datetime import datetime
import os

def individual_frequency_plot(individual: individual_participant_class, save: bool = False):
    figures = []  # List to store the figures
    types = ["before", "after"]
    for when, _raw in dict(Before=individual.raw_haemo_unfiltered, After=individual.raw_haemo).items():
        fig = _raw.compute_psd().plot(
            average=True, amplitude=False, picks="data", exclude="bads", show=False
        )
        fig.suptitle(f"{when} filtering", weight="bold", size="x-large")
        figures.append(fig)  # Store the figure
        
    if save:
        os.makedirs("Plots", exist_ok=True)
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for fig, type in zip(figures, types):
            filename = os.path.join("Plots", f"Individual_frequency_plot_{individual.name}_{current_datetime}_{type}.pdf")
            fig.savefig(filename)
            print(f"Plot saved as {filename}")
    
    # Always close the figures to prevent memory accumulation or re-display
    for fig in figures:
        plt.close(fig)
    
    return figures  # Return the list of figures
