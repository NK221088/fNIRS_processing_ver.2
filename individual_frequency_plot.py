from Participant_class import individual_participant_class
import mne

def individual_frequency_plot(individual: individual_participant_class):
    figures = []  # List to store the figures
    for when, _raw in dict(Before=individual.raw_haemo_unfiltered, After=individual.raw_haemo).items():
        fig = _raw.compute_psd().plot(
            average=True, amplitude=False, picks="data", exclude="bads", show=False
        )
        fig.suptitle(f"{when} filtering", weight="bold", size="x-large")
        figures.append(fig)  # Store the figure
    
    return figures  # Return the list of figures
