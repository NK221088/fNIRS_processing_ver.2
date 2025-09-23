from Participant_class import individual_participant_class
import mne
import matplotlib.pyplot as plt
from datetime import datetime
import os

def paradigm_plot(individual: individual_participant_class, picks_ : list = ["all"], duration: int= 500, show_scrollbars: bool=True, haemo_type : str = "hbo", save: bool = False):

    """Plot channels for one patient along the paradigm

    PARAMETERS
    ----------
    individual : individual instance
        The indivual for who the data should be plotted.
    duration : int
        
    show_scrollbars : boolean
        Whether the scroolbars should be shown or not.
    """
    # Identify the channel types in the picks
    if picks_ == ["all"]:
        picks_ = []
        picks_ = individual.raw_haemo.ch_names
    # If multiple types are selected, split into separate type lists
    
    

    # Create separate plots for each type
    plots = []
    
    if haemo_type == "hbo":
        hbo_picks = [ch for ch in picks_ if ch.lower().endswith('hbo')]
        individual_hbo_copy = individual.raw_haemo.copy()
        individual_hbo_copy.pick(picks_)
        individual_hbo_copy.pick(hbo_picks)
        picks = mne.pick_types(individual_hbo_copy.info, meg=False, fnirs=True)
        dists = mne.preprocessing.nirs.source_detector_distances(individual_hbo_copy.info, picks=picks)
        individual_hbo_copy.pick(picks[dists > 0.01])
        hbo_plots = individual_hbo_copy.plot(n_channels=len(individual_hbo_copy.ch_names), duration=duration, show_scrollbars=show_scrollbars, show=False)
        plots.extend(hbo_plots if isinstance(hbo_plots, list) else [hbo_plots])
    if haemo_type == "hbr":
        hbr_picks = [ch for ch in picks_ if ch.lower().endswith('hbr')]
        individual_hbr_copy = individual.raw_haemo.copy()
        individual_hbr_copy.pick(picks_)
        individual_hbr_copy.pick(hbr_picks)
        picks = mne.pick_types(individual_hbr_copy.info, meg=False, fnirs=True)
        dists = mne.preprocessing.nirs.source_detector_distances(individual_hbr_copy.info, picks=picks)
        individual_hbr_copy.pick(picks[dists > 0.01])
        hbr_plots = individual_hbr_copy.plot(n_channels=len(individual_hbr_copy.ch_names), duration=duration, show_scrollbars=show_scrollbars, show = False)
        plots.extend(hbr_plots if isinstance(hbr_plots, list) else [hbr_plots])
    

    if save:
        os.makedirs("Plots", exist_ok=True)
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for i, fig in enumerate(plots):
            filename = os.path.join("Plots", f"Paradigm_plot_{current_datetime}_{haemo_type}.pdf")
            fig.savefig(filename)
            print(f"Plot saved as {filename}")

    # Always close the figures to prevent memory accumulation or re-display
    for fig in plots:
        plt.close(fig)
        
    return plots