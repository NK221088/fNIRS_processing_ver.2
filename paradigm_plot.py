from Participant_class import individual_participant_class
import mne

def paradigm_plot(individual: individual_participant_class, duration: int= 600, show_scrollbars: bool=True):

    """Plot channels for one patient along the paradigm

    PARAMETERS
    ----------
    individual : individual instance
        The indivual for who the data should be plotted.
    duration : int
        
    show_scrollbars : boolean
        Whether the scroolbars should be shown or not.
    """
    plot = individual.raw_intensity.plot(n_channels=len(individual.raw_intensity.ch_names), duration=duration, show_scrollbars=show_scrollbars)
    return plot