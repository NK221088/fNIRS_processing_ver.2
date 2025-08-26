import numpy as np

def compute_p2p(epochs, data_types, percentile):
    hbo_data = epochs.copy()[data_types].pick("hbo").get_data()
    hbr_data = epochs.copy()[data_types].pick("hbr").get_data()
    hbo_percentile_p2p = np.percentile((hbo_data.max(axis=-1)-hbo_data.min(axis=-1)).ravel(), percentile)
    hbr_percentile_p2p = np.percentile((hbr_data.max(axis=-1)-hbr_data.min(axis=-1)).ravel(), percentile)
    return {"hbo": hbo_percentile_p2p, "hbr": hbr_percentile_p2p}