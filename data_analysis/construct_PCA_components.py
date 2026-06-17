import numpy as np
import mne
import mne_nirs
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()
save_path = Path(os.getenv(rf"PCA_component_visualizations_path"))

def extract_data(patient_data):
    names = [ind.name for ind in patient_data]
    first_names = {name.split("_")[0] for name in names}
    name_idx = {first_name: [idx for idx, n in enumerate(names) if n.split("_")[0] == first_name] for first_name in first_names}
    patient_raw_haemo = {name: [patient_data[i].raw_haemo.copy() for i in idxs] for name, idxs in name_idx.items()}
    for name, haemos in patient_raw_haemo.items():
        bad_channels = list(set(channel for haemo in haemos for channel in haemo.info['bads']))
        for haemo in haemos:
            haemo.info['bads'] = bad_channels
        patient_raw_haemo[name] = mne_nirs.channels.get_long_channels(mne.concatenate_raws(haemos).copy()).pick("hbo").get_data()
    return patient_raw_haemo

def construct_PCA_components(patient_data):
    patient_raw_haemo = extract_data(patient_data)
    PCA_components = {}
    
    for name, raw in patient_raw_haemo.items():
        raw -= raw.mean(axis=1, keepdims=True)
        pca = PCA()
        pca.fit(raw.T)
        
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        plt.figure(figsize=(7, 4))
        plt.plot(range(1, len(cumvar) + 1), cumvar, marker="o")
        plt.axhline(0.95, color="r", linestyle="--", label="95% threshold")
        plt.xlabel("Number of components")
        plt.ylabel("Cumulative explained variance")
        plt.title(f"PCA elbow plot — {name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{name}_PCA_explained.pdf"))
        plt.close()
        
        # Components for 95% variance
        n_components = 9 # np.searchsorted(cumvar, 0.95) + 1; 9 is chosen as the average number of components
        
        # Project
        loadings = pca.components_[:n_components]  # (n_components, 15)
        PCA_components[name] = loadings   

        # pca is your fitted PCA object for a given patient
        channel_names = mne_nirs.channels.get_long_channels(patient_data[0].raw_haemo.copy()).pick("hbo").ch_names

        fig, axes = plt.subplots(n_components, 1, figsize=(10, 3*n_components))
        for i, ax in enumerate(axes):
            ax.bar(range(len(channel_names)), pca.components_[i])
            ax.set_xticks(range(len(channel_names)))
            ax.set_xticklabels(channel_names, rotation=90, fontsize=7)
            ax.set_title(f"PC{i+1} loadings ({pca.explained_variance_ratio_[i]*100:.1f}% variance)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{name}_PCA_components.pdf"))
        plt.close()

    
    return PCA_components

        
        