import numpy as np
import mne
import mne_nirs
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import pandas as pd
import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()
save_path = Path(os.getenv(rf"PCA_component_visualizations_path"))

def extract_data(patient_data, type):
    names = [ind.name for ind in patient_data]
    first_names = {name.split("_")[0] for name in names}
    name_idx = {first_name: [idx for idx, n in enumerate(names) if n.split("_")[0] == first_name] for first_name in first_names}
    patient_raw_haemo = {name: [patient_data[i].raw_haemo.copy() for i in idxs] for name, idxs in name_idx.items()}
    for name, haemos in patient_raw_haemo.items():
        bad_channels = list(set(channel for haemo in haemos for channel in haemo.info['bads']))
        for haemo in haemos:
            haemo.info['bads'] = bad_channels
        if type == "long":
            patient_raw_haemo[name] = mne_nirs.channels.get_long_channels(mne.concatenate_raws(haemos).copy()).pick("hbo").get_data()
        elif type == "short":
            patient_raw_haemo[name] = mne_nirs.channels.get_short_channels(mne.concatenate_raws(haemos).copy()).get_data()
    return patient_raw_haemo

def construct_PCA_components(patient_data):
    patient_long_raw_haemo = extract_data(patient_data, "long")
    patient_short_raw_haemo = extract_data(patient_data, "short")
    
    PCA_long_components = {}
    top_long_channels = {}
    PCA_short_components = {}
    top_long_channels = {}
    
    for i in range 2:
        if i == 0:
            patient_raw_haemo = patient_long_raw_haemo
            PCA
    
    for idx, (name, raw) in enumerate(patient_raw_haemo.items()):
        raw = raw - raw.mean(axis=1, keepdims=True)
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
        channel_names = mne_nirs.channels.get_long_channels(patient_data[idx].raw_haemo.copy()).pick("hbo").ch_names
        
        top_channels[name] = {}
        for i, component in enumerate(loadings):
            top3_idx = np.argsort(np.abs(component))[-3:][::-1]  # descending
            top_channels[name][f"PC{i+1}"] = [channel_names[j] for j in top3_idx]
        
        

        fig, axes = plt.subplots(n_components, 1, figsize=(10, 3*n_components))
        for i, ax in enumerate(axes):
            ax.bar(range(len(channel_names)), pca.components_[i])
            ax.set_xticks(range(len(channel_names)))
            ax.set_xticklabels(channel_names, rotation=90, fontsize=7)
            ax.set_title(f"PC{i+1} loadings ({pca.explained_variance_ratio_[i]*100:.1f}% variance)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{name}_PCA_components.pdf"))
        plt.close()

    rows = [
        {"patient": patient, "PC": pc, "rank1": chs[0], "rank2": chs[1], "rank3": chs[2]}
        for patient, pcs in top_channels.items()
        for pc, chs in pcs.items()
    ]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_path, "top_channels.csv"), index=False)


    fig, axes = plt.subplots(n_components, 3, figsize=(10, 3))
    PC_names = [f"PC{i+1}" for i in range(n_components)]

    for PC_i, PC in enumerate(PC_names):

        ranks = ["rank1", "rank2", "rank3"]

        for idx, rank in enumerate(ranks):
            df[df["PC"] == PC][rank].value_counts().plot(
                kind="bar",
                ax=axes[PC_i, idx]
            )
            if PC_i == 0:
                axes[PC_i, idx].set_title(rank)
                axes[PC_i, idx].set_xlabel("Category")
                axes[PC_i, idx].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "top_3_contributors.pdf"))
    plt.close()

    return PCA_components