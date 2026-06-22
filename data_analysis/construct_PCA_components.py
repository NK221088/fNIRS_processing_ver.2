import numpy as np
import mne
import mne_nirs
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()
save_path = Path(os.getenv(rf"PCA_component_visualizations_path"))

def extract_data(patient_data, channel_type):
    names = [ind.name for ind in patient_data]
    first_names = {name.split("_")[0] for name in names}
    name_idx = {first_name: [idx for idx, n in enumerate(names) if n.split("_")[0] == first_name] for first_name in first_names}
    if channel_type == "long":
        patient_raw_haemo = {name: [patient_data[i].raw_haemo.copy() for i in idxs] for name, idxs in name_idx.items()}
    elif channel_type == "short":
        patient_raw_haemo = {name: [patient_data[i].raw_haemo_unfiltered.copy() for i in idxs] for name, idxs in name_idx.items()}
    for name, haemos in patient_raw_haemo.items():
        bad_channels = list(set(channel for haemo in haemos for channel in haemo.info['bads']))
        for haemo in haemos:
            haemo.info['bads'] = bad_channels
        if channel_type == "long":
            patient_raw_haemo[name] = mne_nirs.channels.get_long_channels(mne.concatenate_raws(haemos).copy()).pick("hbo").get_data()
        elif channel_type == "short":
            patient_raw_haemo[name] = mne_nirs.channels.get_short_channels(mne.concatenate_raws(haemos).copy()).pick("hbo").get_data()
    return patient_raw_haemo
def channel_importance(loadings, explained_variance_ratio, n_components):
    """Weighted sum of |loading| across kept components, weighted by each
    component's explained variance ratio. Collapses all PCs into one
    per-channel score — no PC-identity matching needed."""
    weights = explained_variance_ratio[:n_components]
    importance = (np.abs(loadings) * weights[:, None]).sum(axis=0)
    return importance / importance.sum()

def construct_PCA_components(patient_data):
    types = ["long", "short"]
    n_short_components = []
    PCA_components = {"long": {}, "short": {}}
    top_channels = {"long": {}, "short": {}}
    importance_scores = {"long": {}, "short": {}}
    for channel_type in tqdm(types, position=0, desc="Channel types"):
        patient_raw_haemo = extract_data(patient_data, channel_type)
        pbar = tqdm(patient_raw_haemo.items(), position=1, leave=False, desc=f"{channel_type}")
        for idx, (name, raw) in enumerate(pbar):
            pbar.set_description(f"Processing {name}")
            raw = raw - raw.mean(axis=1, keepdims=True)
            pca = PCA()
            pca.fit(raw.T)
            for i in range(pca.components_.shape[0]):
                if pca.components_[i, np.argmax(np.abs(pca.components_[i]))] < 0:
                    pca.components_[i] *= -1

            cumvar = np.cumsum(pca.explained_variance_ratio_)
            plt.figure(figsize=(7, 4))
            plt.plot(range(1, len(cumvar) + 1), cumvar, marker="o")
            plt.axhline(0.95, color="r", linestyle="--", label="95% threshold")
            plt.xlabel("Number of components")
            plt.ylabel("Cumulative explained variance")
            plt.title(f"PCA elbow plot — {name}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(os.path.join(save_path, channel_type), f"{name}_PCA_explained.pdf"))
            plt.close()
            
            # Components for 95% variance
            if channel_type == "long":
                # n_components = np.searchsorted(cumvar, 0.95) + 1 #; 
                n_components = 9 # is chosen as the average number of components
            elif channel_type == "short":
                n_components = np.searchsorted(cumvar, 0.95) + 1 #; 
                n_short_components.append(n_components)
                # n_components = 7 # is chosen as the average number of components

            # Project
            loadings = pca.components_[:n_components]  # (n_components, 15)
            PCA_components[channel_type][name] = loadings   

            # pca is your fitted PCA object for a given patient
            if channel_type == "long":
                channel_names = mne_nirs.channels.get_long_channels(patient_data[idx].raw_haemo.copy()).pick("hbo").ch_names
            elif channel_type == "short":
                channel_names = mne_nirs.channels.get_short_channels(patient_data[idx].raw_haemo.copy()).pick("hbo").ch_names

            importance = channel_importance(loadings, pca.explained_variance_ratio_, n_components)
            importance_scores[channel_type][name] = dict(zip(channel_names, importance))
            
            top_channels[channel_type][name] = {}
            for i, component in enumerate(loadings):
                top3_idx = np.argsort(np.abs(component))[-3:][::-1]  # descending
                top_channels[channel_type][name][f"PC{i+1}"] = [channel_names[j] for j in top3_idx]

            fig, axes = plt.subplots(n_components, 1, figsize=(10, 3*n_components), squeeze=False)
            for i, ax in enumerate(axes[:, 0]):
                ax.bar(range(len(channel_names)), pca.components_[i])
                ax.set_xticks(range(len(channel_names)))
                ax.set_xticklabels(channel_names, rotation=90, fontsize=7)
                ax.set_title(f"PC{i+1} loadings ({pca.explained_variance_ratio_[i]*100:.1f}% variance)")
            plt.tight_layout()
            plt.savefig(os.path.join(os.path.join(save_path, channel_type), f"{name}_PCA_components.pdf"))
            plt.close()
        
            importance_df = pd.DataFrame.from_dict(importance_scores[channel_type], orient="index")
            importance_df = importance_df[sorted(importance_df.columns)]  # consistent channel order

            plt.figure(figsize=(max(8, 0.4 * len(importance_df.columns)),
                                max(4, 0.3 * len(importance_df))))
        sns.heatmap(importance_df, cmap="viridis", cbar_kws={"label": "Relative importance"})
        plt.xlabel("Channel")
        plt.ylabel("Patient")
        plt.title(f"Channel importance across patients — {channel_type}")
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.join(save_path, channel_type), "channel_importance_heatmap.pdf"))
        plt.close()

        rows = [
            {"patient": patient, "PC": pc, "rank1": chs[0], "rank2": chs[1], "rank3": chs[2]}
            for patient, pcs in top_channels[channel_type].items()
            for pc, chs in pcs.items()
        ]
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(os.path.join(save_path, channel_type), "top_channels.csv"), index=False)


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
        plt.savefig(os.path.join(os.path.join(save_path, channel_type), "top_3_contributors.pdf"))
        plt.close()

    return PCA_components