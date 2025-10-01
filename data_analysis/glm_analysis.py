import pandas as pd
import mne
from mne_nirs.experimental_design import make_first_level_design_matrix, longest_inter_annotation_interval
from mne_nirs.statistics import run_glm
from mne_nirs.channels import get_short_channels

from rpy2.robjects import r, globalenv
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
import numpy as np


def run_glm_analysis(subjects, class_instance):
    
    print("Running GLM analysis")

    all_betas = []

    for idx, subject in enumerate(subjects):
        print(f"Constructing design matrix and running GLM on subject {idx+1}/{len(subjects)}")
        haemo = subject.raw_haemo.copy()
        annotations = [x for x in np.unique(haemo.annotations.description) if x not in set(class_instance.data_types)]
        for annotation in annotations:
            haemo.annotations.delete(haemo.annotations.description == annotation)
        isis, names = longest_inter_annotation_interval(haemo)
        high_pass_value = 1/(max(isis)*2)
        
        short_channel_haemo = get_short_channels(subject.raw_haemo_unfiltered)
        
        design_matrix = make_first_level_design_matrix(
            haemo,
            stim_dur=class_instance.stimulus_duration,
            hrf_model="glover",
            drift_model="cosine",
            high_pass = high_pass_value
        )
        
        # Add short channels as regressor in GLM:
        for chan in range(short_channel_haemo.ch_names):
            design_matrix[f"short_{chan}"] = short_channel_haemo.get_data(chan).T

        glm_estimates = run_glm(haemo, design_matrix)

        # Get column labels (conditions, drift terms, constant, etc.)
        regressor_names = design_matrix.columns  

        for ch_name, result in glm_estimates.data.items():
            betas = result.theta.flatten()  # 1 beta per regressor
            for cond_name, beta in zip(regressor_names, betas):
                all_betas.append({
                    "participant": subject.name,
                    "channel": ch_name,
                    "condition": cond_name,
                    "beta": beta
                })

    betas_df = pd.DataFrame(all_betas)
    betas_df.to_csv(rf"C:\Users\NKUE0003\OneDrive - Region Hovedstaden\Bachelor\results\glm_betas.csv", index=False) # Not permanent

    with localconverter(pandas2ri.converter):
        globalenv["rdf"] = betas_df

    lme4 = importr("lme4")

    r('''
    library(lme4)
    model <- lmer(beta ~ condition + channel + (1 | participant), data=rdf)
    nullModel <- lmer(beta ~ channel + (1 | participant), data=rdf)
    print(summary(model))
    print(summary(nullModel))
    
    anova_result <- anova(model, nullModel)
    print(anova_result)
    ''')