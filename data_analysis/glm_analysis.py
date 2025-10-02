import pandas as pd
import mne
from mne_nirs.experimental_design import longest_inter_annotation_interval
from nilearn.glm.first_level import make_first_level_design_matrix
from mne_nirs.statistics import run_glm
from mne_nirs.channels import get_short_channels
from pandas import DataFrame

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
        redundant_annotations = [x for x in np.unique(haemo.annotations.description) if x not in set(class_instance.data_types)]
        if len(redundant_annotations) != 0:
            for annotation in redundant_annotations:
                haemo.annotations.delete(haemo.annotations.description == annotation)
        renames = {cond: cond.split("/")[0] if "/" in cond else cond for cond in haemo.annotations.description}
        haemo.annotations.rename(renames)
        isis, names = longest_inter_annotation_interval(haemo)
        high_pass_value = 1/(max(isis)*2)
        
        short_channel_haemo = get_short_channels(subject.raw_haemo_unfiltered)
        
        # Here we modify the inputs just like they do in the MNE wrapper - but do it here, as the MNE wrapper doesn't support varying stimulus durations for different events
        frame_times = haemo.times
         # Create events for nilearn
        conditions = haemo.annotations.description
        onsets = haemo.annotations.onset - haemo.first_time
        duration = haemo.annotations.duration
        events = DataFrame({'trial_type': conditions,
                            'onset': onsets,
                            'duration': duration})
        drift_model="cosine"
        hrf_model="glover"
        min_onset = 0 # Normally used for fMRI in case events are coded relative to a trigger that happens before scanning. Not relevant here.
        high_pass = high_pass_value
        add_regs = short_channel_haemo.get_data().T
        oversampling = 50 # Default value.
        drift_order = 1 # When we use the cosine drift model this parameter doesn't really matter, as the drift order is then actually determined by the high_pass argument
        add_reg_names = short_channel_haemo.ch_names
        fir_delays = None # Default when we don't use a FIR model
        
        
        design_matrix = make_first_level_design_matrix(frame_times, events,
                                        drift_model=drift_model,
                                        drift_order=drift_order,
                                        hrf_model=hrf_model,
                                        min_onset=min_onset,
                                        high_pass=high_pass,
                                        add_regs=add_regs,
                                        oversampling=oversampling,
                                        add_reg_names=add_reg_names,
                                        fir_delays=fir_delays)

        glm_estimates = run_glm(haemo, design_matrix)

        # Get column labels (conditions, drift terms, constant, etc.)
        regressor_names = [regressor_name for regressor_name in design_matrix.columns if ("drift" not in regressor_name) & ("constant" not in regressor_name)]

        for ch_name, result in glm_estimates.data.items():
            betas = result.theta.flatten()  # 1 beta per regressor
            for cond_name, beta in zip(regressor_names, betas):
                if cond_name in np.unique(haemo.annotations.description):
                    all_betas.append({
                        "participant": subject.name,
                        "channel": ch_name,
                        "condition": cond_name,
                        "beta": beta
                    })

    betas_df = pd.DataFrame(all_betas)
    betas_df.to_csv(rf"C:\Users\NTres\OneDrive - Danmarks Tekniske Universitet\Bachelor_projekt\glm_betas.csv", index=False) # Not permanent

    with localconverter(pandas2ri.converter):
        globalenv["rdf"] = betas_df

    lme4 = importr("lme4")

    r('''
    library(lme4)
    model <- lmer(beta ~ condition + channel + (1 | participant), data=rdf, REML=FALSE)
    nullModel <- lmer(beta ~ channel + (1 | participant), data=rdf, REML=FALSE)
    print(summary(model))
    print(summary(nullModel))
    
    anova_result <- anova(model, nullModel)
    print(anova_result)
    ''')