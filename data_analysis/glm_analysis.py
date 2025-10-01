import pandas as pd
import mne
from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_glm

from rpy2.robjects import r, globalenv
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri

def run_glm_analysis(subjects):

    all_betas = []

    subjects = [
        ("P00", 'c:/Users/NKUE0003/Documents/GitHub/fNIRS_processing_ver.2/test0_raw.fif'),
        ("P01", 'c:/Users/NKUE0003/Documents/GitHub/fNIRS_processing_ver.2/test1_raw.fif'),
        ("P02", 'c:/Users/NKUE0003/Documents/GitHub/fNIRS_processing_ver.2/test2_raw.fif'),
        ("P03", 'c:/Users/NKUE0003/Documents/GitHub/fNIRS_processing_ver.2/test3_raw.fif')
        
        # Add more subjects here
    ]

    for subj_id, epochs_file in subjects:
        epochs = mne.io.read_raw_fif(epochs_file, preload=True)

        design_matrix = make_first_level_design_matrix(
            epochs,
            stim_dur=5.0,
            hrf_model="spm",
            drift_order=1
        )

        glm_estimates = run_glm(epochs, design_matrix)

        # Get column labels (conditions, drift terms, constant, etc.)
        regressor_names = design_matrix.columns  

        for ch_name, result in glm_estimates.data.items():
            betas = result.theta.flatten()  # 1 beta per regressor
            for cond_name, beta in zip(regressor_names, betas):
                all_betas.append({
                    "participant": subj_id,
                    "channel": ch_name,
                    "condition": cond_name,
                    "beta": beta
                })

    betas_df = pd.DataFrame(all_betas)

    with localconverter(pandas2ri.converter):
        globalenv["rdf"] = betas_df

    lme4 = importr("lme4")

    r('''
    library(lme4)
    model <- lmer(beta ~ condition + channel + (1 | participant), data=rdf)
    print(summary(model))
    ''')