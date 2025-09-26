from rpy2.robjects import r, globalenv
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri

all_betas = []

for subj_id, epochs_file in [("P01", "P01_epochs-epo.fif"), ("P02", "P02_epochs-epo.fif")]:
    epochs = mne.read_epochs(epochs_file, preload=True)
    design_matrix = mne.stats.make_first_level_design_matrix(
        sfreq=epochs.info["sfreq"],
        events=epochs.events,
        event_id=epochs.event_id,
        hrf_model="spm",
        drift_order=1
    )
    glm_estimates = mne.stats.run_glm(epochs, design_matrix)
    for ch_name, results in glm_estimates.items():
        for cond in conditions_of_interest:
            if cond in results.theta:
                all_betas.append({
                    "participant": subj_id,
                    "channel": ch_name,
                    "condition": cond,
                    "beta": results.theta[cond]
                })

betas_df = pd.DataFrame(all_betas)

with localconverter(pandas2ri.converter):
    globalenv["rdf"] = betas_df

lme4 = importr("lme4")

r('''
library(lme4)
model <- lmer(beta ~ condition + channel + (1 + condition | participant), data=rdf)
print(summary(model))
''')
