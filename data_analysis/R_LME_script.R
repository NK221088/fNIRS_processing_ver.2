## ============================================================================
## LME analysis of drug effects on Math/Hard_Math vs Control response contrast
##
## Model, agreed structure:
##   response_diff ~ Condition + Drug * Recording + (1 | ID_prefix/Session)
##
##   - response_diff : mean(active - matched Control) per (recording x Condition),
##                      already computed in Python (build_lme_dataframe.py) and
##                      exported to lme_input.csv
##   - Condition      : Math / Hard_Math, additive control (no interaction --
##                      no hypothesis that drug effect differs by task difficulty)
##   - Drug * Recording : the actual hypothesis -- does drug effect depend on
##                      time since dose (different drugs peak at different times)
##   - (1 | ID_prefix/Session) : session nested within patient, absorbs
##                      session-to-session variation not explained by Drug/Recording
##
## Procedure (agreed): fit the ONE pre-specified model once. No stepwise
## deletion/refitting. Two-step reporting:
##   1) Omnibus LRT: does Drug matter at all (main effect + all its interactions)?
##   2) If (1) is significant, Type III table for every term, from the SAME fit.
##   3) If a specific interaction is significant, follow up with emmeans
##      contrasts (FDR-adjusted), as descriptive characterization only.
## ============================================================================

suppressMessages({
  library(lme4)
  library(afex)
  library(car)
  library(emmeans)
})

rdf <- read.csv("C:/Users/NKUE0003/Documents/GitHub/fNIRS_processing_ver.2/lme_input_subset.csv", stringsAsFactors = TRUE)

# Lock factor levels/reference categories explicitly -- don't rely on
# alphabetical default, since e.g. "Apomorphine" would alphabetically precede
# "Placebo" and silently become the reference level otherwise.
rdf$Recording <- factor(rdf$Recording, levels = c("T0", "T15", "T60"))
rdf$Condition <- factor(rdf$Condition, levels = c("Math", "Hard_Math"))
rdf$Drug      <- factor(rdf$Drug,      levels = c("Placebo", "Apomorphine", "Methylphenidate"))

cat("Rows:", nrow(rdf), " | Patients:", length(unique(rdf$ID_prefix)), "\n")
cat("Rows per Drug x Recording x Condition cell:\n")
print(table(rdf$Drug, rdf$Recording, rdf$Condition))

## ----------------------------------------------------------------------------
## Fit the full, pre-specified model ONCE (afex::mixed sets contr.sum
## automatically, so Type III tests below are valid without extra setup)
## ----------------------------------------------------------------------------
full <- afex::mixed(
  response_diff ~ Condition + Drug * Recording + (1 | ID_prefix),
  data = rdf,
  method = "LRT"
)

cat("\n================ Type III table (single fit, no refitting) ================\n")
print(full)

## ----------------------------------------------------------------------------
## Step 1: omnibus test -- does Drug matter AT ALL (main effect + interaction)?
## Compare the full model against the same model with Drug and everything
## involving Drug removed in one go.
##
## NOTE: both models are fit INDEPENDENTLY here with direct lmer() calls (not by
## reusing/update()-ing the object afex::mixed() returned above). Reusing afex's
## internal fit causes lme4's anova() to fail with "all models must be fit to
## the same data object", even though the data is in fact identical -- an
## internal call/environment mismatch from afex's wrapper, not a real data
## discrepancy. Fitting fresh avoids it.
## ----------------------------------------------------------------------------
full_ml <- lmer(
  response_diff ~ Condition + Drug * Recording + (1 | ID_prefix),
  data = rdf, REML = FALSE
)
no_drug_ml <- lmer(
  response_diff ~ Condition + Recording + (1 | ID_prefix),
  data = rdf, REML = FALSE
)

four_ml <- lmer(
  response_diff ~ Condition + Consciousness + (1 | ID_prefix) + (1 | ID_prefix:Session:Recording),
  data = rdf, REML = FALSE
)

no_four_ml <- lmer(
  response_diff ~ Condition + (1 | ID_prefix) + (1 | ID_prefix:Session:Recording),
  data = rdf, REML = FALSE
)

anova(no_four_ml, four_ml)

cat("\n================ Step 1: omnibus test for Drug (any form) ================\n")
print(anova(no_drug_ml, full_ml))

## ----------------------------------------------------------------------------
## Step 2 (only if Step 1 is significant): the Type III table above already
## tells you WHICH term(s) drive it (Drug, Drug:Recording). Follow up with
## FDR-adjusted pairwise contrasts within Recording, to see which drug(s) and
## which timepoint(s) specifically differ -- descriptive/exploratory, not a
## fresh confirmatory test.
##
## Fit a fresh REML=TRUE model for this (REML is preferred for point estimates/
## contrasts; ML above was only needed for the LRT comparison in Step 1) --
## again independent of afex's returned object, for the same reason as Step 1.
## ----------------------------------------------------------------------------
full_reml <- lmer(
  response_diff ~ Condition + Drug * Recording + (1 | ID_prefix),
  data = rdf, REML = TRUE
)

cat("\n================ Follow-up: Drug contrasts within each Recording ================\n")
emm <- emmeans(full_reml, ~ Drug | Recording)
print(pairs(emm, adjust = "fdr"))