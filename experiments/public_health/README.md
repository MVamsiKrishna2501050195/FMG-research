# Public-health benchmark extension

This folder contains a **new, separate** extension of the FMG audit to two
public-health datasets. It does not overwrite the archived primary benchmark
tables, figures, or manuscript result values. It is a controlled, retrospective
benchmark—not a clinical study, prospective prediction study, or deployment
validation.

## Datasets, tasks, and groups

1. **CDC 2024 BRFSS combined landline/cell core survey**: binary task is
   self-reported diabetes (`DIABETE4`: yes=1, no=3); pregnancy-only and
   unknown/refused outcome responses are excluded. For an interpretable binary
   audit contrast, the evaluation cohort is restricted to the CDC's
   non-Hispanic White (`_RACEGR3=1`) and non-Hispanic Black (`_RACEGR3=2`)
   categories. Focus group = non-Hispanic Black; comparison = non-Hispanic
   White. Race is audit metadata, never a model feature. Core survey predictors
   are predeclared in `run_public_health_fmg_audit.py`; the protected label and
   survey design variables are not predictors. CDC final person weights
   (`_LLCPWT`) are used for held-out metrics; stratum (`_STSTR`) and PSU
   (`_PSU`) are used for unit-bootstrap intervals. Model fitting is unweighted.
   This is a cross-sectional classification benchmark, not prospective
   diabetes-risk prediction.
2. **UCI Diabetes 130-US Hospitals (1999–2008)**: binary task is early
   readmission (`readmitted == '<30'` versus `NO` or `>30`). Focus group =
   African American; comparison = all other records with a reported race.
   `patient_nbr` is used only as a grouping key for patient-disjoint splits and
   cluster resampling; neither it nor `encounter_id` or race is a model input.
   Encounter rows can repeat patients, so split folds keep every patient's
   records together. The task is retrospective and not a prospective clinical
   prediction claim.

The split-level audit fails fast if either broad group has fewer than 30
positive or 30 negative held-out cases. A prespecified intersectional
descriptive audit further compares Black with White women in BRFSS, and
African-American with other reported-race women in UCI. Sex and gender, like
race, are excluded from predictors. The intersectional audit is emitted only
when each of its four group-by-outcome cells has at least 30 test observations;
denominators and an estimability flag are reported in all cases. This is not a
claim that every intersectional population is represented or adequately
powered.

## Prespecified protocol

- Five classifiers: logistic regression reference (LR), random forest (RF),
  XGBoost (XGB), LightGBM (LGB), and multilayer perceptron (MLP). Models use
  fixed configurations recorded verbatim in the run manifest; there is no
  tuning on test results. The families contrast a linear reference, bagged
  trees, gradient-boosted trees, and a feed-forward neural network.
- Main holdout: target-stratified 80/20. Sensitivity: 70/30. For UCI, the
  stratified group folds approximate these ratios while guaranteeing that
  `patient_nbr` does not cross the split.
- Seeds: 42, 123, 456 by default. They are repeated benchmark splits, not
  independent populations.
- The preprocessing and transformed-space median imputer are fitted on the
  training split only. Classifiers train on clean training data and remain
  frozen. The stressor masks **test-only cells after the training-fitted
  encoder/scaler**; the frozen training-fitted imputer handles those cells.
- MCAR selects cells uniformly. The constructed MAR condition samples cells
  with a 2:1 focus-to-comparison group weight, calibrated by weighted sampling
  to an exact overall cell count. The constructed MNAR condition favors test
  cells whose observed transformed values are farther from training medians.
  It is a synthetic value-dependent mechanism, not evidence about naturally
  occurring MNAR. Rates are 5%, 10%, and 20%; five mask replicates per
  condition by default.
- BRFSS held-out metrics are evaluated with the CDC final survey weights.
  Paired 95% intervals at the 20% MCAR condition resample PSUs within strata.
  UCI intervals resample patients. These intervals are conditional on a fitted
  model and split; do not read them as population generalization across seeds.
- Outputs include the FMG and both loss components, accuracy, focus-group
  precision/recall/F1/balanced accuracy/AUROC, focus/comparison recall and FPR
  gaps, group and intersectional denominators, descriptive intersectional
  metrics when estimable, exploratory threshold counts, and optional TreeSHAP
  shifts. Thresholds are flags for sensitivity description only, not
  recommendations. TreeSHAP is exploratory and runs for RF/XGB/LGB on a fixed,
  deterministic sample when explicitly requested.
- Runtime and sampled peak process RSS, software versions, hardware strings,
  input archive hashes, hashes of the exact XPT/CSV member read from each ZIP,
  the audit-script hash, and output hashes are recorded in `run_manifest.json`.

## Obtain public data

Download data only from the providers and retain their attribution/terms:

- CDC 2024 BRFSS annual data page: <https://www.cdc.gov/brfss/annual_data/annual_2024.html>
  Download the combined landline/cell SAS Transport ZIP. The analysis uses
  the CDC 2024 core person weight, stratum and PSU documentation; see the
  official complex-sampling guide linked from that page.
- UCI Diabetes 130-US Hospitals dataset: <https://archive.ics.uci.edu/dataset/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008>
  UCI identifies its license as CC BY 4.0 and requests citation of the dataset
  and its associated publication.

Keep downloaded source files outside Git. For example, save the two archives
under `.local_data/` (already ignored by this repository).

## Computational cost

The run manifest records observed wall-clock time and sampled peak process
memory for the exact environment. For a dataset with `n` rows and `p` model
features, transformation and masking scale approximately with `n × p` per
condition; scoring is approximately `O(n × p)` for the models used, while model
fit cost depends on the classifier family and is typically the dominant term
for the tree ensembles. With `S` split seeds, `R` missingness rates, `M`
mechanisms, and `K` mask replicates, the evaluation workload is proportional
to `S × R × M × K` predictions per model and split. PSU/patient bootstrap work
is run for one explicitly labeled 20%-MCAR mask replicate per model/split/seed
to keep computation bounded. Runtime and memory are hardware-dependent and
should not be extrapolated from the smoke run.

## Reproduce

From the repository root in PowerShell, create an isolated environment and
install only the extension dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r experiments/public_health/requirements.txt
python experiments/public_health/run_public_health_fmg_audit.py `
  --brfss .local_data/LLCP2024XPT.zip `
  --diabetes-zip .local_data/diabetes+130-us-hospitals.zip
```

Use Python 3.11 with the pinned versions in `requirements.txt`. For a smaller
smoke run, pass `--seeds 42 --mask-repeats 1 --bootstrap-reps 20`.
To add fixed-sample exploratory SHAP comparisons, append `--run-shap`.
Outputs are written to ignored `experiments/public_health/outputs/` unless
`--output-dir` is provided. No data downloads or results are silently
overwritten outside that output directory. After each model/split/seed unit,
the script appends CSV rows and updates `checkpoint.json`; rerunning the exact
same command with the same output directory resumes completed units. A
protocol or input hash mismatch requires a new output directory.

The CDC XPT is re-extracted from its uploaded ZIP on every run, so an XPT
cached from an earlier upload cannot be silently reused. Checkpoint signatures
also include the audit-script hash.

The full default run is computationally substantial: it fits five model
families for each dataset, split ratio, and seed, then evaluates paired mask
replicates. Start with the smoke run and use the manifest to record the exact
full-run environment. Do not replace the primary paper's archived results with
this extension until the run has completed and passed independent checks.

## Completed full run (23 October 2026)

The completed Colab run is archived at
`experiments/public_health/outputs_full_2026_10_23/FMG_public_health_full_outputs.zip`.
It contains 2,700 per-mask rows, 60 complete dataset/split/seed/model units,
60 denominator records, summary and threshold-sensitivity tables, and paired
unit-bootstrap intervals. The manifest records the exact audit-code SHA-256,
input ZIP and inner source-member SHA-256 values, software versions, hardware,
runtime, and output hashes. The archived audit-code hash matches
`run_public_health_fmg_audit.py` in this repository.

At the prespecified 80/20 split and 20% MCAR condition, BRFSS mean FMG ranged
from 0.0272 to 0.0455 across five models, with all three seed-level means
positive and above the unvalidated 0.01 descriptive reference. UCI Diabetes
mean FMG ranged from 0.0000 to 0.0038, and none of its three seed-level means
exceeded 0.01. Means and standard deviations summarize 15 seed-by-mask-repeat
records per dataset/model and are descriptive. The contrast is evidence of
dataset- and task-dependent behavior within this benchmark, not a universal
pattern or evidence of clinical or deployment utility. These results are
incorporated as a separate extension in the revised manuscript and supplement;
they do not replace the original primary results.

The run took 9,129 seconds on a two-logical-CPU Colab runtime; sampled peak
process RSS was about 2.00 GB. The BRFSS cohort was 353,573 eligible records;
the UCI cohort was 99,493 encounters. The full run archive SHA-256 is
`75072a2bb81dbc390ce42cde04cc6ed888b72eb3b7ce8121e9ec79d57aa846ad`.

## Interpretation and boundaries

FMG is a signed contrast between focus-group recall loss and overall accuracy
loss under the same test-time stressor. Positive values mean the focus group's
recall loss exceeded the aggregate accuracy loss in that benchmark condition.
It is not a fairness certificate, causal estimator, operational threshold, or
deployment-safety guarantee. The new datasets broaden benchmark settings but
do not establish a benefit for real decisions. The results remain specific to
these datasets, target definitions, groups, models, preprocessing, splits,
and the three tested synthetic/controlled mechanisms. Repeated masks and
seeds do not create independent populations.

## Data citations and use

- Centers for Disease Control and Prevention. *Behavioral Risk Factor
  Surveillance System, 2024 Survey Data and Documentation*. CDC. Cite the
  dataset and documentation in the final journal's required style; do not imply
  CDC endorsement.
- Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). *Diabetes 130-US
  Hospitals for Years 1999–2008* [Data set]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C5230J (CC BY 4.0).
- Centers for Disease Control and Prevention. (2025). *2024 BRFSS Survey Data
  and Documentation*. https://www.cdc.gov/brfss/annual_data/annual_2024.html.

Confirm dataset terms and citation text against the provider pages at the time
of submission. The manuscript should identify the datasets as de-identified,
public-use secondary data and must not imply ethics approval or exemption
without institutional confirmation.
