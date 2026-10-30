Masking Gap reproducibility package

## Current public release

The current submission release is [`fmg-submission-release-v4-2026-10-29`](https://github.com/MVamsiKrishna2501050195/FMG-research/tree/fmg-submission-release-v4-2026-10-29). It supersedes the v2 and v3 release tags; use v4 for the current reproducibility package rather than the historical materials described below. The v4 tag tree and release ZIP were checked from a fresh clone against all 51 SHA-256 manifest entries.

## Historical v2 snapshot (superseded by v4)

Release snapshot: `fmg-submission-release-v2-2026-10-25`. The canonical analysis
notebook is `FMG_research_final_REVIEWER_FIXED.ipynb`; the current manuscript
is `manuscript/FMG_manuscript_final_release_2026-10-25.docx`.
The submission bundle and SHA-256 inventory are
`FMG_final_submission_release_v2_2026-10-25.zip` and
`FMG_final_submission_release_v2_2026-10-25_manifest.json`.

Canonical source
----------------
`FMG_research_final.ipynb` is the historical source notebook for the archived
release results. `FMG_research_final_REVIEWER_FIXED.ipynb` is the corrected
source notebook. It fixes Adult missing-token parsing, the group-conditional
MAR mechanism, stale dataset/runtime descriptions, and the frozen-model check;
it also exports a run manifest with input/frame hashes, package versions,
hardware/runtime information, design settings, and output hashes. The executed
Colab notebook copy is preserved at
`reproducibility/FMG_research_final_REVIEWER_FIXED_EXECUTED_2026-10-25.ipynb`;
all 34 code cells have execution counts and there are no error outputs. The
submitted ZIP is preserved at `reproducibility/FMG_full_run_outputs_2026-10-25.zip`,
with its extracted contents and manifest in `incoming_run_outputs_20261025/`.
The archive contains
the 800-row primary result table for four datasets (German Credit exploratory),
five classifiers, ten seeds, and clean/5%/10%/20% evaluation conditions. The
manifest reports Python 3.13.16 on Linux x86_64 with two logical CPUs and a
runtime of about 2,033 seconds. The exact Colab CPU model and memory allocation
were not recorded. All 10 CSV output hashes and byte counts match the manifest.
The corrected Adult parser excludes rows with `?` as missing, leaving 30,162
records from the 32,561-row UCI training file. The primary 20%-MCAR summaries
were recalculated from the raw rows and match the revised manuscript table.
The superseded `cc3_masking_gap_fmg_audit_method_methodological_revised.ipynb`
and legacy root-level figure PNGs have been removed from the current package;
their prior versions remain recoverable from Git history. Use the corrected
notebook and numbered files under `figures/` for current work.

Manuscript package
------------------
- `manuscript/FMG_manuscript_final_release_2026-10-25.docx` is the current
  release manuscript; `manuscript/FMG_supplementary_adult_sensitivities_reconciled.docx`
  is its Adult-harmonized supplement. Both include the corrected Adult primary
  results and rerun Adult-dependent age-conditioned and imputation sensitivities.
- `manuscript/FMG_supplementary_public_health_extension.docx` contains the
  BRFSS/UCI extension tables, including the revised Table S7 group-recall
  components and comparison.
- `manuscript/FMG_manuscript_reviewer_comments_with_multiseed_shap_final_revisions.docx`
  is preserved as the immediately preceding draft for traceability.
- `manuscript/FMG_supplementary_reviewer_comments_with_multiseed_shap.docx`
  is the preceding supplement. It has **not** yet been harmonized with the
  corrected Adult sample and must not be submitted alongside the corrected
  manuscript until its Adult-dependent results are updated or removed.
  Historical p-value columns in archived sensitivity CSVs remain provenance
  only and are not interpreted as inferential evidence.
- `manuscript/reviewer_comment_response_matrix.md` records which comments are
  addressed in text, partly addressed, or still require evidence outside this
  benchmark.

Included material
-----------------
- Primary and supplementary result CSV files
- subgroup_denominators_per_seed.csv
- `fmg_bootstrap_ci_primary.csv` with descriptive seed-level bootstrap
  intervals (not population-level confidence guarantees)
- Numbered figures for the revised manuscript, plus the additional primary
  seed-distribution plot

Figure map for the revised manuscript
-------------------------------------
| Manuscript figure | Repository file |
|---|---|
| Fig. 1 Audit architecture | `figures/fig01_architecture_clean.png` |
| Fig. 2 Audit data flow | `figures/fig02_data_flow_clean.png` |
| Fig. 3 MCAR masking-gap curves | `figures/fig03_masking_gap_curves.png` |
| Fig. 4 Accuracy and focus-group recall loss | `figures/fig04_accuracy_vs_recall.png` |
| Fig. 5 Group-conditional MAR-like versus MCAR sensitivity | `figures/fig05_group_conditional_vs_mcar.png` |
| Fig. 6 Class-rate matched-subsample analysis | `figures/fig06_class_imbalance.png` |
| Fig. 7 Supplementary COMPAS extended-seed analysis | `figures/fig07_compas_extended_seeds.png` |
| Fig. 8 COMPAS-only exploratory SHAP shifts | `figures/fig08_shap_shifts.png` |
| Fig. 9 Multi-seed type-aware imputation sensitivity | `figures/fig09_imputation_ablation.png` |
| Additional primary seed-level FMG distributions | `figures/fmg_primary_seed_distributions.png` |

The superseded root-level figure PNGs are not part of the current figure set.
Figures 3–6 and 8–9 are regenerated using the corrected Adult rows. The legacy
Adult SHAP panel is intentionally not reused.

Denominator schema
------------------
The denominator CSV records dataset, seed, group, group role, test-group total,
positive-label count, negative-label count, positive-label rate,
recall denominator, no-positive-label indicator, and exploratory status.

Output verification
------------------
Use the pinned environment in `requirements.txt` for a reproducible local
setup. The primary rerun manifest records pandas 2.2.3; categorical column
assignment in the reviewer-fix notebook is also written to avoid the pandas
3.x `.loc` assignment incompatibility.

Run `python verify_primary_outputs.py` after installing NumPy and pandas to
recompute the current root-level corrected full-run seed-level summaries. Run
`python sensitivity/verify_group_conditional_mar.py` after installing NumPy,
pandas, and SciPy to check the expanded group-conditional sensitivity tables
and recompute descriptive summaries. These checks validate saved results; they
do not retrain models. Seed-level p-values in archived CSVs are not used as
inferential support because repeated splits reuse observations.
`python regenerate_primary_plots_from_archived_results.py` reproduces plots
from the current root `results_raw_10seeds.csv`. Corrected manuscript figures
and their lineage are in `adult_corrected_rerun/corrected_figures/`. The
previous independent plot-set manifest documents figures regenerated from
saved CSVs; it does not refit models. To rerun the complete
The legacy `rerun_primary_training_from_source_notebook.py` wrapper points to
`FMG_research_final.ipynb`, which is the historical notebook and does not
include the corrected Adult parsing. Do not use that wrapper to reproduce the
final manuscript results. The corrected full run was executed from
`FMG_research_final_REVIEWER_FIXED.ipynb`; its executed notebook, manifest, and
outputs are preserved in the locations listed above. A rerun from the
corrected notebook should use the same pinned environment and matching input
files, and write to a new output directory rather than replacing the archived
results.

Adult correction and traceability
---------------------------------
The supplied UCI Adult archive was parsed from its `adult.data` member only;
`adult.test` was not appended. The archive SHA-256 is
`7537312dd56c2b98035880805ce99e68183a30ee468aa5329d6df0fbb3cc21bb`. Of
32,561 source rows, 2,399 contained at least one `?` missing marker. Under the
existing complete-case protocol, 30,162 rows remain. The corrected Adult-only
rerun evaluated five models × ten seeds × clean, 5%, 10%, and 20% MCAR
conditions, yielding 200 rows and a separate manifest in
`adult_corrected_rerun/`. It was followed by a complete corrected notebook
rerun; see `incoming_run_outputs_20261025/` and the Independent regeneration
status below. The corrected full-run outputs in the repository root are the
current canonical primary CSVs and match the 2026-10-25 run manifest; they
replace the pre-correction Adult outputs previously tracked there. Earlier
versions remain available in Git history.

The correction package includes Adult primary metrics and denominators,
Table 3 components, matched-subsample results, group-conditional MAR
rows/summaries, type-aware imputation sensitivity, and corrected figures. The
type-aware results remain limited to RF, three datasets, ten repeated splits,
and 20% MCAR. The completed full rerun also generated an Adult and COMPAS
primary SHAP output for RF seed 42; these single-seed attributions remain
exploratory and are not evidence of attribution stability. The manuscript's
separate multi-model, multi-seed SHAP sensitivity concerns the public-health
extension and is not interchangeable with this output.

Adult-dependent age-conditioned MAR and type-aware imputation sensitivities
were rerun using the corrected `adult.data` sample. Per-seed and summary files
ending in `_reconciled_latest.csv` combine the new Adult rows with retained,
unchanged COMPAS and Taiwan Credit rows. Across 72 age-conditioned
dataset/model/rate/strength contrasts, mean FMG differences range from
-0.0092 to 0.0318; the maximum is COMPAS/XGB at 20% expected missingness and
age odds ratio 4, positive in 10/10 splits. Corrected Adult type-aware mean
FMG ranges from 0.026972 to 0.032747 across the three tested strategies. The
current KNN result used scikit-learn 1.9.1, while the earlier output used
1.6.1; the manuscript reports this modest version sensitivity and avoids
describing KNN/mode and median/mode as tied.

The latest submission files are
`manuscript/FMG_manuscript_adult_sensitivities_reconciled.docx` and
`manuscript/FMG_supplementary_adult_sensitivities_reconciled.docx`. The
abstract and introduction explicitly state that FMG is not a between-group
disparity and that g0 recall loss exceeded g1 in only 4/12 primary
dataset-model pairs. Notebook cell 35 is documented as median/mode only;
KNN/mode and iterative/mode results come from the separate type-aware script.

Analysis conventions
--------------------
- Test-time MCAR masking is applied after model training.
- The same trained model is evaluated on clean and stressed test data.
- Preprocessing and imputation are fitted using training data only.
- Across MCAR rates, the same seeded uniform draws create nested masks; within
  a split, that mask is shared across model families for paired comparisons.
- The constructed group-conditional sensitivity uses the observed audit-group
  label to set row-level cell-mask probabilities. It is one simulated MAR
  mechanism, not a representative model of every MAR or MNAR process.
- FMG compares positive-class recall loss with aggregate accuracy loss; a
  positive FMG does not by itself establish greater loss for the focus group
  than for the comparison group. The two subgroup recall losses must be
  compared directly for that claim.
- Recall uses the implementation's zero_division=0 convention.
- Primary seeds are 42, 123, 456, 1, 7, 13, 21, 37, 55, and 99.
- Primary masking rates are 5%, 10%, and 20%.

New public-health extension (separate from primary results)
------------------------------------------------------------
`experiments/public_health/` contains a pre-specified supplemental audit for
CDC 2024 BRFSS and UCI Diabetes 130-US Hospitals. It adds no result to the
primary benchmark until the new pipeline has been run, checked, and the
manuscript revised from verified outputs. The extension includes LR, RF, XGB,
LGB, and MLP; patient-disjoint grouping for UCI; BRFSS survey-weighted held-out
metrics and stratum/PSU bootstrap intervals; MCAR plus explicitly constructed
group-conditional MAR and value-conditional MNAR; 5%, 10%, and 20% rates;
descriptive threshold sensitivity; and optional fixed-sample TreeSHAP shifts.

Its full protocol, task/group definitions, data attributions, limitations,
dependencies, and run instructions are in
`experiments/public_health/README.md`. Downloaded datasets belong under the
ignored `.local_data/` folder. The extension writes only to the ignored
`experiments/public_health/outputs/` folder and records hashes and environment
details in a run manifest. The completed full run is archived at
`experiments/public_health/outputs_full_2026_10_23/FMG_public_health_full_outputs.zip`.
It contains 2,700 metric rows and all 60 denominator records; its input-member,
code, and output hashes were checked against the manifest. At 80/20 and 20%
MCAR, mean FMG ranged from 0.0272 to 0.0455 across BRFSS models, while it
ranged from 0.0000 to 0.0038 across UCI Diabetes models. The extension is
included separately in the revised manuscript and supplement, without changing
archived primary results. It is a retrospective benchmark, not deployment or
clinical validation.

### Multi-model, multi-seed TreeSHAP sensitivity

`experiments/public_health/run_multiseed_tree_shap.py` adds a separate
exploratory attribution sensitivity for the two public-health datasets. It
re-fits RF, XGBoost, and LightGBM under the extension's 80/20 split,
preprocessing, and hyperparameters for seeds 42, 123, and 456, then compares
positive-class attributions for the same up-to-20 test records per audit group
under clean and one deterministic 20% MCAR masked condition. Because the
original Colab estimator objects were not serialized, this is a controlled
re-fit rather than an explanation of those exact stored objects. The local
runtime and its package versions are recorded in the run manifest.

RF and XGBoost use approximate Saabas TreeExplainer values; LightGBM uses exact
TreeSHAP because the installed SHAP API does not support the approximation
option for LightGBM. The methods and raw output scales differ, so attribution
magnitudes are interpreted within each model only. The analysis is descriptive,
uses a small fixed subgroup sample, and does not establish causal drivers or
explanation stability. It does not change primary benchmark results.

The completed run is in
`experiments/public_health/shap_multiseed_2026_10_23/`:
`multiseed_tree_shap_shifts.csv` contains per-feature/per-seed rows,
`top_feature_summary.csv` contains the top two features per dataset/model/group,
and `run_manifest.json` records design, software, input hashes, and output hash.
Reproduce with:

```powershell
python experiments/public_health/run_multiseed_tree_shap.py `
  --brfss <path-to-2024-BRFSS-XPT> `
  --diabetes-zip .local_data/diabetes+130-us-hospitals.zip `
  --seeds 42 123 456 --sample-size 20 `
  --output-dir experiments/public_health/shap_multiseed_rerun
```

Then regenerate the compact summary with
`python experiments/public_health/summarize_multiseed_tree_shap.py
<input-csv> <summary-csv> --top-k 2`.
Verify the archived run and its hashes with
`python experiments/public_health/verify_multiseed_tree_shap.py`.

Post-review sensitivity analyses
--------------------------------
The following analyses are supplemental and do not replace or alter the primary
results in `results_raw_10seeds.csv`:

- `sensitivity/typeaware_imputation_sensitivity_per_seed.csv` records a
  10-seed Random Forest sensitivity analysis at 20% MCAR for Adult Income,
  COMPAS, and Taiwan Credit. It masks raw test feature cells before applying a
  training-fitted mixed-type pipeline: numeric median, numeric 5-nearest
  neighbours, or numeric iterative imputation; categorical values use a
  training-fitted most-frequent imputer and one-hot encoding.
- `sensitivity/typeaware_imputation_sensitivity_summary.csv` aggregates the
  per-seed results and denominator ranges. These summaries are descriptive;
  no additional inferential tests were performed for this sensitivity family.
- `sensitivity/complementary_group_sensitivity_per_seed.csv` and
  `sensitivity/complementary_group_sensitivity_summary.csv` report the
  complementary-group FMG calculation from the already-published primary
  per-seed results. The four primary model families are included; logistic
  regression is excluded because it is exploratory.
- `sensitivity/baseline_recall_sensitivity_per_seed.csv` pairs each primary
  model's clean-test focus-group recall with its 20%-MCAR FMG for the same
  dataset and seed. `sensitivity/baseline_recall_sensitivity_summary.csv`
  reports within-model, within-dataset Spearman correlations (10 paired seeds
  per combination). These are descriptive checks only; no hypothesis tests or
  causal interpretation are intended. The script
  `sensitivity/summarize_baseline_recall_sensitivity.py` reproduces them.
- `sensitivity/typeaware_imputation_sensitivity_environment.txt` records the
  local execution environment. It is distinct from the original Colab run,
  whose exact hardware allocation was not captured.
- `sensitivity/run_typeaware_imputation.py` and
  `sensitivity/summarize_complementary_group.py` reproduce the imputation and
  complementary-group summaries.
- `sensitivity/group_conditional_mar_per_seed.csv` and
  `sensitivity/group_conditional_mar_statistics.csv` contain the expanded
  group-conditional MAR stress-test outputs. They cover three datasets,
  all four primary model families, ten fixed seeds, and 10%/20% nominal
  missingness (240 per-seed condition rows and 24 summaries).
- `sensitivity/expanded_group_conditional_mar_colab_executed.ipynb` is the
  executed Colab notebook used to produce those outputs. It records cell
  outputs and package versions. `sensitivity/group_conditional_mar_delta_mg.png`
  visualizes the paired seed-level change in FMG.
- `sensitivity/age_conditional_mar_outputs/` contains the initial exploratory
  age-conditioned MAR versus matched-eligible-feature MCAR run at an odds ratio
  of 2 per standard deviation of age. Its 240 per-seed rows and 24 summaries
  preserve the original fixed-strength analysis.
- `sensitivity/age_mar_strength_outputs/` contains the expanded propensity-
  strength sensitivity at age odds ratios 1.5, 2, and 4 per standard
  deviation. Age remains observed in both conditions; all other test-feature
  cells are eligible for masking. The frozen-model analysis covers three
  datasets, four primary model families, ten fixed seeds, and 10%/20% rates
  (720 per-seed rows and 72 summaries). This mechanism-specific contrast is
  summarized descriptively with paired means and positive-split counts. Earlier
  archived test outputs are retained for provenance but are not interpreted as
  inferential evidence because repeated splits reuse observations. This is
  exploratory benchmark evidence—not
  evidence for all MAR mechanisms, MNAR, independent populations, deployment
  benefit, or causal effects. Details and runtime are in that folder's README
  and `run_manifest.json`; reproduce with
  `python sensitivity/run_age_conditional_mar_sensitivity.py --taiwan-csv
  <path-to-UCI_Credit_Card.csv> --age-odds-ratios 1.5 2 4 --output-dir
  sensitivity/age_mar_strength_outputs`.
- `sensitivity/group_conditional_mar_run_manifest.json` records the design,
  captured runtime versions, known environment omissions, and SHA-256 hashes
  for the notebook and generated outputs.
- `sensitivity/verify_group_conditional_mar.py` checks the archived per-seed
  and summary tables, recomputes FMG identities and descriptive statistics, and
  ignores historical inferential-test columns. Run it with
  `python sensitivity/verify_group_conditional_mar.py` after installing NumPy
  and pandas.

Group-conditional MAR implementation note
-----------------------------------------
This sensitivity analysis evaluates one deliberately constructed
group-conditional MAR mechanism; it is not an evaluation of every possible MAR
mechanism. The sensitive-group variable is retained as fully observed audit
metadata and excluded from the model feature matrix in all three datasets.
For each test feature cell, masking probability depends on this observed group
and not on the value of the feature being masked. The group-specific
probabilities are calibrated so their weighted expected average equals the
requested overall rate; the focus group receives the higher probability.
Random masks are applied only to test features. Each classifier is fitted on
clean training data and remains unchanged during both clean and masked scoring;
preprocessing/imputation is training-fitted. The paired comparison is the
group-conditional MAR FMG minus MCAR FMG for the same dataset, model, seed, and
nominal rate. Repeated seeds are benchmark resampling conditions, not
independent deployment populations. Results are summarized with paired means,
standard deviations, and positive-split counts; archived test-statistic and
p-value columns are retained only for provenance and are not interpreted as
inferential evidence.

In this run, 12/12 mean paired contrasts were positive at 10% and 11/12 at
20%. These descriptive sensitivity findings apply only to this constructed
mechanism; they do not establish robustness to all MAR mechanisms or universal
subgroup behavior.
The executed public-health extension notebook output reports Python 3.11 (kernel metadata), NumPy
2.1.3, pandas 2.2.3, scikit-learn 1.6.1, XGBoost 3.2.0, LightGBM 4.6.0, and
SHAP 0.51.0. The original run used a Google Colab CPU runtime, but the exact
CPU model/count, memory allocation, and SciPy version were not recorded. The
notebook downloads Adult and COMPAS from their upstream sources
and requires the user to upload the Taiwan Credit CSV; those source snapshots
are not bundled here, so exact reruns also require matching input data files.

Independent regeneration status
--------------------------------
A previous 800-row run in `primary_training_regeneration/` used an Adult parser
that did not treat `?` as missing; its Adult results have been superseded. The
corrected notebook has now been rerun end-to-end, and its manifest and outputs
are in `incoming_run_outputs_20261025/`; the executed notebook copy and original
ZIP are preserved in `reproducibility/`. The 800 raw result rows include the
corrected 30,162-record Adult cohort, and each archived CSV's SHA-256 and byte
count match the run manifest. Recalculation of the 12 primary dataset-model
summaries at 20% MCAR reproduces the revised manuscript values at the displayed
precision. The manifest identifies this fresh run as Linux x86_64 / Python
3.13.16 / two logical CPUs; it does not report a CPU model or memory allocation.
This is distinct from the historical CPU-based Colab run, whose exact backend
CPU model/count and memory allocation remain unknown, and from the earlier
local run. Earlier root-level CSV versions are preserved in Git history.

The root-level `mar_probe_results.csv` and `mar_probe_statistics.csv` are
retained for provenance as the earlier exploratory cellwise heterogeneous-rate
probe. They are not the expanded group-conditional MAR results above and should
not be cited as a formal MAR analysis.

The type-aware sensitivity uses the original ten split seeds and a shared
20%-MCAR cell mask across imputation strategies within each dataset and seed.
It is limited to Random Forest and one masking rate. It does not establish
general preprocessing robustness. A positive FMG remains a conditional
diagnostic, not a fairness certificate or a deployment criterion.

Repository and manuscript
-------------------------
This release is publicly available at:
https://github.com/MVamsiKrishna2501050195/FMG-research/tree/fmg-submission-release-v2-2026-10-25

Before journal submission, archive this release in a persistent service such
as Zenodo if the journal requires a DOI.
