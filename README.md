Masking Gap reproducibility package

Canonical source
----------------
`FMG_research_final.ipynb` is the release notebook for the reported results,
including the empirical diagnostic-discordance analysis. The earlier
`cc3_masking_gap_fmg_audit_method_methodological_revised.ipynb` filename is
retained for compatibility; its code cells mirror the release notebook.

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
| Fig. 8 Subgroup SHAP shifts | `figures/fig08_shap_shifts.png` |
| Fig. 9 Single-seed imputation ablation | `figures/fig09_imputation_ablation.png` |
| Additional primary seed-level FMG distributions | `figures/fmg_primary_seed_distributions.png` |

The legacy root-level `fig3_mar_vs_mcar.png` is the earlier narrow RF/two-dataset
probe and is retained for provenance only; it is not revised-manuscript Fig. 5.

Denominator schema
------------------
The denominator CSV records dataset, seed, group, group role, test-group total,
positive-label count, negative-label count, positive-label rate,
recall denominator, no-positive-label indicator, and exploratory status.

Output verification
------------------
Run `python verify_primary_outputs.py` after installing NumPy and pandas to
recompute the twelve primary seed-level FMG summaries, exact one-sided
Wilcoxon p-values, effect sizes, denominator checks, and type-aware sensitivity
summary checks. Run `python sensitivity/verify_group_conditional_mar.py` after
installing NumPy, pandas, and SciPy to independently verify the expanded
group-conditional sensitivity tables and statistical tests. These checks
validate saved results; they do not retrain models. Run
`python regenerate_primary_plots_from_archived_results.py` to regenerate the
two primary MCAR figures and the seed-distribution plot from the archived
`results_raw_10seeds.csv`; the script stores outputs and hashes in
`primary_plot_regeneration/`. All eight quantitative plot-producing notebook
cells (Figures 1–7 and the seed-distribution plot) have also been rerun from the
archived primary and supplementary CSV tables. Their outputs and SHA-256 hashes
are in `primary_plot_regeneration/independent_full_plot_set/` and its
`plot_regeneration_manifest.json`. This plotting check does not refit models or
independently rerun supplementary sensitivity analyses. To rerun the complete
800-row primary training pipeline
from the canonical notebook, install the notebook's scientific dependencies
and run `python rerun_primary_training_from_source_notebook.py --taiwan-csv
<path-to-UCI_Credit_Card.csv>`. The wrapper executes the notebook's
preprocessing, training, masking, and scoring code with ten fixed seeds over
four datasets and five classifiers; it checkpoints each dataset/seed block and
writes a separate, non-destructive output package to
`primary_training_regeneration/`. That package contains regenerated
results/statistics/plots and `run_manifest.json`, which records input hashes,
dataset fingerprints, software versions, and hardware observed for this
rerun. It does not replace the archived root-level results.

Analysis conventions
--------------------
- Test-time MCAR masking is applied after model training.
- The same trained model is evaluated on clean and stressed test data.
- Preprocessing and imputation are fitted using training data only.
- Recall uses the implementation's zero_division=0 convention.
- Primary seeds are 42, 123, 456, 1, 7, 13, 21, 37, 55, and 99.
- Primary masking rates are 5%, 10%, and 20%.

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
  (720 per-seed rows and 72 summaries). Exact two-sided Wilcoxon tests compare
  paired age-MAR versus matched-eligible-feature MCAR FMG; Holm correction is
  applied within each rate across all 36 dataset/model/strength comparisons.
  No contrast remains below .05 after this correction (minimum adjusted
  p=.0703). This is exploratory, mechanism-specific benchmark evidence—not
  evidence for all MAR mechanisms, MNAR, independent populations, deployment
  benefit, or causal effects. Details and runtime are in that folder's README
  and `run_manifest.json`; reproduce with
  `python sensitivity/run_age_conditional_mar_sensitivity.py --taiwan-csv
  <path-to-UCI_Credit_Card.csv> --age-odds-ratios 1.5 2 4 --output-dir
  sensitivity/age_mar_strength_outputs`.
- `sensitivity/group_conditional_mar_run_manifest.json` records the design,
  captured runtime versions, known environment omissions, and SHA-256 hashes
  for the notebook and generated outputs.
- `sensitivity/verify_group_conditional_mar.py` independently checks the
  archived per-seed and summary tables, recomputes FMG identities, descriptive
  statistics, exact paired Wilcoxon tests, and Holm-adjusted p-values. Run it
  with `python sensitivity/verify_group_conditional_mar.py` after installing
  NumPy, pandas, and SciPy.

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
nominal rate. Two-sided exact Wilcoxon tests are reported, with Holm adjustment
across the 12 dataset/model comparisons within each rate. The directional
test is supplemental. Repeated seeds are benchmark resampling conditions, not
independent deployment populations.

In this run, 12/12 mean paired contrasts were positive at 10% and 11/12 at
20%; 9/12 comparisons at each rate had Holm-adjusted two-sided p < .05. These
are sensitivity-analysis findings under this specific constructed mechanism;
they do not establish robustness to all MAR mechanisms or universal subgroup
behavior.
The executed notebook output reports Python 3.11 (kernel metadata), NumPy
2.1.3, pandas 2.2.3, scikit-learn 1.6.1, XGBoost 3.2.0, LightGBM 4.6.0, and
SHAP 0.51.0. The original run used a Google Colab CPU runtime, but the exact
CPU model/count, memory allocation, and SciPy version were not recorded. The
notebook downloads Adult and COMPAS from their upstream sources
and requires the user to upload the Taiwan Credit CSV; those source snapshots
are not bundled here, so exact reruns also require matching input data files.

Independent regeneration status
--------------------------------
The complete primary training loop was rerun from the canonical notebook using
the user-provided Taiwan Credit CSV and the notebook's Adult, COMPAS, and German
Credit sources. The fresh run produced 800 rows (4 datasets x 5 models x 10
seeds x 4 conditions including clean baseline) and regenerated the primary
statistics and plots in `primary_training_regeneration/`. Its 800-row result
table matches the archived `results_raw_10seeds.csv` exactly at the archived
six-decimal precision for accuracy, focus-group recall, comparison-group recall,
and FMG (zero absolute difference for each metric). This supports reproducibility
of the saved benchmark results under the documented inputs and rerun
environment; it does not establish independent deployment-population evidence.
The run manifest records SHA-256 fingerprints for the uploaded Taiwan CSV and
the downloaded inputs, dataset/group fingerprints, Python/package versions, and
the observed local hardware (Windows 10, 4 logical CPUs, approximately 7.74 GiB
RAM). These are details of the new rerun, not the historical Colab execution.
The original run is known to have used Google Colab on CPU; its exact CPU model,
CPU count, and RAM allocation were not recorded and cannot be reconstructed
reliably after the fact. The notebook's primary-training
dependency versions are documented in the manifest; they may differ from the
original Colab runtime. The historical environment therefore remains only
partially specified even though the primary result table has now been
independently regenerated and matched.

The root-level `mar_probe_results.csv`, `mar_probe_statistics.csv`, and
`fig3_mar_vs_mcar.png` are retained for provenance as the earlier exploratory
cellwise heterogeneous-rate probe. They are not the expanded group-conditional
MAR results above and should not be cited as a formal MAR analysis.

The type-aware sensitivity uses the original ten split seeds and a shared
20%-MCAR cell mask across imputation strategies within each dataset and seed.
It is limited to Random Forest and one masking rate. It does not establish
general preprocessing robustness. A positive FMG remains a conditional
diagnostic, not a fairness certificate or a deployment criterion.

Repository and manuscript
-------------------------
This release is publicly available at:
https://github.com/MVamsiKrishna2501050195/FMG-research

Before journal submission, archive this release in a persistent service such
as Zenodo if the journal requires a DOI.
