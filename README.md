Focus-Group Masking Gap (FMG) reproducibility package

Canonical source
----------------
cc3_masking_gap_fmg_audit_method_methodological_revised.ipynb is the latest
audit-method notebook and contains the primary benchmark, audit helpers, and
reproducibility diagnostics. `cc3_masking_gap_publication_ready.ipynb` is
retained as the publication-ready baseline export.

Included material
-----------------
- Primary and supplementary result CSV files
- subgroup_denominators_per_seed.csv with per-seed subgroup counts
- fmg_bootstrap_ci_primary.csv with descriptive seed-level intervals
- masking_gap_per_seed.csv with the primary per-seed FMG values
- Figures 1-7 as PNG files
- fmg_primary_seed_distributions.png for the 12 primary comparisons

Latest audit outputs
--------------------
The revised notebook adds diagnostics without changing the primary FMG
definition or benchmark values:

- `masking_diagnostics_10seeds.csv` records the target and observed
  cell-level masking rates, masked-cell counts, test dimensions, and the
  fraction of test rows containing at least one masked feature.
- `fmg_bootstrap_ci_primary.csv` reports descriptive 95% bootstrap intervals
  for the mean FMG over the ten repeated seeds. These are benchmark
  uncertainty summaries, not population-level deployment intervals.
- `fmg_primary_seed_distributions.png` shows the ten seed-level FMG values for
  each of the 12 primary dataset--model comparisons.
- The notebook includes a type-aware preprocessing sensitivity analysis. It
  is supplementary and should be run explicitly when a new sensitivity export
  is required.

Denominator schema
------------------
The denominator CSV records dataset, seed, exploratory status, focus-group
definition, subgroup total, positive-label count, negative-label count,
positive-label rate, recall denominator, and a no-positive-label indicator.
The recall denominator is the number of positive labels in the selected
subgroup for that seed; these values support reproducibility and are not
replaced by aggregate accuracy denominators.

Analysis conventions
--------------------
- MCAR masking is an independent per-feature-cell probability. For a test row
  with (p) eligible features, the expected probability of at least one
  masked feature is (1-(1-\epsilon)^p); it is not the masking rate itself.
- Test-time MCAR masking is applied after model training and only to test
  features.
- The same trained model is evaluated on clean and stressed test data.
- Preprocessing and imputation are fitted using training data only.
- The classifier is frozen after clean training and is never retrained after
  masking.
- Recall uses the implementation's zero_division=0 convention.
- Primary seeds are 42, 123, 456, 1, 7, 13, 21, 37, 55, and 99.
- Primary masking rates are 5%, 10%, and 20%.
- The 12 primary comparisons are Adult Income, COMPAS, and Taiwan Credit
  crossed with RF, XGB, LGB, and MLP.
- The zero-FMG test is a one-sided paired Wilcoxon test over seeds. The
  prespecified 12-comparison threshold is (0.05/12=0.0041667), displayed as
  0.0042 in result tables. Cohen's (d_z) uses unrounded per-seed FMG values.

Re-running the notebook
-----------------------
Run the notebook in an environment with the documented Python packages and
the dataset download links used by the manuscript. The primary experiment
fits preprocessing on each training split, trains each classifier once, and
then evaluates clean and masked test paths with the same frozen model. The
diagnostic and bootstrap cells can be run after the primary exports are
available. Do not treat the ten seeds as independent populations or live
deployments.

Repository and manuscript
-------------------------
This release is publicly available at:
https://github.com/MVamsiKrishna2501050195/FMG-research

The IJMLC-compliant manuscript in this repository cites the same repository in
its Data Availability statement. Before journal submission, archive this
release in a persistent service such as Zenodo if the journal requires a DOI.
