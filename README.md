Masking Gap reproducibility package

Canonical source
----------------
cc3_masking_gap_publication_ready.ipynb is the canonical notebook for the
reported results. Earlier alignment notebooks are not part of this package.

Included material
-----------------
- Primary and supplementary result CSV files
- subgroup_denominators_per_seed.csv
- Figures 1-7 as PNG files

Denominator schema
------------------
The denominator CSV records dataset, seed, group, group role, test-group total,
positive-label count, negative-label count, positive-label rate,
recall denominator, no-positive-label indicator, and exploratory status.

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
- `sensitivity/typeaware_imputation_sensitivity_environment.txt` records the
  local execution environment. It is distinct from the original Colab run,
  whose exact hardware allocation was not captured.
- `sensitivity/run_typeaware_imputation.py` and
  `sensitivity/summarize_complementary_group.py` reproduce these summaries.

The type-aware sensitivity uses the original ten split seeds and a shared
20%-MCAR cell mask across imputation strategies within each dataset and seed.
It is limited to Random Forest and one masking rate. It does not establish
general preprocessing robustness. A positive FMG remains a conditional
diagnostic, not a fairness certificate or a deployment criterion.

Repository and manuscript
-------------------------
This release is publicly available at:
https://github.com/MVamsiKrishna2501050195/FMG-research

The IJMLC-compliant manuscript in this repository cites the same repository in
its Data Availability statement. Before journal submission, archive this
release in a persistent service such as Zenodo if the journal requires a DOI.
