# Observed-age-conditioned MAR sensitivity

This directory contains a completed, exploratory comparison of a single
observed-covariate-dependent missingness mechanism against MCAR. It is a
supplementary benchmark analysis; it does not replace the primary results or
the existing group-conditional MAR analysis.

## Design

- Datasets: Adult Income, COMPAS, and Taiwan Credit.
- Classifiers: Random Forest, XGBoost, LightGBM, and MLP.
- Seeds: 42, 123, 456, 1, 7, 13, 21, 37, 55, and 99.
- Expected cell rates: 10% and 20%.
- Split/preprocessing: stratified 70/30 train/test split; category encoders
  and MinMax scaling fit on training data only; median imputation fit on the
  training matrix only.
- Model handling: each classifier is trained on clean training data once per
  dataset/seed/model and kept frozen for the clean, MCAR, and age-conditioned
  test evaluations.
- Mask eligibility: age is retained as fully observed in both conditions;
  only the other feature cells in the test matrix can be masked. Thus MCAR is
  matched to MAR on the same eligible feature set, rather than compared to a
  condition in which age itself is also missing.
- Age-conditioned propensity: each eligible cell has a Bernoulli missingness
  probability determined by the row's observed, scaled age. The logistic
  intercept is solved so the mean probability across test rows equals the
  nominal rate. The age coefficient is fixed at ln(2) per standard deviation,
  i.e. the conditional odds double per SD increase in age. This is one
  specified mechanism, not a general family of MAR mechanisms.
- Outcome: the paired difference is FMG(age-conditioned MAR) minus
  FMG(matched-eligible-feature MCAR), using each condition's contrast against
  the same clean-test baseline.
- Inference: two-sided exact Wilcoxon signed-rank tests over the ten paired
  seed-level differences; Holm adjustment within rate over 12 model/dataset
  comparisons.

## Outputs and observed result

- `age_conditional_mar_per_seed.csv`: 240 rows (3 datasets × 4 models × 10
  seeds × 2 rates), including subgroup positive denominators, realized
  condition-specific masking rates, baseline/stressed metric losses, FMG, and
  a check that the trained classifier remained unchanged.
- `age_conditional_mar_statistics.csv`: 24 paired summaries and adjusted
  p-values.
- `run_manifest.json`: software versions, execution environment, Taiwan CSV
  SHA-256, analysis configuration, and explicit scope limitations.
- `../run_age_conditional_mar_sensitivity.py`: source script.

In this initial fixed-strength run, 1 of 24 dataset/model/rate comparisons had
Holm-adjusted p < .05:
COMPAS/XGBoost at 20% expected missingness, with a mean paired FMG difference
of +0.0198 and adjusted p=.0234. The remaining comparisons did not meet the
adjusted .05 threshold. This is exploratory, mechanism-specific evidence from
the tested benchmark splits. It does not establish that age-dependent
missingness generally increases FMG, does not cover other MAR or MNAR
mechanisms, and does not validate FMG for operational decisions or monitoring.
This fixed-strength result is retained for reproducibility. For interpretation
alongside alternative propensity strengths, use the expanded
`../age_mar_strength_outputs/` analysis, which corrects within each rate across
all 36 dataset/model/strength comparisons and finds no adjusted p < .05.

## Reproduction

Install NumPy, pandas, SciPy, scikit-learn, XGBoost, and LightGBM, then run:

```bash
python sensitivity/run_age_conditional_mar_sensitivity.py --taiwan-csv /path/to/UCI_Credit_Card.csv
```

By default, outputs are written to `sensitivity/age_conditional_mar_outputs/`.
The run uses the public Adult and COMPAS dataset sources coded in the script;
the Taiwan CSV path and checksum are recorded in the manifest. Reproduction
with a different Taiwan data file or different software versions should be
reported as a distinct run.
