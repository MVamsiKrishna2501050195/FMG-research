# Age-conditioned MAR propensity-strength sensitivity

This exploratory extension compares one observed-age-conditioned logistic MAR
mechanism with matched-eligible-feature MCAR while varying the strength of the
age relationship. It supplements, and does not replace, the primary MCAR
analysis or the separate group-conditional MAR analysis.

## Design

- Datasets: Adult Income, COMPAS, and Taiwan Credit.
- Classifiers: RF, XGBoost, LightGBM, and MLP.
- Seeds: 42, 123, 456, 1, 7, 13, 21, 37, 55, and 99.
- Expected cell-missingness rates: 10% and 20%.
- Age propensity strengths: conditional odds ratios 1.5, 2, and 4 per standard
  deviation of observed test age. For each strength and rate, the logistic
  intercept is calibrated so the expected cell-missingness rate equals the
  nominal rate.
- Age is kept observed in both MAR and MCAR. All other encoded/scaled test
  feature cells are eligible in both conditions, allowing paired comparison
  on the same eligible feature set.
- The existing stratified split, training-only label encoding/scaling and
  training-fitted median imputer are retained. The classifier is trained on
  clean training data once for each dataset/seed/model and remains frozen for
  all test-time conditions.
- The outcome is the paired difference FMG(age-conditioned MAR) minus
  FMG(matched-eligible-feature MCAR), each measured against the same clean-test
  baseline.
- Each dataset/model/rate/strength summary uses ten paired seed values and an
  exact two-sided Wilcoxon signed-rank test. Holm correction is applied within
  each missingness rate across all 36 dataset-model-strength comparisons.

## Results and limits

The run contains 720 per-seed rows and 72 summaries. No comparison remained
below adjusted p = .05 after the full within-rate correction; the smallest
adjusted p was .0703. The largest positive mean difference was COMPAS/XGB at
20% expected missingness and odds ratio 4 per age SD (mean paired difference
+0.0318; unadjusted exact p = .00195; adjusted p = .0703; all ten seed
differences positive). The earlier odds-ratio-2 COMPAS/XGB/20% contrast had a
mean paired difference of +0.0198 and the same adjusted p = .0703 under the
expanded family.

These findings are exploratory and specific to the tested datasets, models,
age feature, logistic propensity form and strengths, rates, and repeated
benchmark splits. They do not establish a general MAR effect, cover all MAR or
MNAR mechanisms, validate an operational threshold, or demonstrate deployment
benefit. The ten seeds are repeated splits, not independent populations.

## Files

- `age_conditional_mar_per_seed.csv`: paired per-seed metrics, denominators,
  realized mask rates, propensity strength, and frozen-model integrity checks.
- `age_conditional_mar_statistics.csv`: 72 paired summaries and adjusted
  p-values.
- `run_manifest.json`: configuration, runtime, cleaned analysis-frame
  SHA-256 hashes for all three datasets, Taiwan source-file hash, versions,
  and limitations. These hashes identify the exact cleaned inputs used by
  this completed run, including the remote Adult and COMPAS snapshots.
- `../run_age_conditional_mar_sensitivity.py`: reproducible source script.

## Reproduction

Install NumPy, pandas, SciPy, scikit-learn, XGBoost, and LightGBM, then run:

```bash
python sensitivity/run_age_conditional_mar_sensitivity.py \
  --taiwan-csv /path/to/UCI_Credit_Card.csv \
  --age-odds-ratios 1.5 2 4 \
  --output-dir sensitivity/age_mar_strength_outputs
```

Adult Income and COMPAS are loaded from the source URLs in the script; the
Taiwan CSV path and SHA-256 are recorded in the manifest. A changed dataset or
software environment is a distinct run and should be reported as such.
