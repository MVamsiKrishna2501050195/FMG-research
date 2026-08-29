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

Before submission
-----------------
Add the package to a permanent repository or journal supplement, record the
software and library versions used for execution, and replace the request-only
distribution wording in the manuscript with the permanent archive link.
