# FMG Rectified Submission Package

This package contains the rectified manuscript and the supplementary files needed to reproduce the additional validation reported in the manuscript.

## Rectifications completed

- Corrected the uncertainty description: the primary intervals are seed-level bootstrap summaries across the ten repeated splits, not patient-level confidence intervals.
- Corrected Proposition 1 so that it uses the hard decision indicator and bounds a hard-recall intervention contrast only.
- Removed unsupported synthetic stress-test and enumeration claims that were not present in the verified public release.
- Added a reproducible threshold-discordance analysis based on the saved primary outputs and narrowed the FMG claim to a paired monitoring diagnostic.
- Clarified that the all-model type-aware, native missing-value, FDA/ReMasker, and related supplementary outputs must be uploaded to the public repository before submission.

## Files

- `JAISCR_FMG_submission_q2_q3_ready_rectified_revision2.docx` — revised manuscript after the adversarial panel review. The earlier rectified DOCX is retained for traceability.
- `typeaware_all_models_per_seed.csv` and its manifest — all-model type-aware sensitivity outputs.
- `native_missing_per_seed.csv` and its manifest — native missing-value baseline outputs.
- ReMasker CSV/JSON/statistics files — stronger external comparison outputs.
- Three resumable Colab notebooks and the native-baseline runner.
- `independent_adversarial_peer_review.md` and `independent_adversarial_peer_review_rectified.md` — audit records and evidence-based revision logs.

The current public GitHub v4 release was not modified by this local packaging step. Upload and verify this package in the public repository before claiming that all manuscript results are publicly reproducible.
