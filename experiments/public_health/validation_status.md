# Public-health extension validation status

## Completed

- The uploaded CDC 2024 BRFSS archive was verified and loaded from its official XPT member.
- The UCI Diabetes 130-US Hospitals loader was exercised in a bounded smoke run.
- The CDC 2024 BRFSS loader was exercised in a bounded smoke run using one seed, one mask repeat, both split settings, three constructed missingness mechanisms, three masking rates, and the four configured model families.
- The CDC smoke run completed on CPU in 602.3 seconds, with peak process RSS of approximately 1.37 GB and 90 per-mask metric rows.
- Five implementation tests pass, including test-only masking, FMG identity, patient-disjoint splitting, identifier/audit-label leakage checks, and checkpoint replacement.
- The CDC smoke outputs are stored under `outputs_cdc_smoke/` and include a run manifest, denominators, per-mask metrics, threshold sensitivity, and unit-bootstrap files.

## Not yet a manuscript result

The smoke run is an implementation and reproducibility check. It is not a replacement for the pre-specified multi-seed primary extension analysis.

## Clean full Colab run (23 October 2026)

The latest full-run archive is stored at
`outputs_full_2026_10_23/FMG_public_health_full_outputs.zip`. It reports 2,700
per-mask metric rows, all 60 dataset/split/seed/model units complete, and all
60 denominator records present with model identifiers. Each dataset/model has
15 observations in the 80/20, 20% MCAR condition (three seeds x five mask
repetitions). Metric identity checks, duplicate-key checks, denominator
coverage, and every output hash in the manifest pass. Maximum absolute FMG
identity discrepancy is approximately 1.1e-16.

The BRFSS manifest now reports 353,573 eligible respondents, matching the
corrected extraction from the uploaded BRFSS ZIP. The run records both the ZIP
hash and the exact inner XPT member hash. The audit-code SHA-256 in the
manifest matches the repository script, and the checkpoint signature includes
that code hash. The archive reports 9,129 seconds runtime on a two-logical-CPU
Colab runtime and sampled peak process RSS of about 2.00 GB.

The earlier stale-XPT and missing-denominator issues have therefore been
resolved for this completed run. The output archive is now used for a separate,
bounded public-health extension in `manuscript/FMG_manuscript_public_health_extension_revised.docx`
and `manuscript/FMG_supplementary_public_health_extension.docx`. This does not
change any saved primary benchmark values. The extension remains descriptive
retrospective benchmark evidence, not deployment or decision validation.

The smoke run was an implementation check; the clean full run now supplies the
descriptive extension results. Neither run demonstrates deployment utility,
improved decisions, or monitoring benefit. Those remain future work requiring
an operational setting and decision-linked evaluation.
