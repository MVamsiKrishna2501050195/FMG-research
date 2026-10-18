# Independent primary-pipeline rerun

This folder contains the fresh, complete run of the canonical notebook's
primary training/evaluation pipeline. It is separate from the archived files
in the repository root.

- `results_raw_10seeds.csv`: 800 rows across four datasets, five models, ten
  fixed seeds, and clean/5%/10%/20% test conditions.
- `stats_results_10seeds.csv`, `eog_results_10seeds.csv`, and
  `masking_gap_per_seed.csv`: regenerated summaries from the fresh run.
- `fmg_bootstrap_ci_primary.csv`: descriptive seed-level bootstrap intervals.
- `fig1_masking_gap_curves.*`, `fig2_acc_vs_recall.*`, and
  `fmg_primary_seed_distributions.png`: plots regenerated from the fresh
  training results.
- `run_manifest.json`: source notebook hash, input hashes, dataset
  fingerprints, package versions, and observed hardware.

The fresh 800-row table matches the archived root-level results at the
archived six-decimal precision. All 800 identifiers match; the absolute
differences for accuracy, both group recalls, and FMG are zero. Do not interpret
this benchmark rerun as independent deployment evidence. The historical
Colab hardware was not recorded; the manifest describes only the new local
rerun.

To reproduce, install the dependencies listed in `run_manifest.json` and run
from the repository root:

```powershell
python rerun_primary_training_from_source_notebook.py --taiwan-csv <path-to-UCI_Credit_Card.csv>
```

The Taiwan CSV itself is not redistributed in this package; obtain and use a
properly licensed copy. The wrapper preserves existing archived outputs and
checkpoints complete dataset/seed blocks in this directory.
