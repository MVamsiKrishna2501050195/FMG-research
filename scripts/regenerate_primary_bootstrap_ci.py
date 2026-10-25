"""Regenerate descriptive primary seed-bootstrap intervals from saved raw rows.

This matches the corrected notebook cell used for Table 3. It resamples the
ten benchmark split values and does not estimate population-level uncertainty.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["Adult Income", "COMPAS", "Taiwan Credit"]
MODELS = ["RF", "XGB", "LGB", "MLP"]
N_BOOT = 10_000
SEED = 20260909


def main() -> None:
    rows = pd.read_csv(ROOT / "masking_gap_per_seed.csv")
    rows = rows[
        rows["dataset"].isin(DATASETS)
        & rows["model"].isin(MODELS)
    ]
    output = []
    for (dataset, model), group in rows.groupby(["dataset", "model"], sort=True):
        values = group["masking_gap"].to_numpy(dtype=float)
        if len(values) != 10:
            raise ValueError(f"Expected ten split values for {dataset}/{model}; got {len(values)}")
        # Match the notebook helper, which re-initializes the same RNG for each
        # dataset/model pair.
        rng = np.random.RandomState(SEED)
        samples = rng.choice(values, size=(N_BOOT, len(values)), replace=True).mean(axis=1)
        output.append(
            {
                "dataset": dataset,
                "model": model,
                "n_seeds": len(values),
                "mean_fmg": float(values.mean()),
                "bootstrap_ci95_low": float(np.quantile(samples, 0.025)),
                "bootstrap_ci95_high": float(np.quantile(samples, 0.975)),
                "interval_scope": "descriptive repeated-seed benchmark",
            }
        )
    if len(output) != 12:
        raise ValueError(f"Expected 12 primary dataset/model pairs; got {len(output)}")
    result = pd.DataFrame(output)
    result.to_csv(ROOT / "fmg_bootstrap_ci_primary.csv", index=False)

    table3 = pd.read_csv(ROOT / "adult_corrected_rerun" / "table3_primary_components_corrected.csv")
    joined = result.merge(table3, on=["dataset", "model"], suffixes=("_boot", "_table"))
    for col in ["mean_fmg", "bootstrap_ci95_low", "bootstrap_ci95_high"]:
        table_col = {
            "mean_fmg": "fmg_mean",
            "bootstrap_ci95_low": "fmg_bootstrap_ci_lower",
            "bootstrap_ci95_high": "fmg_bootstrap_ci_upper",
        }[col]
        if not np.allclose(joined[col], joined[table_col], atol=5e-6, rtol=0):
            raise AssertionError(f"Bootstrap output does not match Table 3 field {table_col}")
    print(f"Wrote {len(result)} Table 3 bootstrap rows; all values match Table 3 exactly.")


if __name__ == "__main__":
    main()
