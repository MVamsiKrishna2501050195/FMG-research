"""Run the low-cost reviewer-requested analyses without retraining.

1. Nadeau--Bengio corrected resampled t-tests of mean FMG against zero for
   the 12 primary dataset-model cells.
2. Complementary-group sensitivity, treating g1 as the focus group using the
   same archived predictions and masks.

The inputs are archived seed-level outputs; no model is retrained here.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "outputs" / "review_additions"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ("Adult Income", "COMPAS", "Taiwan Credit")
MODELS = ("RF", "XGB", "LGB", "MLP")
K = 10
N_TEST_OVER_N_TRAIN = 0.30 / 0.70


def corrected_resampled_t(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    corrected_se = float(np.sqrt((1.0 / len(values) + N_TEST_OVER_N_TRAIN) * sd**2))
    t_stat = mean / corrected_se if corrected_se else np.nan
    df = len(values) - 1
    p_two = float(2.0 * student_t.sf(abs(t_stat), df)) if np.isfinite(t_stat) else np.nan
    crit = float(student_t.ppf(0.975, df))
    return {
        "n_splits": len(values),
        "mean_fmg": mean,
        "sd_fmg": sd,
        "corrected_se": corrected_se,
        "t_statistic": t_stat,
        "df": df,
        "p_two_sided": p_two,
        "ci95_low": mean - crit * corrected_se,
        "ci95_high": mean + crit * corrected_se,
        "positive_splits": int(np.sum(values > 0)),
    }


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjustment without an extra dependency."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p_values[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def corrected_tests() -> pd.DataFrame:
    raw = pd.read_csv(ROOT / "masking_gap_per_seed.csv")
    raw = raw[(raw.dataset.isin(DATASETS)) & (raw.model.isin(MODELS)) & (~raw.exploratory.astype(bool))]
    rows = []
    for (dataset, model), group in raw.groupby(["dataset", "model"], sort=True):
        row = {"dataset": dataset, "model": model}
        row.update(corrected_resampled_t(group["masking_gap"].to_numpy()))
        rows.append(row)
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out["p_two_sided"].to_numpy())
    out.to_csv(OUT / "corrected_resampled_t_primary.csv", index=False)
    return out


def complementary_group() -> pd.DataFrame:
    raw = pd.read_csv(ROOT / "sensitivity" / "complementary_group_sensitivity_per_seed.csv")
    raw = raw[raw.dataset.isin(DATASETS) & raw.model.isin(MODELS)].copy()
    rows = []
    for (dataset, model), group in raw.groupby(["dataset", "model"], sort=True):
        g0 = group["fmg_g0"].to_numpy(float)
        g1 = group["fmg_g1"].to_numpy(float)
        rows.append({
            "dataset": dataset,
            "model": model,
            "n_seeds": len(group),
            "prespecified_g0_mean_fmg": float(g0.mean()),
            "alternative_g1_mean_fmg": float(g1.mean()),
            "alternative_g1_sd": float(g1.std(ddof=1)),
            "alternative_g1_min": float(g1.min()),
            "alternative_g1_max": float(g1.max()),
            "alternative_g1_positive_splits": int(np.sum(g1 > 0)),
            "same_sign_as_g0_splits": int(np.sum(np.sign(g0) == np.sign(g1))),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "alternative_focus_group_g1_summary.csv", index=False)
    return out


if __name__ == "__main__":
    a = corrected_tests()
    b = complementary_group()
    manifest = {
        "analysis": "reviewer-requested low-cost additions",
        "corrected_test": "Nadeau-Bengio corrected resampled t-test of FMG against zero",
        "test_ratio": N_TEST_OVER_N_TRAIN,
        "primary_cells": 12,
        "alternative_group": "same archived Adult/COMPAS/Taiwan predictions with g1 treated as focus group",
        "outputs": [str(p.relative_to(ROOT)) for p in OUT.glob("*.csv")],
    }
    (OUT / "review_additions_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(a.to_string(index=False))
    print(b.to_string(index=False))
