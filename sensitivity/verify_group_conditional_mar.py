"""Reproduce integrity checks for the group-conditional MAR-like sensitivity.

Run from any directory with pandas, numpy, and scipy installed:
    python sensitivity/verify_group_conditional_mar.py

This validates the archived outputs; it does not rerun the Colab model fits.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parent
PER_SEED = ROOT / "group_conditional_mar_per_seed.csv"
SUMMARY = ROOT / "group_conditional_mar_statistics.csv"
SEEDS = [42, 123, 456, 1, 7, 13, 21, 37, 55, 99]
DATASETS = ["Adult Income", "COMPAS", "Taiwan Credit"]
MODELS = ["RF", "XGB", "LGB", "MLP"]
RATES = [0.10, 0.20]
KEYS = ["dataset", "model", "rate_nominal"]


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values, returned in original order."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted_sorted = np.maximum.accumulate(
        (len(p_values) - np.arange(len(p_values))) * p_values[order]
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=1e-9, atol=1e-12):
        raise AssertionError(f"{label}: recomputed {actual}, saved {expected}")


def main() -> None:
    per = pd.read_csv(PER_SEED)
    summary = pd.read_csv(SUMMARY)
    expected_rows = len(DATASETS) * len(MODELS) * len(SEEDS) * len(RATES)
    assert len(per) == expected_rows, (len(per), expected_rows)
    assert len(summary) == len(DATASETS) * len(MODELS) * len(RATES)
    assert per.isna().sum().sum() == 0
    assert summary.isna().sum().sum() == 0
    assert not per.duplicated(KEYS + ["seed"]).any()
    assert set(per["dataset"]) == set(DATASETS)
    assert set(per["model"]) == set(MODELS)
    assert set(per["rate_nominal"]) == set(RATES)

    for _, group in per.groupby(KEYS, sort=False):
        assert sorted(group["seed"].tolist()) == sorted(SEEDS)
        # The study statistic is defined as subgroup recall loss minus overall
        # accuracy loss in each condition.
        close(
            np.max(np.abs(group["mg_mcar"] - (
                group["recall_g0_drop_mcar"] - group["accuracy_drop_mcar"]
            ))),
            0.0,
            "MCAR FMG identity",
        )
        close(
            np.max(np.abs(group["mg_mar"] - (
                group["recall_g0_drop_mar"] - group["accuracy_drop_mar"]
            ))),
            0.0,
            "MAR-like FMG identity",
        )
        close(
            np.max(np.abs(group["mar_minus_mcar_mg"] - (
                group["mg_mar"] - group["mg_mcar"]
            ))),
            0.0,
            "paired FMG contrast identity",
        )

    # The planned mechanism raises group-0 missingness while calibrating the
    # expected overall cell-masking probability to the nominal rate.
    assert (per["p_mar_group0"] > per["p_mar_group1"]).all()
    assert (per["rate_mar_group0_realized"] > per["rate_mar_group1_realized"]).all()
    rate_means = per.groupby(["dataset", "rate_nominal"])["rate_mar_realized"].mean()
    assert np.max(np.abs(rate_means.to_numpy() - rate_means.index.get_level_values(1))) < 0.001

    # Recompute descriptive summaries and paired Wilcoxon statistics from the
    # ten seed-level paired differences, then reproduce Holm within each rate.
    for rate in RATES:
        rows = summary[summary["rate_nominal"] == rate].copy()
        assert len(rows) == 12
        raw_p = []
        for idx, row in rows.iterrows():
            sample = per[
                (per["dataset"] == row["dataset"])
                & (per["model"] == row["model"])
                & (per["rate_nominal"] == rate)
            ].sort_values("seed")
            diff = sample["mar_minus_mcar_mg"].to_numpy(dtype=float)
            close(float(sample["mg_mcar"].mean()), row["mean_mg_mcar"], "mean MCAR FMG")
            close(float(sample["mg_mar"].mean()), row["mean_mg_mar"], "mean MAR-like FMG")
            close(float(diff.mean()), row["mean_mar_minus_mcar_mg"], "mean paired contrast")
            close(float(diff.std(ddof=1)), row["sd_mar_minus_mcar_mg"], "SD paired contrast")
            two_sided = wilcoxon(diff, alternative="two-sided", method="exact")
            greater = wilcoxon(diff, alternative="greater", method="exact")
            close(float(two_sided.statistic), row["wilcoxon_statistic_two_sided"], "Wilcoxon statistic")
            close(float(two_sided.pvalue), row["wilcoxon_p_two_sided"], "two-sided Wilcoxon p")
            close(float(greater.pvalue), row["wilcoxon_p_directional_mar_gt_mcar"], "directional Wilcoxon p")
            raw_p.append(float(two_sided.pvalue))
        expected_holm = holm_adjust(np.asarray(raw_p))
        for actual, expected in zip(rows["holm_p_two_sided_within_rate"], expected_holm):
            close(float(actual), float(expected), "Holm-adjusted p")

    print(f"PASS: {len(per)} per-seed rows; {len(summary)} dataset/model/rate tests.")
    print("PASS: no missing or duplicate condition/seed records; all expected seeds present.")
    print("PASS: FMG identities, paired contrasts, nominal rates, and group-rate direction.")
    print("PASS: means, SDs, exact paired Wilcoxon tests, and within-rate Holm adjustment.")
    for rate, rows in summary.groupby("rate_nominal"):
        print(
            f"{rate:.0%}: positive mean contrasts="
            f"{int((rows['mean_mar_minus_mcar_mg'] > 0).sum())}/12; "
            f"Holm p < .05="
            f"{int((rows['holm_p_two_sided_within_rate'] < .05).sum())}/12"
        )


if __name__ == "__main__":
    main()
