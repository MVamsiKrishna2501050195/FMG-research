"""Check saved primary FMG outputs against their per-seed components.

Run from any directory with NumPy and pandas installed:
    python verify_primary_outputs.py

This validates archived results; it does not rerun model fitting or regenerate
figures from the notebook.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / name)


def exact_positive_wilcoxon(values: np.ndarray) -> float:
    ranks = pd.Series(np.abs(values)).rank(method="average").to_numpy()
    observed = float(ranks[values > 0].sum())
    totals = (
        float(sum(rank for rank, bit in zip(ranks, bits) if bit))
        for bits in itertools.product((0, 1), repeat=len(values))
    )
    return sum(total >= observed - 1e-12 for total in totals) / (2 ** len(values))


def main() -> None:
    raw = read("results_raw_10seeds.csv")
    stats = read("stats_results_10seeds.csv")
    primary = raw[(raw["dataset"] != "German Credit") & (raw["model"] != "LR")]
    main_stats = stats[stats["Main"].astype(str).str.lower().isin(["true", "1"])]
    checked = 0

    for (dataset, model), group in primary.groupby(["dataset", "model"]):
        clean = group[group.rate == 0].sort_values("seed").set_index("seed")
        masked = group[group.rate == 0.2].sort_values("seed").set_index("seed")
        if len(clean) != 10 or len(masked) != 10 or not clean.index.equals(masked.index):
            raise AssertionError(f"Expected ten paired seeds: {dataset}/{model}")
        accuracy_drop = clean.accuracy.to_numpy() - masked.accuracy.to_numpy()
        recall_drop = clean.recall_g0.to_numpy() - masked.recall_g0.to_numpy()
        fmg = recall_drop - accuracy_drop
        if not np.allclose(fmg, masked.masking_gap.to_numpy(), atol=2.1e-6, rtol=0):
            raise AssertionError(f"Per-seed FMG mismatch: {dataset}/{model}")

        match = main_stats[(main_stats.Dataset == dataset) & (main_stats.Model == model)]
        if len(match) != 1:
            raise AssertionError(f"Missing/duplicate summary: {dataset}/{model}")
        row = match.iloc[0]
        expected = {
            "Acc Drop mean": float(np.mean(accuracy_drop)),
            "Recall Drop mean": float(np.mean(recall_drop)),
            "MG mean": float(np.mean(fmg)),
            "MG std": float(np.std(fmg, ddof=1)),
            "p2 (MG test)": exact_positive_wilcoxon(fmg),
            "MG Cohen's dz": float(np.mean(fmg) / np.std(fmg, ddof=1)),
        }
        for column, value in expected.items():
            tolerance = 5e-4 if column == "MG Cohen's dz" else 5e-6
            if abs(float(row[column]) - value) > tolerance:
                raise AssertionError(f"{column} mismatch for {dataset}/{model}")
        checked += 1

    if checked != 12:
        raise AssertionError(f"Expected 12 primary comparisons; found {checked}")

    denominators = read("subgroup_denominators_per_seed.csv")
    expected_denominator_keys = {
        (dataset, seed)
        for dataset in raw["dataset"].drop_duplicates()
        for seed in raw["seed"].drop_duplicates()
    }
    observed_denominator_keys = set(
        zip(denominators["dataset"], denominators["seed"])
    )
    if (
        len(denominators) != len(expected_denominator_keys)
        or denominators.duplicated(["dataset", "seed"]).any()
        or observed_denominator_keys != expected_denominator_keys
        or not (
            denominators.recall_denominator == denominators.positive_label_count
        ).all()
        or not (
            denominators.subgroup_total
            == denominators.positive_label_count + denominators.negative_label_count
        ).all()
    ):
        raise AssertionError("Subgroup denominator checks failed")

    sensitivity = pd.read_csv(
        ROOT / "sensitivity" / "typeaware_imputation_sensitivity_per_seed.csv"
    )
    sensitivity_summary = pd.read_csv(
        ROOT / "sensitivity" / "typeaware_imputation_sensitivity_summary.csv"
    )
    if len(sensitivity) != 90 or len(sensitivity_summary) != 9:
        raise AssertionError("Type-aware sensitivity row counts do not match")
    for _, row in sensitivity_summary.iterrows():
        subset = sensitivity[
            (sensitivity.dataset == row.dataset)
            & (sensitivity.imputation_strategy == row.imputation_strategy)
        ]
        if len(subset) != 10 or abs(subset.fmg_g0.mean() - row.mean_fmg_g0) > 5e-7:
            raise AssertionError(
                f"Type-aware summary mismatch: {row.dataset}/{row.imputation_strategy}"
            )

    figures = [
        "figures/fig01_architecture_clean.png",
        "figures/fig02_data_flow_clean.png",
        "figures/fig03_masking_gap_curves.png",
        "figures/fig04_accuracy_vs_recall.png",
        "figures/fig05_group_conditional_vs_mcar.png",
        "figures/fig06_class_imbalance.png",
        "figures/fig07_compas_extended_seeds.png",
        "figures/fig08_shap_shifts.png",
        "figures/fig09_imputation_ablation.png",
        "figures/fmg_primary_seed_distributions.png",
    ]
    missing = [
        name for name in figures
        if not (ROOT / name).is_file() or (ROOT / name).stat().st_size == 0
    ]
    if missing:
        raise AssertionError("Missing/empty figure files: " + ", ".join(missing))

    print("PASS: 12 primary summaries match the saved per-seed metric components.")
    print("PASS: exact one-sided Wilcoxon p-values, standard deviations, and Cohen's dz recompute.")
    print(f"PASS: all {len(denominators)} subgroup denominator records reconcile.")
    print("PASS: 90 type-aware sensitivity rows reconcile with 9 summaries.")
    print("PASS: all 9 numbered manuscript figures and the additional seed plot are present.")
    print("LIMIT: this checks saved outputs; it does not rerun primary training or regenerate figures.")


if __name__ == "__main__":
    main()
