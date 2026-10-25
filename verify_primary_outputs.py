"""Check corrected saved primary and type-aware FMG outputs.

Run from any directory with NumPy and pandas installed:
    python verify_primary_outputs.py

This validates saved results; it does not rerun model fitting or regenerate
figures from the notebook.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / name)


def main() -> None:
    raw = read("results_raw_10seeds.csv")
    summaries = read("descriptive_primary_summaries.csv")
    main_stats = summaries[
        summaries["Main"].astype(str).str.lower().isin(["true", "1"])
    ]
    primary = raw[(raw["dataset"] != "German Credit") & (raw["model"] != "LR")]
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
            raise AssertionError(f"Missing/duplicate archived descriptive row: {dataset}/{model}")
        row = match.iloc[0]
        expected = {
            "Acc Drop mean": float(np.mean(accuracy_drop)),
            "Recall Drop mean": float(np.mean(recall_drop)),
            "FMG mean": float(np.mean(fmg)),
            "FMG SD": float(np.std(fmg, ddof=1)),
        }
        # Recompute descriptive statistics directly from raw paired records.
        # Historical p-value/effect-size columns in the archived summary are
        # intentionally excluded from validation and interpretation.
        for column, value in expected.items():
            if abs(float(row[column]) - value) > 5e-6:
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

    bootstrap = read("fmg_bootstrap_ci_primary.csv")
    components = read("adult_corrected_rerun/table3_primary_components_corrected.csv")
    bootstrap_check = bootstrap.merge(
        components, on=["dataset", "model"], suffixes=("_csv", "_table")
    )
    if len(bootstrap_check) != 12:
        raise AssertionError("Primary bootstrap/Table 3 pairs are incomplete")
    for csv_column, table_column in [
        ("mean_fmg", "fmg_mean"),
        ("bootstrap_ci95_low", "fmg_bootstrap_ci_lower"),
        ("bootstrap_ci95_high", "fmg_bootstrap_ci_upper"),
    ]:
        if not np.allclose(
            bootstrap_check[csv_column],
            bootstrap_check[table_column],
            atol=5e-6,
            rtol=0,
        ):
            raise AssertionError(f"Bootstrap/Table 3 mismatch: {csv_column}")

    sensitivity = pd.read_csv(
        ROOT / "adult_corrected_rerun" / "typeaware_imputation_per_seed_reconciled_latest.csv"
    )
    sensitivity_summary = pd.read_csv(
        ROOT / "adult_corrected_rerun" / "typeaware_imputation_summary_reconciled_latest.csv"
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
    print("PASS: paired FMG identities and descriptive statistics recompute from raw rows.")
    print("NOTE: legacy inferential-test columns are not validated or interpreted.")
    print(f"PASS: all {len(denominators)} subgroup denominator records reconcile.")
    print("PASS: bootstrap intervals reconcile with Table 3 at displayed precision.")
    print("PASS: 90 type-aware sensitivity rows reconcile with 9 summaries.")
    print("PASS: all 9 numbered manuscript figures and the additional seed plot are present.")
    print("LIMIT: this checks saved outputs; it does not rerun primary training or regenerate figures.")


if __name__ == "__main__":
    main()
