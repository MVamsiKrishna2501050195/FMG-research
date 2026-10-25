"""Produce a bounded top-feature summary from the exploratory TreeSHAP CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    data = pd.read_csv(args.input_csv)
    required = {
        "dataset", "seed", "model", "audit_group", "feature",
        "mean_signed_shap_shift_stress_minus_clean", "mean_absolute_shap_difference",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    summary = data.groupby(
        ["dataset", "model", "audit_group", "feature"], as_index=False
    ).agg(
        mean_signed_shift=("mean_signed_shap_shift_stress_minus_clean", "mean"),
        mean_absolute_shift=("mean_absolute_shap_difference", "mean"),
        sd_absolute_shift=("mean_absolute_shap_difference", "std"),
        n_seeds=("seed", "nunique"),
        seed_signs=("mean_signed_shap_shift_stress_minus_clean", lambda x: ";".join(map(str, x))),
    )
    def direction_consistency(row):
        mean_sign = row["mean_signed_shift"]
        if mean_sign == 0:
            return 0
        return sum(
            (float(value) > 0) if mean_sign > 0 else (float(value) < 0)
            for value in row["seed_signs"].split(";")
        )

    summary["direction_consistent_seeds"] = summary.apply(direction_consistency, axis=1)
    summary = summary.sort_values(
        ["dataset", "model", "audit_group", "mean_absolute_shift"],
        ascending=[True, True, True, False],
    )
    summary = summary.groupby(["dataset", "model", "audit_group"], group_keys=False).head(args.top_k)
    summary = summary.drop(columns="seed_signs")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(summary)} top-feature rows to {args.output_csv}")


if __name__ == "__main__":
    main()
