"""Summarize the within-seed association between clean subgroup recall and FMG."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "FMG-research-git" / "results_raw_10seeds.csv"
OUT = Path(__file__).resolve().parent
DATASETS = ["Adult Income", "COMPAS", "Taiwan Credit"]
MODELS = ["RF", "XGB", "LGB", "MLP"]


def spearman(x: pd.Series, y: pd.Series) -> float:
    return float(x.rank(method="average").corr(y.rank(method="average")))


def main() -> None:
    data = pd.read_csv(SOURCE)
    selected = data[
        data["dataset"].isin(DATASETS) & data["model"].isin(MODELS)
    ]
    baseline = selected.loc[selected["rate"] == 0, [
        "dataset", "model", "seed", "recall_g0"
    ]].rename(columns={"recall_g0": "clean_baseline_recall_g0"})
    stressed = selected.loc[selected["rate"] == 0.2, [
        "dataset", "model", "seed", "masking_gap", "accuracy", "recall_g0"
    ]].rename(columns={
        "masking_gap": "fmg_at_20pct_mcar",
        "accuracy": "masked_accuracy",
        "recall_g0": "masked_recall_g0",
    })
    paired = baseline.merge(stressed, on=["dataset", "model", "seed"], validate="one_to_one")
    paired.to_csv(OUT / "baseline_recall_sensitivity_per_seed.csv", index=False)

    rows = []
    for (dataset, model), group in paired.groupby(["dataset", "model"], sort=False):
        rows.append({
            "dataset": dataset,
            "model": model,
            "n_seed_pairs": len(group),
            "baseline_recall_min": group["clean_baseline_recall_g0"].min(),
            "baseline_recall_max": group["clean_baseline_recall_g0"].max(),
            "spearman_rho_baseline_recall_vs_fmg": spearman(
                group["clean_baseline_recall_g0"], group["fmg_at_20pct_mcar"]
            ),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "baseline_recall_sensitivity_summary.csv", index=False)
    print(f"paired values: {len(paired)}; model-dataset summaries: {len(summary)}")
    print(summary.groupby("dataset")["spearman_rho_baseline_recall_vs_fmg"].agg(["min", "max"]))


if __name__ == "__main__":
    main()
