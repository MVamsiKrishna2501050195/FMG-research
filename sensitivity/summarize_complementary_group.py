from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "FMG-research-git" / "results_raw_10seeds.csv"
OUT = Path(__file__).resolve().parent
PRIMARY_MODELS = {"RF", "XGB", "LGB", "MLP"}

raw = pd.read_csv(SOURCE)
raw = raw[(raw["exploratory"] == False) & raw["model"].isin(PRIMARY_MODELS) & raw["rate"].isin([0.0, 0.2])]
wide = raw.pivot(
    index=["dataset", "model", "seed"],
    columns="rate",
    values=["accuracy", "recall_g0", "recall_g1"],
)
wide.columns = [f"{metric}_{int(rate * 100)}" for metric, rate in wide.columns]
wide = wide.reset_index()
wide["accuracy_drop"] = wide["accuracy_0"] - wide["accuracy_20"]
wide["fmg_g0"] = wide["recall_g0_0"] - wide["recall_g0_20"] - wide["accuracy_drop"]
wide["fmg_g1"] = wide["recall_g1_0"] - wide["recall_g1_20"] - wide["accuracy_drop"]
wide.to_csv(OUT / "complementary_group_sensitivity_per_seed.csv", index=False, float_format="%.8f")
summary = wide.groupby(["dataset", "model"], as_index=False).agg(
    n_seeds=("fmg_g1", "size"),
    mean_fmg_g0=("fmg_g0", "mean"),
    mean_fmg_g1=("fmg_g1", "mean"),
    sd_fmg_g1=("fmg_g1", "std"),
    positive_fmg_g1_seeds=("fmg_g1", lambda x: int((x > 0).sum())),
)
summary.to_csv(OUT / "complementary_group_sensitivity_summary.csv", index=False, float_format="%.8f")
print(summary.to_string(index=False))
