"""Compute a genuine Adult Income race-group sensitivity from archived predictions.

The archived row-level export is ordered within the original sex subgroup and
preserves the test-row order. We verify labels against the reconstructed seed
split before attaching the alternative race grouping. No model is retrained.
"""
from __future__ import annotations

from pathlib import Path
import json
import tempfile
import zipfile

import numpy as np
import pandas as pd

from _primary_core import SEEDS, load_adult, preprocess


ROOT = Path(__file__).resolve().parents[1]
DATA_ARCHIVE = Path(r"C:\Users\mvk80\Downloads\FMG_Data.zip")
ROW_EXPORT = Path(r"C:\Users\mvk80\Downloads\fmg_row_level_predictions.csv")
OUT = ROOT / "analysis" / "outputs" / "review_additions"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ("RF", "XGB", "LGB", "MLP")


def recall(y, pred):
    y = np.asarray(y); pred = np.asarray(pred)
    pos = y == 1
    return float(np.sum(pred[pos] == 1) / np.sum(pos)) if np.any(pos) else np.nan


def metrics(y, pred, g):
    return float(np.mean(np.asarray(y) == np.asarray(pred))), recall(np.asarray(y)[g], np.asarray(pred)[g])


def load_adult_source(tmp):
    with zipfile.ZipFile(DATA_ARCHIVE) as outer:
        outer.extract("FMG_Data/adult.zip", tmp)
    inner = Path(tmp) / "FMG_Data" / "adult.zip"
    target = Path(tmp) / "adult"
    with zipfile.ZipFile(inner) as zf:
        zf.extractall(target)
    return next(target.rglob("adult.data"))


def main():
    chunks = []
    for chunk in pd.read_csv(ROW_EXPORT, chunksize=500_000):
        q = chunk[(chunk.dataset == "Adult Income") & chunk.model.isin(MODELS)]
        if len(q): chunks.append(q)
    raw = pd.concat(chunks, ignore_index=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="fmg_adult_race_") as td:
        data = load_adult(load_adult_source(td))
        for seed in SEEDS:
            x_train, x_test, y_test_train, y_test, sex = preprocess(data, seed)
            race = data.frame.loc[x_test.index, "race"].astype(str).to_numpy()
            race_focus = race != "White"
            for model in MODELS:
                q = raw[(raw.seed == seed) & (raw.model == model)].copy()
                if set(q.condition.unique()) != {"complete", "masked_0.20"}:
                    raise RuntimeError(f"Unexpected conditions for {seed}/{model}: {q.condition.unique()}")
                # The export preserves test-row order within the original sex group.
                for primary_group in (0, 1):
                    idx = np.flatnonzero(sex == primary_group)
                    for condition in ("complete", "masked_0.20"):
                        z = q[(q.subgroup == primary_group) & (q.condition == condition)]
                        if len(z) != len(idx):
                            raise RuntimeError(f"Length mismatch for seed={seed}, model={model}, subgroup={primary_group}, condition={condition}")
                        if not np.array_equal(z.y.to_numpy(int), y_test[idx]):
                            raise RuntimeError(f"Label-order mismatch for seed={seed}, model={model}, subgroup={primary_group}, condition={condition}")
                        z = z.copy()
                        z["race_focus"] = race_focus[idx]
                        rows.append(z)
    aligned = pd.concat(rows, ignore_index=True)
    per_seed = []
    for (model, seed), group in aligned.groupby(["model", "seed"], sort=True):
        clean = group[group.condition == "complete"]
        masked = group[group.condition == "masked_0.20"]
        # One row per test record after the sex-segment alignment.
        if len(clean) != len(masked):
            raise RuntimeError("Clean/masked row counts differ")
        ac, rc = metrics(clean.y, clean.pred, clean.race_focus)
        am, rm = metrics(masked.y, masked.pred, masked.race_focus)
        accuracy_loss = ac - am
        recall_loss = rc - rm
        per_seed.append({
            "dataset": "Adult Income",
            "model": model,
            "seed": seed,
            "focus_definition": "non-White race (g0) versus White race (g1)",
            "clean_accuracy": ac,
            "masked_accuracy": am,
            "accuracy_loss": accuracy_loss,
            "clean_focus_recall": rc,
            "masked_focus_recall": rm,
            "focus_recall_loss": recall_loss,
            "fmg": recall_loss - accuracy_loss,
            "focus_positive_n": int(np.sum(clean.race_focus & (clean.y.to_numpy() == 1))),
        })
    detail = pd.DataFrame(per_seed)
    detail.to_csv(OUT / "adult_race_group_per_seed.csv", index=False)
    summary = detail.groupby("model", as_index=False).agg(
        n_seeds=("seed", "size"), mean_fmg=("fmg", "mean"), sd_fmg=("fmg", "std"),
        min_fmg=("fmg", "min"), max_fmg=("fmg", "max"),
        positive_splits=("fmg", lambda x: int(np.sum(x > 0))),
        mean_focus_positive_n=("focus_positive_n", "mean"),
    )
    summary.insert(0, "dataset", "Adult Income")
    summary.to_csv(OUT / "adult_race_group_summary.csv", index=False)
    (OUT / "adult_race_group_manifest.json").write_text(json.dumps({
        "dataset": "Adult Income", "focus_group": "non-White race", "comparison_group": "White race",
        "models": list(MODELS), "seeds": list(SEEDS), "mask_rate": 0.20,
        "protocol": "archived row-level predictions; no retraining; labels verified against reconstructed test splits",
        "raw_prediction_export": str(ROW_EXPORT),
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
