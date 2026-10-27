"""Summarize a per-seed FMG CSV using explicit descriptive reading rules."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Named constants required by the reporting protocol.
FMG_RECALL_FRAGILITY_THRESHOLD = 0.01
DID_FOCUS_LOSS_THRESHOLD = 0.01
CLEAN_FOCUS_RECALL_FLOOR = 0.05


def checklist_reading(fmg: float, did: float, clean_recall: float) -> str:
    readings: list[str] = []
    if clean_recall < CLEAN_FOCUS_RECALL_FLOOR:
        readings.append("floor effect: clean recall < 0.05")
    if fmg > FMG_RECALL_FRAGILITY_THRESHOLD and did <= 0:
        readings.append("recall fragility, not group-specific")
    if did > DID_FOCUS_LOSS_THRESHOLD:
        readings.append("focus group loses more recall")
    return "; ".join(readings) if readings else "no checklist flag at stated thresholds"


def prepare_metrics(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize new analysis, public-health S7, or archived raw-result CSVs."""
    df = data.copy()
    if {"FMG_g0", "DiD", "clean_focus_recall", "dAcc", "dTPR_g0", "dTPR_g1"}.issubset(df.columns):
        df["accuracy_loss"] = df["dAcc"]
        df["focus_recall_loss"] = df["dTPR_g0"]
        df["comparison_recall_loss"] = df["dTPR_g1"]
        df["fmg_value"] = df["FMG_g0"]
        df["did_value"] = df["DiD"]
        df["clean_recall_value"] = df["clean_focus_recall"]
    elif {"FMG_focus", "DiD_focus_minus_comparison", "clean_focus_recall", "accuracy_loss", "focus_recall_loss", "comparison_recall_loss"}.issubset(df.columns):
        df["fmg_value"] = df["FMG_focus"]
        df["did_value"] = df["DiD_focus_minus_comparison"]
        df["clean_recall_value"] = df["clean_focus_recall"]
    elif {"dataset", "model", "seed", "rate", "accuracy", "recall_g0", "recall_g1", "masking_gap"}.issubset(df.columns):
        keys = ["dataset", "model", "seed"]
        clean = df.loc[df.rate == 0, keys + ["accuracy", "recall_g0", "recall_g1"]].rename(
            columns={"accuracy": "clean_accuracy", "recall_g0": "clean_focus_recall",
                     "recall_g1": "clean_comparison_recall"}
        )
        df = df.loc[df.rate > 0].merge(clean, on=keys, how="left", validate="many_to_one")
        df["accuracy_loss"] = df.clean_accuracy - df.accuracy
        df["focus_recall_loss"] = df.clean_focus_recall - df.recall_g0
        df["comparison_recall_loss"] = df.clean_comparison_recall - df.recall_g1
        df["fmg_value"] = df.masking_gap
        df["did_value"] = df.focus_recall_loss - df.comparison_recall_loss
        df["clean_recall_value"] = df.clean_focus_recall
    else:
        raise ValueError("CSV schema not recognized; expected decomposition, extension S7, or raw primary results")
    df["reading"] = [checklist_reading(float(f), float(d), float(r))
                      for f, d, r in zip(df.fmg_value, df.did_value, df.clean_recall_value)]
    if "rate" in df and "nominal_mask_rate" not in df:
        df["nominal_mask_rate"] = df["rate"]
    return df


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    df = prepare_metrics(data)
    keys = [c for c in ("dataset", "model", "rate", "nominal_mask_rate", "mechanism", "split") if c in df]
    aggregations = {
        "accuracy_loss": ("accuracy_loss", "mean"),
        "g0_recall_loss": ("focus_recall_loss", "mean"),
        "g1_recall_loss": ("comparison_recall_loss", "mean"),
        "clean_g0_recall": ("clean_recall_value", "mean"),
        "FMG": ("fmg_value", "mean"),
        "DiD": ("did_value", "mean"),
        "n_records": ("fmg_value", "size"),
    }
    result = df.groupby(keys, dropna=False).agg(**aggregations).reset_index()
    result["reading"] = [checklist_reading(float(f), float(d), float(r))
                          for f, d, r in zip(result.FMG, result.DiD, result.clean_g0_recall)]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Per-seed CSV from decomposition, public-health S7, or primary run")
    args = parser.parse_args()
    result = summarize(pd.read_csv(args.csv))
    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
