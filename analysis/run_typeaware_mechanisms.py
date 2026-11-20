"""Compact type-aware sensitivity extension across masking rates and mechanisms.

This is a supplemental, descriptive analysis. The MAR/MNAR generators are
constructed stress scenarios and must not be interpreted as natural missingness.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score

import sys
sys.path.insert(0, r"C:\Users\mvk80\Documents\Codex\2026-11-05\search-lacuna\work\FMG-research\sensitivity")
from run_typeaware_imputation import load_datasets, fitted_transformer, SEEDS, STRATEGIES

# Laptop-feasible pilot; this does not replace the primary ten-seed analysis.
SEEDS = [42, 123, 456, 1, 7]
STRATEGIES = ("median_mode",)


def _numeric_signal(X: pd.DataFrame) -> np.ndarray:
    vals = []
    for c in X.columns:
        s = pd.to_numeric(X[c], errors="coerce")
        if s.notna().any():
            a = s.to_numpy(dtype=float)
            med = np.nanmedian(a)
            a = np.where(np.isfinite(a), a, med)
            lo, hi = np.nanmin(a), np.nanmax(a)
            if hi > lo:
                vals.append((a - lo) / (hi - lo))
    if not vals:
        return np.zeros(len(X))
    return np.mean(np.vstack(vals), axis=0)


def mask_frame(X: pd.DataFrame, rate: float, seed: int, mechanism: str):
    rng = np.random.RandomState(seed)
    n, p = X.shape
    if mechanism == "MCAR":
        probs = np.full((n, p), rate)
    else:
        signal = _numeric_signal(X)
        if mechanism == "MAR":
            row_p = rate * (0.50 + signal)
        elif mechanism == "MNAR":
            row_p = rate * (0.35 + 1.30 * signal)
        else:
            raise ValueError(mechanism)
        row_p = np.clip(row_p, 0.02, 0.95)
        row_p = np.clip(row_p * rate / row_p.mean(), 0.001, 0.95)
        probs = np.repeat(row_p[:, None], p, axis=1)
    mask = rng.rand(n, p) < probs
    out = X.copy()
    for j, c in enumerate(X.columns):
        out.loc[mask[:, j], c] = np.nan
    return out, int(mask.sum()), float(mask.mean())


def score(model, X, y, group):
    pred = model.predict(X)
    g0 = group == 0
    g1 = group == 1
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "recall_g0": float(recall_score(y[g0], pred[g0], zero_division=0)),
        "recall_g1": float(recall_score(y[g1], pred[g1], zero_division=0)),
        "g0_positive_denominator": int(np.sum(y[g0] == 1)),
        "g1_positive_denominator": int(np.sum(y[g1] == 1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=Path("mechanism_sensitivity_outputs"))
    ap.add_argument("--adult-zip", type=Path)
    ap.add_argument("--adult-only", action="store_true")
    args = ap.parse_args()
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    datasets = load_datasets(adult_zip=args.adult_zip, adult_only=args.adult_only)
    rows = []
    for dname, (df, features, sensitive_col, target_col, declared_cat) in datasets.items():
        print(f"starting {dname}", flush=True)
        X, y, group = df[features].copy(), df[target_col].to_numpy(int), df[sensitive_col].to_numpy(int)
        cats = list(X.select_dtypes(include=["object", "category"]).columns) if declared_cat is None else list(declared_cat)
        for seed in SEEDS:
            Xt, Xv, yt, yv, gt, gv = train_test_split(X, y, group, test_size=.30, random_state=seed, stratify=y)
            for rate in (.05, .10, .20):
                masked_by_mechanism = {mech: mask_frame(Xv, rate, seed, mech) for mech in ("MCAR", "MAR", "MNAR")}
                for strategy in STRATEGIES:
                    tr = fitted_transformer(Xt, cats, strategy)
                    Xtrain = tr.fit_transform(Xt)
                    m = RandomForestClassifier(n_estimators=30, random_state=seed, n_jobs=-1).fit(Xtrain, yt)
                    clean_s = score(m, tr.transform(Xv), yv, gv)
                    for mech, (Xm, cells, realized) in masked_by_mechanism.items():
                        masked = score(m, tr.transform(Xm), yv, gv)
                        ad = clean_s["accuracy"] - masked["accuracy"]
                        rows.append({
                            "dataset": dname, "model": "RF", "seed": seed,
                            "masking_rate": rate, "mechanism": mech,
                            "imputation_strategy": strategy,
                            "realized_mask_fraction": realized, "masked_cells": cells,
                            "accuracy_drop": ad,
                            "recall_drop_g0": clean_s["recall_g0"] - masked["recall_g0"],
                            "fmg_g0": (clean_s["recall_g0"] - masked["recall_g0"]) - ad,
                            "recall_drop_g1": clean_s["recall_g1"] - masked["recall_g1"],
                            "fmg_g1": (clean_s["recall_g1"] - masked["recall_g1"]) - ad,
                            "g0_positive_denominator": clean_s["g0_positive_denominator"],
                            "g1_positive_denominator": clean_s["g1_positive_denominator"],
                        })
        pd.DataFrame(rows).to_csv(out / "typeaware_mechanisms_per_seed.csv", index=False, float_format="%.8f")
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "typeaware_mechanisms_per_seed.csv", index=False, float_format="%.8f")
    summary = raw.groupby(["dataset", "mechanism", "masking_rate", "imputation_strategy"], as_index=False).agg(
        n_seeds=("fmg_g0", "size"), mean_fmg_g0=("fmg_g0", "mean"), sd_fmg_g0=("fmg_g0", "std"),
        positive_fmg_g0_seeds=("fmg_g0", lambda x: int((x > 0).sum())),
        mean_accuracy_drop=("accuracy_drop", "mean"), mean_recall_drop_g0=("recall_drop_g0", "mean"),
        mean_fmg_g1=("fmg_g1", "mean"), positive_fmg_g1_seeds=("fmg_g1", lambda x: int((x > 0).sum())),
    )
    summary.to_csv(out / "typeaware_mechanisms_summary.csv", index=False, float_format="%.8f")
    (out / "typeaware_mechanisms_manifest.txt").write_text(
        "Rates: 5%, 10%, 20%. Mechanisms: MCAR; constructed MAR stress scenario; constructed MNAR stress scenario.\n"
        "Pilot classifier: 30-tree Random Forest, five fixed seeds. Strategy: numeric median + categorical mode.\n"
        "MAR/MNAR are simulations, not evidence about naturally occurring missingness.\n", encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
