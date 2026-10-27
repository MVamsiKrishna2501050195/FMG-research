"""Decompose FMG into accuracy, recall, specificity and predicted-positive shifts.

Run with the same pinned environment and raw inputs used by the primary notebook.
This script is additive: it never edits existing results or the notebook.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score

from _primary_core import (
    MODELS, RATES, SEEDS, Dataset, fit_model, group_mar_mask,
    impute_train_median, load_adult, load_compas, load_taiwan,
    load_german_openml, mcar_mask, preprocess,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MANIFEST = ROOT / "run_manifest_20261025.json"
PRIMARY_RESULTS = ROOT / "results_raw_10seeds.csv"
MAR_RESULTS = ROOT / "group_conditional_mar_full_per_seed.csv"
IDENTITY_TOL = 1e-12
ARCHIVE_TOL = 1e-9


def expected_versions() -> dict[str, str]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    section = manifest.get("environment") or manifest.get("runtime")
    return section["packages"]


def installed_versions() -> dict[str, str]:
    import importlib.metadata as metadata
    names = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
             "scikit-learn": "scikit-learn", "xgboost": "xgboost",
             "lightgbm": "lightgbm", "shap": "shap", "matplotlib": "matplotlib",
             "seaborn": "seaborn"}
    return {key: metadata.version(dist) for key, dist in names.items()}


def check_primary_versions() -> None:
    expected = expected_versions()
    actual = installed_versions()
    mismatches = {name: (expected[name], actual.get(name))
                  for name in expected if actual.get(name) != expected[name]}
    if mismatches:
        details = ", ".join(f"{n}: required {want}, installed {got}"
                             for n, (want, got) in mismatches.items())
        raise RuntimeError("Primary decomposition stopped: environment does not match "
                           f"run_manifest_20261025.json ({details}). Use its pinned "
                           "versions; do not compare cross-version values as exact.")


def confusion_rates(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    prevalence = float(y.mean())
    tpr = float(recall_score(y, pred, zero_division=0))
    negatives = y == 0
    tnr = float(np.mean(pred[negatives] == 0)) if negatives.any() else float("nan")
    acc = float(accuracy_score(y, pred))
    return acc, tpr, tnr


def positive_rate_delta(clean: np.ndarray, stressed: np.ndarray) -> float:
    return float(np.mean(clean == 1) - np.mean(stressed == 1))


def decompose_predictions(
    y: np.ndarray, groups: np.ndarray, clean_pred: np.ndarray, masked_pred: np.ndarray,
    clean_accuracy: float | None = None,
) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups, dtype=int)
    clean_pred = np.asarray(clean_pred, dtype=int)
    masked_pred = np.asarray(masked_pred, dtype=int)
    if set(np.unique(groups)) != {0, 1}:
        raise ValueError("Both focus group g0 and comparison group g1 are required")
    acc0 = float(accuracy_score(y, clean_pred)) if clean_accuracy is None else float(clean_accuracy)
    acc1 = float(accuracy_score(y, masked_pred))
    d_acc = acc0 - acc1
    _, tpr0, tnr0 = confusion_rates(y, clean_pred)
    _, tpr1, tnr1 = confusion_rates(y, masked_pred)
    d_tpr_all, d_tnr_all = tpr0 - tpr1, tnr0 - tnr1
    prevalence = float(y.mean())
    rows: dict[str, float] = {
        "dAcc": d_acc,
        "dTPR_all": d_tpr_all,
        "dTNR_all": d_tnr_all,
        "prevalence_pi": prevalence,
        "FMG_all": d_tpr_all - d_acc,
        "dPPR_all": positive_rate_delta(clean_pred, masked_pred),
        "identity_rhs": (1.0 - prevalence) * (d_tpr_all - d_tnr_all),
    }
    group_losses: dict[int, float] = {}
    for group_value, label in ((0, "g0"), (1, "g1")):
        subset = groups == group_value
        if not subset.any():
            raise ValueError(f"No observations in {label}")
        clean_tpr = float(recall_score(y[subset], clean_pred[subset], zero_division=0))
        masked_tpr = float(recall_score(y[subset], masked_pred[subset], zero_division=0))
        group_loss = clean_tpr - masked_tpr
        group_losses[group_value] = group_loss
        rows[f"dTPR_{label}"] = group_loss
        rows[f"FMG_{label}"] = group_loss - d_acc
        rows[f"dPPR_{label}"] = positive_rate_delta(clean_pred[subset], masked_pred[subset])
        if group_value == 0:
            rows["clean_focus_recall"] = clean_tpr
    rows["DiD"] = group_losses[0] - group_losses[1]
    return rows


def compute_row(data: Dataset, model_name: str, seed: int, rate: float,
                mechanism: str) -> dict[str, object]:
    x_train, x_test, y_train, y_test, groups = preprocess(data, seed)
    model = fit_model(model_name, seed, x_train, y_train)
    clean_pred = np.asarray(model.predict(x_test), dtype=int)
    clean_reference = clean_pred.copy()
    if mechanism == "MCAR":
        stressed = mcar_mask(x_test, rate, seed)
    elif mechanism == "MAR_group_conditional":
        stressed = group_mar_mask(x_test, rate, groups, seed)
    else:
        raise ValueError(mechanism)
    masked_pred = np.asarray(model.predict(impute_train_median(x_train, stressed)), dtype=int)
    if not np.array_equal(clean_reference, model.predict(x_test)):
        raise AssertionError("Frozen classifier clean predictions changed during masked evaluation")
    return {"dataset": data.name, "model": model_name, "seed": int(seed),
            "rate": float(rate), "mechanism": mechanism,
            "exploratory": bool(data.exploratory),
            **decompose_predictions(y_test, groups, clean_pred, masked_pred)}


def assert_identity(rows: pd.DataFrame) -> float:
    errors = np.abs(rows["FMG_all"].to_numpy() - rows["identity_rhs"].to_numpy())
    max_error = float(np.nanmax(errors)) if len(errors) else 0.0
    if max_error >= IDENTITY_TOL:
        raise AssertionError(f"FMG identity failed: max absolute error {max_error:.3g} >= {IDENTITY_TOL}")
    return max_error


def assert_archived_match(rows: pd.DataFrame) -> None:
    if not PRIMARY_RESULTS.exists():
        raise FileNotFoundError(f"Required archive is missing: {PRIMARY_RESULTS}")
    archived = pd.read_csv(PRIMARY_RESULTS)
    mc = rows[rows["mechanism"] == "MCAR"].copy()
    ref = archived[(archived["rate"].isin(RATES)) & archived["model"].isin(MODELS)]
    keys = ["dataset", "model", "seed", "rate"]
    merged = mc.merge(ref[keys + ["masking_gap"]], on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(mc):
        missing = len(mc) - len(merged)
        raise AssertionError(f"Archived MCAR matching rows missing: {missing}")
    error = np.abs(merged["FMG_g0"] - merged["masking_gap"])
    if len(error) and float(error.max()) > ARCHIVE_TOL:
        failing = merged.loc[error > ARCHIVE_TOL, keys + ["FMG_g0", "masking_gap"]]
        raise AssertionError("FMG_g0 did not reproduce archived masking_gap within 1e-9 "
                             f"precision (max error {float(error.max()):.3g}).\n{failing.head(10)}")
    if MAR_RESULTS.exists():
        mar = rows[rows["mechanism"] == "MAR_group_conditional"].copy()
        archived_mar = pd.read_csv(MAR_RESULTS)
        for key in ("dataset", "model", "seed", "rate"):
            archive_key = "rate_nominal" if key == "rate" else key
            if archive_key not in archived_mar:
                raise AssertionError(f"MAR archive lacks required key {archive_key}")
        mar_count = len(mar)
        mar = mar.merge(
            archived_mar[["dataset", "model", "seed", "rate_nominal", "mg_mar"]],
            left_on=["dataset", "model", "seed", "rate"],
            right_on=["dataset", "model", "seed", "rate_nominal"],
            how="inner", validate="one_to_one",
        )
        if len(mar) != mar_count:
            raise AssertionError(f"Group-conditional MAR archive missing {mar_count-len(mar)} paired results")
        mar_error = np.abs(mar["FMG_g0"] - mar["mg_mar"])
        if len(mar_error) and float(mar_error.max()) > ARCHIVE_TOL:
            raise AssertionError("Group-conditional MAR FMG_g0 failed the archived "
                                 f"per-seed check; max error {float(mar_error.max()):.3g}")


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "model", "rate", "mechanism", "exploratory"]
    summary = rows.groupby(keys, dropna=False).agg(
        n_splits=("FMG_g0", "size"),
        FMG_g0_mean=("FMG_g0", "mean"), FMG_g0_sd=("FMG_g0", "std"),
        FMG_g0_positive_splits=("FMG_g0", lambda x: int((x > 0).sum())),
        FMG_g1_mean=("FMG_g1", "mean"), FMG_all_mean=("FMG_all", "mean"),
        DiD_mean=("DiD", "mean"), dAcc_mean=("dAcc", "mean"),
        dTPR_all_mean=("dTPR_all", "mean"), dTNR_all_mean=("dTNR_all", "mean"),
        dPPR_all_mean=("dPPR_all", "mean"),
    ).reset_index()
    return summary


def correlations(rows: pd.DataFrame) -> pd.DataFrame:
    result = []
    for (dataset, rate, mechanism), part in rows.groupby(["dataset", "rate", "mechanism"]):
        for x, y, label in (("FMG_all", "dPPR_all", "corr_FMG_all_dPPR"),
                            ("FMG_g0", "dPPR_all", "corr_FMG_g0_dPPR")):
            corr = part[x].corr(part[y]) if len(part) >= 2 else np.nan
            result.append({"dataset": dataset, "rate": rate, "mechanism": mechanism,
                           "correlation": label, "value": corr, "n": len(part)})
        delta_ppr = part["dPPR_g0"] - part["dPPR_g1"]
        corr = part["DiD"].corr(delta_ppr) if len(part) >= 2 else np.nan
        result.append({"dataset": dataset, "rate": rate, "mechanism": mechanism,
                       "correlation": "corr_DiD_dPPR_g0_minus_g1", "value": corr,
                       "n": len(part)})
    return pd.DataFrame(result)


def load_available(args: argparse.Namespace) -> tuple[list[Dataset], list[str]]:
    data: list[Dataset] = []
    skipped: list[str] = []
    for name, path, loader in (("Adult Income", args.adult, load_adult),
                               ("COMPAS", args.compas, load_compas),
                               ("Taiwan Credit", args.taiwan, load_taiwan)):
        if path:
            p = Path(path)
            if not p.is_file():
                skipped.append(f"{name}: input not found: {p}")
                continue
            data.append(loader(p))
        else:
            skipped.append(f"{name}: raw source not supplied")
    if args.german_openml:
        data.append(load_german_openml())
    else:
        skipped.append("German Credit: --german-openml not selected (exploratory)")
    return data, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adult", help="Official Adult adult.data or adult.zip")
    parser.add_argument("--compas", help="ProPublica compas-scores-two-years.csv")
    parser.add_argument("--taiwan", help="Original Taiwan Credit CSV")
    parser.add_argument("--german-openml", action="store_true", help="Load exploratory OpenML credit-g data_id=31")
    parser.add_argument("--output-dir", default=str(HERE / "outputs"))
    parser.add_argument("--allow-partial-data", action="store_true", help="Run supplied primary datasets and log all others as skipped")
    args = parser.parse_args()
    check_primary_versions()
    datasets, skipped = load_available(args)
    if skipped and not args.allow_partial_data:
        raise SystemExit("Missing primary dataset inputs; supply all paths or pass --allow-partial-data.\n" + "\n".join(skipped))
    if not datasets:
        raise SystemExit("No primary dataset could be loaded.\n" + "\n".join(skipped))
    start = time.perf_counter()
    records: list[dict[str, object]] = []
    for data in datasets:
        for model in MODELS:
            for seed in SEEDS:
                for rate in RATES:
                    for mechanism in ("MCAR", "MAR_group_conditional"):
                        records.append(compute_row(data, model, seed, rate, mechanism))
    per_seed = pd.DataFrame.from_records(records)
    max_error = assert_identity(per_seed)
    assert_archived_match(per_seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(out_dir / "fmg_decomposition_per_seed.csv", index=False, float_format="%.12g")
    summary = summarize(per_seed)
    summary.to_csv(out_dir / "fmg_decomposition_summary.csv", index=False, float_format="%.12g")
    corr = correlations(per_seed)
    corr.to_csv(out_dir / "fmg_decomposition_correlations.csv", index=False, float_format="%.12g")
    print(f"Identity maximum absolute error: {max_error:.3g}")
    print("Correlation table:")
    print(corr.to_string(index=False))
    print("Skipped:")
    print("\n".join(skipped) if skipped else "None")
    print(f"Elapsed seconds: {time.perf_counter() - start:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
