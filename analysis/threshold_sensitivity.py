"""Run train-only operating-point sensitivities without test-label tuning.

Primary data use the notebook's exact preprocessing, 70:30 split, seed, MCAR and
group-conditional MAR logic. Optional BRFSS/UCI data use the existing extension
script's loader, split, transformed-space imputation, and MCAR/MAR/MNAR masks.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score

from _primary_core import (
    MODELS, RATES, SEEDS, build_model, fit_model, group_mar_mask,
    impute_train_median, load_adult, load_compas, load_taiwan,
    load_german_openml, mcar_mask, preprocess,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PRIMARY_MANIFEST = ROOT / "run_manifest_20261025.json"
EXTENSION_SCRIPT = ROOT / "experiments" / "public_health" / "run_public_health_fmg_audit.py"
EXTENSION_METRICS = ROOT / "experiments" / "public_health" / "outputs" / "metrics_per_mask.csv"
MECHANISMS = ("MCAR", "MAR_group_conditional")
EXTENSION_MECHANISMS = ("MCAR", "MAR_group_conditional", "MNAR_value_conditional")


def check_versions(manifest_path: Path, required: tuple[str, ...]) -> None:
    import importlib.metadata as metadata
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = (manifest.get("environment", {}).get("packages") or
                manifest.get("runtime", {}).get("packages") or
                manifest.get("software", {}).get("packages"))
    if not expected:
        raise ValueError(f"No package pins in {manifest_path}")
    mismatch = []
    for name in required:
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = "not-installed"
        if actual != expected[name]:
            mismatch.append(f"{name}: required {expected[name]}, installed {actual}")
    if mismatch:
        raise RuntimeError("Environment does not match the pinned run manifest; "
                           "no sensitivity numbers were generated. " + "; ".join(mismatch))


def calibrate_threshold(train_probabilities: np.ndarray, train_labels: np.ndarray) -> tuple[float, float]:
    """Choose a scalar cutpoint using only clean-train scores and prevalence.

    Returns the cutpoint and the achieved in-sample training positive rate.
    Ties can prevent the scalar cutpoint from exactly matching prevalence;
    that achieved rate is preserved in output rather than hidden.
    """
    scores = np.asarray(train_probabilities, dtype=float)
    labels = np.asarray(train_labels, dtype=int)
    if scores.ndim != 1 or len(scores) != len(labels) or not len(scores):
        raise ValueError("Training probabilities and labels must be aligned nonempty vectors")
    k = int(round(float(labels.mean()) * len(labels)))
    k = min(max(k, 1), len(scores) - 1)
    ordered = np.sort(scores)[::-1]
    high, low = ordered[k - 1], ordered[k]
    threshold = float((high + low) / 2.0) if high > low else float(high)
    achieved = float(np.mean(scores >= threshold))
    return threshold, achieved


def prediction_metrics(y: np.ndarray, group: np.ndarray, clean: np.ndarray,
                       masked: np.ndarray) -> dict[str, float]:
    y, group, clean, masked = map(np.asarray, (y, group, clean, masked))
    if set(np.unique(group)) != {0, 1}:
        raise ValueError("Both focus g0 and comparison g1 groups are required")
    acc_loss = float(accuracy_score(y, clean) - accuracy_score(y, masked))
    g0 = group == 0
    g1 = group == 1
    loss0 = float(recall_score(y[g0], clean[g0], zero_division=0) -
                  recall_score(y[g0], masked[g0], zero_division=0))
    loss1 = float(recall_score(y[g1], clean[g1], zero_division=0) -
                  recall_score(y[g1], masked[g1], zero_division=0))
    return {"accuracy_loss": acc_loss, "focus_recall_loss": loss0,
            "comparison_recall_loss": loss1, "clean_focus_recall": float(
                recall_score(y[g0], clean[g0], zero_division=0)),
            "FMG_g0": loss0 - acc_loss, "FMG_g1": loss1 - acc_loss,
            "DiD": loss0 - loss1}


def balanced_primary_model(name: str, seed: int, x_train, y_train):
    model = build_model(name, seed, balanced=(name != "MLP"))
    if name == "XGB":
        model.set_params(scale_pos_weight=float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1))))
    model.fit(x_train, y_train)
    return model


def run_primary(data, model_name: str, seed: int) -> list[dict[str, object]]:
    x_train, x_test, y_train, y_test, group = preprocess(data, seed)
    records = []
    for method in ("class_weight_balanced", "train_prevalence_threshold"):
        is_balanced = method == "class_weight_balanced"
        # MLP has no supported class_weight/sample_weight in the pinned notebook
        # implementation. Its balanced alternative is the sample-weight-free
        # threshold proxy calibrated on clean training scores.
        model = (balanced_primary_model(model_name, seed, x_train, y_train)
                 if is_balanced else fit_model(model_name, seed, x_train, y_train))
        if is_balanced and model_name == "MLP":
            method_name = "class_weight_balanced_mlp_threshold_proxy"
        else:
            method_name = method
        if method == "train_prevalence_threshold" or (is_balanced and model_name == "MLP"):
            train_prob = model.predict_proba(x_train)[:, 1]
            threshold, achieved = calibrate_threshold(train_prob, y_train)
        else:
            threshold, achieved = 0.5, float(np.mean(model.predict(x_train) == 1))
        clean_pred = (model.predict_proba(x_test)[:, 1] >= threshold).astype(int)
        for mechanism in MECHANISMS:
            for rate in RATES:
                x_mask = (mcar_mask(x_test, rate, seed) if mechanism == "MCAR"
                          else group_mar_mask(x_test, rate, group, seed))
                x_imp = impute_train_median(x_train, x_mask)
                masked_pred = (model.predict_proba(x_imp)[:, 1] >= threshold).astype(int)
                records.append({"dataset": data.name, "model": model_name, "seed": seed,
                                "mechanism": mechanism, "rate": rate,
                                "operating_point": method_name, "threshold": threshold,
                                "train_prevalence": float(np.mean(y_train)),
                                "train_predicted_positive_rate": achieved,
                                "exploratory": data.exploratory,
                                **prediction_metrics(y_test, group, clean_pred, masked_pred)})
    return records


def load_extension_module():
    spec = importlib.util.spec_from_file_location("fmg_public_health_audit", EXTENSION_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load extension code at {EXTENSION_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def set_extension_balance(model, name: str, y_train: np.ndarray) -> None:
    if name in {"LR", "RF"}:
        model.set_params(class_weight="balanced")
    elif name == "XGB":
        model.set_params(scale_pos_weight=float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1))))
    elif name == "LGB":
        model.set_params(is_unbalance=True)


def run_extension_dataset(audit, data: pd.DataFrame, dataset_name: str,
                          seeds: list[int]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for split_fraction in (0.20, 0.30):
        split = "80_20" if split_fraction == 0.20 else "70_30_sensitivity"
        for seed in seeds:
            train_idx, test_idx = audit.make_split(
                data, seed, split_fraction, dataset_name == "UCI_Diabetes130"
            )
            train, test = data.iloc[train_idx], data.iloc[test_idx]
            y_train, y_test = train.target.to_numpy(int), test.target.to_numpy(int)
            group = test.group.to_numpy(int)
            weights = test.weight.to_numpy(float) if "weight" in test else np.ones(len(test))
            prep = audit.make_preprocessor(data, dataset_name)
            x_train = np.asarray(prep.fit_transform(train), dtype=np.float32)
            x_test = np.asarray(prep.transform(test), dtype=np.float32)
            imp = audit.SimpleImputer(strategy="median")
            x_train = imp.fit_transform(x_train).astype(np.float32, copy=False)
            x_test = x_test.astype(np.float32, copy=False)
            for model_index, model_name in enumerate(audit.MODELS):
                for method in ("class_weight_balanced", "train_prevalence_threshold"):
                    model = audit.build_model(model_name, seed)
                    if method == "class_weight_balanced" and model_name != "MLP":
                        set_extension_balance(model, model_name, y_train)
                    model.fit(x_train, y_train)
                    if method == "train_prevalence_threshold" or model_name == "MLP":
                        threshold, achieved = calibrate_threshold(model.predict_proba(x_train)[:, 1], y_train)
                    else:
                        threshold, achieved = 0.5, float(np.mean(model.predict(x_train) == 1))
                    clean_prob = model.predict_proba(x_test)[:, 1]
                    clean_pred = (clean_prob >= threshold).astype(int)
                    for mechanism_index, mechanism in enumerate(EXTENSION_MECHANISMS):
                        for rate in audit.RATES:
                            rep = 0
                            mask_seed = int(np.random.SeedSequence([
                                seed, rep, int(rate * 1000), model_index, mechanism_index
                            ]).generate_state(1)[0])
                            x_mask, _ = audit.mask_matrix(x_test, x_train, group, mechanism, rate, mask_seed)
                            x_mask = imp.transform(x_mask).astype(np.float32, copy=False)
                            masked_pred = (model.predict_proba(x_mask)[:, 1] >= threshold).astype(int)
                            # Public-health data code their focus group as 1;
                            # map it to g0 for a consistent output schema.
                            score = prediction_metrics(y_test, 1 - group, clean_pred, masked_pred)
                            focus = group == 1
                            comp = group == 0
                            clean_focus_recall = float(audit.recall_score(
                                y_test[focus], clean_pred[focus], sample_weight=weights[focus], zero_division=0))
                            masked_focus_recall = float(audit.recall_score(
                                y_test[focus], masked_pred[focus], sample_weight=weights[focus], zero_division=0))
                            clean_comp = float(audit.recall_score(
                                y_test[comp], clean_pred[comp], sample_weight=weights[comp], zero_division=0))
                            masked_comp = float(audit.recall_score(
                                y_test[comp], masked_pred[comp], sample_weight=weights[comp], zero_division=0))
                            # Weighted extension evaluation; replace unweighted losses from the helper.
                            acc0 = float(audit.accuracy_score(y_test, clean_pred, sample_weight=weights))
                            acc1 = float(audit.accuracy_score(y_test, masked_pred, sample_weight=weights))
                            g0loss, g1loss, accloss = clean_focus_recall-masked_focus_recall, clean_comp-masked_comp, acc0-acc1
                            # Use weighted metrics for the BRFSS extension; the
                            # same weights are all ones for UCI.
                            records.append({"dataset": dataset_name, "split": split, "seed": seed,
                                            "model": model_name, "mechanism": mechanism,
                                            "rate": rate, "operating_point": method,
                                            "threshold": threshold, "train_prevalence": float(y_train.mean()),
                                            "train_predicted_positive_rate": achieved,
                                            "clean_focus_recall": clean_focus_recall,
                                            "accuracy_loss": accloss,
                                            "focus_recall_loss": g0loss,
                                            "comparison_recall_loss": g1loss,
                                            "FMG_g0": g0loss-accloss, "FMG_g1": g1loss-accloss,
                                            "DiD": g0loss-g1loss,
                                            "UCI_floor_effect": bool(dataset_name == "UCI_Diabetes130" and clean_focus_recall < 0.05)})
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adult", help="Adult adult.data or adult.zip")
    parser.add_argument("--compas", help="COMPAS CSV")
    parser.add_argument("--taiwan", help="Taiwan Credit CSV")
    parser.add_argument("--german-openml", action="store_true")
    parser.add_argument("--brfss", help="Official CDC 2024 XPT or ZIP")
    parser.add_argument("--diabetes-zip", help="UCI Diabetes 130-US Hospitals ZIP/CSV")
    parser.add_argument("--primary-only", action="store_true", help="Permit partial primary input set")
    parser.add_argument("--extension-only", action="store_true", help="Skip primary data and run supplied public-health data")
    parser.add_argument("--output", default=str(HERE / "outputs" / "threshold_sensitivity_summary.csv"))
    args = parser.parse_args()
    start = time.perf_counter()
    records: list[dict[str, object]] = []
    skipped: list[str] = []
    if not args.extension_only:
        check_versions(PRIMARY_MANIFEST, ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "lightgbm"))
        data_items = []
        for name, path, loader in (("Adult Income", args.adult, load_adult),
                                   ("COMPAS", args.compas, load_compas),
                                   ("Taiwan Credit", args.taiwan, load_taiwan)):
            if path and Path(path).is_file():
                data_items.append(loader(Path(path)))
            else:
                skipped.append(f"{name}: input not supplied or missing")
        if args.german_openml:
            data_items.append(load_german_openml())
        else:
            skipped.append("German Credit: exploratory OpenML input not selected")
        if skipped and not args.primary_only and not (args.brfss or args.diabetes_zip):
            # extension-only may be combined with partial primary data; primary-only names this opt-in.
            if not (args.brfss or args.diabetes_zip):
                raise SystemExit("Primary raw data missing; supply all primary paths or pass --primary-only to run available data. " + "; ".join(skipped))
        for data in data_items:
            for model in MODELS:
                for seed in SEEDS:
                    records.extend(run_primary(data, model, seed))
    if args.brfss or args.diabetes_zip:
        audit = load_extension_module()
        extension_manifest_path = ROOT / "experiments" / "public_health" / "outputs" / "run_manifest.json"
        check_versions(extension_manifest_path, tuple(json.loads(extension_manifest_path.read_text(encoding="utf-8"))["software"]["packages"].keys()))
        saved = pd.read_csv(EXTENSION_METRICS)
        for name, path, loader in (("BRFSS2024", args.brfss, audit.load_brfss),
                                   ("UCI_Diabetes130", args.diabetes_zip, audit.load_uci_diabetes)):
            if not path:
                skipped.append(f"{name}: raw input path not supplied")
                continue
            source = Path(path)
            if not source.is_file():
                skipped.append(f"{name}: input not found at {source}")
                continue
            data = loader(source)
            seeds = sorted(saved.loc[saved.dataset == name, "seed"].astype(int).unique().tolist())
            if not seeds:
                raise ValueError(f"No recorded extension seed for {name}; refusing to invent one")
            records.extend(run_extension_dataset(audit, data, name, seeds))
    if not records:
        raise SystemExit("No dataset ran. Skipped: " + "; ".join(skipped))
    detail = pd.DataFrame(records)
    group_cols = [c for c in ("dataset", "model", "mechanism", "rate", "operating_point", "split") if c in detail]
    summary = detail.groupby(group_cols, dropna=False).agg(
        n_conditions=("FMG_g0", "size"), clean_focus_recall_mean=("clean_focus_recall", "mean"),
        FMG_g0_mean=("FMG_g0", "mean"), FMG_g1_mean=("FMG_g1", "mean"),
        DiD_mean=("DiD", "mean"), positive_splits=("FMG_g0", lambda s: int((s > 0).sum())),
        threshold_mean=("threshold", "mean"), train_prevalence_mean=("train_prevalence", "mean"),
        achieved_train_positive_rate_mean=("train_predicted_positive_rate", "mean"),
    ).reset_index()
    if "UCI_floor_effect" in detail:
        floor = detail.groupby(group_cols, dropna=False).UCI_floor_effect.any().rename("UCI_floor_effect").reset_index()
        summary = summary.merge(floor, on=group_cols, how="left", validate="one_to_one")
    summary["new_not_in_manuscript"] = True
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False, float_format="%.12g")
    detail_path = path.with_name(path.stem.replace("summary", "per_seed") + ".csv")
    detail.to_csv(detail_path, index=False, float_format="%.12g")
    print(f"Wrote NEW threshold summary: {path} ({len(summary)} rows)")
    print(f"Wrote NEW threshold per-seed: {detail_path} ({len(detail)} rows)")
    print("Skipped: " + ("; ".join(skipped) if skipped else "None"))
    print(f"Elapsed seconds: {time.perf_counter() - start:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
