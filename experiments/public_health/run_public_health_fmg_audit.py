"""Frozen-classifier feature-missingness audit on two public-health datasets.

This is a supplemental extension. It does not overwrite the archived FMG
benchmark outputs. All masking is test-only and all classifiers are frozen
after training on the clean training split.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import threading
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pyreadstat
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


RATES = (0.05, 0.10, 0.20)
MECHANISMS = ("MCAR", "MAR_group_conditional", "MNAR_value_conditional")
MODELS = ("LR", "RF", "XGB", "LGB", "MLP")
SEEDS_DEFAULT = (42, 123, 456)
THRESHOLDS = (0.005, 0.01, 0.02, 0.05, 0.10)

BRFSS_FEATURES = (
    "GENHLTH", "PHYSHLTH", "MENTHLTH", "PRIMINS2", "PERSDOC3",
    "MEDCOST1", "CHECKUP1", "EXERANY2", "CVDINFR4", "ASTHMA3",
    "CHCKDNY2", "HAVARTH4", "EDUCA", "INCOME3", "EMPLOY1",
    "_AGEG5YR", "_BMI5CAT",
)
BRFSS_CATEGORICAL = tuple(c for c in BRFSS_FEATURES if c not in {"PHYSHLTH", "MENTHLTH", "_AGEG5YR"})
BRFSS_NUMERIC = tuple(c for c in BRFSS_FEATURES if c not in BRFSS_CATEGORICAL)
BRFSS_REQUIRED = (*BRFSS_FEATURES, "DIABETE4", "_RACEGR3", "SEXVAR", "_LLCPWT", "_STSTR", "_PSU")

UCI_CATEGORICAL = (
    "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "medical_specialty", "payer_code", "diabetesMed",
    "max_glu_serum", "A1Cresult", "metformin", "repaglinide", "nateglinide",
    "chlorpropamide", "glimepiride", "acetohexamide", "glipizide",
    "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone",
    "acarbose", "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone", "change",
)
UCI_NUMERIC = (
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient",
    "number_diagnoses",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def archive_member_sha256(path: Path, suffix: str) -> tuple[str, str]:
    """Return the name and SHA-256 of a source file inside a provider ZIP."""
    with zipfile.ZipFile(path) as zf:
        matches = sorted(name for name in zf.namelist() if name.lower().endswith(suffix.lower()))
        if not matches:
            raise ValueError(f"Archive {path.name} contains no {suffix} member")
        member = matches[0]
        digest = hashlib.sha256()
        with zf.open(member) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return member, digest.hexdigest()


class PeakRSS:
    """Sample process RSS while the experiment runs (portable to Windows)."""
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.peak = psutil.Process().memory_info().rss
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        proc = psutil.Process()
        while not self.stop_event.wait(self.interval):
            try:
                self.peak = max(self.peak, proc.memory_info().rss)
            except psutil.Error:
                pass

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()
        self.peak = max(self.peak, psutil.Process().memory_info().rss)


def load_brfss(path: Path) -> pd.DataFrame:
    """Read only documented core variables from CDC's 2024 XPT or ZIP."""
    xpt = path
    temp_dir = None
    if path.suffix.lower() == ".zip":
        temp_dir = path.parent / "_brfss_extract"
        temp_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as zf:
            candidates = [n for n in zf.namelist() if n.lower().endswith(".xpt")]
            if not candidates:
                raise ValueError("CDC ZIP contains no XPT file")
            member = candidates[0]
            # Always refresh the extracted source. Reusing a file by basename
            # can silently read stale data after a new archive is uploaded.
            xpt = Path(zf.extract(member, temp_dir))
    try:
        frame, _meta = pyreadstat.read_xport(str(xpt), usecols=list(BRFSS_REQUIRED))
    except Exception as exc:
        raise RuntimeError(
            "Could not read CDC XPT. Check that this is the official 2024 combined "
            "landline/cell file and that pyreadstat is installed."
        ) from exc

    # CDC codes are explicitly mapped per variable; do not blanket-recode 7/9,
    # since those are valid values for some BRFSS items.
    for col in ("GENHLTH", "PERSDOC3", "MEDCOST1", "EXERANY2", "CVDINFR4",
                "ASTHMA3", "CHCKDNY2", "HAVARTH4"):
        frame[col] = frame[col].replace({7: np.nan, 9: np.nan})
    for col in ("PHYSHLTH", "MENTHLTH"):
        frame[col] = frame[col].replace({88: 0, 77: np.nan, 99: np.nan})
    frame["PRIMINS2"] = frame["PRIMINS2"].replace({77: np.nan, 99: np.nan})
    frame["CHECKUP1"] = frame["CHECKUP1"].replace({7: np.nan, 9: np.nan})
    frame["EDUCA"] = frame["EDUCA"].replace({9: np.nan})
    frame["INCOME3"] = frame["INCOME3"].replace({77: np.nan, 99: np.nan})
    frame["EMPLOY1"] = frame["EMPLOY1"].replace({9: np.nan})
    frame["_AGEG5YR"] = frame["_AGEG5YR"].where(frame["_AGEG5YR"].between(1, 14))
    frame["_BMI5CAT"] = frame["_BMI5CAT"].replace({9: np.nan})

    # For the prespecified race contrast, retain only non-Hispanic White and
    # non-Hispanic Black respondents. Race/group variables never enter X.
    frame = frame.loc[frame["_RACEGR3"].isin([1, 2])].copy()
    frame["group"] = (frame["_RACEGR3"] == 2).astype(int)  # focus: Black; comparison: White
    frame["intersection_group"] = np.where(frame["SEXVAR"].eq(2), frame["group"], np.nan)
    # DIABETE4: 1=diabetes, 2=pregnancy-only diabetes, 3=no, 4=prediabetes,
    # 7=don't know, 9=refused. Only 1 and 3 define the binary outcome.
    frame["target"] = np.select(
        [frame["DIABETE4"].eq(1), frame["DIABETE4"].eq(3)], [1, 0], default=np.nan
    )
    frame = frame.loc[frame["target"].notna() & frame["_LLCPWT"].gt(0)].copy()
    frame["target"] = frame["target"].astype(int)
    frame["weight"] = frame["_LLCPWT"].astype(float)
    frame["cluster"] = frame["_PSU"].astype(str)
    frame["stratum"] = frame["_STSTR"].astype(str)
    result = frame[[*BRFSS_FEATURES, "target", "group", "intersection_group", "weight", "cluster", "stratum"]].copy()
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.reset_index(drop=True)


def load_uci_diabetes(path: Path) -> pd.DataFrame:
    """Read UCI 130-US Hospitals CSV or ZIP; patient_nbr is split metadata only."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            candidates = [n for n in zf.namelist() if Path(n).name.lower() == "diabetic_data.csv"]
            if not candidates:
                raise ValueError("UCI ZIP contains no diabetic_data.csv")
            with zf.open(candidates[0]) as f:
                frame = pd.read_csv(f, na_values=["?"], low_memory=False)
    else:
        frame = pd.read_csv(path, na_values=["?"])
    required = {"patient_nbr", "readmitted", "race", "gender", *UCI_CATEGORICAL, *UCI_NUMERIC}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"UCI file is missing expected columns: {missing}")
    frame = frame.loc[frame["race"].notna() & frame["patient_nbr"].notna()].copy()
    frame["group"] = (frame["race"].astype(str).str.casefold() == "africanamerican").astype(int)
    frame["intersection_group"] = np.where(
        frame["gender"].astype(str).str.casefold().eq("female"), frame["group"], np.nan
    )
    frame["target"] = (frame["readmitted"].astype(str) == "<30").astype(int)
    frame["cluster"] = frame["patient_nbr"].astype(str)
    frame["stratum"] = "all"
    feature_cols = [c for c in (*UCI_CATEGORICAL, *UCI_NUMERIC) if c in frame]
    # Exclude patient_nbr, encounter_id, race (audit label), target and any
    # outcome-derived values. The encounter ID is never a predictor.
    result = frame[[*feature_cols, "target", "group", "intersection_group", "cluster", "stratum"]].copy()
    result = result.replace([np.inf, -np.inf], np.nan)
    return result.reset_index(drop=True)


def make_split(data: pd.DataFrame, seed: int, fraction: float, patient_disjoint: bool):
    y = data["target"].to_numpy(dtype=int)
    idx = np.arange(len(data))
    if not patient_disjoint:
        train_idx, test_idx = train_test_split(
            idx, test_size=fraction, random_state=seed, stratify=y
        )
        return np.asarray(train_idx), np.asarray(test_idx)
    # Ten stratified group folds; combine two or three folds for approximate
    # 80:20 and 70:30 holdouts while preventing patient overlap.
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    folds = [test for _, test in splitter.split(idx, y, groups=data["cluster"].to_numpy())]
    n_test_folds = 2 if fraction == 0.20 else 3
    test_idx = np.sort(np.concatenate(folds[:n_test_folds]))
    keep = np.ones(len(data), dtype=bool)
    keep[test_idx] = False
    return idx[keep], test_idx


def build_model(name: str, seed: int):
    if name == "LR":
        return LogisticRegression(max_iter=600, solver="lbfgs", random_state=seed)
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=160, min_samples_leaf=2, max_features="sqrt",
            n_jobs=-1, random_state=seed,
        )
    if name == "XGB":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=160, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            tree_method="hist", n_jobs=-1, random_state=seed,
            eval_metric="logloss",
        )
    if name == "LGB":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=160, learning_rate=0.05, num_leaves=31,
            n_jobs=-1, random_state=seed, verbosity=-1,
        )
    if name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=140, early_stopping=True,
            random_state=seed,
        )
    raise ValueError(name)


def make_preprocessor(data: pd.DataFrame, dataset: str):
    if dataset == "BRFSS2024":
        numeric = list(BRFSS_NUMERIC)
        categorical = list(BRFSS_CATEGORICAL)
    else:
        categorical = [c for c in UCI_CATEGORICAL if c in data.columns]
        numeric = [c for c in UCI_NUMERIC if c in data.columns]
    num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
    ])
    return ColumnTransformer(
        [("num", num, numeric), ("cat", cat, categorical)],
        remainder="drop", sparse_threshold=0.0,
    )


def mask_matrix(x_test: np.ndarray, x_train: np.ndarray, group: np.ndarray,
                mechanism: str, rate: float, seed: int) -> tuple[np.ndarray, int]:
    """Mask exactly round(rate*cells) transformed test cells without replacement."""
    rng = np.random.default_rng(seed)
    n, p = x_test.shape
    k = int(round(rate * n * p))
    if k == 0:
        return x_test.copy(), 0
    weights = np.ones(n * p, dtype=float)
    if mechanism == "MAR_group_conditional":
        # Probability depends on observed audit group only, with 2:1 odds in
        # the focus group. Group codes are metadata, excluded from model input.
        cell_weights = np.where(group[:, None] == 1, 2.0, 1.0) * np.ones((1, p), dtype=float)
        weights = cell_weights.ravel()
    elif mechanism == "MNAR_value_conditional":
        # Constructed value-dependent stressor: more extreme observed feature
        # values are more likely to be selected; not an observed field process.
        train_sd = np.nanstd(x_train, axis=0)
        train_sd[~np.isfinite(train_sd) | (train_sd == 0)] = 1.0
        train_center = np.nanmedian(x_train, axis=0)
        z = np.abs((x_test - train_center) / train_sd)
        weights = np.exp(np.clip(0.65 * np.nan_to_num(z, nan=0.0, posinf=4), 0, 5)).ravel()
    elif mechanism != "MCAR":
        raise ValueError(mechanism)
    probs = weights / weights.sum()
    selected = rng.choice(n * p, size=min(k, n * p), replace=False, p=probs)
    masked = x_test.copy()
    masked.reshape(-1)[selected] = np.nan
    return masked, len(selected)


def metric_bundle(y: np.ndarray, pred: np.ndarray, prob: np.ndarray,
                  group: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    focus = group == 1
    comp = group == 0
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y, pred, sample_weight=weights)),
        "focus_recall": float(recall_score(y[focus], pred[focus], sample_weight=weights[focus], zero_division=0)),
        "comparison_recall": float(recall_score(y[comp], pred[comp], sample_weight=weights[comp], zero_division=0)),
        "focus_precision": float(precision_score(y[focus], pred[focus], sample_weight=weights[focus], zero_division=0)),
        "focus_f1": float(f1_score(y[focus], pred[focus], sample_weight=weights[focus], zero_division=0)),
        "focus_balanced_accuracy": float(balanced_accuracy_score(y[focus], pred[focus], sample_weight=weights[focus])),
        "focus_auroc": float(roc_auc_score(y[focus], prob[focus], sample_weight=weights[focus])) if len(np.unique(y[focus])) == 2 else float("nan"),
        "focus_fpr": float(np.average(pred[focus][y[focus] == 0], weights=weights[focus][y[focus] == 0])) if np.any(focus & (y == 0)) else float("nan"),
        "comparison_fpr": float(np.average(pred[comp][y[comp] == 0], weights=weights[comp][y[comp] == 0])) if np.any(comp & (y == 0)) else float("nan"),
        "n_focus": int(focus.sum()),
        "n_focus_positive": int(np.sum(focus & (y == 1))),
        "n_comparison": int(comp.sum()),
        "n_comparison_positive": int(np.sum(comp & (y == 1))),
        "weighted_focus_positive_denominator": float(weights[focus & (y == 1)].sum()),
        "weighted_comparison_positive_denominator": float(weights[comp & (y == 1)].sum()),
    }
    out["recall_gap_focus_minus_comparison"] = out["focus_recall"] - out["comparison_recall"]
    out["fpr_gap_focus_minus_comparison"] = out["focus_fpr"] - out["comparison_fpr"]
    return out


def fmg_from_predictions(y, pred_clean, pred_stress, group, weights):
    acc_clean = accuracy_score(y, pred_clean, sample_weight=weights)
    acc_stress = accuracy_score(y, pred_stress, sample_weight=weights)
    focus = group == 1
    rec_clean = recall_score(y[focus], pred_clean[focus], sample_weight=weights[focus], zero_division=0)
    rec_stress = recall_score(y[focus], pred_stress[focus], sample_weight=weights[focus], zero_division=0)
    recall_loss = rec_clean - rec_stress
    accuracy_loss = acc_clean - acc_stress
    return float(recall_loss - accuracy_loss), float(recall_loss), float(accuracy_loss)


def cluster_bootstrap_ci(y, pred_clean, pred_stress, group, weights, cluster,
                         stratum, n_boot: int, seed: int) -> tuple[float, float]:
    """Paired unit bootstrap; BRFSS resamples PSUs within strata."""
    rng = np.random.default_rng(seed)
    cluster = np.asarray(cluster).astype(str)
    stratum = np.asarray(stratum).astype(str)
    if np.all(stratum == "all"):
        strata_values = ["all"]
    else:
        strata_values = np.unique(stratum)
    groups_by_stratum = {
        s: np.unique(cluster[stratum == s]) for s in strata_values
    }
    values = []
    for _ in range(n_boot):
        multiplicity: Counter[tuple[str, str]] = Counter()
        for s, psus in groups_by_stratum.items():
            picked = rng.choice(psus, size=len(psus), replace=True)
            multiplicity.update((s, str(x)) for x in picked)
        row_mult = np.fromiter(
            (multiplicity[(s, c)] for s, c in zip(stratum, cluster)),
            dtype=float, count=len(cluster),
        )
        bw = weights * row_mult
        if bw.sum() == 0 or bw[(group == 1) & (y == 1)].sum() == 0:
            continue
        fmg, _, _ = fmg_from_predictions(y, pred_clean, pred_stress, group, bw)
        values.append(fmg)
    if not values:
        return float("nan"), float("nan")
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(lo), float(hi)


def _score(model, x, y, group, weights):
    pred = model.predict(x)
    prob = model.predict_proba(x)[:, 1]
    return pred, prob, metric_bundle(y, pred, prob, group, weights)


def _append_checkpoint_rows(out_dir: Path, filename: str, rows: list[dict]):
    """Append a completed model's rows so an interrupted long run can resume."""
    if not rows:
        return
    path = out_dir / filename
    pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def _write_checkpoint(path: Path, payload: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def run(args):
    brfss_path = Path(args.brfss).resolve() if args.brfss else None
    uci_path = Path(args.diabetes_zip).resolve() if args.diabetes_zip else None
    datasets = []
    if brfss_path:
        datasets.append(("BRFSS2024", load_brfss(brfss_path), brfss_path, False))
    if uci_path:
        datasets.append(("UCI_Diabetes130", load_uci_diabetes(uci_path), uci_path, True))
    if not datasets:
        raise SystemExit("Provide --brfss and/or --diabetes-zip")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.json"
    code_sha256 = sha256(Path(__file__).resolve())
    signature_payload = {
        "datasets": [(name, sha256(source_path)) for name, _, source_path, _ in datasets],
        "audit_code_sha256": code_sha256,
        "models": list(MODELS), "seeds": list(args.seeds), "rates": list(RATES),
        "mechanisms": list(MECHANISMS), "mask_repeats": args.mask_repeats,
        "bootstrap_reps": args.bootstrap_reps, "run_shap": args.run_shap,
        "shap_sample_size": args.shap_sample_size,
        "split_fractions": [0.20, 0.30],
    }
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
    checkpoint = None
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("signature") != signature:
            raise ValueError("Existing checkpoint was created with a different dataset or protocol; use a new output directory.")
        if checkpoint.get("complete"):
            raise ValueError("This output directory already contains a completed run; choose a new output directory.")
    elif any((out_dir / name).exists() for name in (
        "metrics_per_mask.csv", "denominators_per_split.csv", "unit_bootstrap_ci_20mcar.csv",
        "intersectional_metrics_per_mask.csv", "intersectional_unit_bootstrap_ci_20mcar.csv",
        "exploratory_shap_shift.csv", "run_manifest.json",
    )):
        raise ValueError("Output files already exist without a compatible checkpoint; choose a new output directory.")

    def _read_rows(filename):
        path = out_dir / filename
        return pd.read_csv(path).to_dict(orient="records") if path.exists() else []

    completed_units = set(checkpoint.get("completed_units", [])) if checkpoint else set()
    metric_rows = _read_rows("metrics_per_mask.csv")
    denominator_rows = _read_rows("denominators_per_split.csv")
    bootstrap_rows = _read_rows("unit_bootstrap_ci_20mcar.csv")
    intersection_rows = _read_rows("intersectional_metrics_per_mask.csv")
    intersection_bootstrap_rows = _read_rows("intersectional_unit_bootstrap_ci_20mcar.csv")
    shap_rows = _read_rows("exploratory_shap_shift.csv")
    if checkpoint:
        def _only_completed(rows, filename):
            if any("model" not in row for row in rows):
                raise ValueError(
                    f"Cannot safely resume because {filename} lacks the model key. "
                    "Use a new output directory; existing files were preserved."
                )
            return [row for row in rows if f"{row['dataset']}|{row['split']}|{int(row['seed'])}|{row['model']}" in completed_units]
        metric_rows = _only_completed(metric_rows, "metrics_per_mask.csv")
        denominator_rows = _only_completed(denominator_rows, "denominators_per_split.csv")
        bootstrap_rows = _only_completed(bootstrap_rows, "unit_bootstrap_ci_20mcar.csv")
        intersection_rows = _only_completed(intersection_rows, "intersectional_metrics_per_mask.csv")
        intersection_bootstrap_rows = _only_completed(intersection_bootstrap_rows, "intersectional_unit_bootstrap_ci_20mcar.csv")
        shap_rows = _only_completed(shap_rows, "exploratory_shap_shift.csv")
        for filename, rows in (
            ("metrics_per_mask.csv", metric_rows),
            ("denominators_per_split.csv", denominator_rows),
            ("unit_bootstrap_ci_20mcar.csv", bootstrap_rows),
            ("intersectional_metrics_per_mask.csv", intersection_rows),
            ("intersectional_unit_bootstrap_ci_20mcar.csv", intersection_bootstrap_rows),
            ("exploratory_shap_shift.csv", shap_rows),
        ):
            path = out_dir / filename
            if rows or path.exists():
                pd.DataFrame(rows).to_csv(path, index=False)
    accumulated_runtime = float(checkpoint.get("runtime_seconds", 0.0)) if checkpoint else 0.0
    prior_peak_rss = int(checkpoint.get("peak_process_rss_bytes_sampled", 0)) if checkpoint else 0
    start = time.perf_counter()
    process = psutil.Process()
    with PeakRSS() as peak:
        for dataset_name, data, source_path, patient_disjoint in datasets:
            print(f"[{dataset_name}] {len(data):,} eligible records", flush=True)
            for split_fraction in (0.20, 0.30):
                split_label = "80_20" if split_fraction == 0.20 else "70_30_sensitivity"
                for seed in args.seeds:
                    unit_keys = [f"{dataset_name}|{split_label}|{seed}|{model_name}" for model_name in MODELS]
                    if all(key in completed_units for key in unit_keys):
                        continue
                    train_idx, test_idx = make_split(data, seed, split_fraction, patient_disjoint)
                    train = data.iloc[train_idx]
                    test = data.iloc[test_idx]
                    y_train = train["target"].to_numpy(dtype=int)
                    y_test = test["target"].to_numpy(dtype=int)
                    g_test = test["group"].to_numpy(dtype=int)
                    for group_value, group_name in ((1, "focus"), (0, "comparison")):
                        for outcome in (0, 1):
                            count = int(np.sum((g_test == group_value) & (y_test == outcome)))
                            if count < 30:
                                raise ValueError(
                                    f"{dataset_name}/{split_label}/seed={seed}: {group_name} "
                                    f"group has only {count} test cases for outcome={outcome}; "
                                    "the prespecified minimum is 30 per group-by-outcome cell."
                                )
                    w_test = test["weight"].to_numpy(dtype=float) if "weight" in test else np.ones(len(test), dtype=float)
                    gi_all = test["intersection_group"].to_numpy(dtype=float)
                    intersection_scope = np.isfinite(gi_all)
                    gi = gi_all[intersection_scope].astype(int)
                    yi = y_test[intersection_scope]
                    wi = w_test[intersection_scope]
                    ti = test.iloc[np.flatnonzero(intersection_scope)]
                    intersection_counts = {
                        "n_intersection_focus": int(np.sum(gi == 1)),
                        "n_intersection_focus_positive": int(np.sum((gi == 1) & (yi == 1))),
                        "n_intersection_focus_negative": int(np.sum((gi == 1) & (yi == 0))),
                        "n_intersection_comparison": int(np.sum(gi == 0)),
                        "n_intersection_comparison_positive": int(np.sum((gi == 0) & (yi == 1))),
                        "n_intersection_comparison_negative": int(np.sum((gi == 0) & (yi == 0))),
                    }
                    intersection_minimum = min(intersection_counts[k] for k in (
                        "n_intersection_focus_positive", "n_intersection_focus_negative",
                        "n_intersection_comparison_positive", "n_intersection_comparison_negative",
                    ))
                    intersection_estimable = intersection_minimum >= 30
                    intersection_definition = (
                        "Black women vs White women" if dataset_name == "BRFSS2024"
                        else "African-American women vs other reported-race women"
                    )
                    intersection_meta = intersection_counts | {
                        "intersection_estimable_min30_per_group_outcome": bool(intersection_estimable),
                        "intersection_minimum_group_outcome_n": intersection_minimum,
                        "intersection_definition": intersection_definition,
                    }
                    prep = make_preprocessor(data, dataset_name)
                    x_train = np.asarray(prep.fit_transform(train), dtype=np.float32)
                    x_test = np.asarray(prep.transform(test), dtype=np.float32)
                    train_imputer = SimpleImputer(strategy="median")
                    x_train = train_imputer.fit_transform(x_train).astype(np.float32, copy=False)
                    x_test = x_test.astype(np.float32, copy=False)
                    y_mask = test["target"].to_numpy(dtype=int)
                    print(f"  split={split_label} seed={seed} train={len(train):,} test={len(test):,} encoded_features={x_train.shape[1]}", flush=True)
                    for model_name in MODELS:
                        unit_key = f"{dataset_name}|{split_label}|{seed}|{model_name}"
                        if unit_key in completed_units:
                            print(f"    {model_name} already checkpointed; skipping", flush=True)
                            continue
                        metric_start = len(metric_rows)
                        denominator_start = len(denominator_rows)
                        bootstrap_start = len(bootstrap_rows)
                        intersection_start = len(intersection_rows)
                        intersection_bootstrap_start = len(intersection_bootstrap_rows)
                        shap_start = len(shap_rows)
                        fit_start = time.perf_counter()
                        model = build_model(model_name, seed)
                        model.fit(x_train, y_train)
                        model_fit_seconds = time.perf_counter() - fit_start
                        pred_clean, prob_clean, clean_m = _score(model, x_test, y_test, g_test, w_test)
                        denominator_rows.append({
                            "dataset": dataset_name, "split": split_label, "seed": seed,
                            "model": model_name,
                            "n_train": len(train), "n_test": len(test),
                            "n_focus": int((g_test == 1).sum()),
                            "n_focus_positive": int(((g_test == 1) & (y_test == 1)).sum()),
                            "n_focus_negative": int(((g_test == 1) & (y_test == 0)).sum()),
                            "n_comparison": int((g_test == 0).sum()),
                            "n_comparison_positive": int(((g_test == 0) & (y_test == 1)).sum()),
                            "patient_overlap": int(len(set(train["cluster"]) & set(test["cluster"]))) if patient_disjoint else 0,
                            "focus_weighted_positive_denominator": float(w_test[(g_test == 1) & (y_test == 1)].sum()),
                            "comparison_weighted_positive_denominator": float(w_test[(g_test == 0) & (y_test == 1)].sum()),
                        } | intersection_meta)
                        for mechanism in MECHANISMS:
                            for rate in RATES:
                                for rep in range(args.mask_repeats):
                                    mask_seed = int(np.random.SeedSequence([seed, rep, int(rate * 1000), MODELS.index(model_name), MECHANISMS.index(mechanism)]).generate_state(1)[0])
                                    x_mask, n_masked = mask_matrix(x_test, x_train, g_test, mechanism, rate, mask_seed)
                                    x_mask = train_imputer.transform(x_mask).astype(np.float32, copy=False)
                                    pred_mask, prob_mask, masked_m = _score(model, x_mask, y_test, g_test, w_test)
                                    fmg, recall_loss, accuracy_loss = fmg_from_predictions(y_test, pred_clean, pred_mask, g_test, w_test)
                                    row = {
                                        "dataset": dataset_name, "split": split_label, "seed": seed,
                                        "model": model_name, "mechanism": mechanism,
                                        "nominal_mask_rate": rate, "mask_repeat": rep,
                                        "masked_cell_fraction": n_masked / (len(test) * x_test.shape[1]),
                                        "clean_accuracy": clean_m["accuracy"], "masked_accuracy": masked_m["accuracy"],
                                        "clean_focus_recall": clean_m["focus_recall"], "masked_focus_recall": masked_m["focus_recall"],
                                        "clean_comparison_recall": clean_m["comparison_recall"], "masked_comparison_recall": masked_m["comparison_recall"],
                                        "focus_recall_loss": recall_loss, "aggregate_accuracy_loss": accuracy_loss,
                                        "FMG": fmg,
                                        "clean_focus_precision": clean_m["focus_precision"], "masked_focus_precision": masked_m["focus_precision"],
                                        "clean_focus_f1": clean_m["focus_f1"], "masked_focus_f1": masked_m["focus_f1"],
                                        "clean_focus_balanced_accuracy": clean_m["focus_balanced_accuracy"],
                                        "masked_focus_balanced_accuracy": masked_m["focus_balanced_accuracy"],
                                        "clean_focus_auroc": clean_m["focus_auroc"], "masked_focus_auroc": masked_m["focus_auroc"],
                                        "clean_recall_gap": clean_m["recall_gap_focus_minus_comparison"],
                                        "masked_recall_gap": masked_m["recall_gap_focus_minus_comparison"],
                                        "clean_fpr_gap": clean_m["fpr_gap_focus_minus_comparison"],
                                        "masked_fpr_gap": masked_m["fpr_gap_focus_minus_comparison"],
                                        "n_focus_positive": clean_m["n_focus_positive"],
                                        "n_comparison_positive": clean_m["n_comparison_positive"],
                                        "model_fit_seconds": model_fit_seconds,
                                    }
                                    metric_rows.append(row)
                                    if intersection_estimable:
                                        pred_intersection = pred_mask[intersection_scope]
                                        prob_intersection = prob_mask[intersection_scope]
                                        clean_intersection = pred_clean[intersection_scope]
                                        int_clean = metric_bundle(yi, clean_intersection, prob_clean[intersection_scope], gi, wi)
                                        int_masked = metric_bundle(yi, pred_intersection, prob_intersection, gi, wi)
                                        int_fmg, int_recall_loss, int_accuracy_loss = fmg_from_predictions(
                                            yi, clean_intersection, pred_intersection, gi, wi
                                        )
                                        intersection_rows.append({
                                            "dataset": dataset_name, "split": split_label, "seed": seed,
                                            "model": model_name, "mechanism": mechanism,
                                            "nominal_mask_rate": rate, "mask_repeat": rep,
                                            "audit_group_definition": intersection_definition,
                                            "FMG": int_fmg, "focus_recall_loss": int_recall_loss,
                                            "aggregate_accuracy_loss_within_intersection": int_accuracy_loss,
                                            "clean_accuracy": int_clean["accuracy"], "masked_accuracy": int_masked["accuracy"],
                                            "clean_focus_recall": int_clean["focus_recall"], "masked_focus_recall": int_masked["focus_recall"],
                                            "clean_focus_precision": int_clean["focus_precision"], "masked_focus_precision": int_masked["focus_precision"],
                                            "clean_focus_f1": int_clean["focus_f1"], "masked_focus_f1": int_masked["focus_f1"],
                                            "clean_focus_balanced_accuracy": int_clean["focus_balanced_accuracy"],
                                            "masked_focus_balanced_accuracy": int_masked["focus_balanced_accuracy"],
                                            "clean_focus_auroc": int_clean["focus_auroc"], "masked_focus_auroc": int_masked["focus_auroc"],
                                            "n_focus_positive": int_clean["n_focus_positive"],
                                            "n_comparison_positive": int_clean["n_comparison_positive"],
                                            "minimum_outcome_cell_n": min(intersection_counts[k] for k in (
                                                "n_intersection_focus_positive", "n_intersection_focus_negative",
                                                "n_intersection_comparison_positive", "n_intersection_comparison_negative",
                                            )),
                                        })
                                    if args.run_shap and split_label == "80_20" and model_name in {"RF", "XGB", "LGB"} and mechanism == "MCAR" and rate == 0.20 and rep == 0:
                                        shap_rows.extend(_shap_shift_rows(
                                            model=model, x_clean=x_test, x_stress=x_mask,
                                            group=g_test, feature_names=prep.get_feature_names_out(),
                                            dataset=dataset_name, seed=seed, model_name=model_name,
                                            sample_size=args.shap_sample_size,
                                        ))
                                    if mechanism == "MCAR" and rate == 0.20 and rep == 0:
                                        low, high = cluster_bootstrap_ci(
                                            y_test, pred_clean, pred_mask, g_test, w_test,
                                            test["cluster"].to_numpy(), test["stratum"].to_numpy(),
                                            args.bootstrap_reps, seed + 100_000 + MODELS.index(model_name),
                                        )
                                        bootstrap_rows.append({
                                            "dataset": dataset_name, "split": split_label,
                                            "seed": seed, "model": model_name,
                                            "condition": "20%_MCAR_rep0",
                                            "FMG": fmg, "unit_bootstrap_95ci_low": low,
                                            "unit_bootstrap_95ci_high": high,
                                            "bootstrap_reps_requested": args.bootstrap_reps,
                                            "resampling_unit": "stratum_PSU" if dataset_name == "BRFSS2024" else "patient_nbr",
                                        })
                                        if intersection_estimable:
                                            int_low, int_high = cluster_bootstrap_ci(
                                                yi, clean_intersection, pred_intersection, gi, wi,
                                                ti["cluster"].to_numpy(), ti["stratum"].to_numpy(),
                                                args.bootstrap_reps, seed + 200_000 + MODELS.index(model_name),
                                            )
                                            intersection_bootstrap_rows.append({
                                                "dataset": dataset_name, "split": split_label, "seed": seed,
                                                "model": model_name, "condition": "20%_MCAR_rep0",
                                                "FMG": int_fmg, "unit_bootstrap_95ci_low": int_low,
                                                "unit_bootstrap_95ci_high": int_high,
                                                "resampling_unit": "stratum_PSU" if dataset_name == "BRFSS2024" else "patient_nbr",
                                                "audit_group_definition": intersection_definition,
                                            })
                        print(f"    {model_name} fit/scoring complete in {time.perf_counter() - fit_start:.1f}s", flush=True)
                        for filename, rows, offset in (
                            ("metrics_per_mask.csv", metric_rows, metric_start),
                            ("denominators_per_split.csv", denominator_rows, denominator_start),
                            ("unit_bootstrap_ci_20mcar.csv", bootstrap_rows, bootstrap_start),
                            ("intersectional_metrics_per_mask.csv", intersection_rows, intersection_start),
                            ("intersectional_unit_bootstrap_ci_20mcar.csv", intersection_bootstrap_rows, intersection_bootstrap_start),
                            ("exploratory_shap_shift.csv", shap_rows, shap_start),
                        ):
                            _append_checkpoint_rows(out_dir, filename, rows[offset:])
                        completed_units.add(unit_key)
                        _write_checkpoint(checkpoint_path, {
                            "signature": signature, "signature_payload": signature_payload,
                            "completed_units": sorted(completed_units), "complete": False,
                            "runtime_seconds": accumulated_runtime + time.perf_counter() - start,
                            "peak_process_rss_bytes_sampled": max(prior_peak_rss, peak.peak),
                            "last_completed_unit": unit_key,
                        })

    metrics = pd.DataFrame(metric_rows)
    denoms = pd.DataFrame(denominator_rows)
    boot = pd.DataFrame(bootstrap_rows)
    metrics.to_csv(out_dir / "metrics_per_mask.csv", index=False)
    denoms.to_csv(out_dir / "denominators_per_split.csv", index=False)
    boot.to_csv(out_dir / "unit_bootstrap_ci_20mcar.csv", index=False)
    pd.DataFrame(intersection_rows).to_csv(out_dir / "intersectional_metrics_per_mask.csv", index=False)
    pd.DataFrame(intersection_bootstrap_rows).to_csv(out_dir / "intersectional_unit_bootstrap_ci_20mcar.csv", index=False)
    if shap_rows:
        pd.DataFrame(shap_rows).to_csv(out_dir / "exploratory_shap_shift.csv", index=False)
    group_cols = ["dataset", "split", "model", "mechanism", "nominal_mask_rate"]
    summary = metrics.groupby(group_cols, dropna=False).agg(
        n_rows=("FMG", "size"), FMG_mean=("FMG", "mean"), FMG_sd=("FMG", "std"),
        FMG_min=("FMG", "min"), FMG_max=("FMG", "max"),
        focus_recall_loss_mean=("focus_recall_loss", "mean"),
        aggregate_accuracy_loss_mean=("aggregate_accuracy_loss", "mean"),
        clean_accuracy_mean=("clean_accuracy", "mean"), masked_accuracy_mean=("masked_accuracy", "mean"),
        clean_focus_recall_mean=("clean_focus_recall", "mean"), masked_focus_recall_mean=("masked_focus_recall", "mean"),
        masked_focus_precision_mean=("masked_focus_precision", "mean"),
        masked_focus_f1_mean=("masked_focus_f1", "mean"),
        masked_focus_balanced_accuracy_mean=("masked_focus_balanced_accuracy", "mean"),
        masked_focus_auroc_mean=("masked_focus_auroc", "mean"),
        masked_recall_gap_mean=("masked_recall_gap", "mean"), masked_fpr_gap_mean=("masked_fpr_gap", "mean"),
        positive_split_count=("FMG", lambda s: int((s > 0).sum())),
    ).reset_index()
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)
    thresholds = []
    for keys, part in metrics.groupby(group_cols, dropna=False):
        for threshold in THRESHOLDS:
            thresholds.append(dict(zip(group_cols, keys)) | {
                "exploratory_threshold": threshold, "n_conditions": len(part),
                "n_fmg_above_threshold": int((part["FMG"] > threshold).sum()),
                "fraction_fmg_above_threshold": float((part["FMG"] > threshold).mean()),
            })
    pd.DataFrame(thresholds).to_csv(out_dir / "threshold_sensitivity.csv", index=False)

    elapsed = accumulated_runtime + time.perf_counter() - start
    manifest = {
        "experiment": "Supplementary public-health FMG benchmark extension",
        "run_time_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "audit_code_sha256": code_sha256,
        "scope_note": "Controlled retrospective benchmark; not deployment or decision validation.",
        "datasets": [],
        "protocol": {
            "models": list(MODELS), "seeds": list(args.seeds), "rates": list(RATES),
            "mechanisms": list(MECHANISMS), "mask_repeats": args.mask_repeats,
            "split_fractions": [0.20, 0.30], "thresholds_exploratory_only": list(THRESHOLDS),
            "frozen_classifier": True, "test_only_masking": True,
            "fit_preprocessing_and_transformed_space_imputer_on_train_only": True,
            "brfss_training_weights": "Unweighted fitting; CDC final weights used in held-out evaluation.",
            "brfss_weighted_uncertainty": "Paired PSU bootstrap within CDC strata, conditional on the split/model.",
            "uci_split": "StratifiedGroupKFold with patient_nbr held out; identifiers excluded from predictors.",
            "seeds_interpretation": "Repeated benchmark splits, not independent populations.",
            "MCAR": "Uniform sampling without replacement of exactly the nominal proportion of transformed test cells.",
            "MAR": "Constructed group-conditional sampling; focus group has 2:1 selection weights; exact overall cell count.",
            "MNAR": "Constructed value-conditional sampling weighted toward larger absolute train-standardized transformed values; exact overall cell count.",
            "repetitions_interpretation": "Mask replicates are repeated stress tests on the same held-out units, not new samples.",
            "metric": "FMG = (focus-group clean recall - masked recall) - (clean overall accuracy - masked overall accuracy).",
            "model_configurations": {
                "LR": "LogisticRegression(max_iter=600, solver=lbfgs)",
                "RF": "RandomForestClassifier(n_estimators=160, min_samples_leaf=2, max_features=sqrt)",
                "XGB": "XGBClassifier(n_estimators=160, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method=hist)",
                "LGB": "LGBMClassifier(n_estimators=160, learning_rate=0.05, num_leaves=31)",
                "MLP": "MLPClassifier(hidden_layer_sizes=(64,32), max_iter=140, early_stopping=True)",
            },
        },
        "dataset_summaries": [],
        "software": {"python": sys.version, "platform": platform.platform(),
                     "packages": {name: _version(name) for name in ["numpy", "pandas", "scikit-learn", "pyreadstat", "xgboost", "lightgbm", "psutil"]}},
        "hardware": {"processor": platform.processor(), "logical_cpu_count": os.cpu_count(),
                     "total_memory_bytes": psutil.virtual_memory().total,
                     "peak_process_rss_bytes_sampled": peak.peak},
        "runtime_seconds": elapsed,
        "outputs": {},
        "inference_caution": "Intervals quantify unit resampling conditional on a fitted model/split. They do not make seeds independent or establish external validity.",
    }
    for name, data, source_path, _ in datasets:
        dataset_record = {"name": name, "input_path_name": source_path.name,
                          "input_sha256": sha256(source_path)}
        if source_path.suffix.lower() == ".zip":
            source_suffix = ".xpt" if name == "BRFSS2024" else "diabetic_data.csv"
            member_name, member_hash = archive_member_sha256(source_path, source_suffix)
            dataset_record["source_member"] = member_name
            dataset_record["source_member_sha256"] = member_hash
        manifest["datasets"].append(dataset_record)
        n = len(data)
        manifest["dataset_summaries"].append({
            "name": name, "eligible_rows": n,
            "model_input_features_before_encoding": len(BRFSS_FEATURES) if name == "BRFSS2024" else sum(c in data.columns for c in (*UCI_CATEGORICAL, *UCI_NUMERIC)),
            "positive_outcome_rate_unweighted": float(data["target"].mean()),
            "focus_group_rows": int((data["group"] == 1).sum()),
            "comparison_group_rows": int((data["group"] == 0).sum()),
            "focus_positive_rows": int(((data["group"] == 1) & (data["target"] == 1)).sum()),
            "focus_negative_rows": int(((data["group"] == 1) & (data["target"] == 0)).sum()),
            "evaluation_weights": "CDC _LLCPWT" if name == "BRFSS2024" else "unit encounter weights",
        })
    for p in sorted(out_dir.iterdir()):
        if p.is_file() and p.name not in {"run_manifest.json", "checkpoint.json"}:
            manifest["outputs"][p.name] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_checkpoint(checkpoint_path, {
        "signature": signature, "signature_payload": signature_payload,
        "completed_units": sorted(completed_units), "complete": True,
        "runtime_seconds": elapsed,
        "peak_process_rss_bytes_sampled": max(prior_peak_rss, peak.peak),
    })
    print(json.dumps({"runtime_seconds": elapsed, "peak_process_rss_bytes": peak.peak,
                      "outputs": str(out_dir), "rows": len(metrics)}, indent=2))


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--brfss", help="Official CDC 2024 combined LLCP XPT file or ZIP")
    p.add_argument("--diabetes-zip", help="Official UCI Diabetes 130-US Hospitals ZIP or CSV")
    p.add_argument("--output-dir", default="experiments/public_health/outputs")
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS_DEFAULT))
    p.add_argument("--mask-repeats", type=int, default=5)
    p.add_argument("--bootstrap-reps", type=int, default=200)
    p.add_argument("--run-shap", action="store_true", help="Optional exploratory TreeSHAP shifts for RF/XGB/LGB at 20% MCAR on the 80/20 splits")
    p.add_argument("--shap-sample-size", type=int, default=50, help="Maximum fixed test units per audit group for each TreeSHAP comparison")
    return p.parse_args()


def _shap_shift_rows(model, x_clean, x_stress, group, feature_names,
                     dataset, seed, model_name, sample_size):
    """Exploratory paired TreeSHAP shifts on a deterministic fixed-size sample."""
    import shap
    explainer = shap.TreeExplainer(model)
    out = []
    rng = np.random.default_rng(seed + 8731)
    names = np.asarray(feature_names, dtype=str)
    for group_value, group_name in ((1, "focus"), (0, "comparison")):
        indices = np.flatnonzero(group == group_value)
        take = min(sample_size, len(indices))
        selected = np.sort(rng.choice(indices, size=take, replace=False))
        vals_clean = explainer.shap_values(x_clean[selected])
        vals_stress = explainer.shap_values(x_stress[selected])
        if isinstance(vals_clean, list):
            vals_clean, vals_stress = vals_clean[-1], vals_stress[-1]
        vals_clean = np.asarray(vals_clean)
        vals_stress = np.asarray(vals_stress)
        if vals_clean.ndim == 3:
            vals_clean, vals_stress = vals_clean[:, :, -1], vals_stress[:, :, -1]
        for j, name in enumerate(names):
            out.append({
                "dataset": dataset, "split": "80_20", "seed": seed,
                "model": model_name, "condition": "20%_MCAR_rep0",
                "audit_group": group_name, "sample_n": take,
                "feature": str(name),
                "mean_signed_shap_shift_stress_minus_clean": float(np.mean(vals_stress[:, j] - vals_clean[:, j])),
                "mean_absolute_shap_difference": float(np.mean(np.abs(vals_stress[:, j] - vals_clean[:, j]))),
            })
    return out


if __name__ == "__main__":
    run(parse_args())
