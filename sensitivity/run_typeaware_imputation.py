"""Multi-seed, mixed-type imputation sensitivity analysis for the FMG audit.

This is a supplemental analysis. It does not overwrite the primary result files.
Run with the isolated review runtime described in the accompanying memo.
"""

from __future__ import annotations

import os
import platform
import sys
import tempfile
import urllib.request
import zipfile
import ctypes
import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEEDS = [42, 123, 456, 1, 7, 13, 21, 37, 55, 99]
RATE = 0.20
STRATEGIES = ("median_mode", "knn_numeric_mode", "iterative_numeric_mode")


def get_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "FMG-reproducibility-analysis/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def load_datasets(adult_zip=None, adult_only=False):
    adult_cols = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
    ]
    if adult_zip is None:
        adult = pd.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
            header=None, names=adult_cols, na_values=["?", " ?"], skipinitialspace=True,
        ).dropna().reset_index(drop=True)
    else:
        with zipfile.ZipFile(adult_zip) as archive:
            member = next(n for n in archive.namelist() if n.lower().endswith("adult.data"))
            adult = pd.read_csv(archive.open(member), header=None, names=adult_cols,
                                na_values=["?", " ?"], skipinitialspace=True).dropna().reset_index(drop=True)
    adult["label"] = (adult["income"].str.strip().str.replace(".", "", regex=False) == ">50K").astype(int)
    adult["sensitive"] = (adult["sex"].str.strip() == "Male").astype(int)
    adult = adult.drop(columns=["income", "sex"])
    adult_features = [c for c in adult.columns if c not in ("label", "sensitive")]
    if adult_only:
        return {"Adult Income": (adult, adult_features, "sensitive", "label", None)}

    compas = pd.read_csv(
        "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    )
    compas = compas[
        compas["days_b_screening_arrest"].between(-30, 30)
        & (compas["is_recid"] != -1)
        & (compas["c_charge_degree"] != "O")
        & (compas["score_text"] != "N/A")
    ].copy()
    compas = compas[[
        "age", "c_charge_degree", "race", "sex", "priors_count",
        "days_b_screening_arrest", "juv_fel_count", "juv_misd_count",
        "juv_other_count", "c_days_from_compas", "is_recid",
    ]].dropna().copy()
    compas["label"] = compas["is_recid"].astype(int)
    compas["sensitive"] = (~(compas["race"] == "African-American")).astype(int)
    compas = compas[compas["race"].isin(["African-American", "Caucasian"])].reset_index(drop=True)
    compas_features = [c for c in compas.columns if c not in ("label", "sensitive", "is_recid", "race")]

    taiwan_url = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
    archive = zipfile.ZipFile(BytesIO(get_bytes(taiwan_url)))
    xls_name = next(n for n in archive.namelist() if n.lower().endswith(".xls"))
    # UCI's workbook has a metadata row followed by the field-name row.
    taiwan = pd.read_excel(BytesIO(archive.read(xls_name)), header=1)
    taiwan.columns = taiwan.columns.astype(str).str.strip().str.lower().str.replace(" ", "_")
    taiwan = taiwan.drop(columns=[c for c in taiwan.columns if c == "id" or c.startswith("unnamed:")])
    target_col = next(c for c in taiwan.columns if "default" in c or "payment" in c)
    taiwan["label"] = taiwan[target_col].astype(int)
    if target_col != "label":
        taiwan = taiwan.drop(columns=[target_col])
    sex_col = next(c for c in taiwan.columns if "sex" in c)
    taiwan["sensitive"] = (taiwan[sex_col] == 1).astype(int)
    taiwan = taiwan.drop(columns=[sex_col]).dropna().reset_index(drop=True)
    taiwan_features = [c for c in taiwan.columns if c not in ("label", "sensitive")]
    taiwan_categorical = [
        c for c in taiwan_features
        if c in {"education", "marriage", "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"}
    ]

    datasets = {
        "Adult Income": (adult, adult_features, "sensitive", "label", None),
        "COMPAS": (compas, compas_features, "sensitive", "label", ["c_charge_degree", "sex"]),
        "Taiwan Credit": (taiwan, taiwan_features, "sensitive", "label", taiwan_categorical),
    }
    return datasets


def fitted_transformer(X_train: pd.DataFrame, categorical_cols: list[str], strategy: str):
    categorical_cols = list(categorical_cols)
    numeric_cols = [c for c in X_train.columns if c not in categorical_cols]
    if strategy == "median_mode":
        numeric_imputer = SimpleImputer(strategy="median")
    elif strategy == "knn_numeric_mode":
        numeric_imputer = KNNImputer(n_neighbors=5, weights="uniform")
    elif strategy == "iterative_numeric_mode":
        numeric_imputer = IterativeImputer(max_iter=10, random_state=42, sample_posterior=False)
    else:
        raise ValueError(strategy)

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_imputer, numeric_cols))
    if categorical_cols:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", categorical_pipe, categorical_cols))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def masked_copy(X: pd.DataFrame, rate: float, seed: int) -> tuple[pd.DataFrame, int]:
    rng = np.random.RandomState(seed)
    mask = rng.rand(*X.shape) < rate
    out = X.copy()
    # Assign through each pandas column explicitly. A mixed-type DataFrame's
    # .values property can be a temporary homogenized array, so mutating it is
    # not guaranteed to update the underlying frame.
    for column_index, column in enumerate(X.columns):
        out.loc[mask[:, column_index], column] = np.nan
    return out, int(mask.sum())


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
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--adult-zip", type=Path, help="Corrected local UCI Adult archive.")
    parser.add_argument("--adult-only", action="store_true", help="Run only Adult; skip remote COMPAS/Taiwan data.")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_dir.expanduser().resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    adult_zip = args.adult_zip.expanduser().resolve() if args.adult_zip else None
    datasets = load_datasets(adult_zip=adult_zip, adult_only=args.adult_only)
    results = []
    for dataset_name, (df, features, sensitive_col, target_col, declared_categorical) in datasets.items():
        print(f"\nBeginning {dataset_name}: rows={len(df)} features={len(features)}", flush=True)
        X = df[features].copy()
        y = df[target_col].to_numpy(dtype=int)
        group = df[sensitive_col].to_numpy(dtype=int)
        if declared_categorical is None:
            categorical_cols = list(X.select_dtypes(include=["object", "category"]).columns)
        else:
            categorical_cols = list(declared_categorical)
        for seed in SEEDS:
            X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
                X, y, group, test_size=0.30, random_state=seed, stratify=y
            )
            X_test_masked, masked_cells = masked_copy(X_test, RATE, seed)
            for strategy in STRATEGIES:
                transformer = fitted_transformer(X_train, categorical_cols, strategy)
                X_train_t = transformer.fit_transform(X_train)
                X_test_clean_t = transformer.transform(X_test)
                X_test_masked_t = transformer.transform(X_test_masked)
                model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
                model.fit(X_train_t, y_train)
                clean = score(model, X_test_clean_t, y_test, group_test)
                masked = score(model, X_test_masked_t, y_test, group_test)
                accuracy_drop = clean["accuracy"] - masked["accuracy"]
                mg_g0 = (clean["recall_g0"] - masked["recall_g0"]) - accuracy_drop
                mg_g1 = (clean["recall_g1"] - masked["recall_g1"]) - accuracy_drop
                results.append({
                    "dataset": dataset_name,
                    "model": "RF",
                    "imputation_strategy": strategy,
                    "seed": seed,
                    "masking_rate": RATE,
                    "n_rows": len(df),
                    "n_features": len(features),
                    "n_categorical_features": len(categorical_cols),
                    "masked_cells": masked_cells,
                    "eligible_cells": int(X_test.shape[0] * X_test.shape[1]),
                    "clean_accuracy": clean["accuracy"],
                    "masked_accuracy": masked["accuracy"],
                    "accuracy_drop": accuracy_drop,
                    "clean_recall_g0": clean["recall_g0"],
                    "masked_recall_g0": masked["recall_g0"],
                    "recall_drop_g0": clean["recall_g0"] - masked["recall_g0"],
                    "fmg_g0": mg_g0,
                    "g0_positive_denominator": clean["g0_positive_denominator"],
                    "clean_recall_g1": clean["recall_g1"],
                    "masked_recall_g1": masked["recall_g1"],
                    "recall_drop_g1": clean["recall_g1"] - masked["recall_g1"],
                    "fmg_g1": mg_g1,
                    "g1_positive_denominator": clean["g1_positive_denominator"],
                })
                print(f"{dataset_name} seed={seed} strategy={strategy} MG0={mg_g0:.4f}", flush=True)
        # Keep completed datasets recoverable if a later dataset fails.
        pd.DataFrame(results).to_csv(
            OUT / "typeaware_imputation_sensitivity_per_seed.csv",
            index=False, float_format="%.8f",
        )

    raw = pd.DataFrame(results)
    raw_path = OUT / "typeaware_imputation_sensitivity_per_seed.csv"
    raw.to_csv(raw_path, index=False, float_format="%.8f")
    summary = raw.groupby(["dataset", "imputation_strategy"], as_index=False).agg(
        n_seeds=("fmg_g0", "size"),
        mean_fmg_g0=("fmg_g0", "mean"),
        sd_fmg_g0=("fmg_g0", "std"),
        positive_fmg_g0_seeds=("fmg_g0", lambda x: int((x > 0).sum())),
        mean_accuracy_drop=("accuracy_drop", "mean"),
        mean_recall_drop_g0=("recall_drop_g0", "mean"),
        mean_fmg_g1=("fmg_g1", "mean"),
        positive_fmg_g1_seeds=("fmg_g1", lambda x: int((x > 0).sum())),
        min_g0_positive_denominator=("g0_positive_denominator", "min"),
        max_g0_positive_denominator=("g0_positive_denominator", "max"),
    )
    summary_path = OUT / "typeaware_imputation_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.8f")

    memory_status_type = type("MEMORYSTATUSEX", (ctypes.Structure,), {"_fields_": [
        ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]})
    memory_status = memory_status_type()
    memory_status.dwLength = ctypes.sizeof(memory_status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status))
    manifest = [
        "Supplemental sensitivity analysis: mixed-type, one-hot categorical representation.",
        f"Python: {platform.python_version()}",
        f"NumPy: {np.__version__}",
        f"pandas: {pd.__version__}",
        f"scikit-learn: {__import__('sklearn').__version__}",
        f"CPU: {platform.processor() or 'not reported by platform API'}",
        f"Logical CPUs: {os.cpu_count()}",
        f"System RAM: {memory_status.ullTotalPhys / 1024**3:.2f} GiB",
        f"Split seeds: {', '.join(map(str, SEEDS))}",
        "Mask: 20% independent MCAR over raw test feature cells; same mask per dataset/seed across imputation strategies.",
        "Classifier: RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1), fit only on clean training data.",
        "Strategies: numeric median + categorical most-frequent; numeric 5-NN + categorical most-frequent; numeric iterative imputation (10 iterations, seed 42) + categorical most-frequent.",
        "Categorical columns are one-hot encoded after train-fitted imputation; categorical cells are masked before encoding; unknown categories are ignored by the encoder.",
        "This sensitivity run is not the primary experiment and is descriptive; it does not revise the primary results.",
        f"Raw per-seed output: {raw_path.name}",
        f"Summary output: {summary_path.name}",
    ]
    (OUT / "typeaware_imputation_sensitivity_environment.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print("\nSUMMARY\n", summary.to_string(index=False), flush=True)
    print(f"\nWrote {raw_path} and {summary_path}", flush=True)


if __name__ == "__main__":
    main()
