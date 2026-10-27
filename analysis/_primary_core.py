"""Shared, notebook-faithful helpers for new FMG analyses.

This module intentionally mirrors the preprocessing, split, model and test-only
masking protocol in FMG_research_final_REVIEWER_FIXED.ipynb. It does not alter
the archived results or the source notebook.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

SEEDS = (42, 123, 456, 1, 7, 13, 21, 37, 55, 99)
RATES = (0.10, 0.20)
MODELS = ("RF", "XGB", "LGB", "MLP")


@dataclass
class Dataset:
    name: str
    frame: pd.DataFrame
    features: list[str]
    source: Path
    exploratory: bool = False


def load_adult(path: Path) -> Dataset:
    columns = ["age", "workclass", "fnlwgt", "education", "education-num",
               "marital-status", "occupation", "relationship", "race", "sex",
               "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [n for n in zf.namelist() if n.lower().endswith("adult.data")]
            if not members:
                raise ValueError(f"{path} does not contain adult.data")
            with zf.open(members[0]) as src:
                frame = pd.read_csv(src, header=None, names=columns,
                                    na_values=["?", " ?"], skipinitialspace=True)
    else:
        frame = pd.read_csv(path, header=None, names=columns,
                            na_values=["?", " ?"], skipinitialspace=True)
    frame = frame.dropna().reset_index(drop=True)
    frame["label"] = (frame["income"].astype(str).str.strip().str.replace(".", "", regex=False) == ">50K").astype(int)
    frame["sensitive"] = (frame["sex"].astype(str).str.strip() == "Male").astype(int)
    frame = frame.drop(columns=["income", "sex"])
    feats = [c for c in frame.columns if c not in ("label", "sensitive")]
    return Dataset("Adult Income", frame, feats, path)


def load_compas(path: Path) -> Dataset:
    frame = pd.read_csv(path)
    frame = frame[(frame["days_b_screening_arrest"] >= -30) &
                  (frame["days_b_screening_arrest"] <= 30) &
                  (frame["is_recid"] != -1) & (frame["c_charge_degree"] != "O") &
                  (frame["score_text"] != "N/A")].copy()
    keep = ["age", "c_charge_degree", "race", "sex", "priors_count",
            "days_b_screening_arrest", "juv_fel_count", "juv_misd_count",
            "juv_other_count", "c_days_from_compas", "is_recid"]
    frame = frame[keep].dropna().copy()
    frame["label"] = frame["is_recid"].astype(int)
    frame["sensitive"] = (~(frame["race"] == "African-American")).astype(int)
    frame = frame[frame["race"].isin(["African-American", "Caucasian"])].copy().reset_index(drop=True)
    feats = [c for c in frame.columns if c not in ("label", "sensitive", "is_recid", "race")]
    return Dataset("COMPAS", frame, feats, path)


def load_taiwan(path: Path) -> Dataset:
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip().str.lower().str.replace(" ", "_")
    if "id" in frame:
        frame = frame.drop(columns="id")
    targets = [c for c in frame if "default" in c or "payment" in c]
    if not targets:
        raise ValueError("Taiwan CSV has no default/payment target column")
    target = targets[-1]
    frame["label"] = frame[target].astype(int)
    if target != "label":
        frame = frame.drop(columns=target)
    sex = next((c for c in frame if "sex" in c), None)
    if sex is None:
        raise ValueError("Taiwan CSV has no sex column")
    frame["sensitive"] = (frame[sex] == 1).astype(int)
    frame = frame.drop(columns=sex).dropna().reset_index(drop=True)
    feats = [c for c in frame if c not in ("label", "sensitive")]
    return Dataset("Taiwan Credit", frame, feats, path)


def load_german_openml() -> Dataset:
    """Load the notebook's OpenML data_id=31 (credit-g), marked exploratory."""
    from sklearn.datasets import fetch_openml
    frame = fetch_openml(data_id=31, as_frame=True, parser="auto").frame.copy()
    frame.columns = frame.columns.str.lower().str.replace("-", "_")
    if "personal_status" not in frame:
        raise ValueError("OpenML credit-g data_id=31 lacks personal_status")
    status = frame["personal_status"]
    female = status.isin(["A92", "A95", "female div/dep/mar",
                          "female : divorced/separated/married", "female single"])
    frame["sensitive"] = (~female).astype(int)
    frame["label"] = (frame["class"] == "good").astype(int)
    frame = frame.drop(columns=["class", "personal_status"]).dropna().reset_index(drop=True)
    feats = [c for c in frame if c not in ("label", "sensitive")]
    return Dataset("German Credit", frame, feats, Path("OpenML:data_id=31"), True)


def preprocess(data: Dataset, seed: int):
    x = data.frame[data.features].copy()
    y = data.frame["label"].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=seed, stratify=y
    )
    categorical = x_train.select_dtypes(include=["object", "category"]).columns
    for col in categorical:
        train_values, test_values = x_train[col].astype(str), x_test[col].astype(str)
        enc = LabelEncoder().fit(train_values)
        mapping = {value: code for code, value in enumerate(enc.classes_)}
        x_train[col] = train_values.map(mapping).astype(float)
        x_test[col] = test_values.map(mapping).fillna(-1).astype(float)
    scaler = MinMaxScaler()
    x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=data.features, index=x_train.index)
    x_test = pd.DataFrame(scaler.transform(x_test), columns=data.features, index=x_test.index)
    group_test = data.frame.loc[x_test.index, "sensitive"].to_numpy()
    return x_train, x_test, np.asarray(y_train), np.asarray(y_test), group_test


def build_model(name: str, seed: int, balanced: bool = False):
    if name == "RF":
        return RandomForestClassifier(n_estimators=100, random_state=seed,
                                      n_jobs=-1, class_weight="balanced" if balanced else None)
    if name == "XGB":
        options: dict[str, Any] = {}
        if balanced:
            options["scale_pos_weight"] = 1.0  # caller replaces with train prevalence ratio
        return XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=seed,
                             verbosity=0, use_label_encoder=False, **options)
    if name == "LGB":
        return LGBMClassifier(n_estimators=100, random_state=seed, verbosity=-1,
                              is_unbalance=balanced)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=500,
                             early_stopping=True, random_state=seed)
    if name == "LR":
        return LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=seed,
                                  class_weight="balanced" if balanced else None)
    raise ValueError(f"Unknown model {name}")


def fit_model(name: str, seed: int, x_train, y_train, balanced: bool = False):
    model = build_model(name, seed, balanced)
    if name == "XGB" and balanced:
        positives = max(1, int(np.sum(y_train == 1)))
        negatives = int(np.sum(y_train == 0))
        model.set_params(scale_pos_weight=negatives / positives)
    model.fit(x_train, y_train)
    return model


def mcar_mask(x_test: pd.DataFrame, rate: float, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    mask = rng.rand(*x_test.shape) < rate
    masked = x_test.copy()
    masked[mask] = np.nan
    return masked


def group_mar_mask(x_test: pd.DataFrame, rate: float, group: np.ndarray, seed: int):
    group = np.asarray(group)
    if set(np.unique(group)) != {0, 1}:
        raise ValueError("Both g0 and g1 must be present for group-conditional MAR")
    q0 = float(np.mean(group == 0))
    q1 = 1.0 - q0
    delta = 0.5 * rate
    p0 = rate + delta * q1
    p1 = rate - delta * q0
    row_p = np.where(group == 0, p0, p1)[:, None]
    rng = np.random.RandomState(int(seed))
    mask = rng.rand(*x_test.shape) < row_p
    masked = x_test.copy()
    masked[mask] = np.nan
    return masked


def impute_train_median(x_train, x_test):
    imp = SimpleImputer(strategy="median")
    imp.fit(x_train)
    return imp.transform(x_test)

