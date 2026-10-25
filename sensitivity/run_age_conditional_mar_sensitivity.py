"""Run a separate age-conditioned MAR-vs-MCAR sensitivity analysis.

This is a bounded, exploratory mechanism comparison, not a replacement for
the primary MCAR analysis or the existing group-conditional MAR sensitivity.
It masks only non-age test features, keeping age observed as the conditioning
covariate in both conditions. All preprocessing and median imputation are fit
on the training split; each classifier is trained once and remains frozen.

Example:
  python run_age_conditional_mar_sensitivity.py --taiwan-csv /path/UCI_Credit_Card.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import expit
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


SEEDS = [42, 123, 456, 1, 7, 13, 21, 37, 55, 99]
RATES = [0.10, 0.20]
MODELS = ["RF", "XGB", "LGB", "MLP"]
AGE_COLUMN = "age"
DEFAULT_AGE_ODDS_RATIOS = [2.0]
DATASET_URLS = {
    "Adult Income": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "COMPAS": "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv",
    "Taiwan Credit": "user-supplied UCI_Credit_Card.csv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_frame(df: pd.DataFrame) -> str:
    """Hash the cleaned analysis frame actually supplied to the split logic."""
    payload = df.to_csv(index=False, lineterminator="\n", float_format="%.17g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_adult(archive_path: Path | None = None) -> tuple[pd.DataFrame, list[str], str, str]:
    cols = ["age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
    if archive_path is None:
        df = pd.read_csv(DATASET_URLS["Adult Income"], header=None, names=cols,
                         na_values=["?", " ?"], skipinitialspace=True)
    else:
        with zipfile.ZipFile(archive_path) as archive:
            member = next(n for n in archive.namelist() if n.lower().endswith("adult.data"))
            df = pd.read_csv(archive.open(member), header=None, names=cols,
                             na_values=["?", " ?"], skipinitialspace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["label"] = (df["income"].str.strip().str.replace(".", "", regex=False) == ">50K").astype(int)
    df.drop(columns=["income"], inplace=True)
    df["sensitive"] = (df["sex"].str.strip() == "Male").astype(int)  # female=0 focus group
    df.drop(columns=["sex"], inplace=True)
    feats = [c for c in df.columns if c not in ("label", "sensitive")]
    return df, feats, "sensitive", "label"


def load_compas() -> tuple[pd.DataFrame, list[str], str, str]:
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    df = pd.read_csv(url)
    df = df[(df["days_b_screening_arrest"] >= -30) &
            (df["days_b_screening_arrest"] <= 30) &
            (df["is_recid"] != -1) & (df["c_charge_degree"] != "O") &
            (df["score_text"] != "N/A")].copy()
    df.reset_index(drop=True, inplace=True)
    keep = ["age", "c_charge_degree", "race", "sex", "priors_count",
            "days_b_screening_arrest", "juv_fel_count", "juv_misd_count",
            "juv_other_count", "c_days_from_compas", "is_recid"]
    df = df[keep].dropna()
    df["label"] = df["is_recid"].astype(int)
    df["sensitive"] = (~(df["race"] == "African-American")).astype(int)
    df = df[df["race"].isin(["African-American", "Caucasian"])].copy()
    df.reset_index(drop=True, inplace=True)
    feats = [c for c in df.columns if c not in ("label", "sensitive", "is_recid", "race")]
    return df, feats, "sensitive", "label"


def load_taiwan(path: Path) -> tuple[pd.DataFrame, list[str], str, str]:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)
    target = [c for c in df.columns if "default" in c or "payment" in c][-1]
    df["label"] = df[target].astype(int)
    if target != "label":
        df.drop(columns=[target], inplace=True)
    sex = [c for c in df.columns if "sex" in c][0]
    df["sensitive"] = (df[sex] == 1).astype(int)  # female=0 focus group
    df.drop(columns=[sex], inplace=True)
    feats = [c for c in df.columns if c not in ("label", "sensitive")]
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df, feats, "sensitive", "label"


def preprocess(df: pd.DataFrame, features: list[str], target: str, seed: int):
    x = df[features].copy()
    y = df[target].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=seed, stratify=y
    )
    categorical = [c for c in x_train.columns if not pd.api.types.is_numeric_dtype(x_train[c])]
    for col in categorical:
        train_values = x_train[col].astype(str)
        test_values = x_test[col].astype(str)
        encoder = LabelEncoder().fit(train_values)
        mapping = {value: code for code, value in enumerate(encoder.classes_)}
        x_train[col] = train_values.map(mapping).astype(float).to_numpy()
        x_test[col] = test_values.map(mapping).fillna(-1).astype(float).to_numpy()
    scaler = MinMaxScaler()
    x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=features, index=x_train.index)
    x_test = pd.DataFrame(scaler.transform(x_test), columns=features, index=x_test.index)
    if not x_train.index.is_unique or not x_test.index.is_unique or len(x_train.index.intersection(x_test.index)):
        raise ValueError("Train/test row-index alignment check failed.")
    return x_train, x_test, y_train, y_test


def make_model(name: str, seed: int):
    # Settings follow the primary notebook's four non-exploratory model families.
    if name == "RF":
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    if name == "XGB":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=seed,
                             verbosity=0, use_label_encoder=False)
    if name == "LGB":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=100, random_state=seed, verbosity=-1)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=500,
                             early_stopping=True, random_state=seed)
    raise ValueError(name)


def score(model, x, y, sensitive):
    prediction = model.predict(x)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "recall_g0": float(recall_score(y[sensitive == 0], prediction[sensitive == 0], zero_division=0)),
        "recall_g1": float(recall_score(y[sensitive == 1], prediction[sensitive == 1], zero_division=0)),
    }


def fmg(clean: dict, stressed: dict) -> float:
    return ((clean["recall_g0"] - stressed["recall_g0"]) -
            (clean["accuracy"] - stressed["accuracy"]))


def calibrated_row_probabilities(age: np.ndarray, rate: float, log_odds_per_sd: float) -> np.ndarray:
    """Logistic age propensity; intercept calibrated to expected cell rate."""
    z = (age - np.mean(age)) / (np.std(age) or 1.0)
    beta = log_odds_per_sd
    lo, hi = -30.0, 30.0
    for _ in range(100):
        intercept = (lo + hi) / 2
        if expit(intercept + beta * z).mean() < rate:
            lo = intercept
        else:
            hi = intercept
    return expit(((lo + hi) / 2) + beta * z)


def create_masks(x_test: pd.DataFrame, age_col: str, rate: float, seed: int, log_odds_per_sd: float):
    age_idx = list(x_test.columns).index(age_col)
    eligible = [i for i in range(x_test.shape[1]) if i != age_idx]
    if not eligible:
        raise ValueError("No non-age features available to mask.")
    rng_mcar = np.random.RandomState(seed + 17011)
    rng_mar = np.random.RandomState(seed + 29009)
    mcar = np.zeros(x_test.shape, dtype=bool)
    mar = np.zeros(x_test.shape, dtype=bool)
    mcar[:, eligible] = rng_mcar.rand(len(x_test), len(eligible)) < rate
    row_probs = calibrated_row_probabilities(x_test[age_col].to_numpy(), rate, log_odds_per_sd)
    mar[:, eligible] = rng_mar.rand(len(x_test), len(eligible)) < row_probs[:, None]
    assert not mcar[:, age_idx].any() and not mar[:, age_idx].any()
    return mcar, mar, age_idx


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taiwan-csv", type=Path)
    parser.add_argument("--adult-zip", type=Path,
                        help="Use a corrected local UCI Adult archive instead of downloading Adult.")
    parser.add_argument("--adult-only", action="store_true",
                        help="Run only Adult and avoid fetching COMPAS and Taiwan sources.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "age_conditional_mar_outputs")
    parser.add_argument("--age-odds-ratios", nargs="+", type=float, default=DEFAULT_AGE_ODDS_RATIOS,
                        help="Per-SD age odds ratios for cell masking; default 2.0 reproduces the original run.")
    args = parser.parse_args()
    if any(value <= 1.0 for value in args.age_odds_ratios):
        parser.error("Each age odds ratio must exceed 1.0.")
    if not args.adult_only and args.taiwan_csv is None:
        parser.error("--taiwan-csv is required unless --adult-only is set.")
    if args.taiwan_csv is not None:
        args.taiwan_csv = args.taiwan_csv.expanduser().resolve()
    if args.adult_zip is not None:
        args.adult_zip = args.adult_zip.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {"Adult Income": (*load_adult(args.adult_zip), args.adult_zip)}
    if not args.adult_only:
        datasets["COMPAS"] = (*load_compas(), None)
        datasets["Taiwan Credit"] = (*load_taiwan(args.taiwan_csv), args.taiwan_csv)
    per_seed = []
    started = time.time()
    for ds_name, (df, features, sensitive_col, target_col, source_path) in datasets.items():
        if AGE_COLUMN not in features:
            raise ValueError(f"{ds_name} lacks the required observed conditioning feature {AGE_COLUMN!r}.")
        print(f"\nDataset {ds_name}: rows={len(df)}, features={len(features)}", flush=True)
        for seed in SEEDS:
            x_train, x_test, y_train, y_test = preprocess(df, features, target_col, seed)
            sensitive = df.loc[x_test.index, sensitive_col].to_numpy()
            positive_denominator_g0 = int(np.sum((sensitive == 0) & (y_test == 1)))
            positive_denominator_g1 = int(np.sum((sensitive == 1) & (y_test == 1)))
            imputer = SimpleImputer(strategy="median").fit(x_train)
            x_train_imp = pd.DataFrame(imputer.transform(x_train), columns=features, index=x_train.index)
            x_test_imp = pd.DataFrame(imputer.transform(x_test), columns=features, index=x_test.index)
            for model_name in MODELS:
                model = make_model(model_name, seed)
                model.fit(x_train_imp, y_train)
                clean = score(model, x_test_imp, y_test, sensitive)
                before = hashlib.sha256(__import__("pickle").dumps(model)).hexdigest()
                for odds_ratio in args.age_odds_ratios:
                    log_odds_per_sd = float(np.log(odds_ratio))
                    for rate in RATES:
                        mask_seed = seed + int(rate * 100000)
                        mcar_mask, _, age_idx = create_masks(x_test, AGE_COLUMN, rate, mask_seed, 0.0)
                        _, mar_mask, _ = create_masks(x_test, AGE_COLUMN, rate, mask_seed, log_odds_per_sd)
                        mcar_raw = x_test.copy()
                        mar_raw = x_test.copy()
                        mcar_raw = mcar_raw.mask(mcar_mask)
                        mar_raw = mar_raw.mask(mar_mask)
                        mcar_imp = pd.DataFrame(imputer.transform(mcar_raw), columns=features, index=x_test.index)
                        mar_imp = pd.DataFrame(imputer.transform(mar_raw), columns=features, index=x_test.index)
                        mcar_score = score(model, mcar_imp, y_test, sensitive)
                        mar_score = score(model, mar_imp, y_test, sensitive)
                        after = hashlib.sha256(__import__("pickle").dumps(model)).hexdigest()
                        if before != after:
                            raise AssertionError("Classifier changed during test-time masking evaluation.")
                        eligible_cells = int(mcar_mask.size - len(x_test))
                        per_seed.append({
                            "dataset": ds_name, "model": model_name, "seed": seed,
                            "rate_nominal": rate, "age_conditioning_feature": AGE_COLUMN,
                            "age_odds_ratio_per_sd": odds_ratio,
                            "age_log_odds_per_sd": log_odds_per_sd,
                            "age_kept_observed": True, "masked_feature_cells_eligible": eligible_cells,
                            "realized_mcar_rate": float(mcar_mask.sum() / eligible_cells),
                            "realized_age_mar_rate": float(mar_mask.sum() / eligible_cells),
                            "age_mar_mask_rate_youngest_quartile": float(mar_mask[np.asarray(x_test[AGE_COLUMN]) <= np.quantile(x_test[AGE_COLUMN], .25)].sum() /
                                max(1, mar_mask[np.asarray(x_test[AGE_COLUMN]) <= np.quantile(x_test[AGE_COLUMN], .25)].size - np.sum(np.asarray(x_test[AGE_COLUMN]) <= np.quantile(x_test[AGE_COLUMN], .25)))),
                            "age_mar_mask_rate_oldest_quartile": float(mar_mask[np.asarray(x_test[AGE_COLUMN]) >= np.quantile(x_test[AGE_COLUMN], .75)].sum() /
                                max(1, mar_mask[np.asarray(x_test[AGE_COLUMN]) >= np.quantile(x_test[AGE_COLUMN], .75)].size - np.sum(np.asarray(x_test[AGE_COLUMN]) >= np.quantile(x_test[AGE_COLUMN], .75)))),
                            "test_rows": len(x_test), "test_features": len(features),
                            "g0_positive_denominator": positive_denominator_g0,
                            "g1_positive_denominator": positive_denominator_g1,
                            "clean_accuracy": clean["accuracy"], "clean_recall_g0": clean["recall_g0"],
                            "clean_recall_g1": clean["recall_g1"],
                            "mcar_accuracy_drop": clean["accuracy"] - mcar_score["accuracy"],
                            "mcar_recall_g0_drop": clean["recall_g0"] - mcar_score["recall_g0"],
                            "mcar_fmg": fmg(clean, mcar_score),
                            "age_mar_accuracy_drop": clean["accuracy"] - mar_score["accuracy"],
                            "age_mar_recall_g0_drop": clean["recall_g0"] - mar_score["recall_g0"],
                            "age_mar_fmg": fmg(clean, mar_score),
                            "age_mar_minus_mcar_fmg": fmg(clean, mar_score) - fmg(clean, mcar_score),
                            "classifier_fingerprint_unchanged": True,
                        })
            print(f"  finished seed {seed}", flush=True)

    per_seed_df = pd.DataFrame(per_seed)
    per_seed_path = args.output_dir / "age_conditional_mar_per_seed.csv"
    per_seed_df.to_csv(per_seed_path, index=False)

    summaries = []
    group_columns = ["dataset", "model", "rate_nominal", "age_odds_ratio_per_sd"]
    for (ds_name, model_name, rate, odds_ratio), sub in per_seed_df.groupby(group_columns, sort=False):
        delta = sub["age_mar_minus_mcar_fmg"].to_numpy()
        if np.allclose(delta, 0.0):
            stat, p = 0.0, 1.0
        else:
            result = wilcoxon(delta, alternative="two-sided", zero_method="wilcox", method="exact")
            stat, p = float(result.statistic), float(result.pvalue)
        summaries.append({
            "dataset": ds_name, "model": model_name, "rate_nominal": rate,
            "age_odds_ratio_per_sd": odds_ratio,
            "n_seed_pairs": int(len(sub)), "mean_mcar_fmg": float(sub["mcar_fmg"].mean()),
            "mean_age_mar_fmg": float(sub["age_mar_fmg"].mean()),
            "mean_age_mar_minus_mcar_fmg": float(np.mean(delta)),
            "sd_age_mar_minus_mcar_fmg": float(np.std(delta, ddof=1)),
            "median_age_mar_minus_mcar_fmg": float(np.median(delta)),
            "n_positive_differences": int(np.sum(delta > 0)),
            "wilcoxon_statistic_two_sided": stat, "wilcoxon_p_two_sided": p,
            "mean_realized_mcar_rate": float(sub["realized_mcar_rate"].mean()),
            "mean_realized_age_mar_rate": float(sub["realized_age_mar_rate"].mean()),
            "interpretation_scope": "exploratory; tested datasets/models/seeds and logistic age-conditioned propensity strengths only",
        })
    summary_df = pd.DataFrame(summaries)
    summary_df["holm_p_two_sided_within_rate"] = np.nan
    for rate, indices in summary_df.groupby("rate_nominal").groups.items():
        summary_df.loc[indices, "holm_p_two_sided_within_rate"] = holm_adjust(
            summary_df.loc[indices, "wilcoxon_p_two_sided"].tolist())
    summary_path = args.output_dir / "age_conditional_mar_statistics.csv"
    summary_df.to_csv(summary_path, index=False)

    input_hashes = {}
    for ds_name, dataset in datasets.items():
        info = {"cleaned_analysis_frame_sha256": sha256_frame(dataset[0])}
        source_path = dataset[4]
        if ds_name == "Adult Income":
            info["source_url"] = DATASET_URLS[ds_name] if source_path is None else None
            if source_path is not None:
                info["path"] = str(source_path)
                info["source_file_sha256"] = sha256_file(source_path)
        elif ds_name == "Taiwan Credit":
            info["path"] = str(source_path)
            info["source_file_sha256"] = sha256_file(source_path)
        else:
            info["source_url"] = DATASET_URLS[ds_name]
        input_hashes[ds_name] = info
    manifest = {
        "analysis": "age-conditioned MAR versus matched-eligible-feature MCAR sensitivity",
        "status": "completed" if len(per_seed_df) == 80 * len(datasets) * len(args.age_odds_ratios) else "incomplete",
        "n_per_seed_rows": len(per_seed_df), "n_summaries": len(summary_df),
        "seeds": SEEDS, "datasets": list(datasets), "models": MODELS, "rates": RATES,
        "age_odds_ratios_per_sd": args.age_odds_ratios,
        "masking": "test-only; age is observed in both conditions; all other feature cells eligible",
        "mar_mechanism": "cellwise Bernoulli with logit propensity intercept calibrated to nominal expected rate; log-odds increase by ln(OR) per SD of observed age for each requested OR",
        "preprocessing": "train-only label encoding/scaling; median imputer fitted on training matrix only; cleaned analysis-frame SHA-256 values identify the dataset snapshots used",
        "classifier": "trained on clean training split once per dataset/seed/model and kept frozen during both stress conditions",
        "inference": "paired exact two-sided Wilcoxon over ten seed-specific FMG contrasts; Holm adjustment within rate across the comparisons included in this run",
        "runtime_seconds": round(time.time() - started, 2),
        "python": sys.version, "platform": platform.platform(), "cpu_model": platform.processor(),
        "logical_cpu_count": __import__("os").cpu_count(),
        "package_versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                             "scikit_learn": sklearn.__version__,
                             "xgboost": __import__("xgboost").__version__,
                             "lightgbm": __import__("lightgbm").__version__},
        "input_hashes": input_hashes,
        "outputs": {"per_seed_csv": per_seed_path.name, "summary_csv": summary_path.name},
        "limitations": ["one observed covariate (age) and one logistic propensity form, evaluated at specified strengths", "not evidence for all MAR or MNAR mechanisms", "benchmark seeds are repeated splits, not independent populations", "exploratory sensitivity, no deployment validation"],
    }
    (args.output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(per_seed_df)} rows: {per_seed_path}", flush=True)
    print(f"Saved {len(summary_df)} summaries: {summary_path}", flush=True)
    print(summary_df[["dataset", "model", "rate_nominal", "mean_age_mar_minus_mcar_fmg", "holm_p_two_sided_within_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
