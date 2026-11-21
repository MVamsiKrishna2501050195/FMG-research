"""COMPAS RF SHAP sensitivity with 100 matched observations per group.

This is an exploratory, seed-42 extension of the archived SHAP analysis. It
reuses the frozen-classifier/test-only masking protocol and reports descriptive
group-wise mean absolute SHAP shifts with uncertainty across the matched rows.
"""
from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import tempfile
import zipfile

import numpy as np
import pandas as pd
import shap

from _primary_core import load_compas, preprocess, fit_model, mcar_mask, impute_train_median


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:\Users\mvk80\Downloads\FMG_Data.zip")
OUT = ROOT / "analysis" / "outputs" / "review_additions"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
N_PER_GROUP = 100


def ci95(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    se = values.std(ddof=1) / np.sqrt(len(values))
    half = 1.96 * se
    return float(values.mean() - half), float(values.mean() + half)


def positive_class_values(explainer, X):
    values = explainer.shap_values(X)
    if isinstance(values, list):
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]
    return values


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="fmg_compas_shap_") as td:
        with zipfile.ZipFile(DATA) as zf:
            member = next(n for n in zf.namelist() if n.endswith("compas-scores-two-years.csv"))
            zf.extract(member, td)
        data = load_compas(Path(td) / member)
        x_train, x_test, y_train, y_test, group = preprocess(data, SEED)
        model = fit_model("RF", SEED, x_train, y_train)

        rng = np.random.RandomState(SEED)
        indices = []
        for g in (0, 1):
            candidates = np.flatnonzero(group == g)
            if len(candidates) < N_PER_GROUP:
                raise RuntimeError(f"Group {g} has only {len(candidates)} eligible test rows")
            indices.extend(rng.choice(candidates, size=N_PER_GROUP, replace=False).tolist())
        indices = np.asarray(indices, dtype=int)
        X_clean = x_test.iloc[indices]
        X_masked = mcar_mask(x_test, 0.20, SEED).iloc[indices]
        X_masked = pd.DataFrame(impute_train_median(x_train, X_masked), columns=x_train.columns)
        selected_group = group[indices]

        explainer = shap.TreeExplainer(model)
        clean = positive_class_values(explainer, X_clean)
        masked = positive_class_values(explainer, X_masked)
        shift = np.abs(masked) - np.abs(clean)

        rows = []
        for g in (0, 1):
            mask = selected_group == g
            for j, feature in enumerate(data.features):
                vals = shift[mask, j]
                lo, hi = ci95(vals)
                rows.append({
                "dataset": "COMPAS",
                "model": "RF",
                "seed": SEED,
                "subgroup": g,
                "n_matched": int(mask.sum()),
                "feature": feature,
                "mean_abs_shap_clean": float(np.abs(clean[mask, j]).mean()),
                "mean_abs_shap_masked": float(np.abs(masked[mask, j]).mean()),
                "mean_abs_shap_shift": float(vals.mean()),
                "sd_abs_shap_shift": float(vals.std(ddof=1)),
                "ci95_low": lo,
                "ci95_high": hi,
                })
        detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "compas_rf_shap_100_per_feature.csv", index=False)

    ranked = detail.sort_values(["subgroup", "mean_abs_shap_shift"], ascending=[True, False])
    ranked.groupby("subgroup", as_index=False).head(5).to_csv(OUT / "compas_rf_shap_100_top5.csv", index=False)
    manifest = {
        "dataset": "COMPAS",
        "model": "Random Forest",
        "seed": SEED,
        "mask_rate": 0.20,
        "n_matched_per_group": N_PER_GROUP,
        "protocol": "training-fitted preprocessing, frozen classifier, test-only MCAR mask, training-fitted median imputation",
        "interpretation": "descriptive attribution sensitivity; not causal explanation",
        "software": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "shap")},
        "input_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
    }
    (OUT / "compas_rf_shap_100_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(detail.groupby("subgroup")["n_matched"].first().to_string())
    print(ranked.groupby("subgroup", as_index=False).head(5)[["subgroup", "feature", "mean_abs_shap_shift"]].to_string(index=False))


if __name__ == "__main__":
    main()
