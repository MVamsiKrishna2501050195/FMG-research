"""Recompute paired TreeSHAP shifts for the public-health extension.

This supplemental analysis fits RF, XGBoost, and LightGBM on the existing
80/20 benchmark splits for the three prespecified seeds, then compares SHAP
values for the same deterministic sample under clean and 20% MCAR test inputs.
It is descriptive and does not alter the primary FMG results.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_public_health_fmg_audit as audit  # noqa: E402

MODELS = ("RF", "XGB", "LGB")
RATE = 0.20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calculate_unit(dataset_name, data, source_path, patient_disjoint, seed, sample_size):
    train_idx, test_idx = audit.make_split(data, seed, 0.20, patient_disjoint)
    train = data.iloc[train_idx]
    test = data.iloc[test_idx]
    y_train = train["target"].to_numpy(dtype=int)
    y_test = test["target"].to_numpy(dtype=int)
    group = test["group"].to_numpy(dtype=int)

    preprocessor = audit.make_preprocessor(data, dataset_name)
    x_train = np.asarray(preprocessor.fit_transform(train), dtype=np.float32)
    x_clean = np.asarray(preprocessor.transform(test), dtype=np.float32)
    encoded_imputer = SimpleImputer(strategy="median")
    x_train = encoded_imputer.fit_transform(x_train).astype(np.float32, copy=False)
    x_clean = encoded_imputer.transform(x_clean).astype(np.float32, copy=False)
    feature_names = preprocessor.get_feature_names_out()

    all_rows = []
    unit_meta = {
        "dataset": dataset_name,
        "seed": int(seed),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_focus": int(np.sum(group == 1)),
        "n_comparison": int(np.sum(group == 0)),
        "encoded_feature_count": int(x_clean.shape[1]),
        "input_sha256": sha256(source_path),
    }
    for model_index, model_name in enumerate(MODELS):
        print(f"  {model_name}: fitting full training split", flush=True)
        start = time.perf_counter()
        model = audit.build_model(model_name, seed)
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - start
        print(f"  {model_name}: fit complete in {fit_seconds:.1f}s; computing paired TreeSHAP", flush=True)
        mask_seed = int(np.random.SeedSequence([
            seed, 0, int(RATE * 1000), audit.MODELS.index(model_name), 0
        ]).generate_state(1)[0])
        x_stress_raw, masked_cells = audit.mask_matrix(
            x_clean, x_train, group, "MCAR", RATE, mask_seed
        )
        x_stress = encoded_imputer.transform(x_stress_raw).astype(np.float32, copy=False)

        import shap
        explainer = shap.TreeExplainer(model)
        rng = np.random.default_rng(seed + 8731)
        for group_value, group_name in ((1, "focus"), (0, "comparison")):
            indices = np.flatnonzero(group == group_value)
            if len(indices) == 0:
                continue
            take = min(sample_size, len(indices))
            selected = np.sort(rng.choice(indices, size=take, replace=False))
            # Saabas approximation keeps RF/XGB tractable, but the current
            # SHAP API does not support approximate=True for LightGBM.
            use_approximation = model_name in {"RF", "XGB"}
            vals_clean = explainer.shap_values(x_clean[selected], approximate=use_approximation)
            vals_stress = explainer.shap_values(x_stress[selected], approximate=use_approximation)
            if isinstance(vals_clean, list):
                vals_clean, vals_stress = vals_clean[-1], vals_stress[-1]
            vals_clean = np.asarray(vals_clean)
            vals_stress = np.asarray(vals_stress)
            if vals_clean.ndim == 3:
                vals_clean, vals_stress = vals_clean[:, :, -1], vals_stress[:, :, -1]
            if vals_clean.ndim != 2 or vals_clean.shape != vals_stress.shape:
                raise ValueError(f"Unexpected SHAP shape for {dataset_name}/{model_name}: {vals_clean.shape}")
            for feature_index, feature_name in enumerate(feature_names):
                difference = vals_stress[:, feature_index] - vals_clean[:, feature_index]
                all_rows.append({
                    **unit_meta,
                    "model": model_name,
                    "condition": "20%_MCAR_rep0",
                    "audit_group": group_name,
                    "sample_n": int(take),
                    "feature": str(feature_name),
                    "mean_signed_shap_shift_stress_minus_clean": float(np.mean(difference)),
                    "mean_absolute_shap_difference": float(np.mean(np.abs(difference))),
                    "model_fit_seconds": float(fit_seconds),
                    "masked_cell_fraction": float(masked_cells / (len(test) * x_clean.shape[1])),
                    "shap_output_class": "positive class",
                    "shap_method": (
                        "approximate TreeSHAP (Saabas; approximate=True)"
                        if use_approximation else "exact TreeSHAP"
                    ),
                })
        print(f"  {model_name}: fit={fit_seconds:.1f}s, SHAP rows={sum(r['model']==model_name for r in all_rows)}", flush=True)
    return all_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brfss", type=Path, required=True)
    parser.add_argument("--diabetes-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--sample-size", type=int, default=50,
                        help="Maximum fixed test units per audit group for paired approximate TreeSHAP")
    args = parser.parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = out_dir / "multiseed_tree_shap_shifts.csv"
    manifest_path = out_dir / "run_manifest.json"
    if output_csv.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite prior output in {out_dir}")

    datasets = [
        ("BRFSS2024", audit.load_brfss(args.brfss.resolve()), args.brfss.resolve(), False),
        ("UCI_Diabetes130", audit.load_uci_diabetes(args.diabetes_zip.resolve()), args.diabetes_zip.resolve(), True),
    ]
    all_rows = []
    start = time.perf_counter()
    for dataset_name, data, source_path, patient_disjoint in datasets:
        print(f"Dataset {dataset_name}: n={len(data):,}, p_raw={data.shape[1]-5}", flush=True)
        for seed in args.seeds:
            print(f" seed={seed}", flush=True)
            all_rows.extend(calculate_unit(
                dataset_name, data, source_path, patient_disjoint, seed, args.sample_size
            ))
            # Persist after each dataset/seed so a later interruption leaves
            # completed units recoverable and clearly marked in the manifest.
            pd.DataFrame(all_rows).to_csv(output_csv, index=False)
    elapsed = time.perf_counter() - start
    manifest = {
        "analysis": "supplementary paired TreeSHAP shift; descriptive only",
        "protocol": {
            "datasets": [name for name, *_ in datasets],
            "models": list(MODELS),
            "seeds": list(args.seeds),
            "split": "80/20; patient-disjoint grouped stratified split for UCI Diabetes",
            "mask": "test-only 20% MCAR, one deterministic mask per dataset/model/seed",
            "sample": f"up to {args.sample_size} observations per audit group, shared clean/stress rows",
            "explanation": "RF/XGB use approximate TreeSHAP (Saabas, no exact SHAP consistency guarantee); LGB uses exact TreeSHAP because SHAP does not support approximate=True for LightGBM. Effects are compared within each model, not across model output scales.",
            "note": "not causal, not a stability guarantee, and not part of primary FMG results",
        },
        "input_sha256": {str(p.resolve()): sha256(p.resolve()) for p in (args.brfss, args.diabetes_zip)},
        "script_sha256": sha256(Path(__file__).resolve()),
        "audit_script_sha256": sha256(HERE / "run_public_health_fmg_audit.py"),
        "software": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "shap", "xgboost", "lightgbm")},
        "python": sys.version,
        "platform": platform.platform(),
        "elapsed_seconds": elapsed,
        "completed_dataset_seed_units": len(datasets) * len(args.seeds),
        "expected_units": len(datasets) * len(args.seeds),
        "completed_model_seed_dataset_units": len(datasets) * len(args.seeds) * len(MODELS),
        "expected_model_seed_dataset_units": len(datasets) * len(args.seeds) * len(MODELS),
        "rows": len(all_rows),
        "output_sha256": sha256(output_csv),
        "output_bytes": output_csv.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"seconds": elapsed, "rows": len(all_rows), "output": str(output_csv)}, indent=2))


if __name__ == "__main__":
    main()
