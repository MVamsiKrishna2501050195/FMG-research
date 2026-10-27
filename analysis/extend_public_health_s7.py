"""Re-score saved public-health audit conditions with S7-style decomposition.

The existing audit script and outputs are read-only. This add-on reconstructs
the exact extension split, fitted model, and masks to obtain predicted-positive
rates that were not stored in metrics_per_mask.csv, then cross-checks its saved
metrics before writing a separate NEW CSV.
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

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SCRIPT = ROOT / "experiments" / "public_health" / "run_public_health_fmg_audit.py"
DEFAULT_METRICS = ROOT / "experiments" / "public_health" / "outputs" / "metrics_per_mask.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs" / "public_health"


def load_audit_module():
    spec = importlib.util.spec_from_file_location("fmg_public_health_audit", PUBLIC_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load existing extension code: {PUBLIC_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def weighted_ppr(pred: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(np.asarray(pred, dtype=float), weights=np.asarray(weights, dtype=float)))


def extension_metrics(audit, data: pd.DataFrame, dataset_name: str, source: Path,
                      old_metrics: pd.DataFrame, seeds: list[int], mask_repeats: int,
                      split_fraction: float = 0.20) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    split_name = "80_20" if split_fraction == 0.20 else "70_30_sensitivity"
    for seed in seeds:
        train_idx, test_idx = audit.make_split(
            data, seed, split_fraction, patient_disjoint=(dataset_name == "UCI_Diabetes130")
        )
        train, test = data.iloc[train_idx], data.iloc[test_idx]
        y_train = train["target"].to_numpy(dtype=int)
        y = test["target"].to_numpy(dtype=int)
        group = test["group"].to_numpy(dtype=int)
        weights = test["weight"].to_numpy(dtype=float) if "weight" in test else np.ones(len(test))
        prep = audit.make_preprocessor(data, dataset_name)
        x_train = np.asarray(prep.fit_transform(train), dtype=np.float32)
        x_test = np.asarray(prep.transform(test), dtype=np.float32)
        imputer = audit.SimpleImputer(strategy="median")
        x_train = imputer.fit_transform(x_train).astype(np.float32, copy=False)
        x_test = x_test.astype(np.float32, copy=False)
        for model_index, model_name in enumerate(audit.MODELS):
            model = audit.build_model(model_name, seed)
            model.fit(x_train, y_train)
            pred_clean, prob_clean, clean = audit._score(model, x_test, y, group, weights)
            clean_copy = pred_clean.copy()
            for mech_index, mechanism in enumerate(audit.MECHANISMS):
                for rate in audit.RATES:
                    for repeat in range(mask_repeats):
                        mask_seed = int(np.random.SeedSequence([
                            seed, repeat, int(rate * 1000), model_index, mech_index
                        ]).generate_state(1)[0])
                        x_mask, masked_cells = audit.mask_matrix(
                            x_test, x_train, group, mechanism, rate, mask_seed
                        )
                        x_mask = imputer.transform(x_mask).astype(np.float32, copy=False)
                        pred_mask, prob_mask, masked = audit._score(model, x_mask, y, group, weights)
                        fmg, g0_loss, acc_loss = audit.fmg_from_predictions(
                            y, pred_clean, pred_mask, group, weights
                        )
                        g1 = group == 0
                        g0 = group == 1
                        g1_loss = float(clean["comparison_recall"] - masked["comparison_recall"])
                        row = {
                            "dataset": dataset_name, "split": split_name, "seed": seed,
                            "model": model_name, "mechanism": mechanism,
                            "nominal_mask_rate": rate, "mask_repeat": repeat,
                            "clean_accuracy": clean["accuracy"], "masked_accuracy": masked["accuracy"],
                            "accuracy_loss": acc_loss,
                            "clean_focus_recall": clean["focus_recall"],
                            "masked_focus_recall": masked["focus_recall"],
                            "focus_recall_loss": g0_loss,
                            "clean_comparison_recall": clean["comparison_recall"],
                            "masked_comparison_recall": masked["comparison_recall"],
                            "comparison_recall_loss": g1_loss,
                            "DiD_focus_minus_comparison": g0_loss - g1_loss,
                            "FMG_focus": fmg,
                            "FMG_comparison": g1_loss - acc_loss,
                            "dPPR_overall": weighted_ppr(pred_clean, weights) - weighted_ppr(pred_mask, weights),
                            "dPPR_focus": weighted_ppr(pred_clean[g0], weights[g0]) - weighted_ppr(pred_mask[g0], weights[g0]),
                            "dPPR_comparison": weighted_ppr(pred_clean[g1], weights[g1]) - weighted_ppr(pred_mask[g1], weights[g1]),
                            "clean_predicted_positive_rate": weighted_ppr(pred_clean, weights),
                            "masked_predicted_positive_rate": weighted_ppr(pred_mask, weights),
                            "masked_cells": masked_cells,
                            "masked_cell_fraction": masked_cells / (len(test) * x_test.shape[1]),
                            "source_sha256": audit.sha256(source),
                            "new_not_in_manuscript": True,
                        }
                        rows.append(row)
                        if not np.array_equal(clean_copy, model.predict(x_test)):
                            raise AssertionError("Frozen model changed during public-health masked scoring")
                        # Compare the metrics already recorded by the public-health run.
                        keys = ((old_metrics["dataset"] == dataset_name) &
                                (old_metrics["split"] == split_name) &
                                (old_metrics["seed"] == seed) &
                                (old_metrics["model"] == model_name) &
                                (old_metrics["mechanism"] == mechanism) &
                                (old_metrics["nominal_mask_rate"] == rate) &
                                (old_metrics["mask_repeat"] == repeat))
                        prior = old_metrics.loc[keys]
                        if len(prior) == 1:
                            prior_row = prior.iloc[0]
                            for new_col, prior_col in (("FMG_focus", "FMG"),
                                                       ("accuracy_loss", "aggregate_accuracy_loss"),
                                                       ("focus_recall_loss", "focus_recall_loss"),
                                                       ("clean_focus_recall", "clean_focus_recall"),
                                                       ("clean_comparison_recall", "clean_comparison_recall")):
                                if not np.isclose(float(row[new_col]), float(prior_row[prior_col]), atol=1e-12, rtol=0):
                                    raise AssertionError(
                                        f"Rebuilt extension metric mismatch {dataset_name}/{model_name}/"
                                        f"{mechanism}/{rate}/{new_col}: {row[new_col]} vs {prior_row[prior_col]}"
                                    )
            # clean records are deliberately evaluated once per model and reuse train-only transforms.
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brfss", help="Official CDC 2024 XPT or ZIP. Omitted datasets are logged as skipped.")
    parser.add_argument("--diabetes-zip", help="UCI Diabetes 130-US Hospitals ZIP or CSV.")
    parser.add_argument("--existing-metrics", default=str(DEFAULT_METRICS), help="Saved extension metrics for exact consistency checks")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT / "public_health_s7_style_new.csv"))
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Defaults to seeds present for each supplied dataset in existing metrics")
    parser.add_argument("--mask-repeats", type=int, default=1,
                        help="Must match the saved run; current archived extension run uses one repeat")
    args = parser.parse_args()
    start = time.perf_counter()
    audit = load_audit_module()
    source_metrics = Path(args.existing_metrics)
    old = pd.read_csv(source_metrics)
    datasets = []
    skipped = []
    for dataset_name, path, loader in (("BRFSS2024", args.brfss, audit.load_brfss),
                                       ("UCI_Diabetes130", args.diabetes_zip, audit.load_uci_diabetes)):
        if not path:
            skipped.append(f"{dataset_name}: raw input path was not supplied")
            continue
        source = Path(path)
        if not source.is_file():
            skipped.append(f"{dataset_name}: input not found at {source}")
            continue
        data = loader(source)
        seeds = args.seeds or sorted(old.loc[old.dataset == dataset_name, "seed"].astype(int).unique().tolist())
        if not seeds:
            raise ValueError(f"No recorded seed available for {dataset_name}; refusing to invent one")
        if args.mask_repeats != int(old.loc[old.dataset == dataset_name, "mask_repeat"].max()) + 1:
            raise ValueError(f"mask_repeats={args.mask_repeats} does not match archived output for {dataset_name}")
        manifest_path = ROOT / "experiments" / "public_health" / "outputs" / "run_manifest.json"
        extension_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = extension_manifest["software"]["packages"]
        import importlib.metadata as metadata
        for package, expected_version in expected.items():
            try:
                current = metadata.version(package)
            except metadata.PackageNotFoundError:
                current = "not-installed"
            if current != expected_version:
                raise RuntimeError(f"Extension environment mismatch for {package}: "
                                   f"expected {expected_version}, installed {current}")
        datasets.append((dataset_name, data, source, seeds))
    if not datasets:
        raise SystemExit("No supplied public-health raw data could be run. Skipped: " + "; ".join(skipped))
    output_frames = []
    for name, data, source, seeds in datasets:
        output_frames.append(extension_metrics(audit, data, name, source, old, seeds,
                                               args.mask_repeats))
    out = pd.concat(output_frames, ignore_index=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, float_format="%.12g")
    print(f"Wrote NEW extension table: {out_path} ({len(out)} rows)")
    print("Summary by dataset/model/mechanism/rate:")
    print(out.groupby(["dataset", "model", "mechanism", "nominal_mask_rate"])[
        ["clean_focus_recall", "comparison_recall_loss", "FMG_focus", "FMG_comparison",
         "DiD_focus_minus_comparison", "dPPR_overall", "dPPR_focus", "dPPR_comparison"]
    ].mean().to_string())
    print("Skipped:", "; ".join(skipped) if skipped else "None")
    print(f"Elapsed seconds: {time.perf_counter() - start:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
