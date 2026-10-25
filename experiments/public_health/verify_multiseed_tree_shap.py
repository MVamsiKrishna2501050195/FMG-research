"""Verify the archived multi-model/multi-seed SHAP sensitivity outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "public_health" / "shap_multiseed_2026_10_23"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    raw_path = OUT / "multiseed_tree_shap_shifts.csv"
    summary_path = OUT / "top_feature_summary.csv"
    manifest_path = OUT / "run_manifest.json"
    raw = pd.read_csv(raw_path)
    summary = pd.read_csv(summary_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    expected_datasets = {"BRFSS2024", "UCI_Diabetes130"}
    expected_models = {"RF", "XGB", "LGB"}
    expected_seeds = {42, 123, 456}
    assert len(raw) == 5430, f"unexpected raw rows: {len(raw)}"
    assert set(raw["dataset"]) == expected_datasets
    assert set(raw["model"]) == expected_models
    assert set(raw["seed"]) == expected_seeds
    assert set(raw["audit_group"]) == {"focus", "comparison"}
    assert set(raw["condition"]) == {"20%_MCAR_rep0"}
    assert set(raw["sample_n"]) == {20}
    assert raw["mean_signed_shap_shift_stress_minus_clean"].notna().all()
    assert raw["mean_absolute_shap_difference"].notna().all()
    assert raw["mean_absolute_shap_difference"].ge(0).all()
    keys = ["dataset", "seed", "model", "audit_group", "feature"]
    assert not raw.duplicated(keys).any(), "duplicate per-feature unit"
    units = raw[["dataset", "seed", "model"]].drop_duplicates()
    group_units = raw[["dataset", "seed", "model", "audit_group"]].drop_duplicates()
    assert len(units) == 18
    assert len(group_units) == 36
    assert len(summary) == 24
    assert set(summary["n_seeds"]) == {3}
    assert set(summary["dataset"]) == expected_datasets
    assert set(summary["model"]) == expected_models
    assert set(summary["audit_group"]) == {"focus", "comparison"}

    runner = ROOT / "experiments" / "public_health" / "run_multiseed_tree_shap.py"
    audit = ROOT / "experiments" / "public_health" / "run_public_health_fmg_audit.py"
    summary_script = ROOT / "experiments" / "public_health" / "summarize_multiseed_tree_shap.py"
    assert manifest["rows"] == len(raw)
    assert manifest["completed_model_seed_dataset_units"] == len(units)
    assert manifest["output_sha256"] == sha256(raw_path)
    assert manifest["top_feature_summary_sha256"] == sha256(summary_path)
    assert manifest["script_sha256"] == sha256(runner)
    assert manifest["audit_script_sha256"] == sha256(audit)
    assert manifest["summary_script_sha256"] == sha256(summary_script)
    assert all(":\\" not in key for key in manifest["input_sha256"]), "local absolute path leaked"

    print(json.dumps({
        "status": "PASS",
        "rows": len(raw),
        "model_seed_dataset_units": len(units),
        "group_units": len(group_units),
        "summary_rows": len(summary),
        "raw_sha256": sha256(raw_path),
        "summary_sha256": sha256(summary_path),
    }, indent=2))


if __name__ == "__main__":
    main()
