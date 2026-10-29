"""Write a hash and environment manifest for the metric-decomposition run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "analysis" / "outputs"
MANIFEST_PATH = OUTPUT_DIR / "metric_decomposition_manifest.json"
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "lightgbm",
            "matplotlib", "seaborn"]
OUTPUTS = [
    ROOT / "analysis" / "outputs" / "fmg_decomposition_per_seed.csv",
    ROOT / "analysis" / "outputs" / "fmg_decomposition_summary.csv",
    ROOT / "analysis" / "outputs" / "fmg_decomposition_correlations.csv",
    ROOT / "manuscript" / "FMG_manuscript_final_release_2026-10-28.docx",
    ROOT / "manuscript" / "FMG_supplementary_final_release_2026-10-28.docx",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[str, object] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    if path.suffix == ".csv":
        result["rows"] = int(len(pd.read_csv(path)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adult", required=True, type=Path)
    parser.add_argument("--compas", required=True, type=Path)
    parser.add_argument("--taiwan", required=True, type=Path)
    parser.add_argument("--runtime-seconds", type=float, default=432.7)
    args = parser.parse_args()

    source_inputs = {}
    for label, path in (("adult", args.adult), ("compas", args.compas), ("taiwan_credit", args.taiwan)):
        source_inputs[label] = {"filename": path.name, "sha256": sha256(path)}

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": __import__("os").cpu_count(),
        "packages": {name: metadata.version(name) for name in PACKAGES},
    }
    outputs = {str(path.relative_to(ROOT)).replace("\\", "/"): record(path) for path in OUTPUTS}
    data = pd.read_csv(ROOT / "analysis" / "outputs" / "fmg_decomposition_per_seed.csv")
    payload = {
        "analysis": "Frozen-classifier metric decomposition",
        "run_date": "2026-10-27",
        "runtime_seconds": args.runtime_seconds,
        "status": "completed",
        "scope": "Descriptive repeated benchmark splits; not independent populations.",
        "environment": environment,
        "design": {
            "datasets": sorted(data["dataset"].unique().tolist()),
            "models": sorted(data["model"].unique().tolist()),
            "seeds": sorted(int(x) for x in data["seed"].unique()),
            "masking_rates": sorted(float(x) for x in data["rate"].unique()),
            "mechanisms": sorted(data["mechanism"].unique().tolist()),
            "rows": int(len(data)),
            "archive_precision_decimals": 6,
        },
        "source_code": {
            "analysis/fmg_decomposition.py": sha256(ROOT / "analysis" / "fmg_decomposition.py"),
            "analysis/_primary_core.py": sha256(ROOT / "analysis" / "_primary_core.py"),
            "analysis/write_metric_decomposition_manifest.py": sha256(Path(__file__).resolve()),
            "scripts/add_metric_decomposition_to_manuscripts.py": sha256(ROOT / "scripts" / "add_metric_decomposition_to_manuscripts.py"),
        },
        "source_archives": {
            "results_raw_10seeds.csv": sha256(ROOT / "results_raw_10seeds.csv"),
            "group_conditional_mar_full_per_seed.csv": sha256(ROOT / "group_conditional_mar_full_per_seed.csv"),
        },
        "input_data_not_redistributed": source_inputs,
        "outputs": outputs,
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {MANIFEST_PATH}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
