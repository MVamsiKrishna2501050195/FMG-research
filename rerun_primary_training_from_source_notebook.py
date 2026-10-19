"""Run the notebook's full primary training loop with checkpoints.

The training/preprocessing/model/metric logic is read from the canonical
notebook. This wrapper only supplies the original Taiwan CSV, caches the
notebook's public inputs, checkpoints completed dataset/seed blocks, and writes
fresh outputs to primary_training_regeneration/ without replacing the archive.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "FMG_research_final.ipynb"
OUTPUT = ROOT / "primary_training_regeneration"
INPUT_CACHE = Path(tempfile.gettempdir()) / "fmg-primary-rerun-inputs"
DATA_CACHE = INPUT_CACHE / "openml"
ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
COMPAS_URL = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
ADULT_LOCAL = INPUT_CACHE / "adult.data"
COMPAS_LOCAL = INPUT_CACHE / "compas-scores-two-years.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_once(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "FMG-reproducibility/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, path.open("wb") as out:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            out.write(block)


def hardware_snapshot() -> dict:
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
    }
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        state = MEMORYSTATUSEX()
        state.dwLength = ctypes.sizeof(state)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
            info["total_ram_bytes"] = int(state.ullTotalPhys)
            info["available_ram_bytes_at_start"] = int(state.ullAvailPhys)
    return info


def run(taiwan_path: Path) -> None:
    if not NOTEBOOK.is_file():
        raise FileNotFoundError(f"Canonical notebook missing: {NOTEBOOK}")
    if not taiwan_path.is_file():
        raise FileNotFoundError(f"Taiwan CSV missing: {taiwan_path}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT / "results_checkpoint.csv"
    INPUT_CACHE.mkdir(parents=True, exist_ok=True)
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["SCIKIT_LEARN_DATA"] = str(DATA_CACHE)
    os.environ["MPLCONFIGDIR"] = str(INPUT_CACHE / "matplotlib")

    download_once(ADULT_URL, ADULT_LOCAL)
    download_once(COMPAS_URL, COMPAS_LOCAL)

    import json as json_module
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    import pandas as pd

    nb = json_module.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__", "__file__": str(NOTEBOOK)}

    # Load notebook libraries and fixed design; keep its published code intact.
    exec(compile("".join(nb["cells"][2]["source"]), f"{NOTEBOOK.name}:cell-2", "exec"), namespace)
    exec(compile("".join(nb["cells"][3]["source"]), f"{NOTEBOOK.name}:cell-3", "exec"), namespace)

    # Redirect the notebook's two URL-based pandas reads to hashed local copies.
    notebook_read_csv = namespace["pd"].read_csv
    source_map = {ADULT_URL: ADULT_LOCAL, COMPAS_URL: COMPAS_LOCAL}
    def cached_read_csv(source, *args, **kwargs):
        return notebook_read_csv(source_map.get(str(source), source), *args, **kwargs)
    namespace["pd"].read_csv = cached_read_csv

    for index in (4, 5):
        code = "".join(nb["cells"][index]["source"])
        exec(compile(code, f"{NOTEBOOK.name}:cell-{index}", "exec"), namespace)

    # The original notebook's only Colab-specific operation is its upload UI;
    # here the same loader receives the user's original CSV as a local path.
    code = "".join(nb["cells"][6]["source"])
    code = code.replace("from google.colab import files as colab_files\n", "")
    code = code.replace(
        "taiwan_df, taiwan_feats, taiwan_sens, taiwan_tgt = load_taiwan()\n", ""
    )
    exec(compile(code, f"{NOTEBOOK.name}:cell-6", "exec"), namespace)
    namespace["taiwan_df"], namespace["taiwan_feats"], namespace["taiwan_sens"], namespace["taiwan_tgt"] = namespace["load_taiwan"](str(taiwan_path))

    code = "".join(nb["cells"][7]["source"])
    code = code.replace(
        "fetch_openml(data_id=31, as_frame=True, parser=\"auto\")",
        "fetch_openml(data_id=31, as_frame=True, parser=\"auto\", data_home=str(DATA_CACHE))",
    )
    namespace["DATA_CACHE"] = DATA_CACHE
    exec(compile(code, f"{NOTEBOOK.name}:cell-7", "exec"), namespace)
    exec(compile("".join(nb["cells"][8]["source"]), f"{NOTEBOOK.name}:cell-8", "exec"), namespace)
    exec(compile("".join(nb["cells"][9]["source"]), f"{NOTEBOOK.name}:cell-9", "exec"), namespace)

    # Log the observed training inputs before the long run starts.
    dataset_fingerprints = {}
    for name, (df, features, sensitive, target, exploratory) in namespace["DATASETS"].items():
        dataset_fingerprints[name] = {
            "rows": int(len(df)), "feature_count": len(features),
            "feature_names_sha256": hashlib.sha256("\n".join(features).encode()).hexdigest(),
            "group_counts": {str(k): int(v) for k, v in df[sensitive].value_counts().sort_index().items()},
            "positive_counts_by_group": {str(k): int(v) for k, v in df.groupby(sensitive)[target].sum().items()},
            "exploratory": bool(exploratory),
        }
    print("DATASET_FINGERPRINTS " + json.dumps(dataset_fingerprints, sort_keys=True), flush=True)

    # Resume only whole dataset/seed blocks (20 rows per block = 5 models x 4 rates).
    all_results = []
    completed_pairs = set()
    if checkpoint_path.exists():
        previous = pd.read_csv(checkpoint_path)
        counts = previous.groupby(["dataset", "seed"]).size()
        completed_pairs = {tuple(k) for k, n in counts.items() if n == 20}
        if completed_pairs:
            all_results = previous[
                previous.apply(lambda r: (r["dataset"], r["seed"]) in completed_pairs, axis=1)
            ].to_dict("records")
        print(f"Resuming {len(completed_pairs)} completed dataset/seed blocks", flush=True)

    source = "".join(nb["cells"][10]["source"])
    skip_marker = "for seed in SEEDS:\n        X_tr, X_te, y_tr, y_te = preprocess(df, feats, tgt_col, seed)"
    checkpoint_marker = "        print(f\"  seed {seed:3d} done\", end=\"  \", flush=True)"
    if skip_marker not in source or checkpoint_marker not in source:
        raise RuntimeError("Notebook main-loop markers changed; refusing a non-equivalent run.")
    source = source.replace(
        "all_results = []   # one row per (dataset, model, seed, rate)",
        "# all_results and completed_pairs are initialized by the restartable wrapper.",
    )
    source = source.replace(
        skip_marker,
        "for seed in SEEDS:\n        if (ds_name, seed) in completed_pairs:\n            continue\n        X_tr, X_te, y_tr, y_te = preprocess(df, feats, tgt_col, seed)",
    )
    source = source.replace(
        checkpoint_marker,
        "        print(f\"  seed {seed:3d} done\", end=\"  \", flush=True)\n"
        "        completed_pairs.add((ds_name, seed))\n"
        "        tmp = checkpoint_path.with_suffix('.tmp.csv')\n"
        "        pd.DataFrame(all_results).to_csv(tmp, index=False)\n"
        "        os.replace(tmp, checkpoint_path)",
    )
    namespace.update({
        "all_results": all_results,
        "completed_pairs": completed_pairs,
        "checkpoint_path": checkpoint_path,
    })
    print("Starting notebook cell 10: full four-dataset/five-model/ten-seed run", flush=True)
    exec(compile(source, f"{NOTEBOOK.name}:cell-10", "exec"), namespace)
    if len(namespace["results_df"]) != 800 or len(completed_pairs) != 40:
        raise RuntimeError(
            f"Expected 800 output rows across 40 dataset/seed blocks; got "
            f"{len(namespace['results_df'])} rows and {len(completed_pairs)} blocks."
        )

    os.chdir(OUTPUT)
    namespace["results_df"].round(6).to_csv("results_raw_10seeds.csv", index=False)
    # The canonical notebook's final print line has conflicting nested quotes
    # and is a syntax error. Correct only that line in-memory; retain all its
    # statistical calculations verbatim.
    stats_source = "".join(nb["cells"][11]["source"])
    stats_source = "\n".join(
        'print("Mean MG Cohen\'s dz:", round(stats_df.loc[stats_df["Main"], "MG Cohen\'s dz"].mean(), 3))'
        if line.startswith('print(f"Mean MG Cohen\'s dz:') else line
        for line in stats_source.splitlines()
    )
    exec(compile(stats_source, f"{NOTEBOOK.name}:cell-11", "exec"), namespace)
    exec(compile("".join(nb["cells"][12]["source"]), f"{NOTEBOOK.name}:cell-12", "exec"), namespace)
    namespace["stats_df"].round(6).to_csv("stats_results_10seeds.csv", index=False)
    namespace["eog_df"].round(6).to_csv("eog_results_10seeds.csv", index=False)
    namespace["results_df"][namespace["results_df"]["rate"] == 0.2][
        ["dataset", "model", "seed", "masking_gap", "exploratory"]
    ].copy().round(6).to_csv("masking_gap_per_seed.csv", index=False)
    namespace["display"] = lambda frame: print(frame.to_string(index=False))
    for index in (13, 14, 33):
        exec(compile("".join(nb["cells"][index]["source"]), f"{NOTEBOOK.name}:cell-{index}", "exec"), namespace)

    input_files = {
        "adult.data": {"sha256": sha256(ADULT_LOCAL), "source": ADULT_URL},
        "compas-scores-two-years.csv": {"sha256": sha256(COMPAS_LOCAL), "source": COMPAS_URL},
        "taiwan_credit_csv": {"sha256": sha256(taiwan_path), "source": "user-provided original Colab upload"},
    }
    openml_files = sorted(p for p in DATA_CACHE.rglob("*") if p.is_file())
    manifest = {
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_type": "full primary notebook training rerun with per-seed checkpoints",
        "source_notebook": NOTEBOOK.name,
        "source_notebook_sha256": sha256(NOTEBOOK),
        "rows": int(len(namespace["results_df"])),
        "expected_rows": 800,
        "input_files": input_files,
        "openml_cache_files_sha256": {str(p.relative_to(DATA_CACHE)): sha256(p) for p in openml_files},
        "dataset_fingerprints": dataset_fingerprints,
        "runtime": {
            "python": platform.python_version(),
            "packages": {name: metadata.version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "lightgbm", "shap", "matplotlib", "seaborn"]},
            "hardware": hardware_snapshot(),
        },
        "historical_colab_hardware": "not recorded in original notebook; this manifest describes only the new rerun",
    }
    (OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("RUN_COMPLETE " + json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--taiwan-csv", required=True, type=Path)
    run(parser.parse_args().taiwan_csv.resolve())
