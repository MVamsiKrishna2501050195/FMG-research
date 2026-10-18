"""Recreate the notebook's primary figures using archived per-seed outputs.

This verifies plotting reproducibility only. It does NOT retrain models or
independently reproduce the archived experiment from raw datasets.
Run from the repository root with the notebook and results CSV present.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "cc3_masking_gap_fmg_audit_method_methodological_revised.ipynb"
RESULTS = ROOT / "results_raw_10seeds.csv"
OUTPUT = ROOT / "primary_plot_regeneration"
OUTPUT.mkdir(exist_ok=True)

with NOTEBOOK.open(encoding="utf-8") as f:
    nb = json.load(f)
ns = {
    "__name__": "__main__",
    "np": np,
    "pd": pd,
    "plt": plt,
    "Path": Path,
    "display": lambda obj: print(obj.to_string(index=False)),
    "results_df": pd.read_csv(RESULTS),
}
os.chdir(OUTPUT)

# Configuration and the three primary plotting cells from the canonical
# notebook: Figures 1-2 and the primary seed-distribution figure.
for index in (3, 13, 14, 33):
    code = "".join(nb["cells"][index].get("source", []))
    exec(compile(code, f"{NOTEBOOK.name}:cell-{index}", "exec"), ns)

input_hashes = {}
for path in (NOTEBOOK, RESULTS):
    input_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
outputs = sorted(p for p in OUTPUT.iterdir() if p.is_file())
output_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}
manifest = {
    "run_type": "plot regeneration from archived result table; no model retraining",
    "notebook_cells": [3, 13, 14, 33],
    "python": platform.python_version(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "matplotlib": matplotlib.__version__,
    "input_sha256": input_hashes,
    "output_sha256": output_hashes,
}
(OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2))
