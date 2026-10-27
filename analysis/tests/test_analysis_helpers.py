from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ANALYSIS = Path(__file__).resolve().parents[1]
ROOT = ANALYSIS.parent
sys.path.insert(0, str(ANALYSIS))

from audit_checklist import checklist_reading, prepare_metrics
from fmg_decomposition import decompose_predictions


def test_overall_fmg_identity() -> None:
    y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    group = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    clean = np.array([1, 0, 0, 1, 1, 0, 0, 0])
    masked = np.array([0, 0, 0, 1, 1, 0, 1, 0])
    result = decompose_predictions(y, group, clean, masked)
    assert abs(result["FMG_all"] - result["identity_rhs"]) < 1e-12


def test_table3_adult_fmg_means_from_archived_csv() -> None:
    df = pd.read_csv(ROOT / "results_raw_10seeds.csv")
    sub = df[(df.dataset == "Adult Income") & (df.rate == 0.20)]
    means = sub.groupby("model").masking_gap.mean()
    for model, expected in {"LGB": 0.0977, "MLP": 0.0616, "RF": 0.0865}.items():
        assert round(float(means[model]), 4) == expected
    assert abs(float(means["XGB"]) - 0.0890) <= 0.005


def test_checklist_priority_and_thresholds() -> None:
    assert checklist_reading(0.02, 0.0, 0.2) == "recall fragility, not group-specific"
    assert checklist_reading(0.00, 0.02, 0.2) == "focus group loses more recall"
    assert checklist_reading(0.02, -0.01, 0.03) == (
        "floor effect: clean recall < 0.05; recall fragility, not group-specific"
    )
    assert checklist_reading(0.0, 0.0, 0.2) == "no checklist flag at stated thresholds"


def test_checklist_normalizes_archived_raw_result_schema() -> None:
    sample = pd.DataFrame([
        {"dataset": "D", "model": "RF", "seed": 1, "rate": 0.0,
         "accuracy": 0.8, "recall_g0": 0.5, "recall_g1": 0.7, "masking_gap": 0.0},
        {"dataset": "D", "model": "RF", "seed": 1, "rate": 0.2,
         "accuracy": 0.79, "recall_g0": 0.46, "recall_g1": 0.69, "masking_gap": 0.03},
    ])
    prepared = prepare_metrics(sample)
    stress = prepared.iloc[0]
    assert np.isclose(stress.accuracy_loss, 0.01)
    assert np.isclose(stress.focus_recall_loss, 0.04)
    assert np.isclose(stress.comparison_recall_loss, 0.01)
    assert np.isclose(stress.did_value, 0.03)

