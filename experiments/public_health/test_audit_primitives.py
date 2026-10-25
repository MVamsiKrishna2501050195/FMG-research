"""Small deterministic tests for the extension's core audit invariants."""
import unittest
import tempfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_public_health_fmg_audit import (
    BRFSS_FEATURES, UCI_CATEGORICAL, UCI_NUMERIC,
    _append_checkpoint_rows, _write_checkpoint,
    fmg_from_predictions, make_split, mask_matrix,
)


class AuditPrimitiveTests(unittest.TestCase):
    def test_mask_rate_is_exact_and_repeatable_for_all_mechanisms(self):
        x_train = np.arange(60, dtype=float).reshape(20, 3)
        x_test = np.arange(60, 120, dtype=float).reshape(20, 3)
        group = np.array([0] * 10 + [1] * 10)
        for mech in ("MCAR", "MAR_group_conditional", "MNAR_value_conditional"):
            with self.subTest(mechanism=mech):
                a, count_a = mask_matrix(x_test, x_train, group, mech, 0.10, 9)
                b, count_b = mask_matrix(x_test, x_train, group, mech, 0.10, 9)
                self.assertEqual(count_a, 6)
                self.assertEqual(count_b, 6)
                np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
                self.assertEqual(int(np.isnan(a).sum()), 6)

    def test_fmg_is_difference_of_the_two_paired_losses(self):
        y = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        clean = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        stress = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        group = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        fmg, recall_loss, accuracy_loss = fmg_from_predictions(
            y, clean, stress, group, np.ones(len(y))
        )
        self.assertAlmostEqual(fmg, recall_loss - accuracy_loss)

    def test_patient_split_keeps_clusters_disjoint(self):
        rows = []
        for patient in range(100):
            label = patient % 2
            for encounter in range(2):
                rows.append({"target": label, "cluster": str(patient)})
        data = pd.DataFrame(rows)
        train, test = make_split(data, 42, 0.20, patient_disjoint=True)
        self.assertFalse(set(data.iloc[train].cluster) & set(data.iloc[test].cluster))
        self.assertAlmostEqual(len(test) / len(data), 0.20, delta=0.02)

    def test_audit_attributes_and_identifiers_are_excluded_from_predictors(self):
        brfss_predictors = set(BRFSS_FEATURES)
        uci_predictors = set(UCI_CATEGORICAL) | set(UCI_NUMERIC)
        for forbidden in ("_RACEGR3", "SEXVAR", "group", "intersection_group", "target", "weight", "cluster", "stratum"):
            self.assertNotIn(forbidden, brfss_predictors)
        for forbidden in ("race", "gender", "group", "intersection_group", "patient_nbr", "encounter_id", "target"):
            self.assertNotIn(forbidden, uci_predictors)

    def test_checkpoint_rows_append_and_checkpoint_json_replaces_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _append_checkpoint_rows(root, "metrics.csv", [{"FMG": 0.2}])
            _append_checkpoint_rows(root, "metrics.csv", [{"FMG": 0.4}])
            _write_checkpoint(root / "checkpoint.json", {"complete": False, "units": ["a"]})
            _write_checkpoint(root / "checkpoint.json", {"complete": True, "units": ["a", "b"]})
            rows = pd.read_csv(root / "metrics.csv")
            self.assertEqual(rows.FMG.tolist(), [0.2, 0.4])
            self.assertIn('"complete": true', (root / "checkpoint.json").read_text().lower())


if __name__ == "__main__":
    unittest.main()
