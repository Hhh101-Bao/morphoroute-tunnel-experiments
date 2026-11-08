import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from morphoroute.metrics.reliability import expected_calibration_error
from morphoroute.metrics.structure import (
    area_relative_error,
    boundary_f1,
    cldice,
    component_count_error,
    crack_break_rate,
    hd95,
)


class StructureMetricTests(unittest.TestCase):
    def test_cldice_is_one_for_identical_line(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 1:4] = True
        self.assertAlmostEqual(cldice(mask, mask), 1.0)

    def test_cldice_is_zero_when_only_prediction_is_empty(self):
        truth = np.zeros((5, 5), dtype=bool)
        truth[2, 1:4] = True
        prediction = np.zeros_like(truth)
        self.assertEqual(cldice(prediction, truth), 0.0)

    def test_boundary_f1_is_zero_for_distant_pixels(self):
        prediction = np.zeros((7, 7), dtype=bool)
        truth = np.zeros((7, 7), dtype=bool)
        prediction[1, 1] = True
        truth[5, 5] = True
        self.assertEqual(boundary_f1(prediction, truth, tolerance=1), 0.0)

    def test_hd95_is_zero_for_identical_masks(self):
        mask = np.zeros((7, 7), dtype=bool)
        mask[2:5, 2:5] = True
        self.assertEqual(hd95(mask, mask), 0.0)

    def test_crack_break_and_component_errors_detect_a_split(self):
        truth = np.zeros((9, 9), dtype=bool)
        truth[4, 1:8] = True
        prediction = truth.copy()
        prediction[4, 4] = False

        self.assertGreater(crack_break_rate(prediction, truth), 0.0)
        self.assertEqual(component_count_error(prediction, truth), 1.0)

    def test_area_relative_error_is_zero_for_identical_masks(self):
        mask = np.zeros((7, 7), dtype=bool)
        mask[2:5, 2:5] = True
        self.assertEqual(area_relative_error(mask, mask), 0.0)


class ReliabilityMetricTests(unittest.TestCase):
    def test_ece_matches_two_hand_derived_bins(self):
        confidence = np.array([0.2, 0.4, 0.8, 1.0])
        correct = np.array([0, 1, 1, 1])
        # Bin 1: |0.50 - 0.30| * 0.5 = 0.10
        # Bin 2: |1.00 - 0.90| * 0.5 = 0.05
        self.assertAlmostEqual(
            expected_calibration_error(confidence, correct, n_bins=2),
            0.15,
        )


if __name__ == "__main__":
    unittest.main()
