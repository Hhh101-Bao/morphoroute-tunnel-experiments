import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from morphoroute.statistics.bootstrap import paired_group_bootstrap
from morphoroute.statistics.effects import standardized_paired_effect
from morphoroute.statistics.multiple_testing import holm_adjust


class StatisticsTests(unittest.TestCase):
    def test_paired_bootstrap_preserves_group_pairs(self):
        records = pd.DataFrame(
            {
                "group_id": ["g1", "g2", "g3", "g4"],
                "method_a": [0.7, 0.8, 0.6, 0.9],
                "method_b": [0.6, 0.7, 0.5, 0.8],
            }
        )
        result = paired_group_bootstrap(
            records,
            group_col="group_id",
            value_a="method_a",
            value_b="method_b",
            n_resamples=20,
            seed=7,
        )
        self.assertAlmostEqual(result["point_estimate"], 0.1)
        self.assertEqual(len(result["resample_groups"]), 20)
        self.assertTrue(
            all(len(sampled_groups) == 4 for sampled_groups in result["resample_groups"])
        )
        self.assertTrue(
            all(set(sampled_groups).issubset({"g1", "g2", "g3", "g4"})
                for sampled_groups in result["resample_groups"])
        )

    def test_holm_adjust_matches_literal_example(self):
        adjusted = holm_adjust([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])

    def test_paired_effect_uses_difference_standard_deviation(self):
        effect = standardized_paired_effect(
            np.array([3.0, 5.0, 8.0]),
            np.array([1.0, 4.0, 4.0]),
        )
        differences = np.array([2.0, 1.0, 4.0])
        expected = differences.mean() / differences.std(ddof=1)
        self.assertAlmostEqual(effect, expected)


if __name__ == "__main__":
    unittest.main()
