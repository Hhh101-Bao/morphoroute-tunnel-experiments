import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from morphoroute.models.morphology import (
    MorphologyPrior,
    extract_morphology,
    extract_observable_morphology,
)


class MorphologyTests(unittest.TestCase):
    def test_line_and_area_masks_have_expected_relative_features(self):
        line = np.zeros((11, 11), dtype=bool)
        line[5, 2:9] = True
        square = np.zeros((11, 11), dtype=bool)
        square[3:8, 3:8] = True
        line_features = extract_morphology(line)
        square_features = extract_morphology(square)

        self.assertGreater(line_features[0], square_features[0])
        self.assertGreater(line_features[1], square_features[1])
        self.assertGreater(square_features[3], line_features[3])
        self.assertGreater(square_features[4], line_features[4])

    def test_empty_mask_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            extract_morphology(np.zeros((5, 5), dtype=bool))

    def test_batch_extraction_excludes_only_unobservable_empty_masks(self):
        empty = np.zeros((7, 7), dtype=bool)
        line = np.zeros((7, 7), dtype=bool)
        line[3, 1:6] = True

        features, observable_indices = extract_observable_morphology(
            [empty, line]
        )

        self.assertEqual(observable_indices.tolist(), [1])
        self.assertEqual(features.shape, (1, 5))
        np.testing.assert_allclose(features[0], extract_morphology(line))

    def test_batch_extraction_handles_all_empty_masks_without_fabrication(self):
        features, observable_indices = extract_observable_morphology(
            [
                np.zeros((3, 3), dtype=bool),
                np.zeros((5, 5), dtype=bool),
            ]
        )

        self.assertEqual(observable_indices.tolist(), [])
        self.assertEqual(features.shape, (0, 5))

    def test_prior_returns_normalized_two_component_targets(self):
        features = np.array(
            [
                [2.2, 0.9, 0.1, 0.2, 0.02],
                [2.0, 0.8, 0.1, 0.25, 0.03],
                [0.1, 0.2, 0.0, 0.8, 0.20],
                [0.2, 0.25, 0.0, 0.75, 0.18],
            ]
        )
        prior = MorphologyPrior.fit(features, seed=3)
        targets = prior.soft_targets(features)
        np.testing.assert_allclose(targets.sum(axis=1), np.ones(4), atol=1e-6)
        self.assertGreater(targets[0, 0], targets[0, 1])
        self.assertGreater(targets[3, 1], targets[3, 0])


if __name__ == "__main__":
    unittest.main()
