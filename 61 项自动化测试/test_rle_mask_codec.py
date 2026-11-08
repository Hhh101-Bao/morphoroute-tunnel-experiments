from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class RLEMaskCodecTests(unittest.TestCase):
    def test_dense_roundtrip_is_pixel_exact(self):
        from morphoroute.evaluation.mask_codec import decode_mask, encode_mask

        dense = np.zeros((17, 23), dtype=bool)
        dense[2:11, 4:19] = True

        encoded = encode_mask(dense)

        self.assertEqual(encoded.size, (17, 23))
        self.assertIsInstance(encoded.counts, str)
        np.testing.assert_array_equal(decode_mask(encoded), dense)

    def test_rle_iou_matches_hand_calculated_dense_reference(self):
        from morphoroute.evaluation.mask_codec import encode_mask, mask_iou

        left = np.zeros((5, 5), dtype=bool)
        right = np.zeros((5, 5), dtype=bool)
        left[1:4, 1:4] = True
        right[2:5, 2:5] = True

        actual = mask_iou(encode_mask(left), encode_mask(right))

        self.assertAlmostEqual(actual, 4.0 / 14.0, places=12)

    def test_two_empty_masks_keep_legacy_iou_of_one(self):
        from morphoroute.evaluation.mask_codec import encode_mask, mask_iou

        empty = encode_mask(np.zeros((6, 8), dtype=bool))

        self.assertEqual(mask_iou(empty, empty), 1.0)

    def test_prediction_instance_keeps_encoded_mask_compressed(self):
        from morphoroute.evaluation.mask_codec import EncodedMask, encode_mask
        from morphoroute.evaluation.schema import PredictionInstance

        encoded = encode_mask(np.ones((10, 10), dtype=bool))

        instance = PredictionInstance(
            class_name="crack",
            score=0.9,
            bbox=(0.0, 0.0, 10.0, 10.0),
            mask=encoded,
        )

        self.assertIs(instance.mask, encoded)
        self.assertIsInstance(instance.mask, EncodedMask)

    def test_polygon_annotation_encodes_without_losing_pixels(self):
        from morphoroute.evaluation.mask_codec import (
            decode_mask,
            encode_coco_segmentation,
            mask_area,
        )

        encoded = encode_coco_segmentation(
            [[2, 2, 15, 2, 15, 12, 2, 12]],
            height=20,
            width=24,
        )

        dense = decode_mask(encoded)
        self.assertEqual(encoded.size, (20, 24))
        self.assertEqual(dense.shape, (20, 24))
        self.assertEqual(int(mask_area(encoded)), 130)
        self.assertEqual(int(dense.sum()), 130)

    def test_malformed_dimensions_are_rejected(self):
        from morphoroute.evaluation.mask_codec import EncodedMask

        with self.assertRaisesRegex(ValueError, "positive"):
            EncodedMask(size=(0, 10), counts="abc")


if __name__ == "__main__":
    unittest.main()
