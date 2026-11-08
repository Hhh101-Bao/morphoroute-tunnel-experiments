import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from morphoroute.data.annotations import (
    dacl_json_to_instances,
    decode_uncompressed_rle,
    tack_mask_to_instances,
)


class AnnotationTests(unittest.TestCase):
    def test_tack_mask_creates_one_instance_with_exact_area_and_bbox(self):
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 1:4] = 40
        instances = tack_mask_to_instances(mask, {40: "crack"})

        self.assertEqual(len(instances), 1)
        instance = instances[0]
        self.assertEqual(instance.class_name, "crack")
        self.assertEqual(instance.area, 3.0)
        self.assertEqual(instance.bbox, (1.0, 2.0, 3.0, 1.0))
        decoded = decode_uncompressed_rle(instance.segmentation)
        np.testing.assert_array_equal(decoded, mask == 40)

    def test_tack_mask_separates_disconnected_components(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[0, 0] = 40
        mask[3, 3] = 40
        instances = tack_mask_to_instances(mask, {40: "crack"})
        self.assertEqual([instance.area for instance in instances], [1.0, 1.0])

    def test_dacl_polygon_maps_class_and_geometry(self):
        payload = {
            "imageWidth": 10,
            "imageHeight": 10,
            "shapes": [
                {
                    "label": "Spalling",
                    "shape_type": "polygon",
                    "points": [[1, 1], [4, 1], [4, 3], [1, 3]],
                }
            ],
        }
        instances = dacl_json_to_instances(payload, {"Spalling": "spalling"})
        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0].class_name, "spalling")
        self.assertEqual(instances[0].area, 6.0)
        self.assertEqual(instances[0].bbox, (1.0, 1.0, 3.0, 2.0))

    def test_dacl_can_ignore_unmapped_non_target_labels(self):
        payload = {
            "imageWidth": 10,
            "imageHeight": 10,
            "shapes": [
                {
                    "label": "Graffiti",
                    "shape_type": "polygon",
                    "points": [[0, 0], [2, 0], [2, 2], [0, 2]],
                },
                {
                    "label": "Crack",
                    "shape_type": "polygon",
                    "points": [[1, 1], [4, 1], [4, 3], [1, 3]],
                },
            ],
        }
        instances = dacl_json_to_instances(
            payload,
            {"Crack": "crack"},
            ignore_unknown_labels=True,
        )
        self.assertEqual([instance.class_name for instance in instances], ["crack"])

    def test_dacl_can_clip_small_boundary_overshoot_when_requested(self):
        payload = {
            "imageWidth": 10,
            "imageHeight": 10,
            "shapes": [
                {
                    "label": "Crack",
                    "shape_type": "polygon",
                    "points": [[-2, 1], [4, 1], [4, 3], [-2, 3]],
                }
            ],
        }
        instances = dacl_json_to_instances(
            payload,
            {"Crack": "crack"},
            clip_out_of_bounds_tolerance=2.0,
        )
        self.assertEqual(instances[0].bbox, (0.0, 1.0, 4.0, 2.0))
        self.assertEqual(instances[0].area, 8.0)

    def test_dacl_still_rejects_overshoot_beyond_tolerance(self):
        payload = {
            "imageWidth": 10,
            "imageHeight": 10,
            "shapes": [
                {
                    "label": "Crack",
                    "shape_type": "polygon",
                    "points": [[-3, 1], [4, 1], [4, 3], [-3, 3]],
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "out-of-bounds"):
            dacl_json_to_instances(
                payload,
                {"Crack": "crack"},
                clip_out_of_bounds_tolerance=2.0,
            )

    def test_dacl_can_remove_an_out_of_bounds_spike_between_duplicate_neighbors(self):
        payload = {
            "imageWidth": 10,
            "imageHeight": 10,
            "shapes": [
                {
                    "label": "Crack",
                    "shape_type": "polygon",
                    "points": [
                        [1, 1], [4, 1], [4, 3], [15, 7], [4, 3], [1, 3]
                    ],
                }
            ],
        }
        instances = dacl_json_to_instances(
            payload,
            {"Crack": "crack"},
            drop_out_of_bounds_spikes=True,
        )
        self.assertEqual(instances[0].bbox, (1.0, 1.0, 3.0, 2.0))
        self.assertEqual(instances[0].area, 6.0)

    def test_dacl_rejects_degenerate_polygon(self):
        payload = {
            "imageWidth": 10,
            "imageHeight": 10,
            "shapes": [
                {
                    "label": "Crack",
                    "shape_type": "polygon",
                    "points": [[1, 1], [2, 2]],
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "at least three"):
            dacl_json_to_instances(payload, {"Crack": "crack"})


if __name__ == "__main__":
    unittest.main()
