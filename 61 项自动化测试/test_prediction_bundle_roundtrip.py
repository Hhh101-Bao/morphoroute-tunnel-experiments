from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class PredictionBundleRoundtripTests(unittest.TestCase):
    def test_targets_and_predictions_survive_roundtrip(self):
        from morphoroute.evaluation.io import read_prediction_bundle
        from morphoroute.evaluation.pipeline import write_prediction_bundle
        from morphoroute.evaluation.schema import (
            PredictionBundle,
            PredictionImage,
            PredictionInstance,
        )

        mask = np.eye(6, dtype=bool)
        instance = PredictionInstance("crack", 0.8, (0, 0, 6, 6), mask)
        bundle = PredictionBundle(
            experiment_id="EXP-09",
            split="D3",
            class_names=("crack",),
            images=(
                PredictionImage(
                    image_id="image",
                    group_id="group",
                    width=6,
                    height=6,
                    predictions=(instance,),
                    targets=(instance,),
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bundle.json"
            write_prediction_bundle(bundle, path)
            restored = read_prediction_bundle(path)

        self.assertEqual(len(restored.images[0].predictions), 1)
        self.assertEqual(len(restored.images[0].targets), 1)
        from morphoroute.evaluation.mask_codec import decode_mask

        np.testing.assert_array_equal(
            decode_mask(restored.images[0].targets[0].mask), mask
        )

    def test_compressed_bundle_roundtrip_stays_compressed_and_pixel_exact(self):
        from morphoroute.evaluation.io import read_prediction_bundle
        from morphoroute.evaluation.mask_codec import (
            EncodedMask,
            decode_mask,
            encode_mask,
        )
        from morphoroute.evaluation.pipeline import write_prediction_bundle
        from morphoroute.evaluation.schema import (
            PredictionBundle,
            PredictionImage,
            PredictionInstance,
        )

        dense = np.zeros((9, 13), dtype=bool)
        dense[2:8, 3:11] = True
        instance = PredictionInstance(
            "spalling",
            0.75,
            (3.0, 2.0, 8.0, 6.0),
            encode_mask(dense),
        )
        bundle = PredictionBundle(
            experiment_id="EXP-06",
            split="D2",
            class_names=("spalling",),
            images=(
                PredictionImage(
                    image_id="image-1",
                    group_id="group-1",
                    width=13,
                    height=9,
                    predictions=(instance,),
                    targets=(instance,),
                ),
            ),
        )

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bundle.json"
            write_prediction_bundle(bundle, path)
            raw = json.loads(path.read_text(encoding="utf-8"))
            restored = read_prediction_bundle(path)

        stored_counts = raw["images"][0]["predictions"][0]["mask_rle"][
            "counts"
        ]
        restored_mask = restored.images[0].predictions[0].mask
        self.assertIsInstance(stored_counts, str)
        self.assertIsInstance(restored_mask, EncodedMask)
        np.testing.assert_array_equal(decode_mask(restored_mask), dense)

    def test_legacy_uncompressed_rle_loads_as_encoded_mask(self):
        from morphoroute.data.annotations import encode_uncompressed_rle
        from morphoroute.evaluation.io import read_prediction_bundle
        from morphoroute.evaluation.mask_codec import EncodedMask, decode_mask

        dense = np.eye(7, dtype=bool)
        instance = {
            "class_name": "crack",
            "score": 0.9,
            "bbox": [0, 0, 7, 7],
            "mask_rle": encode_uncompressed_rle(dense),
        }
        payload = {
            "experiment_id": "EXP-03",
            "split": "D3",
            "class_names": ["crack"],
            "images": [
                {
                    "image_id": "legacy",
                    "width": 7,
                    "height": 7,
                    "predictions": [instance],
                    "targets": [instance],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            restored = read_prediction_bundle(path)

        restored_mask = restored.images[0].targets[0].mask
        self.assertIsInstance(restored_mask, EncodedMask)
        np.testing.assert_array_equal(decode_mask(restored_mask), dense)

    def test_failed_stream_write_does_not_replace_existing_file(self):
        from morphoroute.evaluation import pipeline
        from morphoroute.evaluation.schema import PredictionBundle

        bundle = PredictionBundle(
            experiment_id="EXP-06",
            split="D2",
            class_names=("crack",),
            images=(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bundle.json"
            path.write_text('{"sentinel": true}', encoding="utf-8")
            with mock.patch.object(
                pipeline,
                "_write_prediction_images",
                side_effect=RuntimeError("simulated write failure"),
                create=True,
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    pipeline.write_prediction_bundle(bundle, path)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"sentinel": True},
            )


if __name__ == "__main__":
    unittest.main()
