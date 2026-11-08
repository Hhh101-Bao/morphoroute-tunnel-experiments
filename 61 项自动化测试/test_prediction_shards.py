from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _identity(checkpoint_hash: str = "a" * 64):
    from morphoroute.runtime.prediction_shards import ShardIdentity

    return ShardIdentity(
        experiment_id="EXP-06",
        variant_id="B2",
        seed=42,
        split="D2",
        checkpoint_sha256=checkpoint_hash,
        dataset_sha256="b" * 64,
        config_sha256="c" * 64,
    )


def _image(image_id: str):
    from morphoroute.evaluation.mask_codec import encode_mask
    from morphoroute.evaluation.schema import (
        PredictionImage,
        PredictionInstance,
    )

    mask = np.zeros((8, 9), dtype=bool)
    mask[2:6, 3:7] = True
    instance = PredictionInstance(
        "crack",
        0.8,
        (3.0, 2.0, 4.0, 4.0),
        encode_mask(mask),
    )
    return PredictionImage(
        image_id=image_id,
        group_id=f"group-{image_id}",
        width=9,
        height=8,
        predictions=(instance,),
        targets=(instance,),
    )


class PredictionShardTests(unittest.TestCase):
    def test_completed_shards_resume_without_duplicates(self):
        from morphoroute.runtime.prediction_shards import PredictionShardStore

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = PredictionShardStore(root, _identity(), shard_size=2)
            first.append((_image("a"), _image("b")), start_index=0)

            resumed = PredictionShardStore(root, _identity(), shard_size=2)
            self.assertEqual(resumed.completed_image_ids(), ("a", "b"))
            resumed.append((_image("c"),), start_index=2)
            restored = resumed.read_all_images(("a", "b", "c"))

        self.assertEqual(
            tuple(image.image_id for image in restored),
            ("a", "b", "c"),
        )

    def test_mismatched_checkpoint_hash_is_rejected(self):
        from morphoroute.runtime.prediction_shards import (
            PredictionShardStore,
            ShardIdentityError,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            PredictionShardStore(root, _identity("a" * 64), 2).append(
                (_image("a"),),
                start_index=0,
            )

            with self.assertRaisesRegex(
                ShardIdentityError,
                "checkpoint_sha256",
            ):
                PredictionShardStore(root, _identity("d" * 64), 2)

    def test_truncated_final_shard_is_rejected(self):
        from morphoroute.runtime.prediction_shards import (
            PredictionShardError,
            PredictionShardStore,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root.mkdir(parents=True, exist_ok=True)
            (root / "part-000000-000001.json").write_text(
                '{"identity":',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(PredictionShardError, "part-000000"):
                PredictionShardStore(root, _identity(), 2)

    def test_shard_payload_uses_compressed_rle_string(self):
        from morphoroute.runtime.prediction_shards import PredictionShardStore

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = PredictionShardStore(root, _identity(), shard_size=2)
            path = store.append((_image("a"),), start_index=0)
            payload = json.loads(path.read_text(encoding="utf-8"))

        counts = payload["images"][0]["predictions"][0]["mask_rle"]["counts"]
        self.assertIsInstance(counts, str)

    def test_route_diagnostics_survive_resume(self):
        from morphoroute.runtime.prediction_shards import PredictionShardStore

        row = {
            "image_id": "a",
            "mean_route_entropy": 0.25,
            "ground_truth_inputs_used": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = PredictionShardStore(root, _identity(), shard_size=2)
            store.append((_image("a"),), start_index=0, route_rows=(row,))
            resumed = PredictionShardStore(root, _identity(), shard_size=2)

            restored = resumed.read_all_route_rows()

        self.assertEqual(restored, (row,))


if __name__ == "__main__":
    unittest.main()
