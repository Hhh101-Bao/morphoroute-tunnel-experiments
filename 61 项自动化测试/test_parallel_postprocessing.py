from __future__ import annotations

import json
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from contextlib import redirect_stdout

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class _FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def item(self):
        return self.value.item()

    def __getitem__(self, index):
        return _FakeTensor(self.value[index])


class _FakeInstances:
    def __init__(self):
        mask = np.zeros((1, 8, 9), dtype=bool)
        mask[0, 2:6, 3:7] = True
        self.masks = _FakeTensor(mask)
        self.labels = _FakeTensor([0])
        self.scores = _FakeTensor([0.85])
        self.bboxes = _FakeTensor([[3.0, 2.0, 7.0, 6.0]])

    def cpu(self):
        return self


class _FakeInference:
    def __init__(self, fail_at: int | None = None):
        self.fail_at = fail_at
        self.paths: list[str] = []

    def __call__(self, model, image_path: str):
        self.paths.append(image_path)
        if self.fail_at is not None and len(self.paths) == self.fail_at:
            raise RuntimeError("simulated inference interruption")
        return SimpleNamespace(pred_instances=_FakeInstances())


def _install_fake_mmdet(inference):
    package = types.ModuleType("mmdet")
    apis = types.ModuleType("mmdet.apis")
    apis.init_detector = lambda *args, **kwargs: SimpleNamespace(
        panoptic_head=None
    )
    apis.inference_detector = inference
    package.apis = apis
    return mock.patch.dict(
        sys.modules,
        {"mmdet": package, "mmdet.apis": apis},
    )


def _fixture(root: Path, image_count: int = 6):
    annotation_path = root / "instances_D2.json"
    images = [
        {
            "id": index,
            "record_id": f"image-{index}",
            "group_id": f"group-{index}",
            "width": 9,
            "height": 8,
            "file_name": str(root / f"image-{index}.png"),
        }
        for index in range(1, image_count + 1)
    ]
    annotations = [
        {
            "id": index,
            "image_id": index,
            "category_id": 1,
            "bbox": [3.0, 2.0, 4.0, 4.0],
            "segmentation": [[3, 2, 7, 2, 7, 6, 3, 6]],
        }
        for index in range(1, image_count + 1)
    ]
    annotation_path.write_text(
        json.dumps(
            {
                "categories": [{"id": 1, "name": "crack"}],
                "images": images,
                "annotations": annotations,
            }
        ),
        encoding="utf-8",
    )
    config_path = root / "resolved.py"
    checkpoint = root / "runs" / "B2" / "seed_42" / "best.pth"
    config_path.write_text("model = {}\n", encoding="utf-8")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    materialized = SimpleNamespace(
        annotation_files={"D2": annotation_path},
        fingerprint=SimpleNamespace(sha256="d" * 64),
    )
    return config_path, checkpoint, materialized


class ParallelPostprocessingTests(unittest.TestCase):
    def test_prediction_returns_only_encoded_masks_and_writes_shards(self):
        from morphoroute.evaluation.mask_codec import EncodedMask
        from morphoroute.runtime.mmdet_predictions import predict_mmdet_bundle

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, checkpoint, materialized = _fixture(root)
            inference = _FakeInference()
            shard_root = root / "shards"
            with _install_fake_mmdet(inference):
                output = io.StringIO()
                with redirect_stdout(output):
                    bundle = predict_mmdet_bundle(
                        config,
                        checkpoint,
                        materialized,
                        split="D2",
                        shard_root=shard_root,
                        cpu_workers=2,
                        shard_size=2,
                        progress_interval=2,
                    )
            shard_files = tuple(shard_root.glob("part-*.json"))

        self.assertEqual(len(bundle.images), 6)
        self.assertEqual(len(shard_files), 3)
        self.assertEqual(output.getvalue().count("MMDetection D2 prediction"), 3)
        self.assertTrue(
            all(
                isinstance(instance.mask, EncodedMask)
                for image in bundle.images
                for instance in (*image.predictions, *image.targets)
            )
        )

    def test_restart_skips_images_from_completed_shards(self):
        from morphoroute.runtime.mmdet_predictions import predict_mmdet_bundle

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, checkpoint, materialized = _fixture(root)
            shard_root = root / "shards"
            interrupted = _FakeInference(fail_at=4)
            with _install_fake_mmdet(interrupted):
                with self.assertRaisesRegex(RuntimeError, "interruption"):
                    predict_mmdet_bundle(
                        config,
                        checkpoint,
                        materialized,
                        split="D2",
                        shard_root=shard_root,
                        cpu_workers=1,
                        shard_size=2,
                        progress_interval=2,
                    )

            resumed = _FakeInference()
            with _install_fake_mmdet(resumed):
                bundle = predict_mmdet_bundle(
                    config,
                    checkpoint,
                    materialized,
                    split="D2",
                    shard_root=shard_root,
                    cpu_workers=1,
                    shard_size=2,
                    progress_interval=2,
                )

        self.assertEqual(
            [Path(value).stem for value in resumed.paths],
            ["image-3", "image-4", "image-5", "image-6"],
        )
        self.assertEqual(
            tuple(image.image_id for image in bundle.images),
            tuple(f"image-{index}" for index in range(1, 7)),
        )


if __name__ == "__main__":
    unittest.main()
