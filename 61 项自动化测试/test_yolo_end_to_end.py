from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class RecordingUltralyticsBackend:
    version = "test-backend"

    def __init__(self, bundle):
        self.bundle = bundle
        self.epochs = None
        self.resume = None
        self.prediction_split = None

    def train(self, plan, *, epochs, resume):
        self.epochs = epochs
        self.resume = resume
        checkpoint = plan.output_dir / "train" / "weights" / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        return checkpoint

    def predict(self, checkpoint, materialized, *, split):
        self.prediction_split = split
        return self.bundle


class YoloEndToEndTests(unittest.TestCase):
    @staticmethod
    def _fake_official_backend(root: Path):
        from morphoroute.runtime.ultralytics_adapter import (
            OfficialUltralyticsBackend,
        )

        calls = []

        class FakeModel:
            def __init__(self, source):
                self.source = str(source)

            def train(self, **kwargs):
                calls.append((self.source, kwargs))
                save_dir = root / "runs" / "train"
                best = save_dir / "weights" / "best.pt"
                best.parent.mkdir(parents=True, exist_ok=True)
                best.write_bytes(b"checkpoint")
                return SimpleNamespace(save_dir=save_dir)

        backend = OfficialUltralyticsBackend.__new__(
            OfficialUltralyticsBackend
        )
        backend._ultralytics = SimpleNamespace(YOLO=FakeModel)
        backend.version = "test"
        return backend, calls

    @staticmethod
    def _official_plan(root: Path):
        return SimpleNamespace(
            model="yolov8n-seg.pt",
            dataset_config=root / "data.yaml",
            input_size=640,
            physical_batch=1,
            effective_batch=8,
            seed=42,
            amp=True,
            output_dir=root / "runs",
        )

    def test_official_backend_uses_effective_batch_as_nbs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            backend, calls = self._fake_official_backend(root)

            backend.train(
                self._official_plan(root),
                epochs=2,
                resume=True,
            )

            source, arguments = calls[0]
            self.assertEqual(source, "yolov8n-seg.pt")
            self.assertEqual(arguments["nbs"], 8)
            self.assertFalse(arguments["resume"])

    def test_official_backend_resumes_only_from_existing_last_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            last = root / "runs" / "train" / "weights" / "last.pt"
            last.parent.mkdir(parents=True, exist_ok=True)
            last.write_bytes(b"last")
            backend, calls = self._fake_official_backend(root)

            backend.train(
                self._official_plan(root),
                epochs=2,
                resume=True,
            )

            source, arguments = calls[0]
            self.assertEqual(source, str(last))
            self.assertTrue(arguments["resume"])

    def test_pipeline_trains_d1_selects_d2_and_predicts_d3(self):
        from morphoroute.core.config import ExperimentConfig
        from morphoroute.data.materialize import (
            DatasetFingerprint,
            MaterializedDataset,
        )
        from morphoroute.evaluation.schema import PredictionBundle
        from morphoroute.runtime.ultralytics_adapter import (
            train_validate_predict_yolo,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "dataset"
            (data_root / "metadata").mkdir(parents=True)
            output_root = root / "outputs"
            materialized_root = root / "derived"
            (materialized_root / "images" / "train").mkdir(parents=True)
            (materialized_root / "images" / "test").mkdir(parents=True)
            (materialized_root / "data.yaml").write_text(
                "path: .\ntrain: images/train\nval: images/val\n",
                encoding="utf-8",
            )
            for index in range(16):
                (materialized_root / "images" / "train" / f"{index}.png").touch()
            bundle = PredictionBundle(
                experiment_id="EXP-03",
                split="D3",
                class_names=("crack",),
                images=(),
            )
            backend = RecordingUltralyticsBackend(bundle)
            config = ExperimentConfig.from_mapping(
                {
                    "experiment_id": "EXP-03",
                    "profile": "local_6gb",
                    "data_root": str(data_root),
                    "output_root": str(output_root),
                    "seed": 42,
                    "allowed_splits": ["D1", "D2", "D3"],
                    "parameters": {
                        "model": "yolov8n-seg.pt",
                        "input_size": 640,
                        "max_updates": 10,
                        "physical_batch": 1,
                        "effective_batch": 8,
                        "gradient_accumulation": 8,
                        "amp": True,
                    },
                }
            )
            materialized = MaterializedDataset(
                root=materialized_root,
                fingerprint=DatasetFingerprint("abc", 16, ("D1", "D2", "D3")),
                reused=False,
                format="yolo",
                data_yaml=materialized_root / "data.yaml",
            )

            result = train_validate_predict_yolo(
                config,
                materialized,
                backend=backend,
            )

            self.assertEqual(result.training_plan.train_split, "D1")
            self.assertEqual(result.training_plan.validation_split, "D2")
            self.assertEqual(backend.prediction_split, "D3")
            self.assertEqual(backend.epochs, 5)
            self.assertTrue(result.best_checkpoint.is_file())
            self.assertEqual(result.predictions.split, "D3")
            self.assertTrue(result.evaluation_outputs.metrics_csv.is_file())


if __name__ == "__main__":
    unittest.main()
