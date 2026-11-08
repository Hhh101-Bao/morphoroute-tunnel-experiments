from __future__ import annotations

import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


REQUIRED_CFG_KEYS = {
    "model",
    "train_dataloader",
    "val_dataloader",
    "test_dataloader",
    "train_cfg",
    "val_cfg",
    "test_cfg",
    "optim_wrapper",
    "param_scheduler",
    "val_evaluator",
    "test_evaluator",
    "default_hooks",
    "visualizer",
    "work_dir",
}


def _config(root: Path, **parameter_overrides):
    from morphoroute.core.config import ExperimentConfig

    data_root = root / "dataset"
    (data_root / "metadata").mkdir(parents=True)
    return ExperimentConfig.from_mapping(
        {
            "experiment_id": "EXP-05",
            "profile": "local_6gb",
            "data_root": str(data_root),
            "output_root": str(root / "outputs"),
            "seed": 42,
            "allowed_splits": ["D1", "D2", "D3"],
            "parameters": {
                "model": "mask2former_r50",
                "input_size": 640,
                "max_updates": 10,
                "physical_batch": 1,
                "effective_batch": 8,
                "amp": True,
                **parameter_overrides,
            },
        }
    )


def _dataset(root: Path):
    from morphoroute.data.materialize import (
        DatasetFingerprint,
        MaterializedDataset,
    )

    dataset_root = root / "coco"
    annotations = {}
    for split in ("D1", "D2", "D3"):
        path = dataset_root / "annotations" / f"{split}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '{"images":[],"annotations":[],"categories":[]}',
            encoding="utf-8",
        )
        annotations[split] = path
    return MaterializedDataset(
        root=dataset_root,
        fingerprint=DatasetFingerprint("abc", 0, ("D1", "D2", "D3")),
        reused=False,
        format="coco",
        annotation_files=annotations,
    )


class RecordingMMDetBackend:
    version = "test-backend"

    def __init__(self, bundle):
        self.bundle = bundle
        self.trained = False
        self.prediction_split = None

    def train(self, resolved, *, resume):
        self.trained = True
        checkpoint = resolved.work_dir / "best_coco_segm_mAP.pth"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
        return checkpoint

    def predict(self, checkpoint, resolved, materialized, *, split):
        self.prediction_split = split
        return self.bundle


class MMDetEndToEndTests(unittest.TestCase):
    def test_mask2former_config_is_runner_complete(self):
        from morphoroute.mmdet.config_builder import build_mask2former_config

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resolved = build_mask2former_config(_config(root), _dataset(root))
            cfg = resolved.config

            self.assertTrue(REQUIRED_CFG_KEYS <= set(cfg))
            self.assertEqual(
                cfg["model"]["panoptic_head"]["num_things_classes"],
                5,
            )
            self.assertTrue(
                cfg["train_dataloader"]["dataset"]["ann_file"].endswith(
                    "D1.json"
                )
            )
            self.assertEqual(
                cfg["model"]["panoptic_head"]["loss_cls"]["class_weight"],
                [1.0] * 5 + [0.1],
            )
            self.assertEqual(
                cfg["model"]["data_preprocessor"]["batch_augments"][0]["size"],
                (640, 640),
            )
            self.assertTrue(
                cfg["val_dataloader"]["dataset"]["ann_file"].endswith(
                    "D2.json"
                )
            )
            self.assertEqual(cfg["optim_wrapper"]["accumulative_counts"], 8)
            self.assertEqual(
                cfg["train_cfg"]["max_iters"],
                10 * 8,
            )
            self.assertEqual(
                cfg["param_scheduler"][1]["end"],
                10 * 8,
            )
            self.assertEqual(cfg["train_cfg"]["val_interval"], 8)
            self.assertEqual(
                cfg["default_hooks"]["checkpoint"]["interval"],
                8,
            )

    def test_formal_validation_interval_uses_optimizer_updates(self):
        from morphoroute.mmdet.config_builder import build_mask2former_config

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resolved = build_mask2former_config(
                _config(root, max_updates=50_000),
                _dataset(root),
            )
            cfg = resolved.config

            self.assertEqual(cfg["train_cfg"]["val_interval"], 40_000)
            self.assertEqual(
                cfg["default_hooks"]["checkpoint"]["interval"],
                40_000,
            )

    def test_validation_interval_accepts_optimizer_update_override(self):
        from morphoroute.mmdet.config_builder import build_mask2former_config

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resolved = build_mask2former_config(
                _config(
                    root,
                    max_updates=50_000,
                    validation_update_interval=2_500,
                ),
                _dataset(root),
            )
            cfg = resolved.config

            self.assertEqual(cfg["train_cfg"]["val_interval"], 20_000)
            self.assertEqual(
                cfg["default_hooks"]["checkpoint"]["interval"],
                20_000,
            )

    def test_validation_interval_rejects_non_positive_override(self):
        from morphoroute.mmdet.config_builder import build_mask2former_config

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            with self.assertRaisesRegex(
                ValueError,
                "validation_update_interval must be positive",
            ):
                build_mask2former_config(
                    _config(root, validation_update_interval=0),
                    _dataset(root),
                )

    def test_mask_rcnn_config_replaces_incompatible_base_fields(self):
        from mmengine.config import Config
        from morphoroute.mmdet.config_builder import build_mask_rcnn_config

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resolved = build_mask_rcnn_config(_config(root), _dataset(root))
            base = Config(
                {
                    "train_cfg": {
                        "type": "EpochBasedTrainLoop",
                        "max_epochs": 12,
                    },
                    "optim_wrapper": {
                        "type": "OptimWrapper",
                        "optimizer": {
                            "type": "SGD",
                            "lr": 0.02,
                            "momentum": 0.9,
                        },
                    },
                    "train_dataloader": {
                        "dataset": {"data_root": "data/coco/"},
                    },
                }
            )

            base.merge_from_dict(resolved.config)

            self.assertNotIn("max_epochs", base.train_cfg)
            self.assertEqual(base.train_cfg.type, "IterBasedTrainLoop")
            self.assertNotIn(
                "momentum",
                base.optim_wrapper.optimizer,
            )
            self.assertNotIn(
                "data_root",
                base.train_dataloader.dataset,
            )

    def test_morphoroute_stages_and_model_counter_use_optimizer_updates(self):
        from morphoroute.mmdet.config_builder import build_morphoroute_config

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resolved = build_morphoroute_config(
                _config(root),
                _dataset(root),
                variant_id="B9",
                morphology_prior_path=None,
            )
            cfg = resolved.config

            self.assertEqual(cfg["train_cfg"]["max_iters"], 80)
            self.assertEqual(cfg["model"]["gradient_accumulation"], 8)
            self.assertEqual(
                cfg["model"]["panoptic_head"]["stage_warmup_end"],
                1,
            )
            self.assertEqual(
                cfg["model"]["panoptic_head"]["stage_sparse_start"],
                8,
            )
            threshold_hook = next(
                hook
                for hook in cfg["custom_hooks"]
                if hook["type"] == "RoutingThresholdSelectionHook"
            )
            self.assertEqual(threshold_hook["trigger_iteration"], 8 * 8)

    def test_pipeline_trains_selects_best_and_predicts_d3(self):
        from morphoroute.evaluation.schema import PredictionBundle
        from morphoroute.runtime.mmdet_adapter import (
            train_validate_predict_mmdet,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = PredictionBundle(
                experiment_id="EXP-05",
                split="D3",
                class_names=("crack",),
                images=(),
            )
            backend = RecordingMMDetBackend(bundle)

            result = train_validate_predict_mmdet(
                _config(root),
                _dataset(root),
                model_kind="mask2former",
                backend=backend,
            )

            self.assertTrue(backend.trained)
            self.assertEqual(backend.prediction_split, "D3")
            self.assertTrue(result.best_checkpoint.is_file())
            self.assertTrue(result.evaluation_outputs.metrics_xlsx.is_file())

    def test_package_base_config_uri_resolves_to_mim_config(self):
        from morphoroute.runtime.mmdet_adapter import (
            _resolve_base_config_path,
        )

        with tempfile.TemporaryDirectory() as temporary:
            package_root = Path(temporary) / "mmdet"
            package_init = package_root / "__init__.py"
            config_path = (
                package_root
                / ".mim"
                / "configs"
                / "mask_rcnn"
                / "mask-rcnn_r50_fpn_1x_coco.py"
            )
            config_path.parent.mkdir(parents=True)
            package_init.write_text("", encoding="utf-8")
            config_path.write_text("model = {}", encoding="utf-8")
            spec = SimpleNamespace(origin=str(package_init))

            with patch("importlib.util.find_spec", return_value=spec):
                resolved = _resolve_base_config_path(
                    "mmdet::mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py"
                )

            self.assertEqual(resolved, config_path)


if __name__ == "__main__":
    unittest.main()
