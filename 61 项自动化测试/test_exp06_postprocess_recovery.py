from __future__ import annotations

import json
import io
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from morphoroute.aliasing import install_module_alias


def _bundle(split: str):
    from morphoroute.evaluation.mask_codec import encode_mask
    from morphoroute.evaluation.schema import (
        PredictionBundle,
        PredictionImage,
        PredictionInstance,
    )

    mask = np.zeros((10, 12), dtype=bool)
    mask[2:8, 3:9] = True
    target = PredictionInstance(
        "spalling",
        1.0,
        (3.0, 2.0, 6.0, 6.0),
        encode_mask(mask),
    )
    prediction = PredictionInstance(
        "spalling",
        0.9,
        target.bbox,
        encode_mask(mask),
    )
    return PredictionBundle(
        experiment_id="EXP-06",
        split=split,
        class_names=("spalling",),
        images=(
            PredictionImage(
                image_id=f"{split}-image",
                group_id=f"{split}-group",
                width=12,
                height=10,
                predictions=(prediction,),
                targets=(target,),
            ),
        ),
        framework="mmdetection",
    )


class _TrainForbiddenBackend:
    version = "test-mmdet"

    def __init__(self):
        self.train_calls = 0
        self.predict_splits: list[str] = []

    def train(self, *args, **kwargs):
        self.train_calls += 1
        raise AssertionError("B2 training must not be called")

    def predict(self, checkpoint, resolved, materialized, *, split):
        self.predict_splits.append(split)
        return _bundle(split)


class _CountingTrainingBackend(_TrainForbiddenBackend):
    def train(self, resolved, *, resume):
        self.train_calls += 1
        checkpoint = resolved.work_dir / "best_coco_segm_mAP_iter_1.pth"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"new-b3-checkpoint")
        return checkpoint


class EXP06PostprocessRecoveryTests(unittest.TestCase):
    def test_checkpoint_only_postprocess_never_calls_train_and_writes_cache(self):
        from morphoroute.core.config import ExperimentConfig
        from morphoroute.data.materialize import (
            DatasetFingerprint,
            MaterializedDataset,
        )
        from morphoroute.mmdet.config_builder import MMDetResolvedConfig
        from morphoroute.runtime.mmdet_adapter import (
            postprocess_trained_mmdet_checkpoint,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "outputs"
            work_dir = (
                output_root
                / "experiments/EXP-06/local_6gb/seed_42/runs/B2/seed_42"
            )
            work_dir.mkdir(parents=True)
            config_path = work_dir / "resolved_mmdet_config.py"
            checkpoint = work_dir / "best_coco_segm_mAP_iter_200000.pth"
            config_path.write_text("model = {}\n", encoding="utf-8")
            checkpoint.write_bytes(b"frozen-b2-checkpoint")
            config = ExperimentConfig(
                experiment_id="EXP-06",
                profile="local_6gb",
                data_root=root / "dataset",
                output_root=output_root,
                seed=42,
                allowed_splits=("D1", "D2", "D3"),
                parameters={"resume": False, "reuse_completed_training": True},
            )
            materialized = MaterializedDataset(
                root=root / "derived",
                fingerprint=DatasetFingerprint(
                    sha256="d" * 64,
                    record_count=3,
                    splits=("D1", "D2", "D3"),
                ),
                reused=True,
                format="coco",
            )
            resolved = MMDetResolvedConfig(
                model_kind="morphoroute",
                base_config="unused.py",
                config={},
                work_dir=work_dir,
                config_path=config_path,
            )
            backend = _TrainForbiddenBackend()
            result = postprocess_trained_mmdet_checkpoint(
                config,
                materialized,
                resolved,
                checkpoint,
                variant_id="B2",
                cache_signature="test-signature",
                backend=backend,
            )
            completed_path = (
                output_root
                / "training_cache/mmdet/test-signature/completed_run.json"
            )
            completed = json.loads(completed_path.read_text(encoding="utf-8"))

        self.assertEqual(backend.train_calls, 0)
        self.assertEqual(backend.predict_splits, ["D2", "D3"])
        self.assertEqual(result.best_checkpoint, checkpoint)
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["signature"], "test-signature")

    def test_recovery_context_rejects_dataset_fingerprint_mismatch(self):
        from morphoroute.core.config import ExperimentConfig
        from morphoroute.data.materialize import (
            DatasetFingerprint,
            MaterializedDataset,
        )
        from morphoroute.mmdet.config_builder import MMDetResolvedConfig

        recovery = install_module_alias(
            "_test_recover_exp06_b2_postprocess",
            PROJECT_ROOT
            / "工程辅助运行脚本/recover_exp06_b2_postprocess.py",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work_dir = root / "runs/B2/seed_42"
            work_dir.mkdir(parents=True)
            config_path = work_dir / "resolved_mmdet_config.py"
            checkpoint = work_dir / "best_coco_segm_mAP_iter_200000.pth"
            config_path.write_text("model = {}\n", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            (work_dir / "mmdet_run_arguments.json").write_text(
                json.dumps(
                    {
                        "dataset_fingerprint": "expected-fingerprint",
                        "training_cache_signature": "original-signature",
                    }
                ),
                encoding="utf-8",
            )
            config = ExperimentConfig(
                experiment_id="EXP-06",
                profile="local_6gb",
                data_root=root / "data",
                output_root=root / "outputs",
                seed=42,
                allowed_splits=("D1", "D2", "D3"),
                parameters={"resume": False},
            )
            materialized = MaterializedDataset(
                root=root / "derived",
                fingerprint=DatasetFingerprint(
                    sha256="actual-fingerprint",
                    record_count=3,
                    splits=("D1", "D2", "D3"),
                ),
                reused=True,
                format="coco",
            )
            resolved = MMDetResolvedConfig(
                model_kind="morphoroute",
                base_config="unused.py",
                config={},
                work_dir=work_dir,
                config_path=config_path,
            )

            with self.assertRaisesRegex(ValueError, "dataset fingerprint"):
                recovery.validate_b2_recovery_context(
                    config,
                    materialized,
                    resolved,
                    checkpoint,
                )

    def test_valid_recovery_context_forbids_training(self):
        from morphoroute.core.config import ExperimentConfig
        from morphoroute.data.materialize import (
            DatasetFingerprint,
            MaterializedDataset,
        )
        from morphoroute.mmdet.config_builder import MMDetResolvedConfig

        recovery = install_module_alias(
            "_test_recover_exp06_b2_postprocess_valid",
            PROJECT_ROOT
            / "工程辅助运行脚本/recover_exp06_b2_postprocess.py",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work_dir = root / "runs/B2/seed_42"
            work_dir.mkdir(parents=True)
            config_path = work_dir / "resolved_mmdet_config.py"
            checkpoint = work_dir / "best_coco_segm_mAP_iter_200000.pth"
            config_path.write_text("model = {}\n", encoding="utf-8")
            checkpoint.write_bytes(b"checkpoint")
            fingerprint = "same-fingerprint"
            (work_dir / "mmdet_run_arguments.json").write_text(
                json.dumps(
                    {
                        "dataset_fingerprint": fingerprint,
                        "training_cache_signature": "original-signature",
                    }
                ),
                encoding="utf-8",
            )
            config = ExperimentConfig(
                experiment_id="EXP-06",
                profile="local_6gb",
                data_root=root / "data",
                output_root=root / "outputs",
                seed=42,
                allowed_splits=("D1", "D2", "D3"),
                parameters={"resume": False},
            )
            materialized = MaterializedDataset(
                root=root / "derived",
                fingerprint=DatasetFingerprint(
                    sha256=fingerprint,
                    record_count=3,
                    splits=("D1", "D2", "D3"),
                ),
                reused=True,
                format="coco",
            )
            resolved = MMDetResolvedConfig(
                model_kind="morphoroute",
                base_config="unused.py",
                config={},
                work_dir=work_dir,
                config_path=config_path,
            )

            context = recovery.validate_b2_recovery_context(
                config,
                materialized,
                resolved,
                checkpoint,
            )

        self.assertEqual(context.training_action, "forbidden")
        self.assertEqual(context.cache_signature, "original-signature")
        self.assertEqual(context.variant_id, "B2")

    def test_completed_b2_cache_skips_training_while_b3_trains_once(self):
        from dataclasses import replace

        from morphoroute.core.config import ExperimentConfig
        from morphoroute.core.io import sha256_file
        from morphoroute.data.materialize import (
            DatasetFingerprint,
            MaterializedDataset,
        )
        from morphoroute.mmdet.config_builder import build_morphoroute_config
        from morphoroute.runtime.acceleration import training_cache_signature
        from morphoroute.runtime.mmdet_adapter import (
            postprocess_trained_mmdet_checkpoint,
            train_validate_predict_mmdet,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "outputs"
            prior = root / "morphology_prior.json"
            prior.write_text("{}", encoding="utf-8")
            base_parameters = {
                "resume": False,
                "reuse_completed_training": True,
                "model": "resource_matched_shared",
                "run_name": "B2/seed_42",
                "variant_id": "B2",
                "input_size": 128,
                "max_updates": 1,
                "physical_batch": 1,
                "effective_batch": 1,
                "gradient_accumulation": 1,
                "query_count": 20,
            }
            config = ExperimentConfig(
                experiment_id="EXP-06",
                profile="local_6gb",
                data_root=root / "dataset",
                output_root=output_root,
                seed=42,
                allowed_splits=("D1", "D2", "D3"),
                parameters=base_parameters,
            )
            annotation_files = {}
            for split in ("D1", "D2", "D3"):
                path = root / "derived" / f"instances_{split}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
                annotation_files[split] = path
            materialized = MaterializedDataset(
                root=root / "derived",
                fingerprint=DatasetFingerprint(
                    sha256="f" * 64,
                    record_count=3,
                    splits=("D1", "D2", "D3"),
                ),
                reused=True,
                format="coco",
                annotation_files=annotation_files,
            )
            resolved_b2 = build_morphoroute_config(
                config,
                materialized,
                variant_id="B2",
                morphology_prior_path=prior,
            )
            resolved_b2.work_dir.mkdir(parents=True, exist_ok=True)
            resolved_b2.config_path.write_text("model = {}\n", encoding="utf-8")
            checkpoint = resolved_b2.work_dir / "best_coco_segm_mAP_iter_1.pth"
            checkpoint.write_bytes(b"existing-b2-checkpoint")
            cache_signature = training_cache_signature(
                model_kind="morphoroute",
                variant_id="B2",
                seed=42,
                dataset_sha256=materialized.fingerprint.sha256,
                morphology_prior_sha256=sha256_file(prior),
                framework_version="test-mmdet",
                parameters=config.parameters,
            )
            postprocess_backend = _TrainForbiddenBackend()
            postprocess_trained_mmdet_checkpoint(
                config,
                materialized,
                resolved_b2,
                checkpoint,
                variant_id="B2",
                cache_signature=cache_signature,
                backend=postprocess_backend,
            )

            b2_reuse_backend = _TrainForbiddenBackend()
            reuse_output = io.StringIO()
            with redirect_stdout(reuse_output):
                reused = train_validate_predict_mmdet(
                    config,
                    materialized,
                    model_kind="morphoroute",
                    variant_id="B2",
                    morphology_prior_path=prior,
                    backend=b2_reuse_backend,
                )

            b3_parameters = {
                **base_parameters,
                "run_name": "B3/seed_42",
                "variant_id": "B3",
            }
            b3_config = replace(config, parameters=b3_parameters)
            b3_backend = _CountingTrainingBackend()
            trained_b3 = train_validate_predict_mmdet(
                b3_config,
                materialized,
                model_kind="morphoroute",
                variant_id="B3",
                morphology_prior_path=prior,
                backend=b3_backend,
            )

        self.assertEqual(b2_reuse_backend.train_calls, 0)
        self.assertEqual(b2_reuse_backend.predict_splits, [])
        self.assertIn("Reusing completed MMDetection cache", reuse_output.getvalue())
        self.assertIn("morphoroute_B2", reuse_output.getvalue())
        self.assertEqual(reused.best_checkpoint, checkpoint)
        self.assertEqual(b3_backend.train_calls, 1)
        self.assertEqual(b3_backend.predict_splits, ["D2", "D3"])
        self.assertEqual(trained_b3.best_checkpoint.name, "best_coco_segm_mAP_iter_1.pth")


if __name__ == "__main__":
    unittest.main()
