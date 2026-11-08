from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class BudgetGateTests(unittest.TestCase):
    def test_patience_starts_at_gate_instead_of_consuming_pre_gate_plateaus(self):
        from morphoroute.runtime.acceleration import (
            BudgetGateMonitor,
            BudgetGatePolicy,
        )

        monitor = BudgetGateMonitor(
            BudgetGatePolicy(
                enabled=True,
                start_update=30_000,
                metric="coco/segm_mAP",
                patience=2,
                min_delta=0.001,
            )
        )

        for update, score in ((5_000, 0.10), (10_000, 0.09), (25_000, 0.08)):
            decision = monitor.observe(update=update, value=score)
            self.assertFalse(decision.should_stop)

        at_gate = monitor.observe(update=30_000, value=0.08)
        self.assertFalse(at_gate.should_stop)
        self.assertEqual(at_gate.bad_validations, 1)

        after_second_gated_plateau = monitor.observe(
            update=35_000,
            value=0.08,
        )
        self.assertTrue(after_second_gated_plateau.should_stop)
        self.assertEqual(after_second_gated_plateau.bad_validations, 2)

    def test_improvement_at_gate_keeps_training(self):
        from morphoroute.runtime.acceleration import (
            BudgetGateMonitor,
            BudgetGatePolicy,
        )

        monitor = BudgetGateMonitor(
            BudgetGatePolicy(True, 30_000, "coco/segm_mAP", 2, 0.001)
        )
        monitor.observe(update=25_000, value=0.10)

        decision = monitor.observe(update=30_000, value=0.102)

        self.assertFalse(decision.should_stop)
        self.assertEqual(decision.best_value, 0.102)

    def test_release_flag_removes_gate_from_mmdet_config(self):
        from test_mmdet_end_to_end import _config, _dataset
        from morphoroute.mmdet.config_builder import build_mask2former_config

        gate = {
            "enabled": True,
            "start_update": 30_000,
            "metric": "coco/segm_mAP",
            "patience": 2,
            "min_delta": 0.001,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active = build_mask2former_config(
                _config(root / "active", max_updates=50_000, budget_gate=gate),
                _dataset(root / "active"),
            )
            released = build_mask2former_config(
                _config(
                    root / "released",
                    max_updates=50_000,
                    budget_gate=gate,
                    release_budget_gate=True,
                ),
                _dataset(root / "released"),
            )

        active_hooks = active.config.get("custom_hooks", [])
        released_hooks = released.config.get("custom_hooks", [])
        gate_hook = next(
            hook for hook in active_hooks
            if hook["type"] == "D2BudgetGateHook"
        )
        self.assertEqual(gate_hook["start_iteration"], 30_000 * 8)
        self.assertFalse(
            any(hook["type"] == "D2BudgetGateHook" for hook in released_hooks)
        )

    def test_local_performance_mode_is_explicit_and_auditable(self):
        from test_mmdet_end_to_end import _config, _dataset
        from morphoroute.mmdet.config_builder import build_mask2former_config

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resolved = build_mask2former_config(
                _config(
                    root,
                    performance_mode=True,
                    dataloader_workers=4,
                ),
                _dataset(root),
            )

        self.assertFalse(resolved.config["randomness"]["deterministic"])
        self.assertTrue(resolved.config["env_cfg"]["cudnn_benchmark"])
        self.assertEqual(resolved.config["train_dataloader"]["num_workers"], 4)
        self.assertTrue(resolved.config["train_dataloader"]["persistent_workers"])
        self.assertTrue(resolved.config["train_dataloader"]["pin_memory"])
        self.assertEqual(
            resolved.config["train_dataloader"]["prefetch_factor"],
            2,
        )
        self.assertTrue(
            resolved.config["acceleration_audit"]["performance_mode"]
        )
        self.assertTrue(resolved.config["acceleration_audit"]["pin_memory"])
        self.assertEqual(
            resolved.config["acceleration_audit"]["prefetch_factor"],
            2,
        )

    def test_zero_worker_dataloader_omits_prefetch_and_persistence(self):
        from test_mmdet_end_to_end import _config, _dataset
        from morphoroute.mmdet.config_builder import build_mask2former_config

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resolved = build_mask2former_config(
                _config(
                    root,
                    performance_mode=True,
                    dataloader_workers=0,
                ),
                _dataset(root),
            )

        dataloader = resolved.config["train_dataloader"]
        self.assertFalse(dataloader["persistent_workers"])
        self.assertNotIn("prefetch_factor", dataloader)


class VariantSeedPolicyTests(unittest.TestCase):
    def test_core_variants_get_confirmation_seeds(self):
        from morphoroute.core.config import ExperimentConfig
        from morphoroute.training.variant_execution import resolve_variant_seeds

        config = ExperimentConfig.from_mapping(
            {
                "experiment_id": "EXP-09",
                "profile": "local_6gb",
                "data_root": "data",
                "output_root": "outputs",
                "seed": 42,
                "allowed_splits": ["D1", "D2", "D3"],
                "parameters": {
                    "model": "ablation_B0_B11",
                    "screening_seeds": [42],
                    "confirmation_seeds": [42, 43, 44, 45, 46],
                    "confirmation_variants": ["B0", "B9", "B10", "B11"],
                },
            }
        )

        selected = resolve_variant_seeds(config, ("B0", "B1", "B9"))

        self.assertEqual(selected["B0"], (42, 43, 44, 45, 46))
        self.assertEqual(selected["B1"], (42,))
        self.assertEqual(selected["B9"], (42, 43, 44, 45, 46))

    def test_only_promoted_variants_release_the_budget_gate(self):
        from morphoroute.core.config import ExperimentConfig
        from morphoroute.training.variant_execution import _variant_config

        config = ExperimentConfig.from_mapping(
            {
                "experiment_id": "EXP-09",
                "profile": "local_6gb",
                "data_root": "data",
                "output_root": "outputs",
                "seed": 42,
                "allowed_splits": ["D1", "D2", "D3"],
                "parameters": {
                    "model": "ablation_B0_B11",
                    "release_budget_gate": False,
                    "release_budget_gate_variants": ["B0", "B9"],
                },
            }
        )

        b9 = _variant_config(config, "B9", 42)
        b1 = _variant_config(config, "B1", 42)

        self.assertTrue(b9.parameters["release_budget_gate"])
        self.assertFalse(b1.parameters["release_budget_gate"])


class TrainingCacheSignatureTests(unittest.TestCase):
    def test_signature_changes_with_dataset_seed_or_training_parameter(self):
        from morphoroute.runtime.acceleration import training_cache_signature

        parameters = {
            "model": "morphoroute_b9",
            "input_size": 640,
            "max_updates": 50_000,
            "run_name": "ignored/output/path",
            "result_status": "ignored-report-label",
        }
        baseline = training_cache_signature(
            model_kind="morphoroute",
            variant_id="B9",
            seed=42,
            dataset_sha256="dataset-a",
            morphology_prior_sha256="prior-a",
            framework_version="3.3.0",
            parameters=parameters,
        )
        renamed = training_cache_signature(
            model_kind="morphoroute",
            variant_id="B9",
            seed=42,
            dataset_sha256="dataset-a",
            morphology_prior_sha256="prior-a",
            framework_version="3.3.0",
            parameters={**parameters, "run_name": "another/path"},
        )
        different_data = training_cache_signature(
            model_kind="morphoroute",
            variant_id="B9",
            seed=42,
            dataset_sha256="dataset-b",
            morphology_prior_sha256="prior-a",
            framework_version="3.3.0",
            parameters=parameters,
        )
        different_report_model_label = training_cache_signature(
            model_kind="morphoroute",
            variant_id="B9",
            seed=42,
            dataset_sha256="dataset-a",
            morphology_prior_sha256="prior-a",
            framework_version="3.3.0",
            parameters={**parameters, "model": "ablation_B0_B11"},
        )
        different_budget = training_cache_signature(
            model_kind="morphoroute",
            variant_id="B9",
            seed=42,
            dataset_sha256="dataset-a",
            morphology_prior_sha256="prior-a",
            framework_version="3.3.0",
            parameters={**parameters, "max_updates": 30_000},
        )

        self.assertEqual(baseline, renamed)
        self.assertEqual(baseline, different_report_model_label)
        self.assertNotEqual(baseline, different_data)
        self.assertNotEqual(baseline, different_budget)

    def test_identical_completed_run_skips_second_training_and_prediction(self):
        from test_mmdet_end_to_end import _config, _dataset
        from morphoroute.evaluation.schema import PredictionBundle
        from morphoroute.runtime.mmdet_adapter import train_validate_predict_mmdet

        class CountingBackend:
            version = "cache-test"

            def __init__(self):
                self.train_calls = 0
                self.predict_calls = 0

            def train(self, resolved, *, resume):
                del resume
                self.train_calls += 1
                checkpoint = resolved.work_dir / "best_coco_segm_mAP.pth"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.write_bytes(b"checkpoint")
                return checkpoint

            def predict(self, checkpoint, resolved, materialized, *, split):
                del checkpoint, resolved, materialized
                self.predict_calls += 1
                return PredictionBundle(
                    experiment_id="EXP-05",
                    split=split,
                    class_names=("crack",),
                    images=(),
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root, reuse_completed_training=True)
            dataset = _dataset(root)
            backend = CountingBackend()

            first = train_validate_predict_mmdet(
                config,
                dataset,
                model_kind="mask2former",
                backend=backend,
            )
            second = train_validate_predict_mmdet(
                config,
                dataset,
                model_kind="mask2former",
                backend=backend,
            )

            self.assertEqual(backend.train_calls, 1)
            self.assertEqual(backend.predict_calls, 2)
            self.assertEqual(first.best_checkpoint, second.best_checkpoint)
            self.assertTrue(
                second.predictions.metadata.get("reused_training_cache")
            )


if __name__ == "__main__":
    unittest.main()
