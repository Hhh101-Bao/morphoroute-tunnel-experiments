from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


try:
    from morphoroute.runtime.pcgrad_fp32 import (  # noqa: E402
        POLICY_REVISION,
        apply_b11_true_fp32_config,
        sanitize_pcgrad_fp32_checkpoint_state,
    )
except ModuleNotFoundError:
    module_path = (
        SOURCE_ROOT
        / "morphoroute"
        / "训练适配、断点恢复和资源记录"
        / "pcgrad_fp32.py"
    )
    spec = importlib.util.spec_from_file_location(
        "exp09_b11_pcgrad_fp32_test_module",
        module_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    POLICY_REVISION = module.POLICY_REVISION
    apply_b11_true_fp32_config = module.apply_b11_true_fp32_config
    sanitize_pcgrad_fp32_checkpoint_state = (
        module.sanitize_pcgrad_fp32_checkpoint_state
    )


class Exp09B11TrueFp32PolicyTests(unittest.TestCase):
    def setUp(self):
        self.configuration = {
            "model": {
                "panoptic_fusion_head": {
                    "type": "MaskFormerFusionHead",
                },
                "data_preprocessor": {
                    "batch_augments": [
                        {
                            "type": "BatchFixedSizePad",
                            "size": (640, 640),
                            "pad_mask": True,
                        }
                    ]
                }
            },
            "optim_wrapper": {
                "type": "PCGradAmpOptimWrapper",
                "accumulative_counts": 8,
                "loss_scale": {
                    "init_scale": 2048,
                    "growth_interval": 2000,
                },
                "clip_grad": {"max_norm": 0.1, "norm_type": 2},
                "optimizer": {
                    "type": "AdamW",
                    "lr": 0.0001,
                    "weight_decay": 0.05,
                },
            },
            "train_cfg": {"max_iters": 400000},
            "param_scheduler": [{"type": "MultiStepLR"}],
            "numerical_stability_audit": {
                "amp_enabled": False,
                "policy_revision": "morphoroute_amp_fp32_v1",
            },
        }

    def test_b11_replaces_only_execution_wrapper(self):
        before_optimizer = copy.deepcopy(
            self.configuration["optim_wrapper"]["optimizer"]
        )
        before_train = copy.deepcopy(self.configuration["train_cfg"])
        before_scheduler = copy.deepcopy(self.configuration["param_scheduler"])

        apply_b11_true_fp32_config(
            self.configuration,
            variant_id="B11",
        )

        wrapper = self.configuration["optim_wrapper"]
        self.assertEqual(wrapper["type"], "PCGradFP32OptimWrapper")
        self.assertEqual(wrapper["accumulative_counts"], 8)
        self.assertEqual(wrapper["clip_grad"]["max_norm"], 0.1)
        self.assertTrue(wrapper["clip_grad"]["error_if_nonfinite"])
        self.assertNotIn("loss_scale", wrapper)
        self.assertNotIn("dtype", wrapper)
        self.assertEqual(wrapper["optimizer"], before_optimizer)
        self.assertEqual(self.configuration["train_cfg"], before_train)
        self.assertEqual(
            self.configuration["param_scheduler"], before_scheduler
        )
        audit = self.configuration["numerical_stability_audit"]
        self.assertEqual(audit["policy_revision"], POLICY_REVISION)
        self.assertFalse(audit["amp_enabled"])
        self.assertFalse(audit["grad_scaler_enabled"])
        self.assertEqual(
            audit["optimizer_wrapper"], "PCGradFP32OptimWrapper"
        )
        augment = self.configuration["model"]["data_preprocessor"][
            "batch_augments"
        ][0]
        self.assertEqual(
            augment["type"], "MorphoRouteSafeBatchFixedSizePad"
        )
        self.assertEqual(augment["size"], (640, 640))
        self.assertEqual(
            audit["batch_padding_policy"],
            "preserve_oversized_dimension_and_record_actual_shape",
        )
        self.assertEqual(
            audit["empty_morphology_policy"],
            "exclude_route_supervision_without_fabricating_target",
        )
        self.assertEqual(
            self.configuration["model"]["panoptic_fusion_head"]["type"],
            "MorphoRouteMemoryEfficientMaskFormerFusionHead",
        )
        self.assertEqual(
            audit["instance_postprocess_policy"],
            "stream_queries_without_batched_fp32_mask_copies",
        )

    def test_non_b11_is_rejected_without_mutation(self):
        original = copy.deepcopy(self.configuration)
        with self.assertRaisesRegex(ValueError, "limited to B11"):
            apply_b11_true_fp32_config(
                self.configuration,
                variant_id="B10",
            )
        self.assertEqual(self.configuration, original)

    def test_incompatible_wrapper_is_rejected(self):
        self.configuration["optim_wrapper"]["type"] = "OptimWrapper"
        with self.assertRaisesRegex(RuntimeError, "unexpected optimizer"):
            apply_b11_true_fp32_config(
                self.configuration,
                variant_id="B11",
            )

    def test_checkpoint_sanitizer_removes_only_amp_scaler(self):
        state = {
            "optimizer": {"state": {7: {"step": 40000}}},
            "loss_scaler": {"scale": 1024.0, "_growth_tracker": 2215},
            "base_param_settings": {"lr": 0.0001},
        }
        sanitized, removed = sanitize_pcgrad_fp32_checkpoint_state(state)
        self.assertTrue(removed)
        self.assertNotIn("loss_scaler", sanitized)
        self.assertEqual(sanitized["optimizer"], state["optimizer"])
        self.assertEqual(
            sanitized["base_param_settings"], state["base_param_settings"]
        )
        self.assertIn("loss_scaler", state)

    def test_runner_contract_is_b11_scoped_and_true_fp32(self):
        runner = (
            PROJECT_ROOT
            / "工程辅助运行脚本"
            / "run_exp09_b11_true_fp32.py"
        )
        text = runner.read_text(encoding="utf-8")
        required = (
            "PCGradFP32OptimWrapper",
            "configure_torch_strict_fp32",
            "install_pcgrad_fp32_optim_wrapper",
            "install_hungarian_finite_guard",
            "variant != \"B11\"",
            "runpy.run_path",
        )
        for token in required:
            self.assertIn(token, text)

    def test_wrapper_has_no_amp_or_scaler_execution_path(self):
        source = (
            PROJECT_ROOT
            / "src"
            / "morphoroute"
            / "训练适配、断点恢复和资源记录"
            / "pcgrad_fp32.py"
        ).read_text(encoding="utf-8")
        class_body = source.split(
            "class PCGradFP32OptimWrapper(OptimWrapper):", 1
        )[1]
        self.assertNotIn("AmpOptimWrapper", class_body)
        self.assertNotIn("loss_scaler.scale", class_body)
        self.assertNotIn("torch.autocast", class_body)
        self.assertIn("self.scale_loss(loss)", class_body)
        self.assertIn("validate_finite_loss(first_loss)", class_body)
        self.assertIn("_validate_gradient_values(first", class_body)
        self.assertIn("project_conflicting_pair(left, right)", class_body)

    def test_default_b11_builder_and_registry_use_true_fp32(self):
        builder = (
            PROJECT_ROOT
            / "src"
            / "morphoroute"
            / "MMDetection 配置与扩展组件"
            / "config_builder.py"
        ).read_text(encoding="utf-8")
        model_init = (
            PROJECT_ROOT
            / "src"
            / "morphoroute"
            / "MMDetection 配置与扩展组件"
            / "models"
            / "__init__.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'result["optim_wrapper"] = {"type": "PCGradFP32OptimWrapper"}',
            builder,
        )
        self.assertIn("install_pcgrad_fp32_optim_wrapper", model_init)
        self.assertIn(
            "MorphoRouteMemoryEfficientMaskFormerFusionHead", model_init
        )


if __name__ == "__main__":
    unittest.main()
