from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from morphoroute.runtime.strict_fp32 import (  # noqa: E402
    POLICY_REVISION,
    NonFiniteTrainingError,
    apply_strict_fp32_config,
    configure_torch_strict_fp32,
    guarded_linear_sum_assignment,
    install_strict_finite_optim_wrapper,
    validate_finite_gradients,
    validate_finite_loss,
)


class StrictFP32PolicyTests(unittest.TestCase):
    def _configuration(self):
        return {
            "model": {"type": "MorphoRouteMask2Former"},
            "train_dataloader": {"batch_size": 1, "num_workers": 4},
            "train_cfg": {"max_iters": 400000, "val_interval": 40000},
            "optim_wrapper": {
                "type": "AmpOptimWrapper",
                "accumulative_counts": 8,
                "loss_scale": {
                    "init_scale": 2048.0,
                    "growth_interval": 1000000,
                },
                "dtype": "float16",
                "optimizer": {
                    "type": "AdamW",
                    "lr": 1e-4,
                    "weight_decay": 0.05,
                },
                "clip_grad": {"max_norm": 0.1, "norm_type": 2},
            },
            "param_scheduler": [{"type": "LinearLR", "end": 8000}],
            "randomness": {"seed": 42, "deterministic": False},
        }

    def test_policy_changes_only_numerical_execution_fields(self):
        configuration = self._configuration()
        original = copy.deepcopy(configuration)

        result = apply_strict_fp32_config(
            configuration,
            variant_id="B4",
            mode="formal",
        )

        self.assertIs(result, configuration)
        wrapper = configuration["optim_wrapper"]
        self.assertEqual(wrapper["type"], "StrictFiniteOptimWrapper")
        self.assertNotIn("loss_scale", wrapper)
        self.assertNotIn("dtype", wrapper)
        self.assertTrue(wrapper["clip_grad"]["error_if_nonfinite"])
        self.assertEqual(
            configuration["numerical_stability_audit"]["policy_revision"],
            POLICY_REVISION,
        )
        self.assertEqual(
            configuration["numerical_stability_audit"]["variant_id"],
            "B4",
        )
        self.assertEqual(configuration["model"], original["model"])
        self.assertEqual(
            configuration["train_dataloader"],
            original["train_dataloader"],
        )
        self.assertEqual(configuration["train_cfg"], original["train_cfg"])
        self.assertEqual(
            configuration["optim_wrapper"]["optimizer"],
            original["optim_wrapper"]["optimizer"],
        )
        self.assertEqual(
            configuration["optim_wrapper"]["accumulative_counts"],
            original["optim_wrapper"]["accumulative_counts"],
        )
        self.assertEqual(
            configuration["param_scheduler"],
            original["param_scheduler"],
        )
        self.assertEqual(configuration["randomness"], original["randomness"])

    def test_b4_b5_b6_have_identical_policy_except_variant_identifier(self):
        audits = []
        for variant in ("B4", "B5", "B6"):
            configuration = self._configuration()
            apply_strict_fp32_config(
                configuration,
                variant_id=variant,
                mode="formal",
            )
            audit = copy.deepcopy(configuration["numerical_stability_audit"])
            self.assertEqual(audit.pop("variant_id"), variant)
            audits.append(audit)
        self.assertEqual(audits[0], audits[1])
        self.assertEqual(audits[1], audits[2])

    def test_rejects_out_of_scope_variant_and_mode(self):
        with self.assertRaises(ValueError):
            apply_strict_fp32_config(
                self._configuration(),
                variant_id="B9",
                mode="formal",
            )
        with self.assertRaises(ValueError):
            apply_strict_fp32_config(
                self._configuration(),
                variant_id="B4",
                mode="resume",
            )

    def test_torch_fp32_flags_are_enforced(self):
        old_matmul = torch.backends.cuda.matmul.allow_tf32
        old_cudnn = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            audit = configure_torch_strict_fp32(torch)
            self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
            self.assertEqual(audit["matmul_precision"], "highest")
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul
            torch.backends.cudnn.allow_tf32 = old_cudnn

    def test_nonfinite_loss_is_rejected(self):
        validate_finite_loss(torch.tensor(1.0))
        with self.assertRaises(NonFiniteTrainingError):
            validate_finite_loss(torch.tensor(float("nan")))

    def test_nonfinite_gradient_is_rejected_before_parameter_change(self):
        parameter = torch.nn.Parameter(torch.tensor([3.0]))
        parameter.grad = torch.tensor([float("inf")])
        original = parameter.detach().clone()
        with self.assertRaises(NonFiniteTrainingError):
            validate_finite_gradients([parameter])
        self.assertTrue(torch.equal(parameter.detach(), original))

    def test_invalid_hungarian_matrix_is_saved_and_never_forwarded(self):
        called = False

        def original_assignment(matrix, *args, **kwargs):
            nonlocal called
            called = True
            return np.array([0]), np.array([0])

        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            matrix = np.array([[0.0, np.nan]], dtype=np.float32)
            with self.assertRaises(NonFiniteTrainingError):
                guarded_linear_sum_assignment(
                    original_assignment,
                    artifact_dir,
                    matrix,
                )
            self.assertFalse(called)
            reports = list(artifact_dir.glob("invalid_hungarian_cost_*.json"))
            arrays = list(artifact_dir.glob("invalid_hungarian_cost_*.npy"))
            self.assertEqual(len(reports), 1)
            self.assertEqual(len(arrays), 1)
            report = json.loads(reports[0].read_text(encoding="utf-8"))
            self.assertEqual(report["event"], "invalid_hungarian_cost_matrix")
            self.assertEqual(report["nan"], 1)

    def test_amp_loss_scaler_state_is_removed_before_strict_resume(self):
        from morphoroute.runtime.strict_fp32 import (
            sanitize_optimizer_checkpoint_state,
        )

        original = {
            "loss_scaler": {"scale": 1024.0},
            "state": {0: {"step": torch.tensor(40000)}},
            "param_groups": [{"params": [0]}],
        }
        sanitized, removed = sanitize_optimizer_checkpoint_state(original)
        self.assertTrue(removed)
        self.assertNotIn("loss_scaler", sanitized)
        self.assertIn("state", sanitized)
        self.assertIn("loss_scaler", original)

    @unittest.skipUnless(
        importlib.util.find_spec("mmengine") is not None,
        "MMEngine is verified on the AutoDL target environment",
    )
    def test_real_mmengine_wrapper_updates_finite_and_rejects_inf_gradient(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper_type = install_strict_finite_optim_wrapper(directory)
            parameter = torch.nn.Parameter(torch.tensor([2.0]))
            optimizer = torch.optim.SGD([parameter], lr=0.1)
            wrapper = wrapper_type(
                optimizer=optimizer,
                clip_grad={
                    "max_norm": 1.0,
                    "norm_type": 2,
                    "error_if_nonfinite": True,
                },
            )
            wrapper.update_params(parameter.square().sum())
            self.assertTrue(torch.isfinite(parameter).all())

            parameter.grad = torch.tensor([float("inf")])
            before = parameter.detach().clone()
            with self.assertRaises(NonFiniteTrainingError):
                wrapper.step()
            self.assertTrue(torch.equal(parameter.detach(), before))


if __name__ == "__main__":
    unittest.main()
