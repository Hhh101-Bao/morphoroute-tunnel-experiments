from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
RUNNER_PATH = (
    PROJECT_ROOT / "工程辅助运行脚本" / "run_exp07_strict_fp32.py"
)
VERIFIER_PATH = (
    PROJECT_ROOT
    / "工程辅助运行脚本"
    / "verify_exp07_strict_fp32_checkpoint.py"
)
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_module("exp07_strict_fp32_runner", RUNNER_PATH)
verifier = load_module("exp07_strict_fp32_verifier", VERIFIER_PATH)


class StrictFP32RunnerTests(unittest.TestCase):
    def _base_configuration(self):
        return {
            "train_cfg": {"max_iters": 400000},
            "optim_wrapper": {
                "type": "AmpOptimWrapper",
                "loss_scale": 2048.0,
                "optimizer": {"type": "AdamW", "lr": 1e-4},
                "clip_grad": {"max_norm": 0.1, "norm_type": 2},
            },
        }

    def test_configuration_wrapper_applies_formal_policy(self):
        resolved = SimpleNamespace(
            model_kind="morphoroute_B5",
            config_path=None,
        )
        configuration = self._base_configuration()
        wrapped = runner.build_strict_configuration_wrapper(
            lambda _: configuration,
            mode="formal",
            stop_iter=None,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            result = wrapped(resolved)
        self.assertIs(result, configuration)
        self.assertEqual(
            result["optim_wrapper"]["type"],
            "StrictFiniteOptimWrapper",
        )
        self.assertEqual(result["train_cfg"]["max_iters"], 400000)
        self.assertIn("variant=B5", output.getvalue())
        self.assertIn("mode=formal", output.getvalue())

    def test_probe_configuration_is_limited_to_b4_and_60000(self):
        configuration = self._base_configuration()
        wrapped = runner.build_strict_configuration_wrapper(
            lambda _: configuration,
            mode="probe",
            stop_iter=60000,
        )
        result = wrapped(
            SimpleNamespace(model_kind="morphoroute_B4", config_path=None)
        )
        self.assertEqual(result["train_cfg"]["max_iters"], 60000)
        self.assertTrue(result["default_hooks"]["checkpoint"]["save_last"])

        with self.assertRaises(RuntimeError):
            wrapped(
                SimpleNamespace(
                    model_kind="morphoroute_B5",
                    config_path=None,
                )
            )

    def test_probe_stop_must_cross_prior_failure_region(self):
        with self.assertRaises(ValueError):
            runner.validate_mode("probe", 50000)
        runner.validate_mode("probe", 60000)
        runner.validate_mode("formal", None)

    def test_checkpoint_verifier_accepts_finite_non_amp_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "finite.pth"
            torch.save(
                {
                    "state_dict": {"weight": torch.tensor([1.0, 2.0])},
                    "optimizer": {"state": {}, "param_groups": []},
                },
                path,
            )
            report = verifier.verify_checkpoint(path)
            self.assertEqual(report["nonfinite_tensor_count"], 0)
            self.assertFalse(report["contains_loss_scaler"])

    def test_checkpoint_verifier_rejects_nonfinite_or_amp_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pth"
            torch.save(
                {
                    "state_dict": {"weight": torch.tensor([float("nan")])},
                    "optimizer": {
                        "loss_scaler": {"scale": 1024.0},
                        "state": {},
                    },
                },
                path,
            )
            with self.assertRaises(verifier.CheckpointVerificationError):
                verifier.verify_checkpoint(path)

    def test_diagnostic_verifier_accepts_finite_amp_state_but_reports_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostic_amp.pth"
            torch.save(
                {
                    "state_dict": {"weight": torch.tensor([1.0])},
                    "optimizer": {
                        "loss_scaler": {"scale": 1024.0},
                        "state": {},
                    },
                },
                path,
            )
            report = verifier.verify_checkpoint(
                path,
                allow_amp_diagnostic=True,
            )
            self.assertTrue(report["contains_loss_scaler"])
            self.assertTrue(report["amp_state_diagnostic_only"])


if __name__ == "__main__":
    unittest.main()
