from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from morphoroute.runtime.strict_fp32 import (
    POLICY_REVISION,
    apply_strict_fp32_config,
)


class MorphoRouteAmpNumericalPolicyTest(unittest.TestCase):
    """B4/B5/B6 all run under the same strict FP32 execution policy."""

    @staticmethod
    def _resolved() -> SimpleNamespace:
        return SimpleNamespace(
            optim_wrapper={
                "type": "AmpOptimWrapper",
                "loss_scale": "dynamic",
                "dtype": "float16",
                "optimizer": {"type": "SGD", "lr": 0.01},
            },
            model={},
            custom_hooks=[],
        )

    def test_b4_b5_b6_use_same_calibrated_policy(self):
        audits = []
        for variant_id in ("B4", "B5", "B6"):
            with self.subTest(variant_id=variant_id):
                resolved = apply_strict_fp32_config(
                    self._resolved(),
                    variant_id=variant_id,
                    mode="formal",
                )

                wrapper = resolved.optim_wrapper
                self.assertEqual(wrapper["type"], "StrictFiniteOptimWrapper")
                self.assertNotIn("loss_scale", wrapper)
                self.assertNotIn("dtype", wrapper)
                self.assertTrue(wrapper["clip_grad"]["error_if_nonfinite"])

                audit = resolved.numerical_stability_audit
                self.assertEqual(audit["policy_revision"], POLICY_REVISION)
                self.assertEqual(audit["variant_id"], variant_id)
                self.assertEqual(audit["mode"], "formal")
                self.assertFalse(audit["amp_enabled"])
                self.assertFalse(audit["grad_scaler_enabled"])
                self.assertEqual(
                    audit["optimizer_wrapper"],
                    "StrictFiniteOptimWrapper",
                )
                self.assertEqual(
                    audit["nonfinite_behavior"],
                    "fail_before_parameter_update",
                )
                audits.append(
                    {key: value for key, value in audit.items() if key != "variant_id"}
                )

        self.assertEqual(audits[0], audits[1])
        self.assertEqual(audits[1], audits[2])

    def test_non_ablation_variants_are_rejected(self):
        with self.assertRaises(ValueError):
            apply_strict_fp32_config(
                self._resolved(),
                variant_id="B9",
                mode="formal",
            )

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_strict_fp32_config(
                self._resolved(),
                variant_id="B4",
                mode="debug",
            )


if __name__ == "__main__":
    unittest.main()
