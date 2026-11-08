import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from morphoroute.models.variants import (
    VARIANT_IDS,
    build_variant_config,
    diff_keys,
)


class VariantTests(unittest.TestCase):
    def test_all_b0_through_b11_exist(self):
        self.assertEqual(VARIANT_IDS, tuple(f"B{index}" for index in range(12)))

    def test_b10_only_disables_dual_activation_from_b9(self):
        self.assertEqual(
            diff_keys(build_variant_config("B9"), build_variant_config("B10")),
            {"router.dual_activation"},
        )

    def test_b11_only_enables_pcgrad_from_b9(self):
        self.assertEqual(
            diff_keys(build_variant_config("B9"), build_variant_config("B11")),
            {"optimizer.pcgrad"},
        )

    def test_unknown_variant_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "variant"):
            build_variant_config("B12")


if __name__ == "__main__":
    unittest.main()
