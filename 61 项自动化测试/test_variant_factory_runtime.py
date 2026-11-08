from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class VariantFactoryRuntimeTests(unittest.TestCase):
    def test_all_variants_build_registered_runtime_config(self):
        from morphoroute.mmdet.models.variant_factory import (
            build_variant_model_cfg,
        )
        from morphoroute.models.variants import VARIANT_IDS

        for variant_id in VARIANT_IDS:
            with self.subTest(variant_id=variant_id):
                config = build_variant_model_cfg(
                    variant_id,
                    in_channels=256,
                    query_dim=256,
                )
                self.assertEqual(
                    config["type"],
                    "MorphoRouteMask2FormerHead",
                )
                self.assertEqual(config["variant_id"], variant_id)


if __name__ == "__main__":
    unittest.main()
