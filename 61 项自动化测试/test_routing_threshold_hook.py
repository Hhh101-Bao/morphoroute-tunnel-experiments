from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class RoutingThresholdSelectionTests(unittest.TestCase):
    def test_selection_maximizes_d2_map_and_uses_sparser_tie_break(self):
        from morphoroute.mmdet.models.routing_threshold_hook import (
            choose_routing_threshold,
        )

        selected = choose_routing_threshold(
            [
                {"threshold": 0.50, "coco/segm_mAP": 0.40},
                {"threshold": 0.60, "coco/segm_mAP": 0.43},
                {"threshold": 0.75, "coco/segm_mAP": 0.43},
            ],
            metric="coco/segm_mAP",
        )

        self.assertEqual(selected, 0.60)


if __name__ == "__main__":
    unittest.main()
