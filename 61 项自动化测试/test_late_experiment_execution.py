from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class LateExperimentExecutionTests(unittest.TestCase):
    def test_quantification_without_scale_never_emits_physical_units(self):
        from morphoroute.data.annotations import encode_uncompressed_rle
        from morphoroute.analysis.quantification_pipeline import run_quantification

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mask = np.zeros((10, 10), dtype=bool)
            mask[4, 1:9] = True
            predictions = root / "predictions.json"
            predictions.write_text(
                json.dumps(
                    {
                        "images": [
                            {
                                "image_id": "image-1",
                                "predictions": [
                                    {
                                        "class_name": "crack",
                                        "mask_rle": encode_uncompressed_rle(mask),
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = run_quantification(
                predictions,
                metadata_path=None,
                output_dir=root / "outputs",
            )

            self.assertFalse(result.physical_complete)
            self.assertTrue(result.template_path.is_file())
            self.assertTrue(
                all(row["unit"] == "pixel" for row in result.measurements)
            )

    def test_final_aggregation_rejects_smoke_runs(self):
        from morphoroute.analysis.final_statistics import (
            aggregate_final_results,
        )

        formal = [{"run_status": "completed", "smoke_test": False}]
        smoke = [{"run_status": "completed", "smoke_test": True}]

        with self.assertRaisesRegex(ValueError, "smoke"):
            aggregate_final_results([*formal, *smoke])


if __name__ == "__main__":
    unittest.main()
