import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from morphoroute.visualization.tables import save_result_bundle


class ResultBundleTests(unittest.TestCase):
    def test_writes_consistent_json_csv_and_xlsx(self):
        records = [
            {"model": "B0", "macro_ap": 0.4, "missing_class_ap": None},
            {"model": "B9", "macro_ap": 0.5, "missing_class_ap": None},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            outputs = save_result_bundle(records, Path(tmp) / "comparison")
            self.assertEqual(set(outputs), {"json", "csv", "xlsx"})
            self.assertTrue(all(path.is_file() for path in outputs.values()))
            payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
            self.assertEqual(payload[1]["model"], "B9")
            self.assertIsNone(payload[0]["missing_class_ap"])


if __name__ == "__main__":
    unittest.main()
