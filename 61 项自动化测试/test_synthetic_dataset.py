import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from morphoroute.data.audit import audit_records
from morphoroute.data.manifest import load_manifest
from morphoroute.aliasing import install_module_alias
from morphoroute.core.layout import ProjectLayout


CREATE_SYNTHETIC = ProjectLayout.discover(PROJECT_ROOT).script_file(
    "五类合成测试数据生成器.py",
    "create_synthetic_dataset.py",
)
create_synthetic_dataset = install_module_alias(
    "_test_create_synthetic_dataset",
    CREATE_SYNTHETIC,
).create_synthetic_dataset


class SyntheticDatasetTests(unittest.TestCase):
    def test_generator_creates_leak_free_five_class_four_split_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = create_synthetic_dataset(Path(tmp) / "synthetic")
            records = load_manifest(
                root / "metadata" / "split_manifest.csv",
                root,
            )
            report = audit_records(records, verify_hashes=True)
            self.assertEqual(report.total_records, 20)
            self.assertEqual(
                report.split_counts,
                {"D1": 5, "D2": 5, "D3": 5, "D4": 5},
            )
            self.assertTrue(report.passed)
            for split in ("D1", "D2", "D3", "D4"):
                self.assertEqual(len(report.class_positive_images[split]), 5)


if __name__ == "__main__":
    unittest.main()
