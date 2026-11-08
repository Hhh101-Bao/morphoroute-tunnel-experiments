import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from morphoroute.data.manifest import load_manifest, parse_manifest_rows


class ManifestTests(unittest.TestCase):
    def _row(self, **overrides):
        row = {
            "record_id": "sample-1",
            "logical_split": "train",
            "source": "TACK",
            "group_key": "group-1",
            "assignment_group_id": "group-1",
            "destination_image_relative": "D1/images/a.png",
            "destination_annotation_relative": "D1/annotations/a.png",
            "width": "5",
            "height": "5",
            "image_sha256": "a" * 64,
            "annotation_sha256": "b" * 64,
            "classes_present": "crack",
            "class_instances_json": '{"crack": 1}',
        }
        row.update(overrides)
        return row

    def test_loads_split_aliases_as_d1_d2_d3(self):
        fixture = Path(__file__).resolve().parent / "fixtures" / "manifest_small.csv"
        records = load_manifest(fixture, PROJECT_ROOT)
        self.assertEqual([record.dataset_id for record in records], ["D1", "D2", "D3"])

    def test_rejects_duplicate_record_id(self):
        with self.assertRaisesRegex(ValueError, "duplicate record_id"):
            parse_manifest_rows([self._row(), self._row()], PROJECT_ROOT)

    def test_rejects_destination_path_escape(self):
        with self.assertRaisesRegex(ValueError, "escapes"):
            parse_manifest_rows(
                [self._row(destination_image_relative="../outside.png")],
                PROJECT_ROOT,
            )

    def test_preserves_class_counts_and_group_id(self):
        record = parse_manifest_rows([self._row()], PROJECT_ROOT)[0]
        self.assertEqual(record.group_id, "group-1")
        self.assertEqual(record.classes_present, ("crack",))
        self.assertEqual(record.class_instances, {"crack": 1})


if __name__ == "__main__":
    unittest.main()
