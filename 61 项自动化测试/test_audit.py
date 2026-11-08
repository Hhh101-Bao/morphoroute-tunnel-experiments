import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from morphoroute.data.audit import audit_records, count_unresolved_duplicates
from morphoroute.data.manifest import parse_manifest_rows


class DatasetAuditTests(unittest.TestCase):
    def _row(self, record_id, split, group_id, image_name, annotation_name):
        return {
            "record_id": record_id,
            "logical_split": split,
            "source": "DACL",
            "group_key": group_id,
            "assignment_group_id": group_id,
            "destination_image_relative": image_name,
            "destination_annotation_relative": annotation_name,
            "width": "10",
            "height": "10",
            "image_sha256": "a" * 64,
            "annotation_sha256": "b" * 64,
            "classes_present": "crack",
            "class_instances_json": '{"crack": 1}',
        }

    def test_reports_missing_files_and_cross_split_group_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.jpg").write_bytes(b"a")
            (root / "a.json").write_text('{"shapes":[]}', encoding="utf-8")
            rows = [
                self._row("a", "train", "shared", "a.jpg", "a.json"),
                self._row("b", "val", "shared", "missing.jpg", "missing.json"),
            ]
            records = parse_manifest_rows(rows, root)
            report = audit_records(records, verify_hashes=False)

            self.assertEqual(report.total_records, 2)
            self.assertFalse(report.hash_verification_performed)
            self.assertEqual(report.missing_images, ("b",))
            self.assertEqual(report.missing_annotations, ("b",))
            self.assertEqual(
                report.cross_split_groups,
                ({"group_id": "shared", "splits": ["D1", "D2"]},),
            )

    def test_reports_class_coverage_per_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                self._row("a", "train", "g1", "a.jpg", "a.json"),
                self._row("b", "val", "g2", "b.jpg", "b.json"),
            ]
            report = audit_records(parse_manifest_rows(rows, root), verify_hashes=False)
            self.assertEqual(report.class_positive_images["D1"]["crack"], 1)
            self.assertEqual(report.class_positive_images["D2"]["crack"], 1)

    def test_duplicate_audit_counts_only_unresolved_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "duplicate_audit.csv"
            path.write_text(
                "decision,kind,left_record_id,right_record_id\n"
                ",exact_duplicate_excluded,a,b\n"
                "distinct_content,near_duplicate_candidate,c,d\n"
                ",near_duplicate_candidate,e,f\n",
                encoding="utf-8",
            )
            self.assertEqual(count_unresolved_duplicates(path), 1)


if __name__ == "__main__":
    unittest.main()
