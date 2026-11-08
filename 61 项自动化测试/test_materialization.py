from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


CLASS_MAPPING = {
    "class_order": [
        "crack",
        "water_leakage",
        "spalling",
        "efflorescence",
        "exposed_rebar",
    ],
    "DACL": {"Crack": "crack"},
    "TACK_mask_values": {},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(root: Path):
    from morphoroute.data.manifest import SampleRecord

    image = root / "image.png"
    annotation = root / "annotation.json"
    Image.new("RGB", (8, 8), "white").save(image)
    annotation.write_text(
        json.dumps(
            {
                "imageWidth": 8,
                "imageHeight": 8,
                "shapes": [
                    {
                        "label": "Crack",
                        "points": [[1, 1], [6, 1], [6, 2], [1, 2]],
                    },
                    {
                        "label": "Graffiti",
                        "points": [[2, 3], [5, 3], [5, 5], [2, 5]],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return SampleRecord(
        record_id="DACL-0001",
        dataset_id="D1",
        source="DACL",
        group_id="tunnel-A/sequence-1",
        image_path=image,
        annotation_path=annotation,
        width=8,
        height=8,
        image_sha256=_sha256(image),
        annotation_sha256=_sha256(annotation),
        classes_present=("crack",),
        class_instances={"crack": 1},
    )


class MaterializationTests(unittest.TestCase):
    def test_coco_preserves_instances_categories_and_provenance(self):
        from morphoroute.data.coco_dataset import ensure_coco_dataset

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = _record(root)

            result = ensure_coco_dataset(
                [record],
                CLASS_MAPPING,
                root / "derived",
                ("D1",),
            )
            payload = json.loads(
                result.annotation_files["D1"].read_text(encoding="utf-8")
            )

            self.assertEqual(
                {item["name"] for item in payload["categories"]},
                set(CLASS_MAPPING["class_order"]),
            )
            self.assertEqual(len(payload["annotations"]), 1)
            self.assertEqual(payload["images"][0]["group_id"], record.group_id)
            self.assertEqual(
                payload["images"][0]["record_id"],
                record.record_id,
            )

    def test_dacl_exif_dimensions_drive_coco_and_yolo_coordinates(self):
        from morphoroute.data.coco_dataset import ensure_coco_dataset
        from morphoroute.data.manifest import SampleRecord
        from morphoroute.data.yolo_dataset import ensure_yolo_dataset

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "oriented.jpg"
            annotation = root / "annotation.json"
            exif = Image.Exif()
            exif[274] = 6
            Image.new("RGB", (8, 12), "white").save(image, exif=exif)
            annotation.write_text(
                json.dumps(
                    {
                        "imageWidth": 12,
                        "imageHeight": 8,
                        "shapes": [
                            {"label": "Crack", "points": [[3, 1], [6, 1], [6, 2], [3, 2]]}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            record = SampleRecord(
                record_id="DACL-EXIF",
                dataset_id="D1",
                source="DACL",
                group_id="tunnel-A/exif",
                image_path=image,
                annotation_path=annotation,
                width=8,
                height=12,
                image_sha256=_sha256(image),
                annotation_sha256=_sha256(annotation),
                classes_present=("crack",),
                class_instances={"crack": 1},
            )
            coco = ensure_coco_dataset([record], CLASS_MAPPING, root / "coco", ("D1",))
            payload = json.loads(coco.annotation_files["D1"].read_text(encoding="utf-8"))
            self.assertEqual((payload["images"][0]["width"], payload["images"][0]["height"]), (12, 8))
            yolo = ensure_yolo_dataset([record], CLASS_MAPPING, root / "yolo", ("D1",))
            values = [float(value) for value in next((yolo.root / "labels" / "train").glob("*.txt")).read_text().split()]
            self.assertEqual(values[1:3], [0.25, 0.125])
            self.assertEqual(values[3:5], [0.5, 0.125])

    def test_repeated_yolo_materialization_reuses_matching_hash(self):
        from morphoroute.data.yolo_dataset import ensure_yolo_dataset

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = _record(root)
            output = root / "derived"

            first = ensure_yolo_dataset(
                [record],
                CLASS_MAPPING,
                output,
                ("D1",),
            )
            second = ensure_yolo_dataset(
                [record],
                CLASS_MAPPING,
                output,
                ("D1",),
            )

            self.assertEqual(first.fingerprint, second.fingerprint)
            self.assertFalse(first.reused)
            self.assertTrue(second.reused)
            self.assertTrue(second.data_yaml.is_file())
            images = list((second.root / "images" / "train").glob("*.png"))
            labels = list((second.root / "labels" / "train").glob("*.txt"))
            self.assertEqual(len(images), 1)
            self.assertEqual(len(labels), 1)

    def test_disallowed_split_is_never_materialized(self):
        from dataclasses import replace

        from morphoroute.data.coco_dataset import ensure_coco_dataset

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            d1 = _record(root)
            d4 = replace(d1, record_id="D4-0001", dataset_id="D4")

            result = ensure_coco_dataset(
                [d1, d4],
                CLASS_MAPPING,
                root / "derived",
                ("D1",),
            )

            self.assertEqual(set(result.annotation_files), {"D1"})
            self.assertNotIn("D4", result.provenance_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
