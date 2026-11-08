from __future__ import annotations

import argparse
import csv
import json
import sys
from io import StringIO
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from morphoroute.core.io import (
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)


CLASS_ORDER = (
    "crack",
    "water_leakage",
    "spalling",
    "efflorescence",
    "exposed_rebar",
)
TACK_VALUES = {"crack": 40, "water_leakage": 160, "efflorescence": 200}
DACL_LABELS = {"spalling": "Spalling", "exposed_rebar": "ExposedRebars"}
SPLITS = {
    "D1": "train",
    "D2": "val",
    "D3": "test_internal",
    "D4": "D4",
}


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "record_id",
        "logical_split",
        "split_path",
        "source",
        "source_role",
        "source_tunnel",
        "source_camera",
        "official_split",
        "group_key",
        "source_image_path",
        "source_annotation_path",
        "destination_image_relative",
        "destination_annotation_relative",
        "width",
        "height",
        "mode",
        "bytes",
        "image_sha256",
        "annotation_sha256",
        "classes_present",
        "class_instances_json",
        "assignment_group_id",
    ]
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _synthetic_image(class_index: int, split_index: int) -> np.ndarray:
    image = np.full((32, 32, 3), 35 + 12 * split_index, dtype=np.uint8)
    image[5:27, 5:27, class_index % 3] = 125 + 20 * class_index
    return image


def create_synthetic_dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    metadata = root / "metadata"
    metadata.mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []
    distribution: list[dict[str, object]] = []

    for split_index, (dataset_id, logical_split) in enumerate(SPLITS.items()):
        for class_index, class_name in enumerate(CLASS_ORDER):
            source = "TACK" if class_name in TACK_VALUES else "DACL"
            record_id = f"synthetic_{dataset_id}_{class_name}"
            group_id = f"group_{dataset_id}_{class_name}"
            split_root = root / dataset_id
            image_path = split_root / "images" / f"{record_id}.png"
            annotation_suffix = ".png" if source == "TACK" else ".json"
            annotation_path = (
                split_root
                / "annotations_original"
                / f"{record_id}{annotation_suffix}"
            )
            image_path.parent.mkdir(parents=True, exist_ok=True)
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(_synthetic_image(class_index, split_index), mode="RGB").save(
                image_path
            )

            if source == "TACK":
                mask = np.zeros((32, 32), dtype=np.uint8)
                if class_name == "crack":
                    mask[15, 5:27] = TACK_VALUES[class_name]
                else:
                    mask[9:24, 8:25] = TACK_VALUES[class_name]
                Image.fromarray(mask, mode="L").save(annotation_path)
            else:
                payload = {
                    "shapes": [
                        {
                            "label": DACL_LABELS[class_name],
                            "points": [[7, 8], [25, 8], [25, 24], [7, 24]],
                            "shape_type": "polygon",
                        }
                    ],
                    "imagePath": image_path.name,
                    "imageHeight": 32,
                    "imageWidth": 32,
                    "imageName": image_path.name,
                    "split": logical_split,
                    "dacl10k_version": "synthetic-v1",
                }
                atomic_write_json(annotation_path, payload)

            rows.append(
                {
                    "record_id": record_id,
                    "logical_split": logical_split,
                    "split_path": dataset_id,
                    "source": source,
                    "source_role": "synthetic",
                    "source_tunnel": f"tunnel_{dataset_id}",
                    "source_camera": "camera_1",
                    "official_split": logical_split,
                    "group_key": group_id,
                    "source_image_path": "",
                    "source_annotation_path": "",
                    "destination_image_relative": image_path.relative_to(root).as_posix(),
                    "destination_annotation_relative": annotation_path.relative_to(
                        root
                    ).as_posix(),
                    "width": 32,
                    "height": 32,
                    "mode": "RGB",
                    "bytes": image_path.stat().st_size,
                    "image_sha256": sha256_file(image_path),
                    "annotation_sha256": sha256_file(annotation_path),
                    "classes_present": class_name,
                    "class_instances_json": json.dumps({class_name: 1}),
                    "assignment_group_id": group_id,
                }
            )
            distribution.append(
                {
                    "logical_split": dataset_id,
                    "source": source,
                    "subdomain": "synthetic",
                    "class_name": class_name,
                    "positive_images": 1,
                    "instances": 1,
                }
            )

    _write_manifest(metadata / "split_manifest.csv", rows)
    class_buffer = StringIO(newline="")
    class_writer = csv.DictWriter(
        class_buffer,
        fieldnames=[
            "logical_split",
            "source",
            "subdomain",
            "class_name",
            "positive_images",
            "instances",
        ],
    )
    class_writer.writeheader()
    class_writer.writerows(distribution)
    atomic_write_text(metadata / "class_distribution.csv", class_buffer.getvalue())
    atomic_write_json(
        metadata / "class_mapping.json",
        {
            "class_order": list(CLASS_ORDER),
            "DACL": {
                "Crack": "crack",
                "Wetspot": "water_leakage",
                "Spalling": "spalling",
                "Efflorescence": "efflorescence",
                "ExposedRebars": "exposed_rebar",
            },
            "TACK_mask_values": {
                str(value): class_name for class_name, value in TACK_VALUES.items()
            },
        },
    )
    atomic_write_json(
        metadata / "split_summary.json",
        {
            "total_images": len(rows),
            "counts_by_logical_split": {dataset_id: 5 for dataset_id in SPLITS},
            "synthetic": True,
        },
    )
    atomic_write_json(
        metadata / "verification_report.json",
        {
            "manifest_rows": len(rows),
            "missing_images": 0,
            "missing_annotations": 0,
            "cross_split_group_overlaps": 0,
        },
    )
    atomic_write_text(
        metadata / "duplicate_audit.csv",
        "decision,kind,left_record_id,right_record_id\n",
    )
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    root = create_synthetic_dataset(arguments.output)
    print(json.dumps({"dataset_root": str(root), "records": 20}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
