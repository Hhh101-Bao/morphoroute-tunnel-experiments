from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

from morphoroute.core.io import atomic_write_json, atomic_write_text

from .annotations import (
    InstanceAnnotation,
    dacl_json_to_instances,
    decode_uncompressed_rle,
    tack_mask_to_instances,
)
from .manifest import SampleRecord


def load_class_mapping(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"class_order", "DACL", "TACK_mask_values"}
    if not required.issubset(payload):
        raise ValueError(f"class mapping lacks fields: {sorted(required - set(payload))}")
    return payload


def export_dimensions(record: SampleRecord) -> tuple[int, int]:
    if record.source == "DACL":
        payload = json.loads(record.annotation_path.read_text(encoding="utf-8"))
        width = int(payload["imageWidth"])
        height = int(payload["imageHeight"])
    else:
        width, height = record.width, record.height
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid export dimensions for {record.record_id}")
    return width, height


def read_instances(
    record: SampleRecord,
    class_mapping: Mapping[str, Any],
) -> list[InstanceAnnotation]:
    if record.source == "TACK":
        mask = np.asarray(Image.open(record.annotation_path))
        value_map = {
            int(value): str(class_name)
            for value, class_name in class_mapping["TACK_mask_values"].items()
        }
        return tack_mask_to_instances(mask, value_map)
    if record.source == "DACL":
        payload = json.loads(record.annotation_path.read_text(encoding="utf-8"))
        return dacl_json_to_instances(
            payload,
            class_mapping["DACL"],
            ignore_unknown_labels=True,
            clip_out_of_bounds_tolerance=8.0,
            drop_out_of_bounds_spikes=True,
        )
    raise ValueError(f"unsupported annotation source: {record.source}")


def export_coco(
    records: Iterable[SampleRecord],
    class_mapping: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    class_order = [str(name) for name in class_mapping["class_order"]]
    category_ids = {name: index + 1 for index, name in enumerate(class_order)}
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1
    for image_id, record in enumerate(records, start=1):
        width, height = export_dimensions(record)
        images.append(
            {
                "id": image_id,
                "file_name": str(record.image_path),
                "width": width,
                "height": height,
                "record_id": record.record_id,
                "source": record.source,
                "group_id": record.group_id,
                "dataset_id": record.dataset_id,
            }
        )
        for instance in read_instances(record, class_mapping):
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_ids[instance.class_name],
                    "segmentation": instance.segmentation,
                    "bbox": list(instance.bbox),
                    "area": instance.area,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": index + 1, "name": name}
            for index, name in enumerate(class_order)
        ],
    }
    return atomic_write_json(output_path, payload)


def _mask_contours(mask: np.ndarray) -> list[np.ndarray]:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "YOLO export for TACK masks requires opencv-python"
        ) from error
    contours, _ = cv2.findContours(
        np.asarray(mask, dtype=np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return [contour.reshape(-1, 2) for contour in contours if len(contour) >= 3]


def export_yolo(
    records: Iterable[SampleRecord],
    class_mapping: Mapping[str, Any],
    output_dir: str | Path,
) -> Path:
    root = Path(output_dir)
    labels_root = root / "labels"
    images_root = root / "images"
    class_order = [str(name) for name in class_mapping["class_order"]]
    class_ids = {name: index for index, name in enumerate(class_order)}
    split_names = {"D1": "train", "D2": "val", "D3": "test", "D4": "external"}

    for record in records:
        width, height = export_dimensions(record)
        split_name = split_names[record.dataset_id]
        safe_record_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", record.record_id)
        destination_image = (
            images_root
            / split_name
            / f"{safe_record_id}{record.image_path.suffix.lower()}"
        )
        destination_image.parent.mkdir(parents=True, exist_ok=True)
        if not destination_image.exists():
            try:
                os.link(record.image_path, destination_image)
            except OSError:
                shutil.copy2(record.image_path, destination_image)

        lines: list[str] = []
        for instance in read_instances(record, class_mapping):
            if isinstance(instance.segmentation, dict):
                polygons = _mask_contours(
                    decode_uncompressed_rle(instance.segmentation)
                )
            else:
                polygons = [
                    np.asarray(instance.segmentation[0], dtype=float).reshape(-1, 2)
                ]
            for polygon in polygons:
                normalized: list[str] = [str(class_ids[instance.class_name])]
                for x, y in polygon:
                    normalized.extend(
                        [f"{float(x) / width:.8f}", f"{float(y) / height:.8f}"]
                    )
                lines.append(" ".join(normalized))
        atomic_write_text(
            labels_root / split_name / f"{safe_record_id}.txt",
            "\n".join(lines) + ("\n" if lines else ""),
        )

    yaml_text = "\n".join(
        [
            f"path: {root.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            f"names: {json.dumps(class_order, ensure_ascii=False)}",
            "",
        ]
    )
    return atomic_write_text(root / "data.yaml", yaml_text)
