from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from morphoroute.core.io import atomic_write_json, sha256_json

from .manifest import SampleRecord


@dataclass(frozen=True)
class DatasetFingerprint:
    sha256: str
    record_count: int
    splits: tuple[str, ...]


@dataclass(frozen=True)
class MaterializedDataset:
    root: Path
    fingerprint: DatasetFingerprint
    reused: bool
    format: str
    annotation_files: Mapping[str, Path] = field(default_factory=dict)
    provenance_file: Path | None = None
    data_yaml: Path | None = None


def select_records(
    records: Iterable[SampleRecord],
    allowed_splits: Iterable[str],
) -> list[SampleRecord]:
    allowed = tuple(dict.fromkeys(str(split) for split in allowed_splits))
    unknown = set(allowed).difference({"D1", "D2", "D3", "D4"})
    if unknown:
        raise ValueError(f"unsupported materialization splits: {sorted(unknown)}")
    selected = [
        record
        for record in records
        if record.dataset_id in set(allowed)
    ]
    return sorted(selected, key=lambda record: record.record_id)


def fingerprint_records(
    records: Iterable[SampleRecord],
    class_mapping: Mapping[str, Any],
    allowed_splits: Iterable[str],
) -> DatasetFingerprint:
    splits = tuple(dict.fromkeys(str(split) for split in allowed_splits))
    selected = select_records(records, splits)
    payload = {
        "schema_version": 2,
        "splits": splits,
        "class_order": list(class_mapping["class_order"]),
        "records": [
            {
                "record_id": record.record_id,
                "split": record.dataset_id,
                "source": record.source,
                "group_id": record.group_id,
                "width": record.width,
                "height": record.height,
                "image_sha256": record.image_sha256,
                "annotation_sha256": record.annotation_sha256,
            }
            for record in selected
        ],
    }
    return DatasetFingerprint(
        sha256=sha256_json(payload),
        record_count=len(selected),
        splits=splits,
    )


def materialization_metadata_path(root: Path) -> Path:
    return root / "materialization.json"


def matching_materialization(
    root: Path,
    fingerprint: DatasetFingerprint,
    *,
    format_name: str,
) -> bool:
    path = materialization_metadata_path(root)
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("format") == format_name
        and payload.get("fingerprint_sha256") == fingerprint.sha256
        and payload.get("record_count") == fingerprint.record_count
        and tuple(payload.get("splits", ())) == fingerprint.splits
    )


def write_materialization_metadata(
    root: Path,
    fingerprint: DatasetFingerprint,
    *,
    format_name: str,
) -> Path:
    return atomic_write_json(
        materialization_metadata_path(root),
        {
            "schema_version": 2,
            "format": format_name,
            "fingerprint_sha256": fingerprint.sha256,
            "record_count": fingerprint.record_count,
            "splits": list(fingerprint.splits),
        },
    )
