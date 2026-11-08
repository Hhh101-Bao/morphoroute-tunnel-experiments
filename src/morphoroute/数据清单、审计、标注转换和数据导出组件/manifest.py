from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


SPLIT_ALIASES = {
    "train": "D1",
    "D1": "D1",
    "val": "D2",
    "validation": "D2",
    "D2": "D2",
    "test_internal": "D3",
    "test": "D3",
    "D3": "D3",
    "D4": "D4",
}


def _resolve_within(root: Path, relative: str, field_name: str) -> Path:
    if not relative:
        raise ValueError(f"{field_name} is empty")
    candidate = root / Path(relative)
    root_resolved = os.path.normcase(str(root.resolve(strict=False)))
    candidate_resolved = os.path.normcase(str(candidate.resolve(strict=False)))
    try:
        contained = os.path.commonpath([root_resolved, candidate_resolved]) == root_resolved
    except ValueError:
        contained = False
    if not contained:
        raise ValueError(f"{field_name} escapes the dataset root: {relative}")
    return candidate


@dataclass(frozen=True)
class SampleRecord:
    record_id: str
    dataset_id: str
    source: str
    group_id: str
    image_path: Path
    annotation_path: Path
    width: int
    height: int
    image_sha256: str
    annotation_sha256: str
    classes_present: tuple[str, ...]
    class_instances: dict[str, int]


def parse_manifest_rows(
    rows: Iterable[Mapping[str, str]],
    dataset_root: str | Path,
) -> list[SampleRecord]:
    root = Path(dataset_root)
    seen: set[str] = set()
    records: list[SampleRecord] = []
    for index, row in enumerate(rows, start=2):
        record_id = str(row.get("record_id", "")).strip()
        if not record_id:
            raise ValueError(f"manifest row {index} has empty record_id")
        if record_id in seen:
            raise ValueError(f"duplicate record_id: {record_id}")
        seen.add(record_id)

        raw_split = str(row.get("logical_split", "")).strip()
        try:
            dataset_id = SPLIT_ALIASES[raw_split]
        except KeyError as error:
            raise ValueError(
                f"manifest row {index} has unsupported logical_split: {raw_split}"
            ) from error

        group_id = str(
            row.get("assignment_group_id") or row.get("group_key") or ""
        ).strip()
        if not group_id:
            raise ValueError(f"manifest row {index} has empty group_id")

        try:
            class_instances_payload = json.loads(
                row.get("class_instances_json") or "{}"
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                f"manifest row {index} has invalid class_instances_json"
            ) from error
        if not isinstance(class_instances_payload, dict):
            raise ValueError(
                f"manifest row {index} class_instances_json must be an object"
            )
        class_instances = {
            str(name): int(count) for name, count in class_instances_payload.items()
        }
        classes_present = tuple(
            name
            for name in str(row.get("classes_present", "")).split(";")
            if name
        )

        records.append(
            SampleRecord(
                record_id=record_id,
                dataset_id=dataset_id,
                source=str(row.get("source", "")).strip(),
                group_id=group_id,
                image_path=_resolve_within(
                    root,
                    str(row.get("destination_image_relative", "")),
                    "destination_image_relative",
                ),
                annotation_path=_resolve_within(
                    root,
                    str(row.get("destination_annotation_relative", "")),
                    "destination_annotation_relative",
                ),
                width=int(row.get("width") or 0),
                height=int(row.get("height") or 0),
                image_sha256=str(row.get("image_sha256", "")).strip(),
                annotation_sha256=str(row.get("annotation_sha256", "")).strip(),
                classes_present=classes_present,
                class_instances=class_instances,
            )
        )
    return records


def load_manifest(
    manifest_path: str | Path,
    dataset_root: str | Path,
) -> list[SampleRecord]:
    with Path(manifest_path).open("r", encoding="utf-8-sig", newline="") as handle:
        return parse_manifest_rows(csv.DictReader(handle), dataset_root)
