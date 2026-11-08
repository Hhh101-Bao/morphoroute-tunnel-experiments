from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from morphoroute.core.io import sha256_file

from .manifest import SampleRecord


@dataclass(frozen=True)
class AuditReport:
    total_records: int
    split_counts: dict[str, int]
    hash_verification_performed: bool
    unresolved_duplicates: int
    missing_images: tuple[str, ...]
    missing_annotations: tuple[str, ...]
    image_hash_mismatches: tuple[str, ...]
    annotation_hash_mismatches: tuple[str, ...]
    cross_split_groups: tuple[dict[str, Any], ...]
    class_positive_images: dict[str, dict[str, int]]
    class_instances: dict[str, dict[str, int]]

    @property
    def passed(self) -> bool:
        return not (
            self.missing_images
            or self.missing_annotations
            or self.image_hash_mismatches
            or self.annotation_hash_mismatches
            or self.cross_split_groups
            or self.unresolved_duplicates
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


def audit_records(
    records: Iterable[SampleRecord],
    *,
    verify_hashes: bool = False,
    unresolved_duplicates: int = 0,
) -> AuditReport:
    records_list = list(records)
    split_counts: dict[str, int] = defaultdict(int)
    groups: dict[str, set[str]] = defaultdict(set)
    positive: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    instances: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    missing_images: list[str] = []
    missing_annotations: list[str] = []
    bad_image_hashes: list[str] = []
    bad_annotation_hashes: list[str] = []

    for record in records_list:
        split_counts[record.dataset_id] += 1
        groups[record.group_id].add(record.dataset_id)
        for class_name in record.classes_present:
            positive[record.dataset_id][class_name] += 1
        for class_name, count in record.class_instances.items():
            instances[record.dataset_id][class_name] += count

        if not record.image_path.is_file():
            missing_images.append(record.record_id)
        elif (
            verify_hashes
            and record.image_sha256
            and sha256_file(record.image_path) != record.image_sha256
        ):
            bad_image_hashes.append(record.record_id)

        if not record.annotation_path.is_file():
            missing_annotations.append(record.record_id)
        elif (
            verify_hashes
            and record.annotation_sha256
            and sha256_file(record.annotation_path) != record.annotation_sha256
        ):
            bad_annotation_hashes.append(record.record_id)

    overlaps = tuple(
        {
            "group_id": group_id,
            "splits": sorted(split_ids),
        }
        for group_id, split_ids in sorted(groups.items())
        if len(split_ids) > 1
    )
    return AuditReport(
        total_records=len(records_list),
        split_counts=dict(sorted(split_counts.items())),
        hash_verification_performed=verify_hashes,
        unresolved_duplicates=int(unresolved_duplicates),
        missing_images=tuple(sorted(missing_images)),
        missing_annotations=tuple(sorted(missing_annotations)),
        image_hash_mismatches=tuple(sorted(bad_image_hashes)),
        annotation_hash_mismatches=tuple(sorted(bad_annotation_hashes)),
        cross_split_groups=overlaps,
        class_positive_images={
            split: dict(sorted(values.items()))
            for split, values in sorted(positive.items())
        },
        class_instances={
            split: dict(sorted(values.items()))
            for split, values in sorted(instances.items())
        },
    )


def count_unresolved_duplicates(path: str | Path) -> int:
    from pathlib import Path

    audit_path = Path(path)
    if not audit_path.is_file():
        raise FileNotFoundError(f"duplicate audit is missing: {audit_path}")
    unresolved = 0
    with audit_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("kind") == "near_duplicate_candidate"
                and not str(row.get("decision", "")).strip()
            ):
                unresolved += 1
    return unresolved
