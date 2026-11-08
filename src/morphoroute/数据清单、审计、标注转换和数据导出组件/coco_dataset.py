from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from morphoroute.core.io import atomic_write_json

from .conversion import export_coco
from .manifest import SampleRecord
from .materialize import (
    MaterializedDataset,
    fingerprint_records,
    matching_materialization,
    select_records,
    write_materialization_metadata,
)


def ensure_coco_dataset(
    records: Iterable[SampleRecord],
    class_mapping: Mapping[str, Any],
    output_root: str | Path,
    allowed_splits: Iterable[str],
) -> MaterializedDataset:
    records_list = list(records)
    fingerprint = fingerprint_records(
        records_list,
        class_mapping,
        allowed_splits,
    )
    root = Path(output_root) / "coco" / fingerprint.sha256[:16]
    annotation_files = {
        split: root / "annotations" / f"{split}.json"
        for split in fingerprint.splits
    }
    provenance_file = root / "provenance.json"
    reused = (
        matching_materialization(root, fingerprint, format_name="coco")
        and provenance_file.is_file()
        and all(path.is_file() for path in annotation_files.values())
    )
    if reused:
        return MaterializedDataset(
            root=root,
            fingerprint=fingerprint,
            reused=True,
            format="coco",
            annotation_files=annotation_files,
            provenance_file=provenance_file,
        )

    selected = select_records(records_list, fingerprint.splits)
    for split, output_path in annotation_files.items():
        split_records = [
            record for record in selected if record.dataset_id == split
        ]
        export_coco(split_records, class_mapping, output_path)
    atomic_write_json(
        provenance_file,
        {
            "fingerprint_sha256": fingerprint.sha256,
            "records": [
                {
                    "record_id": record.record_id,
                    "split": record.dataset_id,
                    "source": record.source,
                    "group_id": record.group_id,
                    "image_path": str(record.image_path),
                    "annotation_path": str(record.annotation_path),
                    "image_sha256": record.image_sha256,
                    "annotation_sha256": record.annotation_sha256,
                }
                for record in selected
            ],
        },
    )
    write_materialization_metadata(
        root,
        fingerprint,
        format_name="coco",
    )
    return MaterializedDataset(
        root=root,
        fingerprint=fingerprint,
        reused=False,
        format="coco",
        annotation_files=annotation_files,
        provenance_file=provenance_file,
    )
