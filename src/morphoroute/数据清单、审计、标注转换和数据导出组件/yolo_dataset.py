from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from .conversion import export_yolo
from .manifest import SampleRecord
from .materialize import (
    MaterializedDataset,
    fingerprint_records,
    matching_materialization,
    select_records,
    write_materialization_metadata,
)


def ensure_yolo_dataset(
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
    root = Path(output_root) / "yolo" / fingerprint.sha256[:16]
    data_yaml = root / "data.yaml"
    reused = (
        matching_materialization(root, fingerprint, format_name="yolo")
        and data_yaml.is_file()
    )
    if reused:
        return MaterializedDataset(
            root=root,
            fingerprint=fingerprint,
            reused=True,
            format="yolo",
            data_yaml=data_yaml,
        )

    selected = select_records(records_list, fingerprint.splits)
    export_yolo(selected, class_mapping, root)
    write_materialization_metadata(
        root,
        fingerprint,
        format_name="yolo",
    )
    return MaterializedDataset(
        root=root,
        fingerprint=fingerprint,
        reused=False,
        format="yolo",
        data_yaml=data_yaml,
    )
