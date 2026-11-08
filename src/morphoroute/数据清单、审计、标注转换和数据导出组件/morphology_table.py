from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from morphoroute.core.io import atomic_write_json
from morphoroute.models.morphology import (
    FEATURE_NAMES,
    MorphologyPrior,
    extract_morphology,
)

from .conversion import read_instances
from .manifest import SampleRecord


def _extract_record_morphology_rows(
    record: SampleRecord,
    class_mapping: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Extract ordered morphology rows for one D1 record."""
    rows: list[dict[str, Any]] = []
    for instance_index, instance in enumerate(
        read_instances(record, class_mapping)
    ):
        from .annotations import decode_uncompressed_rle

        if isinstance(instance.segmentation, dict):
            mask = decode_uncompressed_rle(instance.segmentation)
        else:
            import numpy as np
            from PIL import Image, ImageDraw

            canvas = Image.new("1", (record.width, record.height), 0)
            coordinates = np.asarray(
                instance.segmentation[0],
                dtype=float,
            ).reshape(-1, 2)
            ImageDraw.Draw(canvas).polygon(
                [tuple(point) for point in coordinates],
                fill=1,
            )
            mask = np.asarray(canvas, dtype=bool)
        features = extract_morphology(mask)
        rows.append(
            {
                "record_id": record.record_id,
                "group_id": record.group_id,
                "dataset_id": "D1",
                "class_name": instance.class_name,
                "instance_index": instance_index,
                "instance_area": instance.area,
                **dict(zip(FEATURE_NAMES, features.tolist())),
            }
        )
    return rows


def _worker_count(pending_count: int) -> int:
    raw = os.environ.get("MORPHOROUTE_MORPHOLOGY_WORKERS", "4")
    try:
        requested = int(raw)
    except ValueError as error:
        raise ValueError(
            "MORPHOROUTE_MORPHOLOGY_WORKERS must be an integer"
        ) from error
    if requested < 1:
        raise ValueError("MORPHOROUTE_MORPHOLOGY_WORKERS must be positive")
    return max(1, min(requested, pending_count or 1))


def _load_partial_rows(
    partial_path: Path,
    expected_record_ids: list[str],
) -> tuple[int, list[dict[str, Any]]]:
    if not partial_path.is_file():
        return 0, []
    payloads: list[dict[str, Any]] = []
    normalized_lines: list[str] = []
    for line_number, line in enumerate(
        partial_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            break
        record_index = len(payloads)
        if record_index >= len(expected_record_ids):
            raise ValueError("morphology partial checkpoint has extra records")
        if payload.get("record_id") != expected_record_ids[record_index]:
            raise ValueError(
                "morphology partial checkpoint does not match D1 manifest order"
            )
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise ValueError(
                f"invalid morphology partial checkpoint line {line_number}"
            )
        payloads.append(payload)
        normalized_lines.append(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
    normalized = "".join(normalized_lines)
    if partial_path.read_text(encoding="utf-8") != normalized:
        temporary = partial_path.with_name(partial_path.name + ".tmp")
        temporary.write_text(normalized, encoding="utf-8")
        os.replace(temporary, partial_path)
    rows = [row for payload in payloads for row in payload["rows"]]
    return len(payloads), rows


def ensure_d1_morphology_prior(
    records: Iterable[SampleRecord],
    class_mapping: Mapping[str, Any],
    output_dir: str | Path,
    *,
    seed: int,
) -> tuple[Path, Path]:
    directory = Path(output_dir)
    table_path = directory / "morphology_features_D1.csv"
    prior_path = directory / "gmm_D1.json"
    if table_path.is_file() and prior_path.is_file():
        return table_path, prior_path
    directory.mkdir(parents=True, exist_ok=True)
    d1_records = [record for record in records if record.dataset_id == "D1"]
    expected_ids = [record.record_id for record in d1_records]
    partial_path = directory / "morphology_features_D1.partial.jsonl"
    completed_count, rows = _load_partial_rows(partial_path, expected_ids)
    pending = d1_records[completed_count:]
    workers = _worker_count(len(pending))
    print(
        "D1形态预处理 "
        f"total={len(d1_records)} resumed={completed_count} workers={workers}",
        flush=True,
    )
    started = time.monotonic()
    if workers == 1:
        results = (
            _extract_record_morphology_rows(record, class_mapping)
            for record in pending
        )
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(
            _extract_record_morphology_rows,
            pending,
            repeat(dict(class_mapping)),
            chunksize=1,
        )
    progress_interval = max(1, min(25, max(1, len(d1_records) // 20)))
    try:
        with partial_path.open("a", encoding="utf-8") as handle:
            for offset, record_rows in enumerate(results, start=1):
                record = pending[offset - 1]
                rows.extend(record_rows)
                payload = {"record_id": record.record_id, "rows": record_rows}
                handle.write(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                handle.flush()
                completed = completed_count + offset
                if completed % progress_interval == 0 or completed == len(d1_records):
                    elapsed = max(time.monotonic() - started, 1e-9)
                    rate = offset / elapsed
                    remaining = (len(d1_records) - completed) / max(rate, 1e-9)
                    print(
                        "D1形态进度 "
                        f"{completed}/{len(d1_records)} | "
                        f"{rate:.2f}图/秒 | 剩余约{remaining / 60:.1f}分钟",
                        flush=True,
                    )
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
    if len(rows) < 2:
        raise ValueError("D1 needs at least two instances to fit morphology GMM")
    table = pd.DataFrame(rows)
    temporary_table = table_path.with_name(table_path.name + ".tmp")
    table.to_csv(temporary_table, index=False)
    os.replace(temporary_table, table_path)
    prior = MorphologyPrior.fit(
        table[list(FEATURE_NAMES)].to_numpy(float),
        seed=seed,
    )
    atomic_write_json(prior_path, prior.to_dict())
    partial_path.unlink(missing_ok=True)
    return table_path, prior_path
