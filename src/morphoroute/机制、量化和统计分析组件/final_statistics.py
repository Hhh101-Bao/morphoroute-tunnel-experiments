from __future__ import annotations

from collections.abc import Iterable, Mapping


def aggregate_final_results(
    records: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    rows = [dict(record) for record in records]
    if any(bool(row.get("smoke_test")) for row in rows):
        raise ValueError("smoke runs cannot enter formal final aggregation")
    missing_status = [
        index for index, row in enumerate(rows) if not row.get("run_status")
    ]
    if missing_status:
        raise ValueError(
            f"every run needs explicit run_status; missing rows: {missing_status}"
        )
    completed = [
        row for row in rows if row["run_status"] == "completed"
    ]
    failed = [
        row for row in rows if row["run_status"] != "completed"
    ]
    return {
        "records": rows,
        "completed_count": len(completed),
        "failed_count": len(failed),
        "failed_runs_retained": True,
    }
