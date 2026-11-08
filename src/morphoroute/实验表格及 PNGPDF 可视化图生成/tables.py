from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from morphoroute.core.io import atomic_write_json, atomic_write_text


def _normalize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_result_bundle(
    records: Iterable[Mapping[str, Any]],
    output_stem: str | Path,
) -> dict[str, Path]:
    stem = Path(output_stem)
    normalized = [
        {str(key): _normalize(value) for key, value in record.items()}
        for record in records
    ]
    dataframe = pd.DataFrame(normalized)
    json_path = stem.with_suffix(".json")
    csv_path = stem.with_suffix(".csv")
    xlsx_path = stem.with_suffix(".xlsx")
    atomic_write_json(json_path, normalized)
    atomic_write_text(csv_path, dataframe.to_csv(index=False, lineterminator="\n"))

    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=xlsx_path.parent,
            prefix=f".{xlsx_path.name}.",
            suffix=".tmp.xlsx",
            delete=False,
        ) as handle:
            temporary_name = handle.name
        dataframe.to_excel(temporary_name, index=False, engine="openpyxl")
        os.replace(temporary_name, xlsx_path)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    return {"json": json_path, "csv": csv_path, "xlsx": xlsx_path}
