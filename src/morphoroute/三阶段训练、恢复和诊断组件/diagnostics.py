from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class QueryDiagnosticWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(dict(record), ensure_ascii=False, default=str))
            handle.write("\n")
