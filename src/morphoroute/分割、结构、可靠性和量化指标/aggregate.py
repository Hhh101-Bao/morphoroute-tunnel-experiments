from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def macro_average(
    per_class: Mapping[str, float | None],
    class_order: Sequence[str],
) -> float | None:
    values = [
        float(per_class[class_name])
        for class_name in class_order
        if per_class.get(class_name) is not None
    ]
    return float(np.mean(values)) if values else None


def worst_class(
    per_class: Mapping[str, float | None],
    class_order: Sequence[str],
) -> dict[str, Any]:
    available = {
        class_name: float(per_class[class_name])
        for class_name in class_order
        if per_class.get(class_name) is not None
    }
    if not available:
        return {"class_name": None, "value": None}
    class_name = min(available, key=available.get)
    return {"class_name": class_name, "value": available[class_name]}
