from __future__ import annotations

import numpy as np


def holm_adjust(p_values: list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must lie in [0, 1]")
    count = len(values)
    order = np.argsort(values)
    ordered = values[order]
    adjusted_ordered = np.empty(count, dtype=float)
    running_max = 0.0
    for rank, p_value in enumerate(ordered):
        candidate = min(1.0, (count - rank) * float(p_value))
        running_max = max(running_max, candidate)
        adjusted_ordered[rank] = running_max
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = adjusted_ordered
    return adjusted
