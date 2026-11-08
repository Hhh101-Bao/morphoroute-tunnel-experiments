from __future__ import annotations

import numpy as np


def standardized_paired_effect(
    values_a: np.ndarray,
    values_b: np.ndarray,
) -> float:
    left = np.asarray(values_a, dtype=float)
    right = np.asarray(values_b, dtype=float)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("paired values must be one-dimensional with equal shapes")
    differences = left - right
    if len(differences) < 2:
        return float("nan")
    standard_deviation = float(differences.std(ddof=1))
    if standard_deviation == 0:
        mean = float(differences.mean())
        return 0.0 if mean == 0 else float(np.sign(mean) * np.inf)
    return float(differences.mean() / standard_deviation)


def relative_difference(value_a: float, value_b: float) -> float:
    if value_b == 0:
        return float("nan")
    return float((value_a - value_b) / abs(value_b))
