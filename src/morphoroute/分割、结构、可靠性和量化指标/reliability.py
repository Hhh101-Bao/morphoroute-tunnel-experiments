from __future__ import annotations

import numpy as np


def expected_calibration_error(
    confidence: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> float:
    confidence_array = np.asarray(confidence, dtype=float)
    correct_array = np.asarray(correct, dtype=float)
    if confidence_array.shape != correct_array.shape:
        raise ValueError("confidence and correct must have identical shapes")
    if confidence_array.ndim != 1:
        raise ValueError("confidence and correct must be one-dimensional")
    if len(confidence_array) == 0:
        return float("nan")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if np.any((confidence_array < 0) | (confidence_array > 1)):
        raise ValueError("confidence values must lie in [0, 1]")
    bin_ids = np.digitize(
        confidence_array,
        np.linspace(0.0, 1.0, n_bins + 1)[1:-1],
        right=False,
    )
    error = 0.0
    for bin_id in range(n_bins):
        members = bin_ids == bin_id
        if not members.any():
            continue
        error += float(members.mean()) * abs(
            float(correct_array[members].mean())
            - float(confidence_array[members].mean())
        )
    return error


def brier_score(confidence: np.ndarray, correct: np.ndarray) -> float:
    confidence_array = np.asarray(confidence, dtype=float)
    correct_array = np.asarray(correct, dtype=float)
    if confidence_array.shape != correct_array.shape:
        raise ValueError("confidence and correct must have identical shapes")
    return float(np.mean((confidence_array - correct_array) ** 2))


def risk_coverage_curve(
    confidence: np.ndarray,
    error: np.ndarray,
) -> dict[str, np.ndarray]:
    confidence_array = np.asarray(confidence, dtype=float)
    error_array = np.asarray(error, dtype=float)
    if confidence_array.shape != error_array.shape:
        raise ValueError("confidence and error must have identical shapes")
    order = np.argsort(-confidence_array)
    sorted_error = error_array[order]
    count = len(sorted_error)
    coverage = np.arange(1, count + 1, dtype=float) / count
    risk = np.cumsum(sorted_error) / np.arange(1, count + 1)
    return {"coverage": coverage, "risk": risk, "order": order}
