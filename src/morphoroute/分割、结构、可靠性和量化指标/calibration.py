from __future__ import annotations

import numpy as np


def _probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(logits, dtype=float) / temperature
    if values.ndim == 1:
        probability = 1.0 / (1.0 + np.exp(-values))
        return np.column_stack([1.0 - probability, probability])
    shifted = values - values.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def negative_log_likelihood(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float,
) -> float:
    if temperature <= 0:
        return float("inf")
    probabilities = _probabilities(logits, temperature)
    labels_array = np.asarray(labels, dtype=int)
    if len(probabilities) != len(labels_array):
        raise ValueError("logits and labels length mismatch")
    if np.any((labels_array < 0) | (labels_array >= probabilities.shape[1])):
        raise ValueError("labels are outside the logits class range")
    selected = probabilities[np.arange(len(labels_array)), labels_array]
    return float(-np.log(selected.clip(1e-12, 1.0)).mean())


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    iterations: int = 80,
) -> float:
    left, right = -4.0, 4.0
    golden = (np.sqrt(5.0) - 1.0) / 2.0
    first = right - golden * (right - left)
    second = left + golden * (right - left)
    for _ in range(iterations):
        first_loss = negative_log_likelihood(logits, labels, np.exp(first))
        second_loss = negative_log_likelihood(logits, labels, np.exp(second))
        if first_loss < second_loss:
            right = second
            second = first
            first = right - golden * (right - left)
        else:
            left = first
            first = second
            second = left + golden * (right - left)
    return float(np.exp((left + right) / 2.0))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return _probabilities(logits, temperature)
