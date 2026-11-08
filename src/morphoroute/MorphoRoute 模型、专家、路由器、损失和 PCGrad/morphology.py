from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from morphoroute.metrics.structure import skeletonize


FEATURE_NAMES = (
    "log_aspect_ratio",
    "skeleton_length_over_area",
    "branch_density",
    "compactness",
    "area_fraction",
)


def _perimeter_four_connected(mask: np.ndarray) -> float:
    binary = np.asarray(mask, dtype=bool)
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    exposed = (
        center & ~padded[:-2, 1:-1],
        center & ~padded[2:, 1:-1],
        center & ~padded[1:-1, :-2],
        center & ~padded[1:-1, 2:],
    )
    return float(sum(edge.sum() for edge in exposed))


def _branch_density(skeleton: np.ndarray) -> float:
    padded = np.pad(skeleton, 1, mode="constant", constant_values=False)
    height, width = skeleton.shape
    neighbor_count = sum(
        padded[dy : dy + height, dx : dx + width].astype(int)
        for dy in range(3)
        for dx in range(3)
        if not (dy == 1 and dx == 1)
    )
    length = int(skeleton.sum())
    if length == 0:
        return 0.0
    return float(np.logical_and(skeleton, neighbor_count >= 3).sum() / length)


def extract_morphology(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("morphology mask must be two-dimensional")
    area = int(binary.sum())
    if area == 0:
        raise ValueError("cannot extract morphology from an empty mask")
    ys, xs = np.nonzero(binary)
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    aspect_ratio = max(width, height) / max(1, min(width, height))
    skeleton = skeletonize(binary)
    perimeter = _perimeter_four_connected(binary)
    compactness = 4.0 * np.pi * area / max(perimeter * perimeter, 1e-12)
    return np.asarray(
        [
            np.log(aspect_ratio),
            float(skeleton.sum() / area),
            _branch_density(skeleton),
            float(compactness),
            float(area / binary.size),
        ],
        dtype=float,
    )


def extract_observable_morphology(
    masks,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features only where post-augmentation morphology is visible.

    An all-zero mask has no defined aspect ratio, skeleton, or compactness.
    Such masks can legitimately arise when a very small annotation disappears
    under random resizing.  Returning the original positions lets callers
    mask only morphology supervision without fabricating a target or changing
    the primary segmentation sample.
    """

    features = []
    observable_indices = []
    for index, mask in enumerate(masks):
        binary = np.asarray(mask, dtype=bool)
        if binary.ndim != 2:
            raise ValueError("morphology mask must be two-dimensional")
        if not bool(binary.any()):
            continue
        features.append(extract_morphology(binary))
        observable_indices.append(index)
    if features:
        feature_array = np.stack(features)
    else:
        feature_array = np.empty((0, len(FEATURE_NAMES)), dtype=float)
    return feature_array, np.asarray(observable_indices, dtype=np.int64)


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(
        maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True)),
        axis=axis,
    )


@dataclass(frozen=True)
class MorphologyPrior:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    converged: bool
    iterations: int
    seed: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MorphologyPrior":
        if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("morphology prior feature order does not match")
        return cls(
            feature_mean=np.asarray(payload["feature_mean"], dtype=float),
            feature_scale=np.asarray(payload["feature_scale"], dtype=float),
            weights=np.asarray(payload["weights"], dtype=float),
            means=np.asarray(payload["means"], dtype=float),
            variances=np.asarray(payload["variances"], dtype=float),
            converged=bool(payload["converged"]),
            iterations=int(payload["iterations"]),
            seed=int(payload["seed"]),
        )

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        *,
        seed: int = 42,
        max_iterations: int = 200,
        tolerance: float = 1e-6,
    ) -> "MorphologyPrior":
        values = np.asarray(features, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"features must have shape [n, {len(FEATURE_NAMES)}]")
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("features need at least two finite rows")
        feature_mean = values.mean(axis=0)
        feature_scale = values.std(axis=0)
        feature_scale[feature_scale < 1e-8] = 1.0
        normalized = (values - feature_mean) / feature_scale

        line_score = (
            normalized[:, 0]
            + normalized[:, 1]
            + 0.25 * normalized[:, 2]
            - normalized[:, 3]
            - normalized[:, 4]
        )
        initial_indices = [
            int(np.argmin(line_score)),
            int(np.argmax(line_score)),
        ]
        means = normalized[initial_indices].copy()
        variances = np.ones((2, normalized.shape[1]), dtype=float)
        weights = np.full(2, 0.5, dtype=float)
        previous_likelihood = -np.inf
        converged = False
        iterations = 0

        for iterations in range(1, max_iterations + 1):
            log_probabilities = []
            for component in range(2):
                log_density = -0.5 * (
                    np.log(2 * np.pi * variances[component]).sum()
                    + (
                        (normalized - means[component]) ** 2
                        / variances[component]
                    ).sum(axis=1)
                )
                log_probabilities.append(np.log(weights[component]) + log_density)
            log_probabilities_array = np.column_stack(log_probabilities)
            log_normalizer = _logsumexp(log_probabilities_array, axis=1)
            responsibilities = np.exp(
                log_probabilities_array - log_normalizer[:, None]
            )
            effective_count = responsibilities.sum(axis=0) + 1e-8
            weights = effective_count / effective_count.sum()
            means = (
                responsibilities.T @ normalized
            ) / effective_count[:, None]
            for component in range(2):
                centered = normalized - means[component]
                variances[component] = (
                    responsibilities[:, component, None] * centered**2
                ).sum(axis=0) / effective_count[component]
            variances = np.maximum(variances, 1e-5)
            likelihood = float(log_normalizer.sum())
            if abs(likelihood - previous_likelihood) < tolerance:
                converged = True
                break
            previous_likelihood = likelihood

        component_means = means * feature_scale + feature_mean
        component_line_scores = (
            component_means[:, 0]
            + component_means[:, 1]
            + 0.25 * component_means[:, 2]
            - component_means[:, 3]
            - component_means[:, 4]
        )
        order = np.argsort(-component_line_scores)
        return cls(
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            weights=weights[order],
            means=means[order],
            variances=variances[order],
            converged=converged,
            iterations=iterations,
            seed=seed,
        )

    def soft_targets(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        normalized = (values - self.feature_mean) / self.feature_scale
        log_probabilities = []
        for component in range(2):
            log_density = -0.5 * (
                np.log(2 * np.pi * self.variances[component]).sum()
                + (
                    (normalized - self.means[component]) ** 2
                    / self.variances[component]
                ).sum(axis=1)
            )
            log_probabilities.append(np.log(self.weights[component]) + log_density)
        stacked = np.column_stack(log_probabilities)
        normalizer = _logsumexp(stacked, axis=1)
        return np.exp(stacked - normalizer[:, None])

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(FEATURE_NAMES),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "weights": self.weights.tolist(),
            "means": self.means.tolist(),
            "variances": self.variances.tolist(),
            "converged": self.converged,
            "iterations": self.iterations,
            "seed": self.seed,
        }
