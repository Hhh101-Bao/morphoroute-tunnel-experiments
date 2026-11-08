from __future__ import annotations

import numpy as np


def binary_erosion(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        height, width = result.shape
        neighborhoods = [
            padded[dy : dy + height, dx : dx + width]
            for dy in range(3)
            for dx in range(3)
        ]
        result = np.logical_and.reduce(neighborhoods)
    return result


def binary_dilation(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        height, width = result.shape
        neighborhoods = [
            padded[dy : dy + height, dx : dx + width]
            for dy in range(3)
            for dx in range(3)
        ]
        result = np.logical_or.reduce(neighborhoods)
    return result


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    return binary & ~binary_erosion(binary)


def skeletonize(mask: np.ndarray) -> np.ndarray:
    image = np.asarray(mask, dtype=bool).copy()
    if not image.any():
        return image
    changed = True
    while changed:
        changed = False
        for first_pass in (True, False):
            padded = np.pad(
                image,
                1,
                mode="constant",
                constant_values=False,
            )
            height, width = image.shape
            p2 = padded[0:height, 1 : width + 1]
            p3 = padded[0:height, 2 : width + 2]
            p4 = padded[1 : height + 1, 2 : width + 2]
            p5 = padded[2 : height + 2, 2 : width + 2]
            p6 = padded[2 : height + 2, 1 : width + 1]
            p7 = padded[2 : height + 2, 0:width]
            p8 = padded[1 : height + 1, 0:width]
            p9 = padded[0:height, 0:width]
            neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)
            neighbor_count = sum(
                neighbor.astype(np.uint8) for neighbor in neighbors
            )
            circular = neighbors + (neighbors[0],)
            transitions = sum(
                np.logical_and(~circular[index], circular[index + 1])
                for index in range(8)
            )
            if first_pass:
                condition_a = ~(p2 & p4 & p6)
                condition_b = ~(p4 & p6 & p8)
            else:
                condition_a = ~(p2 & p4 & p8)
                condition_b = ~(p2 & p6 & p8)
            remove = (
                image
                & (neighbor_count >= 2)
                & (neighbor_count <= 6)
                & (transitions == 1)
                & condition_a
                & condition_b
            )
            if remove.any():
                changed = True
                image[remove] = False
    return image


def cldice(prediction: np.ndarray, truth: np.ndarray, epsilon: float = 1e-8) -> float:
    prediction_binary = np.asarray(prediction, dtype=bool)
    truth_binary = np.asarray(truth, dtype=bool)
    if not prediction_binary.any() and not truth_binary.any():
        return 1.0
    if not prediction_binary.any() or not truth_binary.any():
        return 0.0
    prediction_skeleton = skeletonize(prediction_binary)
    truth_skeleton = skeletonize(truth_binary)
    topology_precision = (
        np.logical_and(prediction_skeleton, truth_binary).sum()
        / (prediction_skeleton.sum() + epsilon)
    )
    topology_sensitivity = (
        np.logical_and(truth_skeleton, prediction_binary).sum()
        / (truth_skeleton.sum() + epsilon)
    )
    return float(
        2
        * topology_precision
        * topology_sensitivity
        / (topology_precision + topology_sensitivity + epsilon)
    )


def boundary_f1(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    tolerance: int = 2,
    epsilon: float = 1e-8,
) -> float:
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    prediction_boundary = mask_boundary(prediction)
    truth_boundary = mask_boundary(truth)
    if not prediction_boundary.any() and not truth_boundary.any():
        return 1.0
    if not prediction_boundary.any() or not truth_boundary.any():
        return 0.0
    truth_region = binary_dilation(truth_boundary, tolerance)
    prediction_region = binary_dilation(prediction_boundary, tolerance)
    precision = np.logical_and(prediction_boundary, truth_region).sum() / (
        prediction_boundary.sum() + epsilon
    )
    recall = np.logical_and(truth_boundary, prediction_region).sum() / (
        truth_boundary.sum() + epsilon
    )
    return float(2 * precision * recall / (precision + recall + epsilon))


def _minimum_distances(
    left: np.ndarray,
    right: np.ndarray,
    *,
    block_size: int = 1024,
) -> np.ndarray:
    if len(left) == 0 or len(right) == 0:
        return np.asarray([], dtype=float)
    right_float = right.astype(np.float32, copy=False)
    minima = []
    for start in range(0, len(left), block_size):
        block = left[start : start + block_size].astype(
            np.float32,
            copy=False,
        )
        squared = (
            (block[:, None, :] - right_float[None, :, :]) ** 2
        ).sum(axis=2)
        minima.append(np.sqrt(squared.min(axis=1)))
    return np.concatenate(minima).astype(float, copy=False)


def hd95(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction_boundary_mask = mask_boundary(prediction)
    truth_boundary_mask = mask_boundary(truth)
    prediction_boundary = np.argwhere(prediction_boundary_mask)
    truth_boundary = np.argwhere(truth_boundary_mask)
    if len(prediction_boundary) == 0 and len(truth_boundary) == 0:
        return 0.0
    if len(prediction_boundary) == 0 or len(truth_boundary) == 0:
        return float("inf")
    try:
        import cv2

        to_truth = cv2.distanceTransform(
            (~truth_boundary_mask).astype(np.uint8),
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
        )[prediction_boundary_mask]
        to_prediction = cv2.distanceTransform(
            (~prediction_boundary_mask).astype(np.uint8),
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
        )[truth_boundary_mask]
        bidirectional = np.concatenate([to_truth, to_prediction])
    except ImportError:
        bidirectional = np.concatenate(
            [
                _minimum_distances(prediction_boundary, truth_boundary),
                _minimum_distances(truth_boundary, prediction_boundary),
            ]
        )
    return float(np.percentile(bidirectional, 95))


def connected_component_count(mask: np.ndarray) -> int:
    """Count 8-connected foreground components in a two-dimensional mask."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("connected-component metrics require a 2-D mask")
    try:
        import cv2

        labels, _ = cv2.connectedComponents(
            binary.astype(np.uint8),
            connectivity=8,
        )
        return int(labels - 1)
    except ImportError:
        pass
    visited = np.zeros_like(binary, dtype=bool)
    height, width = binary.shape
    count = 0
    for start_y, start_x in np.argwhere(binary):
        if visited[start_y, start_x]:
            continue
        count += 1
        visited[start_y, start_x] = True
        stack = [(int(start_y), int(start_x))]
        while stack:
            y, x = stack.pop()
            for offset_y in (-1, 0, 1):
                for offset_x in (-1, 0, 1):
                    if offset_y == 0 and offset_x == 0:
                        continue
                    neighbor_y = y + offset_y
                    neighbor_x = x + offset_x
                    if not (
                        0 <= neighbor_y < height
                        and 0 <= neighbor_x < width
                    ):
                        continue
                    if (
                        binary[neighbor_y, neighbor_x]
                        and not visited[neighbor_y, neighbor_x]
                    ):
                        visited[neighbor_y, neighbor_x] = True
                        stack.append((neighbor_y, neighbor_x))
    return count


def component_count_error(
    prediction: np.ndarray,
    truth: np.ndarray,
) -> float:
    """Absolute error in the number of 8-connected components."""
    return float(
        abs(
            connected_component_count(prediction)
            - connected_component_count(truth)
        )
    )


def crack_break_rate(
    prediction: np.ndarray,
    truth: np.ndarray,
) -> float:
    """Fraction of excess predicted skeleton components caused by breaks.

    A completely missing non-empty crack has a break rate of one. Merges are
    captured separately by ``component_count_error`` and do not count as
    breaks.
    """
    prediction_count = connected_component_count(skeletonize(prediction))
    truth_count = connected_component_count(skeletonize(truth))
    if truth_count == 0:
        return 0.0
    if prediction_count == 0:
        return 1.0
    excess = max(0, prediction_count - truth_count)
    return float(excess / max(prediction_count, truth_count, 1))


def area_relative_error(
    prediction: np.ndarray,
    truth: np.ndarray,
) -> float:
    """Absolute relative foreground-area error."""
    prediction_area = float(np.asarray(prediction, dtype=bool).sum())
    truth_area = float(np.asarray(truth, dtype=bool).sum())
    if truth_area == 0:
        return 0.0 if prediction_area == 0 else 1.0
    return float(abs(prediction_area - truth_area) / truth_area)
