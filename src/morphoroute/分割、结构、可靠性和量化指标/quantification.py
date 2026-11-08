from __future__ import annotations

import math
from typing import Any

import numpy as np

from .structure import skeletonize


def skeleton_arc_length(skeleton: np.ndarray) -> float:
    binary = np.asarray(skeleton, dtype=bool)
    length = 0.0
    for y, x in np.argwhere(binary):
        for dy, dx, distance in (
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
        ):
            next_y, next_x = y + dy, x + dx
            if (
                0 <= next_y < binary.shape[0]
                and 0 <= next_x < binary.shape[1]
                and binary[next_y, next_x]
            ):
                length += distance
    return length


def local_skeleton_widths(mask: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    skeleton_points = np.argwhere(skeleton)
    background_points = np.argwhere(~binary)
    if len(skeleton_points) == 0 or len(background_points) == 0:
        return np.asarray([], dtype=float)
    widths: list[float] = []
    for point in skeleton_points:
        distances = np.sqrt(((background_points - point) ** 2).sum(axis=1))
        widths.append(2.0 * float(distances.min()))
    return np.asarray(widths, dtype=float)


def measure_mask(
    mask: np.ndarray,
    *,
    scale_per_pixel: float | None,
) -> dict[str, Any]:
    binary = np.asarray(mask, dtype=bool)
    area_pixels = int(binary.sum())
    if area_pixels == 0:
        return {
            "measurable": False,
            "reason": "empty_mask",
            "length_pixels": 0.0,
            "width_median_pixels": None,
            "width_p95_pixels": None,
            "area_pixels": 0,
            "length_physical": None,
            "width_median_physical": None,
            "width_p95_physical": None,
            "area_physical": None,
        }
    skeleton = skeletonize(binary)
    length_pixels = skeleton_arc_length(skeleton)
    widths = local_skeleton_widths(binary, skeleton)
    result = {
        "measurable": scale_per_pixel is not None,
        "reason": None if scale_per_pixel is not None else "missing_scale",
        "length_pixels": length_pixels,
        "width_median_pixels": float(np.median(widths)) if len(widths) else None,
        "width_p95_pixels": float(np.percentile(widths, 95)) if len(widths) else None,
        "area_pixels": area_pixels,
        "length_physical": None,
        "width_median_physical": None,
        "width_p95_physical": None,
        "area_physical": None,
    }
    if scale_per_pixel is not None:
        scale = float(scale_per_pixel)
        result.update(
            {
                "length_physical": length_pixels * scale,
                "width_median_physical": (
                    result["width_median_pixels"] * scale
                    if result["width_median_pixels"] is not None
                    else None
                ),
                "width_p95_physical": (
                    result["width_p95_pixels"] * scale
                    if result["width_p95_pixels"] is not None
                    else None
                ),
                "area_physical": area_pixels * scale * scale,
            }
        )
    return result
