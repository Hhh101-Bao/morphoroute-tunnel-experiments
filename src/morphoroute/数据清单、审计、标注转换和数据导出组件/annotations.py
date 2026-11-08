from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class InstanceAnnotation:
    class_name: str
    segmentation: Any
    bbox: tuple[float, float, float, float]
    area: float


def encode_uncompressed_rle(mask: np.ndarray) -> dict[str, Any]:
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2:
        raise ValueError("RLE mask must be two-dimensional")
    pixels = binary.flatten(order="F")
    counts: list[int] = []
    current = 0
    run_length = 0
    for pixel in pixels:
        value = int(pixel != 0)
        if value == current:
            run_length += 1
        else:
            counts.append(run_length)
            run_length = 1
            current = value
    counts.append(run_length)
    return {"size": [int(binary.shape[0]), int(binary.shape[1])], "counts": counts}


def decode_uncompressed_rle(rle: Mapping[str, Any]) -> np.ndarray:
    height, width = (int(value) for value in rle["size"])
    values: list[int] = []
    current = 0
    for count in rle["counts"]:
        values.extend([current] * int(count))
        current = 1 - current
    if len(values) != height * width:
        raise ValueError("RLE count total does not match mask dimensions")
    return np.asarray(values, dtype=bool).reshape((height, width), order="F")


def _connected_components(binary: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(binary, dtype=bool)
    visited = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            component = np.zeros_like(mask, dtype=bool)
            stack = [(y, x)]
            visited[y, x] = True
            while stack:
                current_y, current_x = stack.pop()
                component[current_y, current_x] = True
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        next_y = current_y + dy
                        next_x = current_x + dx
                        if (
                            0 <= next_y < height
                            and 0 <= next_x < width
                            and mask[next_y, next_x]
                            and not visited[next_y, next_x]
                        ):
                            visited[next_y, next_x] = True
                            stack.append((next_y, next_x))
            components.append(component)
    return components


def tack_mask_to_instances(
    label_mask: np.ndarray,
    value_to_class: Mapping[int, str],
) -> list[InstanceAnnotation]:
    array = np.asarray(label_mask)
    if array.ndim != 2:
        raise ValueError("TACK annotation must be a single-channel mask")
    instances: list[InstanceAnnotation] = []
    for raw_value, class_name in sorted(value_to_class.items()):
        for component in _connected_components(array == int(raw_value)):
            ys, xs = np.nonzero(component)
            if len(xs) == 0:
                continue
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            instances.append(
                InstanceAnnotation(
                    class_name=class_name,
                    segmentation=encode_uncompressed_rle(component),
                    bbox=(
                        float(x_min),
                        float(y_min),
                        float(x_max - x_min + 1),
                        float(y_max - y_min + 1),
                    ),
                    area=float(component.sum()),
                )
            )
    return instances


def _polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
        )
    ) / 2.0


def _same_point(
    first: tuple[float, float],
    second: tuple[float, float],
    *,
    tolerance: float = 1e-6,
) -> bool:
    return (
        abs(first[0] - second[0]) <= tolerance
        and abs(first[1] - second[1]) <= tolerance
    )


def dacl_json_to_instances(
    payload: Mapping[str, Any],
    label_to_class: Mapping[str, str],
    *,
    ignore_unknown_labels: bool = False,
    clip_out_of_bounds_tolerance: float = 1e-6,
    drop_out_of_bounds_spikes: bool = False,
) -> list[InstanceAnnotation]:
    width = float(payload["imageWidth"])
    height = float(payload["imageHeight"])
    instances: list[InstanceAnnotation] = []
    for index, shape in enumerate(payload.get("shapes", [])):
        label = str(shape.get("label", ""))
        if label not in label_to_class:
            if ignore_unknown_labels:
                continue
            raise ValueError(f"unknown DACL label at shape {index}: {label}")
        raw_points = shape.get("points", [])
        if len(raw_points) < 3:
            raise ValueError(f"DACL polygon {index} requires at least three points")
        converted = [(float(raw_x), float(raw_y)) for raw_x, raw_y in raw_points]
        points: list[tuple[float, float]] = []
        for position, (x, y) in enumerate(converted):
            excess = max(-x, -y, x - width, y - height, 0.0)
            if excess > clip_out_of_bounds_tolerance:
                previous_point = converted[(position - 1) % len(converted)]
                next_point = converted[(position + 1) % len(converted)]
                if (
                    drop_out_of_bounds_spikes
                    and _same_point(previous_point, next_point)
                ):
                    continue
                raise ValueError(f"DACL polygon {index} has out-of-bounds coordinates")
            clipped = (min(max(x, 0.0), width), min(max(y, 0.0), height))
            if not points or not _same_point(points[-1], clipped):
                points.append(clipped)
        if len(points) > 1 and _same_point(points[0], points[-1]):
            points.pop()
        if len(points) < 3:
            raise ValueError(
                f"DACL polygon {index} requires at least three distinct points"
            )
        area = _polygon_area(points)
        if area <= 1e-9:
            raise ValueError(f"DACL polygon {index} is degenerate")
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        instances.append(
            InstanceAnnotation(
                class_name=label_to_class[label],
                segmentation=[[coordinate for point in points for coordinate in point]],
                bbox=(x_min, y_min, x_max - x_min, y_max - y_min),
                area=area,
            )
        )
    return instances
