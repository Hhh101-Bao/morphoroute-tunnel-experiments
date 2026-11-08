from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TypeAlias

import numpy as np
from pycocotools import mask as mask_utils


@dataclass(frozen=True)
class EncodedMask:
    size: tuple[int, int]
    counts: str

    def __post_init__(self) -> None:
        height, width = (int(value) for value in self.size)
        if height <= 0 or width <= 0:
            raise ValueError("encoded mask dimensions must be positive")
        if not isinstance(self.counts, str):
            raise TypeError("encoded mask counts must be an ASCII string")
        object.__setattr__(self, "size", (height, width))


MaskLike: TypeAlias = np.ndarray | EncodedMask


def _compressed_rle(mask: EncodedMask) -> dict[str, object]:
    return {
        "size": [mask.size[0], mask.size[1]],
        "counts": mask.counts.encode("ascii"),
    }


def _from_compressed_rle(rle: Mapping[str, object]) -> EncodedMask:
    raw_size = rle["size"]
    if not isinstance(raw_size, (list, tuple)) or len(raw_size) != 2:
        raise ValueError("RLE size must contain height and width")
    raw_counts = rle["counts"]
    if isinstance(raw_counts, bytes):
        counts = raw_counts.decode("ascii")
    elif isinstance(raw_counts, str):
        counts = raw_counts
    else:
        raise TypeError("compressed RLE counts must be bytes or string")
    return EncodedMask(
        size=(int(raw_size[0]), int(raw_size[1])),
        counts=counts,
    )


def encode_mask(mask: MaskLike) -> EncodedMask:
    if isinstance(mask, EncodedMask):
        return mask
    dense = np.asarray(mask, dtype=np.uint8)
    if dense.ndim != 2:
        raise ValueError("instance mask must be two-dimensional")
    encoded = mask_utils.encode(np.asfortranarray(dense))
    return _from_compressed_rle(encoded)


def mask_from_json_rle(payload: Mapping[str, object]) -> EncodedMask:
    raw_size = payload.get("size")
    if not isinstance(raw_size, (list, tuple)) or len(raw_size) != 2:
        raise ValueError("RLE size must contain height and width")
    height, width = int(raw_size[0]), int(raw_size[1])
    raw_counts = payload.get("counts")
    if isinstance(raw_counts, list):
        compressed = mask_utils.frPyObjects(
            {"size": [height, width], "counts": raw_counts},
            height,
            width,
        )
        return _from_compressed_rle(compressed)
    return _from_compressed_rle(
        {"size": [height, width], "counts": raw_counts}
    )


def encode_coco_segmentation(
    segmentation: object,
    height: int,
    width: int,
) -> EncodedMask:
    height, width = int(height), int(width)
    if height <= 0 or width <= 0:
        raise ValueError("segmentation dimensions must be positive")
    if isinstance(segmentation, Mapping):
        payload = dict(segmentation)
        payload.setdefault("size", [height, width])
        return mask_from_json_rle(payload)
    if not isinstance(segmentation, list):
        raise TypeError("COCO segmentation must be an RLE object or polygon list")
    if not segmentation:
        return encode_mask(np.zeros((height, width), dtype=bool))
    polygons = segmentation
    if polygons and isinstance(polygons[0], (int, float)):
        polygons = [polygons]
    rles = mask_utils.frPyObjects(polygons, height, width)
    merged = mask_utils.merge(rles) if isinstance(rles, list) else rles
    return _from_compressed_rle(merged)


def decode_mask(mask: MaskLike) -> np.ndarray:
    if not isinstance(mask, EncodedMask):
        dense = np.asarray(mask, dtype=bool)
        if dense.ndim != 2:
            raise ValueError("instance mask must be two-dimensional")
        return dense
    decoded = mask_utils.decode(_compressed_rle(mask))
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return np.asarray(decoded, dtype=bool)


def mask_iou(left: MaskLike, right: MaskLike) -> float:
    left_encoded = encode_mask(left)
    right_encoded = encode_mask(right)
    if left_encoded.size != right_encoded.size:
        raise ValueError("mask dimensions must match")
    left_area = mask_utils.area(_compressed_rle(left_encoded))
    right_area = mask_utils.area(_compressed_rle(right_encoded))
    if float(left_area) == 0.0 and float(right_area) == 0.0:
        return 1.0
    overlap = mask_utils.iou(
        [_compressed_rle(left_encoded)],
        [_compressed_rle(right_encoded)],
        [0],
    )
    return float(overlap[0, 0])


def mask_area(mask: MaskLike) -> float:
    encoded = encode_mask(mask)
    return float(mask_utils.area(_compressed_rle(encoded)))


def mask_shape(mask: MaskLike) -> tuple[int, int]:
    if isinstance(mask, EncodedMask):
        return mask.size
    dense = np.asarray(mask)
    if dense.ndim != 2:
        raise ValueError("instance mask must be two-dimensional")
    return int(dense.shape[0]), int(dense.shape[1])


def mask_to_json_rle(mask: MaskLike) -> dict[str, object]:
    encoded = encode_mask(mask)
    return {
        "size": [encoded.size[0], encoded.size[1]],
        "counts": encoded.counts,
    }
