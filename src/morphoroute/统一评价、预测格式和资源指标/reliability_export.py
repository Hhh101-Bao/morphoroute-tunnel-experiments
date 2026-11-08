from __future__ import annotations

import json
import math

import numpy as np

from .mask_codec import MaskLike, mask_iou
from .schema import PredictionBundle, PredictionInstance


def _mask_iou(left: MaskLike, right: MaskLike) -> float:
    return mask_iou(left, right)


def _class_logits(
    instance: PredictionInstance | None,
    class_names: tuple[str, ...],
) -> list[float]:
    if instance is None:
        return [0.0] * len(class_names)
    if len(instance.logits) == len(class_names):
        return [float(value) for value in instance.logits]
    count = len(class_names)
    selected = class_names.index(instance.class_name)
    score = min(max(float(instance.score), 1e-6), 1.0 - 1e-6)
    remainder = (1.0 - score) / max(count - 1, 1)
    probabilities = [remainder] * count
    probabilities[selected] = score
    return [math.log(max(value, 1e-12)) for value in probabilities]


def reliability_rows_from_bundle(
    bundle: PredictionBundle,
    *,
    model: str,
    seed: int,
) -> list[dict[str, object]]:
    """Create target-anchored reliability rows from a prediction bundle."""

    rows: list[dict[str, object]] = []
    for image in bundle.images:
        for target_index, target in enumerate(image.targets):
            ranked = sorted(
                (
                    (_mask_iou(target.mask, prediction.mask), prediction)
                    for prediction in image.predictions
                ),
                key=lambda pair: pair[0],
                reverse=True,
            )
            best_iou, best = ranked[0] if ranked else (0.0, None)
            correct = bool(
                best is not None
                and best_iou >= 0.5
                and best.class_name == target.class_name
            )
            rows.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "split": bundle.split,
                    "image_id": image.image_id,
                    "group_id": image.group_id,
                    "target_index": target_index,
                    "label": bundle.class_names.index(target.class_name),
                    "logits_json": json.dumps(
                        _class_logits(best, bundle.class_names),
                        ensure_ascii=False,
                    ),
                    "matched_iou": best_iou,
                    "predicted_class": (
                        best.class_name if best is not None else "__missed__"
                    ),
                    "high_risk": not correct,
                }
            )
    return rows
