from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from morphoroute.data.materialize import MaterializedDataset
from morphoroute.evaluation.schema import (
    PredictionBundle,
    PredictionImage,
    PredictionInstance,
)


def _polygon_mask(
    points: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    canvas = Image.new("1", (width, height), 0)
    ImageDraw.Draw(canvas).polygon(
        [(float(x), float(y)) for x, y in points],
        fill=1,
    )
    return np.asarray(canvas, dtype=bool)


def _targets_for_image(
    image_path: Path,
    materialized: MaterializedDataset,
    *,
    width: int,
    height: int,
    class_names: tuple[str, ...],
    split_directory: str,
) -> tuple[PredictionInstance, ...]:
    label_path = (
        materialized.root
        / "labels"
        / split_directory
        / f"{image_path.stem}.txt"
    )
    if not label_path.is_file():
        return ()
    targets = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 7:
            continue
        class_id = int(values[0])
        normalized = np.asarray(
            [float(value) for value in values[1:]],
            dtype=float,
        ).reshape(-1, 2)
        points = normalized * np.asarray([width, height], dtype=float)
        mask = _polygon_mask(points, width, height)
        ys, xs = np.nonzero(mask)
        bbox = (
            float(xs.min()),
            float(ys.min()),
            float(xs.max() - xs.min() + 1),
            float(ys.max() - ys.min() + 1),
        )
        targets.append(
            PredictionInstance(
                class_name=class_names[class_id],
                score=1.0,
                bbox=bbox,
                mask=mask,
            )
        )
    return tuple(targets)


def convert_ultralytics_results(
    results: Iterable[object],
    materialized: MaterializedDataset,
    *,
    split: str,
    experiment_id: str,
) -> PredictionBundle:
    result_list = list(results)
    if not result_list:
        return PredictionBundle(
            experiment_id=experiment_id,
            split=split,
            class_names=(),
            images=(),
            framework="ultralytics",
        )
    names_payload = getattr(result_list[0], "names", {})
    if isinstance(names_payload, dict):
        class_names = tuple(
            str(names_payload[index]) for index in sorted(names_payload)
        )
    else:
        class_names = tuple(str(name) for name in names_payload)
    split_directory = {"D2": "val", "D3": "test", "D4": "external"}[split]
    images = []
    for result in result_list:
        image_path = Path(str(result.path))
        height, width = (int(value) for value in result.orig_shape)
        boxes = result.boxes
        mask_polygons = result.masks.xy if result.masks is not None else []
        predictions = []
        for index, points in enumerate(mask_polygons):
            class_id = int(boxes.cls[index].item())
            score = float(boxes.conf[index].item())
            xyxy = boxes.xyxy[index].detach().cpu().numpy()
            predictions.append(
                PredictionInstance(
                    class_name=class_names[class_id],
                    score=score,
                    bbox=(
                        float(xyxy[0]),
                        float(xyxy[1]),
                        float(xyxy[2] - xyxy[0]),
                        float(xyxy[3] - xyxy[1]),
                    ),
                    mask=_polygon_mask(
                        np.asarray(points),
                        width,
                        height,
                    ),
                )
            )
        images.append(
            PredictionImage(
                image_id=image_path.stem,
                group_id=image_path.stem,
                width=width,
                height=height,
                predictions=tuple(predictions),
                targets=_targets_for_image(
                    image_path,
                    materialized,
                    width=width,
                    height=height,
                    class_names=class_names,
                    split_directory=split_directory,
                ),
                image_path=str(image_path),
            )
        )
    return PredictionBundle(
        experiment_id=experiment_id,
        split=split,
        class_names=class_names,
        images=tuple(images),
        framework="ultralytics",
    )
