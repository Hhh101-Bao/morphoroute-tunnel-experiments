from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from statistics import fmean
import tempfile

import numpy as np
from PIL import Image

from morphoroute.core.io import atomic_write_json
from morphoroute.metrics.structure import (
    area_relative_error,
    boundary_f1,
    cldice,
    component_count_error,
    crack_break_rate,
    hd95,
)
from morphoroute.visualization.plots import plot_prediction_overlay
from morphoroute.visualization.tables import save_result_bundle

from .io import prediction_image_to_payload
from .mask_codec import decode_mask, mask_iou as _encoded_mask_iou
from .schema import PredictionBundle, PredictionImage, PredictionInstance


AREA_CLASSES = frozenset(
    {"water_leakage", "spalling", "efflorescence", "exposed_rebar"}
)


@dataclass(frozen=True)
class EvaluationResult:
    experiment_id: str
    split: str
    image_count: int
    prediction_count: int
    target_count: int
    box_ap: float
    box_ap50: float
    box_ap75: float
    mask_ap: float
    mask_ap50: float
    mask_ap75: float
    macro_mask_ap: float
    worst_class_ap: float | None
    crack_cldice: float | None
    crack_break_rate: float | None
    crack_component_error: float | None
    area_boundary_f1: float | None
    area_hd95: float | None
    area_relative_error: float | None
    per_class_ap: dict[str, float | None]
    per_class_ap50: dict[str, float | None]


@dataclass(frozen=True)
class EvaluationOutputPaths:
    metrics_csv: Path
    metrics_xlsx: Path
    summary_json: Path
    overlay_png: Path
    overlay_pdf: Path


def mask_iou(left, right) -> float:
    return _encoded_mask_iou(left, right)


def bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_x1, left_y1, left_w, left_h = left
    right_x1, right_y1, right_w, right_h = right
    left_x2, left_y2 = left_x1 + left_w, left_y1 + left_h
    right_x2, right_y2 = right_x1 + right_w, right_y1 + right_h
    intersection_w = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection_h = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = intersection_w * intersection_h
    union = left_w * left_h + right_w * right_h - intersection
    return float(intersection / union) if union > 0 else 0.0


def _class_overlap_rows(
    image: PredictionImage,
    class_name: str,
    *,
    overlap,
) -> tuple[int, list[tuple[float, str, tuple[float, ...]]]]:
    targets = [
        target for target in image.targets if target.class_name == class_name
    ]
    rows = []
    for prediction in image.predictions:
        if prediction.class_name != class_name:
            continue
        rows.append(
            (
                float(prediction.score),
                image.image_id,
                tuple(overlap(prediction, target) for target in targets),
            )
        )
    return len(targets), rows


def _average_precision_from_overlap_rows(
    image_rows: list[tuple[int, list[tuple[float, str, tuple[float, ...]]]]],
    *,
    threshold: float,
) -> float | None:
    total_targets = sum(target_count for target_count, _ in image_rows)
    if total_targets == 0:
        return None
    ranked = sorted(
        (row for _, rows in image_rows for row in rows),
        key=lambda item: item[0],
        reverse=True,
    )
    matched: dict[str, set[int]] = {}
    true_positive = []
    false_positive = []
    for _, image_id, overlaps in ranked:
        matched.setdefault(image_id, set())
        candidates = [
            (value, index)
            for index, value in enumerate(overlaps)
            if index not in matched[image_id]
        ]
        best_overlap, best_index = max(candidates, default=(0.0, -1))
        is_true = best_index >= 0 and best_overlap >= threshold
        if is_true:
            matched[image_id].add(best_index)
        true_positive.append(1.0 if is_true else 0.0)
        false_positive.append(0.0 if is_true else 1.0)
    if not ranked:
        return 0.0
    tp = np.cumsum(np.asarray(true_positive))
    fp = np.cumsum(np.asarray(false_positive))
    recall = tp / total_targets
    precision = tp / np.maximum(tp + fp, 1e-12)
    interpolated = [
        float(precision[recall >= level].max())
        if np.any(recall >= level)
        else 0.0
        for level in np.linspace(0.0, 1.0, 101)
    ]
    return float(np.mean(interpolated))


def _class_average_precisions(
    bundle: PredictionBundle,
    class_name: str,
    *,
    thresholds: tuple[float, ...],
    overlap,
    workers: int,
) -> dict[float, float | None]:
    if workers <= 0:
        raise ValueError("evaluation workers must be positive")
    if workers == 1:
        image_rows = [
            _class_overlap_rows(
                image,
                class_name,
                overlap=overlap,
            )
            for image in bundle.images
        ]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            image_rows = list(
                executor.map(
                    lambda image: _class_overlap_rows(
                        image,
                        class_name,
                        overlap=overlap,
                    ),
                    bundle.images,
                )
            )
    return {
        threshold: _average_precision_from_overlap_rows(
            image_rows,
            threshold=threshold,
        )
        for threshold in thresholds
    }


def _resolve_evaluation_workers(explicit: int | None) -> int:
    if explicit is None:
        raw = os.environ.get("MORPHOROUTE_EVALUATION_WORKERS")
        explicit = min(8, os.cpu_count() or 1) if raw is None else int(raw)
    if explicit <= 0:
        raise ValueError("evaluation workers must be positive")
    return explicit


def _match_instances(
    image: PredictionImage,
    class_name: str,
    *,
    iou_threshold: float = 0.5,
) -> tuple[
    list[tuple[PredictionInstance, PredictionInstance]],
    int,
    int,
]:
    predictions = sorted(
        (
            item
            for item in image.predictions
            if item.class_name == class_name
        ),
        key=lambda item: item.score,
        reverse=True,
    )
    targets = [
        item for item in image.targets if item.class_name == class_name
    ]
    unmatched = set(range(len(targets)))
    matches: list[tuple[PredictionInstance, PredictionInstance]] = []
    false_positives = 0
    for prediction in predictions:
        candidates = [
            (mask_iou(prediction.mask, targets[index].mask), index)
            for index in unmatched
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_index >= 0 and best_iou >= iou_threshold:
            unmatched.remove(best_index)
            matches.append((prediction, targets[best_index]))
        else:
            false_positives += 1
    return matches, false_positives, len(unmatched)


def evaluate_bundle(
    bundle: PredictionBundle,
    *,
    mask_iou_workers: int | None = None,
) -> EvaluationResult:
    mask_iou_workers = _resolve_evaluation_workers(mask_iou_workers)
    thresholds = tuple(float(value) for value in np.arange(0.50, 0.96, 0.05))
    per_class_mask_thresholds: dict[str, dict[float, float | None]] = {}
    per_class_box_thresholds: dict[str, dict[float, float | None]] = {}
    crack_scores: list[float] = []
    crack_break_scores: list[float] = []
    crack_component_errors: list[float] = []
    area_scores: list[float] = []
    area_hd95_scores: list[float] = []
    area_relative_errors: list[float] = []
    for class_name in bundle.class_names:
        per_class_mask_thresholds[class_name] = _class_average_precisions(
            bundle,
            class_name,
            thresholds=thresholds,
            overlap=lambda prediction, target: mask_iou(
                prediction.mask,
                target.mask,
            ),
            workers=mask_iou_workers,
        )
        per_class_box_thresholds[class_name] = _class_average_precisions(
            bundle,
            class_name,
            thresholds=thresholds,
            overlap=lambda prediction, target: bbox_iou(
                prediction.bbox,
                target.bbox,
            ),
            workers=1,
        )
        for image in bundle.images:
            matches, _, _ = _match_instances(image, class_name)
            for prediction, target in matches:
                prediction_mask = decode_mask(prediction.mask)
                target_mask = decode_mask(target.mask)
                if class_name == "crack":
                    crack_scores.append(cldice(prediction_mask, target_mask))
                    crack_break_scores.append(
                        crack_break_rate(prediction_mask, target_mask)
                    )
                    crack_component_errors.append(
                        component_count_error(
                            prediction_mask,
                            target_mask,
                        )
                    )
                elif class_name in AREA_CLASSES:
                    area_scores.append(
                        boundary_f1(prediction_mask, target_mask)
                    )
                    area_hd95_scores.append(
                        hd95(prediction_mask, target_mask)
                    )
                    area_relative_errors.append(
                        area_relative_error(
                            prediction_mask,
                            target_mask,
                        )
                    )
    per_class_ap50 = {
        name: scores[thresholds[0]]
        for name, scores in per_class_mask_thresholds.items()
    }
    per_class_ap = {
        name: (
            fmean(
                value for value in scores.values() if value is not None
            )
            if any(value is not None for value in scores.values())
            else None
        )
        for name, scores in per_class_mask_thresholds.items()
    }

    def threshold_macro(table, threshold):
        values = [
            scores[threshold]
            for scores in table.values()
            if scores[threshold] is not None
        ]
        return fmean(values) if values else 0.0

    mask_by_threshold = [
        threshold_macro(per_class_mask_thresholds, threshold)
        for threshold in thresholds
    ]
    box_by_threshold = [
        threshold_macro(per_class_box_thresholds, threshold)
        for threshold in thresholds
    ]
    available_ap = [
        value for value in per_class_ap.values() if value is not None
    ]
    return EvaluationResult(
        experiment_id=bundle.experiment_id,
        split=bundle.split,
        image_count=len(bundle.images),
        prediction_count=sum(len(image.predictions) for image in bundle.images),
        target_count=sum(len(image.targets) for image in bundle.images),
        box_ap=fmean(box_by_threshold),
        box_ap50=box_by_threshold[0],
        box_ap75=box_by_threshold[5],
        mask_ap=fmean(mask_by_threshold),
        mask_ap50=mask_by_threshold[0],
        mask_ap75=mask_by_threshold[5],
        macro_mask_ap=fmean(available_ap) if available_ap else 0.0,
        worst_class_ap=min(available_ap) if available_ap else None,
        crack_cldice=fmean(crack_scores) if crack_scores else None,
        crack_break_rate=(
            fmean(crack_break_scores) if crack_break_scores else None
        ),
        crack_component_error=(
            fmean(crack_component_errors)
            if crack_component_errors
            else None
        ),
        area_boundary_f1=fmean(area_scores) if area_scores else None,
        area_hd95=(
            fmean(area_hd95_scores) if area_hd95_scores else None
        ),
        area_relative_error=(
            fmean(area_relative_errors)
            if area_relative_errors
            else None
        ),
        per_class_ap=per_class_ap,
        per_class_ap50=per_class_ap50,
    )


def _metric_rows(result: EvaluationResult) -> list[dict[str, object]]:
    rows = [
        {
            "split": result.split,
            "scope": "overall",
            "metric": "box_ap",
            "value": result.box_ap,
        },
        {
            "split": result.split,
            "scope": "overall",
            "metric": "box_ap50",
            "value": result.box_ap50,
        },
        {
            "split": result.split,
            "scope": "overall",
            "metric": "box_ap75",
            "value": result.box_ap75,
        },
        {
            "split": result.split,
            "scope": "overall",
            "metric": "mask_ap",
            "value": result.mask_ap,
        },
        {
            "split": result.split,
            "scope": "overall",
            "metric": "mask_ap75",
            "value": result.mask_ap75,
        },
        {
            "split": result.split,
            "scope": "overall",
            "metric": "worst_class_ap",
            "value": result.worst_class_ap,
        },
        {
            "split": result.split,
            "scope": "overall",
            "metric": "mask_ap50",
            "value": result.mask_ap50,
        },
        {
            "split": result.split,
            "scope": "crack",
            "metric": "clDice",
            "value": result.crack_cldice,
        },
        {
            "split": result.split,
            "scope": "crack",
            "metric": "break_rate",
            "value": result.crack_break_rate,
        },
        {
            "split": result.split,
            "scope": "crack",
            "metric": "connected_component_error",
            "value": result.crack_component_error,
        },
        {
            "split": result.split,
            "scope": "area_diseases",
            "metric": "boundary_f1",
            "value": result.area_boundary_f1,
        },
        {
            "split": result.split,
            "scope": "area_diseases",
            "metric": "HD95_pixels",
            "value": result.area_hd95,
        },
        {
            "split": result.split,
            "scope": "area_diseases",
            "metric": "relative_area_error",
            "value": result.area_relative_error,
        },
    ]
    rows.extend(
        {
            "split": result.split,
            "scope": class_name,
            "metric": "mask_ap50",
            "value": value,
        }
        for class_name, value in result.per_class_ap50.items()
    )
    rows.extend(
        {
            "split": result.split,
            "scope": class_name,
            "metric": "mask_ap",
            "value": value,
        }
        for class_name, value in result.per_class_ap.items()
    )
    return rows


def write_evaluation_outputs(
    result: EvaluationResult,
    bundle: PredictionBundle,
    output_dir: str | Path,
) -> EvaluationOutputPaths:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    table_paths = save_result_bundle(
        _metric_rows(result),
        directory / "metrics",
    )
    summary_json = atomic_write_json(
        directory / "evaluation_summary.json",
        asdict(result),
    )
    if bundle.images:
        first = bundle.images[0]
        image_path = Path(first.image_path) if first.image_path else None
        if image_path is not None and image_path.is_file():
            image = np.asarray(
                Image.open(image_path).convert("RGB"),
                dtype=np.uint8,
            )
        else:
            image = np.zeros(
                (first.height, first.width, 3),
                dtype=np.uint8,
            )
        figure_paths = plot_prediction_overlay(
            image,
            [decode_mask(item.mask) for item in first.predictions],
            directory / "prediction_overlay",
        )
    else:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        figure_paths = plot_prediction_overlay(
            image,
            [],
            directory / "prediction_overlay",
        )
    return EvaluationOutputPaths(
        metrics_csv=table_paths["csv"],
        metrics_xlsx=table_paths["xlsx"],
        summary_json=summary_json,
        overlay_png=figure_paths["png"],
        overlay_pdf=figure_paths["pdf"],
    )


def write_prediction_bundle(
    bundle: PredictionBundle,
    output_path: str | Path,
) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
            newline="\n",
        ) as handle:
            temporary_name = handle.name
            header = {
                "schema_version": 2,
                "experiment_id": bundle.experiment_id,
                "split": bundle.split,
                "framework": bundle.framework,
                "checkpoint_sha256": bundle.checkpoint_sha256,
                "class_names": list(bundle.class_names),
                "metadata": dict(bundle.metadata),
            }
            handle.write("{\n")
            for key, value in header.items():
                handle.write(f"{json.dumps(key)}: ")
                json.dump(value, handle, ensure_ascii=False, sort_keys=True)
                handle.write(",\n")
            handle.write('"images": [')
            _write_prediction_images(handle, bundle.images)
            handle.write("]\n}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    return target


def _write_prediction_images(handle, images) -> None:
    for index, image in enumerate(images):
        if index:
            handle.write(",")
        handle.write("\n")
        json.dump(
            prediction_image_to_payload(image),
            handle,
            ensure_ascii=False,
            sort_keys=True,
        )
    if images:
        handle.write("\n")
