from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

from morphoroute.core.io import sha256_file
from morphoroute.data.materialize import MaterializedDataset
from morphoroute.evaluation.mask_codec import (
    encode_coco_segmentation,
    encode_mask,
)
from morphoroute.evaluation.schema import (
    PredictionBundle,
    PredictionImage,
    PredictionInstance,
)

from .prediction_shards import PredictionShardStore, ShardIdentity


def route_diagnostics_from_model(
    model,
    *,
    image_id: str,
) -> dict[str, object] | None:
    """Reduce the latest MorphoRoute query diagnostics to one image row."""
    head = getattr(model, "panoptic_head", None)
    diagnostics = getattr(head, "latest_route_diagnostics", None)
    if not isinstance(diagnostics, dict):
        return None
    required = {
        "route_probabilities",
        "active_experts",
        "route_entropy",
    }
    if not required.issubset(diagnostics):
        return None
    probabilities = (
        diagnostics["route_probabilities"]
        .detach()
        .float()
        .cpu()
        .numpy()
        .reshape(-1, 2)
    )
    active = (
        diagnostics["active_experts"]
        .detach()
        .float()
        .cpu()
        .numpy()
        .reshape(-1)
    )
    entropy = (
        diagnostics["route_entropy"]
        .detach()
        .float()
        .cpu()
        .numpy()
        .reshape(-1)
    )
    if len(probabilities) == 0:
        return None
    top = probabilities.argmax(axis=1)
    return {
        "image_id": image_id,
        "query_count": int(len(probabilities)),
        "mean_route_entropy": float(entropy.mean()),
        "mean_active_experts": float(active.mean()),
        "dual_activation_rate": float((active == 2).mean()),
        "line_probability_load": float(probabilities[:, 0].mean()),
        "area_probability_load": float(probabilities[:, 1].mean()),
        "line_top1_fraction": float((top == 0).mean()),
        "area_top1_fraction": float((top == 1).mean()),
        "training_stage": str(
            getattr(head, "latest_training_stage", "inference")
        ),
        "ground_truth_inputs_used": False,
    }


@dataclass(frozen=True)
class _CPUInferencePayload:
    image_record: dict
    annotations: tuple[dict, ...]
    categories: dict[int, str]
    instances: object
    route_row: dict[str, object] | None


def _ordered_parallel_map(
    function,
    values: Iterable[object],
    *,
    max_workers: int,
    max_pending: int,
):
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if max_pending < max_workers:
        raise ValueError("max_pending must be at least max_workers")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = deque()
        for value in values:
            pending.append(executor.submit(function, value))
            if len(pending) >= max_pending:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def _prediction_image_from_cpu_payload(
    payload: _CPUInferencePayload,
) -> tuple[PredictionImage, dict[str, object] | None]:
    image_record = payload.image_record
    instances = payload.instances
    masks = (
        instances.masks.numpy()
        if hasattr(instances, "masks")
        else np.zeros(
            (0, image_record["height"], image_record["width"]),
            dtype=bool,
        )
    )
    predictions = []
    for index, mask in enumerate(masks):
        label = int(instances.labels[index].item())
        score = float(instances.scores[index].item())
        xyxy = instances.bboxes[index].numpy()
        predictions.append(
            PredictionInstance(
                class_name=payload.categories[label + 1],
                score=score,
                bbox=(
                    float(xyxy[0]),
                    float(xyxy[1]),
                    float(xyxy[2] - xyxy[0]),
                    float(xyxy[3] - xyxy[1]),
                ),
                mask=encode_mask(mask),
            )
        )
    targets = tuple(
        PredictionInstance(
            class_name=payload.categories[int(annotation["category_id"])],
            score=1.0,
            bbox=tuple(float(value) for value in annotation["bbox"]),
            mask=encode_coco_segmentation(
                annotation["segmentation"],
                int(image_record["height"]),
                int(image_record["width"]),
            ),
        )
        for annotation in payload.annotations
    )
    image = PredictionImage(
        image_id=str(image_record["record_id"]),
        group_id=str(image_record["group_id"]),
        width=int(image_record["width"]),
        height=int(image_record["height"]),
        predictions=tuple(predictions),
        targets=targets,
        source=str(image_record.get("source", "")),
        image_path=str(Path(str(image_record["file_name"]))),
    )
    return image, payload.route_row


def _resolve_prediction_workers(explicit: int | None) -> int:
    if explicit is None:
        raw = os.environ.get("MORPHOROUTE_PREDICTION_WORKERS")
        explicit = min(8, os.cpu_count() or 1) if raw is None else int(raw)
    if explicit <= 0:
        raise ValueError("prediction CPU workers must be positive")
    return explicit


def _experiment_id_from_checkpoint(checkpoint: Path) -> str:
    parts = checkpoint.parts
    try:
        index = parts.index("experiments")
        return parts[index + 1]
    except (ValueError, IndexError):
        return "EXP-UNKNOWN"


def _variant_seed_from_checkpoint(checkpoint: Path) -> tuple[str, int]:
    seed_directory = checkpoint.parent
    variant_directory = seed_directory.parent
    seed_text = seed_directory.name.removeprefix("seed_")
    return variant_directory.name, int(seed_text) if seed_text.isdigit() else 0


def _rss_mib() -> float:
    status = Path("/proc/self/status")
    if not status.is_file():
        return 0.0
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return 0.0


def predict_mmdet_bundle(
    config_path: Path,
    checkpoint: Path,
    materialized: MaterializedDataset,
    *,
    split: str,
    shard_root: Path | None = None,
    cpu_workers: int | None = None,
    shard_size: int = 25,
    progress_interval: int = 25,
) -> PredictionBundle:
    from mmdet.apis import inference_detector, init_detector

    annotation_file = materialized.annotation_files[split]
    coco = json.loads(annotation_file.read_text(encoding="utf-8"))
    categories = {
        int(category["id"]): str(category["name"])
        for category in coco["categories"]
    }
    annotations_by_image: dict[int, list[dict]] = {}
    for annotation in coco["annotations"]:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(
            annotation
        )
    workers = _resolve_prediction_workers(cpu_workers)
    if shard_size <= 0 or progress_interval <= 0:
        raise ValueError("shard_size and progress_interval must be positive")
    checkpoint_hash = sha256_file(checkpoint)
    config_hash = sha256_file(config_path)
    variant_id, seed = _variant_seed_from_checkpoint(checkpoint)
    identity = ShardIdentity(
        experiment_id=_experiment_id_from_checkpoint(checkpoint),
        variant_id=variant_id,
        seed=seed,
        split=split,
        checkpoint_sha256=checkpoint_hash,
        dataset_sha256=materialized.fingerprint.sha256,
        config_sha256=config_hash,
    )
    selected_shard_root = (
        Path(shard_root)
        if shard_root is not None
        else checkpoint.parent / "prediction_shards" / split
    )
    store = PredictionShardStore(
        selected_shard_root,
        identity,
        shard_size,
    )
    expected_ids = tuple(
        str(record["record_id"]) for record in coco["images"]
    )
    completed_ids = store.completed_image_ids()
    if completed_ids != expected_ids[: len(completed_ids)]:
        raise ValueError("completed prediction shards are not a dataset prefix")
    remaining_records = coco["images"][len(completed_ids) :]
    model = None
    if remaining_records:
        model = init_detector(
            str(config_path),
            str(checkpoint),
            device="cuda:0",
        )

    def inference_payloads():
        for image_record in remaining_records:
            image_path = Path(str(image_record["file_name"]))
            result = inference_detector(model, str(image_path))
            route_row = route_diagnostics_from_model(
                model,
                image_id=str(image_record["record_id"]),
            )
            yield _CPUInferencePayload(
                image_record=image_record,
                annotations=tuple(
                    annotations_by_image.get(int(image_record["id"]), ())
                ),
                categories=categories,
                instances=result.pred_instances.cpu(),
                route_row=route_row,
            )

    started = perf_counter()
    batch: list[PredictionImage] = []
    batch_route_rows: list[dict[str, object]] = []
    completed_count = len(completed_ids)
    last_reported = completed_count
    processed = _ordered_parallel_map(
        _prediction_image_from_cpu_payload,
        inference_payloads(),
        max_workers=workers,
        max_pending=max(workers, workers * 2),
    )
    for image, route_row in processed:
        batch.append(image)
        if route_row is not None:
            batch_route_rows.append(route_row)
        if len(batch) == shard_size:
            store.append(
                batch,
                start_index=completed_count,
                route_rows=batch_route_rows,
            )
            completed_count += len(batch)
            batch = []
            batch_route_rows = []
        if completed_count > last_reported and (
            completed_count % progress_interval == 0
            or completed_count == len(expected_ids)
        ):
            elapsed = max(perf_counter() - started, 1e-9)
            newly_completed = completed_count - len(completed_ids)
            seconds_per_image = elapsed / max(newly_completed, 1)
            eta = seconds_per_image * (len(expected_ids) - completed_count)
            print(
                f"MMDetection {split} prediction "
                f"[{completed_count}/{len(expected_ids)}] "
                f"{seconds_per_image:.2f}s/image, "
                f"ETA {eta / 60.0:.1f}min, RSS {_rss_mib():.0f}MiB",
                flush=True,
            )
            last_reported = completed_count
    if batch:
        store.append(
            batch,
            start_index=completed_count,
            route_rows=batch_route_rows,
        )
        completed_count += len(batch)
    if completed_count > last_reported:
        elapsed = max(perf_counter() - started, 1e-9)
        newly_completed = completed_count - len(completed_ids)
        seconds_per_image = elapsed / max(newly_completed, 1)
        print(
            f"MMDetection {split} prediction "
            f"[{completed_count}/{len(expected_ids)}] "
            f"{seconds_per_image:.2f}s/image, ETA 0.0min, "
            f"RSS {_rss_mib():.0f}MiB",
            flush=True,
        )
    images = store.read_all_images(expected_ids)
    router_diagnostics = store.read_all_route_rows()
    return PredictionBundle(
        experiment_id=identity.experiment_id,
        split=split,
        class_names=tuple(categories[index] for index in sorted(categories)),
        images=images,
        framework="mmdetection",
        metadata={
            "router_diagnostics": router_diagnostics,
            "ground_truth_inputs_used": False,
            "prediction_cpu_workers": workers,
            "prediction_shard_size": shard_size,
            "prediction_shard_root": str(selected_shard_root),
            "checkpoint_sha256": checkpoint_hash,
        },
    )
