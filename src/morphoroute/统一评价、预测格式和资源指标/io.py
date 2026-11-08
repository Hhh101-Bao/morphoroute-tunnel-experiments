from __future__ import annotations

import json
from pathlib import Path

from .mask_codec import mask_from_json_rle, mask_to_json_rle
from .schema import (
    PredictionBundle,
    PredictionImage,
    PredictionInstance,
)


def prediction_instance_to_payload(instance: PredictionInstance) -> dict:
    return {
        "class_name": instance.class_name,
        "score": instance.score,
        "bbox": list(instance.bbox),
        "mask_rle": mask_to_json_rle(instance.mask),
        "logits": list(instance.logits),
        "query_id": instance.query_id,
        "route_probabilities": list(instance.route_probabilities),
        "active_experts": list(instance.active_experts),
        "latency_ms": instance.latency_ms,
    }


def prediction_instance_from_payload(payload: dict) -> PredictionInstance:
    return PredictionInstance(
        class_name=str(payload["class_name"]),
        score=float(payload.get("score", 1.0)),
        bbox=tuple(float(value) for value in payload["bbox"]),
        mask=mask_from_json_rle(payload["mask_rle"]),
        logits=tuple(float(value) for value in payload.get("logits", ())),
        query_id=payload.get("query_id"),
        route_probabilities=tuple(
            float(value)
            for value in payload.get("route_probabilities", ())
        ),
        active_experts=tuple(
            str(value) for value in payload.get("active_experts", ())
        ),
        latency_ms=(
            float(payload["latency_ms"])
            if payload.get("latency_ms") is not None
            else None
        ),
    )


def prediction_image_to_payload(image: PredictionImage) -> dict:
    return {
        "image_id": image.image_id,
        "group_id": image.group_id,
        "source": image.source,
        "domain": image.domain,
        "width": image.width,
        "height": image.height,
        "image_path": image.image_path,
        "predictions": [
            prediction_instance_to_payload(instance)
            for instance in image.predictions
        ],
        "targets": [
            prediction_instance_to_payload(instance)
            for instance in image.targets
        ],
    }


def prediction_image_from_payload(image: dict) -> PredictionImage:
    return PredictionImage(
        image_id=str(image["image_id"]),
        group_id=str(image.get("group_id", image["image_id"])),
        width=int(image["width"]),
        height=int(image["height"]),
        predictions=tuple(
            prediction_instance_from_payload(item)
            for item in image.get("predictions", ())
        ),
        targets=tuple(
            prediction_instance_from_payload(item)
            for item in image.get("targets", ())
        ),
        source=str(image.get("source", "")),
        domain=str(image.get("domain", "")),
        image_path=image.get("image_path"),
    )


def read_prediction_bundle(path: str | Path) -> PredictionBundle:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    images = [
        prediction_image_from_payload(image)
        for image in payload.get("images", [])
    ]
    return PredictionBundle(
        experiment_id=str(payload["experiment_id"]),
        split=str(payload["split"]),
        class_names=tuple(
            str(value) for value in payload.get("class_names", ())
        ),
        images=tuple(images),
        framework=str(payload.get("framework", "")),
        checkpoint_sha256=payload.get("checkpoint_sha256"),
        metadata=dict(payload.get("metadata", {})),
    )
