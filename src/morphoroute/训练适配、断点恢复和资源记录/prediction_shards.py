from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from morphoroute.core.io import atomic_write_json
from morphoroute.evaluation.io import (
    prediction_image_from_payload,
    prediction_image_to_payload,
)
from morphoroute.evaluation.schema import PredictionImage


class PredictionShardError(RuntimeError):
    pass


class ShardIdentityError(PredictionShardError):
    pass


@dataclass(frozen=True)
class ShardIdentity:
    experiment_id: str
    variant_id: str
    seed: int
    split: str
    checkpoint_sha256: str
    dataset_sha256: str
    config_sha256: str
    schema_version: int = 1


@dataclass(frozen=True)
class _ShardPart:
    path: Path
    start_index: int
    end_index: int
    image_ids: tuple[str, ...]
    images: tuple[PredictionImage, ...]
    route_rows: tuple[dict[str, object], ...]


class PredictionShardStore:
    def __init__(
        self,
        directory: str | Path,
        identity: ShardIdentity,
        shard_size: int,
    ) -> None:
        if int(shard_size) <= 0:
            raise ValueError("shard_size must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.shard_size = int(shard_size)
        self._parts = self._load_parts()

    def _load_parts(self) -> tuple[_ShardPart, ...]:
        parts = tuple(
            self._load_part(path)
            for path in sorted(self.directory.glob("part-*.json"))
        )
        expected_start = 0
        for part in parts:
            if part.start_index != expected_start:
                raise PredictionShardError(
                    f"non-contiguous prediction shard at {part.path}: "
                    f"expected start {expected_start}, got {part.start_index}"
                )
            expected_start = part.end_index + 1
        return parts

    def _load_part(self, path: Path) -> _ShardPart:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PredictionShardError(
                f"invalid prediction shard {path}: {error}"
            ) from error
        actual_identity = payload.get("identity")
        expected_identity = asdict(self.identity)
        if not isinstance(actual_identity, dict):
            raise PredictionShardError(f"prediction shard lacks identity: {path}")
        for key, expected in expected_identity.items():
            actual = actual_identity.get(key)
            if actual != expected:
                raise ShardIdentityError(
                    f"prediction shard {path} has incompatible {key}: "
                    f"expected {expected!r}, got {actual!r}"
                )
        try:
            start_index = int(payload["start_index"])
            end_index = int(payload["end_index"])
            image_ids = tuple(str(value) for value in payload["image_ids"])
            images = tuple(
                prediction_image_from_payload(value)
                for value in payload["images"]
            )
            route_rows = tuple(
                dict(value) for value in payload.get("route_diagnostics", ())
            )
        except (KeyError, TypeError, ValueError) as error:
            raise PredictionShardError(
                f"invalid prediction shard fields in {path}: {error}"
            ) from error
        expected_count = end_index - start_index + 1
        actual_ids = tuple(image.image_id for image in images)
        if expected_count <= 0 or len(images) != expected_count:
            raise PredictionShardError(
                f"prediction shard count mismatch in {path}"
            )
        if image_ids != actual_ids:
            raise PredictionShardError(
                f"prediction shard image IDs disagree in {path}"
            )
        route_image_ids = tuple(
            str(row.get("image_id", "")) for row in route_rows
        )
        if len(set(route_image_ids)) != len(route_image_ids) or not set(
            route_image_ids
        ).issubset(image_ids):
            raise PredictionShardError(
                f"prediction shard route diagnostics disagree in {path}"
            )
        expected_name = f"part-{start_index:06d}-{end_index:06d}.json"
        if path.name != expected_name:
            raise PredictionShardError(
                f"prediction shard filename mismatch: {path.name!r} "
                f"!= {expected_name!r}"
            )
        return _ShardPart(
            path=path,
            start_index=start_index,
            end_index=end_index,
            image_ids=image_ids,
            images=images,
            route_rows=route_rows,
        )

    def completed_image_ids(self) -> tuple[str, ...]:
        return tuple(
            image_id for part in self._parts for image_id in part.image_ids
        )

    def append(
        self,
        images: Iterable[PredictionImage],
        *,
        start_index: int,
        route_rows: Iterable[dict[str, object]] = (),
    ) -> Path:
        selected = tuple(images)
        selected_route_rows = tuple(dict(value) for value in route_rows)
        if not selected:
            raise ValueError("prediction shard must contain at least one image")
        if len(selected) > self.shard_size:
            raise ValueError("prediction shard exceeds configured shard_size")
        start_index = int(start_index)
        expected_start = (
            self._parts[-1].end_index + 1 if self._parts else 0
        )
        if start_index != expected_start:
            raise PredictionShardError(
                f"prediction shard must start at {expected_start}, "
                f"got {start_index}"
            )
        end_index = start_index + len(selected) - 1
        path = self.directory / (
            f"part-{start_index:06d}-{end_index:06d}.json"
        )
        payload = {
            "identity": asdict(self.identity),
            "start_index": start_index,
            "end_index": end_index,
            "image_ids": [image.image_id for image in selected],
            "images": [
                prediction_image_to_payload(image) for image in selected
            ],
            "route_diagnostics": list(selected_route_rows),
        }
        atomic_write_json(path, payload)
        part = self._load_part(path)
        self._parts = (*self._parts, part)
        return path

    def read_all_route_rows(self) -> tuple[dict[str, object], ...]:
        return tuple(
            row for part in self._parts for row in part.route_rows
        )

    def read_all_images(
        self,
        expected_order: Iterable[str],
    ) -> tuple[PredictionImage, ...]:
        images = tuple(image for part in self._parts for image in part.images)
        actual_order = tuple(image.image_id for image in images)
        requested_order = tuple(str(value) for value in expected_order)
        if actual_order != requested_order:
            raise PredictionShardError(
                "prediction shard image order does not match the dataset: "
                f"expected {requested_order!r}, got {actual_order!r}"
            )
        return images
