from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from morphoroute.data.annotations import decode_uncompressed_rle
from morphoroute.metrics.quantification import measure_mask


@dataclass(frozen=True)
class QuantificationResult:
    measurements: tuple[dict[str, object], ...]
    physical_complete: bool
    measurements_path: Path
    template_path: Path


def run_quantification(
    prediction_bundle_path: str | Path,
    *,
    metadata_path: str | Path | None,
    output_dir: str | Path,
) -> QuantificationResult:
    bundle = json.loads(
        Path(prediction_bundle_path).read_text(encoding="utf-8")
    )
    metadata: dict[str, dict] = {}
    if metadata_path is not None and Path(metadata_path).is_file():
        metadata_table = pd.read_csv(Path(metadata_path))
        if "image_id" not in metadata_table:
            raise ValueError("quantification metadata requires image_id")
        metadata = {
            str(row["image_id"]): row
            for row in metadata_table.to_dict("records")
        }
    rows = []
    template_rows = []
    for image in bundle.get("images", []):
        image_id = str(image["image_id"])
        image_metadata = metadata.get(image_id, {})
        raw_scale = image_metadata.get("scale_per_pixel")
        scale = (
            None
            if raw_scale is None or pd.isna(raw_scale)
            else float(raw_scale)
        )
        if scale is not None and scale <= 0:
            raise ValueError(f"invalid scale_per_pixel for {image_id}")
        for instance_index, instance in enumerate(
            image.get("predictions", [])
        ):
            mask = decode_uncompressed_rle(instance["mask_rle"])
            measured = measure_mask(mask, scale_per_pixel=scale)
            rows.append(
                {
                    "instance_id": f"{image_id}:{instance_index}",
                    "image_id": image_id,
                    "class_name": instance["class_name"],
                    "unit": "physical" if scale is not None else "pixel",
                    "scale_per_pixel": scale,
                    **measured,
                }
            )
        if image_id not in metadata:
            template_rows.append(
                {
                    "image_id": image_id,
                    "scale_per_pixel": None,
                    "scale_unit": None,
                    "reference_source": None,
                    "quality_grade": None,
                }
            )
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    measurements_path = directory / "measurements_pred.csv"
    template_path = directory / "quantification_metadata_template.csv"
    pd.DataFrame(rows).to_csv(measurements_path, index=False)
    pd.DataFrame(
        template_rows,
        columns=(
            "image_id",
            "scale_per_pixel",
            "scale_unit",
            "reference_source",
            "quality_grade",
        ),
    ).to_csv(template_path, index=False)
    physical_complete = bool(rows) and all(
        row["scale_per_pixel"] is not None for row in rows
    )
    return QuantificationResult(
        measurements=tuple(rows),
        physical_complete=physical_complete,
        measurements_path=measurements_path,
        template_path=template_path,
    )
