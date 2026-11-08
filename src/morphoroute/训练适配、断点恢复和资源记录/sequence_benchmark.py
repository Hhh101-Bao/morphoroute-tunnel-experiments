from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

from morphoroute.data.manifest import SampleRecord


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def _hardware_id(torch) -> str:
    gpu = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )
    return f"{platform.system()}|{platform.machine()}|{gpu}"


def benchmark_frozen_models(
    checkpoint_index_path: str | Path,
    records: list[SampleRecord],
    *,
    warmup: int = 5,
) -> list[dict[str, object]]:
    """Benchmark exact frozen models with separately synchronized stages."""

    import cv2
    import torch
    from mmcv.transforms import Compose
    from mmdet.apis import init_detector
    from mmdet.utils import get_test_pipeline_cfg

    d3_records = sorted(
        (record for record in records if record.dataset_id == "D3"),
        key=lambda record: (record.group_id, record.record_id),
    )
    if not d3_records:
        raise ValueError("sequence benchmark requires D3 images")
    index = json.loads(Path(checkpoint_index_path).read_text(encoding="utf-8"))
    output = []
    for frozen in index["models"]:
        model_name = str(frozen["model"])
        model = init_detector(
            str(frozen["config"]),
            str(frozen["checkpoint"]),
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
        pipeline_cfg = get_test_pipeline_cfg(model.cfg)
        pipeline_cfg[0].type = "mmdet.LoadImageFromNDArray"
        pipeline = Compose(pipeline_cfg)

        def prepare(record):
            raw = record.image_path.read_bytes()
            image = cv2.imdecode(
                np.frombuffer(raw, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if image is None:
                raise ValueError(f"cannot decode image: {record.image_path}")
            data = pipeline({"img": image, "img_id": record.record_id})
            data["inputs"] = [data["inputs"]]
            data["data_samples"] = [data["data_samples"]]
            return raw, image, data

        _, _, first_data = prepare(d3_records[0])
        for _ in range(max(0, warmup)):
            with torch.no_grad():
                model.test_step(first_data)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        for record in d3_records:
            started = time.perf_counter()
            raw = record.image_path.read_bytes()
            input_ms = _elapsed_ms(started)

            started = time.perf_counter()
            image = cv2.imdecode(
                np.frombuffer(raw, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            data = pipeline({"img": image, "img_id": record.record_id})
            data["inputs"] = [data["inputs"]]
            data["data_samples"] = [data["data_samples"]]
            preprocess_ms = _elapsed_ms(started)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                result = model.test_step(data)[0]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_ms = _elapsed_ms(started)

            started = time.perf_counter()
            instances = result.pred_instances.cpu()
            prediction_count = int(len(instances))
            postprocess_ms = _elapsed_ms(started)

            started = time.perf_counter()
            json.dumps(
                {
                    "labels": instances.labels.tolist(),
                    "scores": instances.scores.tolist(),
                    "bboxes": instances.bboxes.tolist(),
                },
                ensure_ascii=False,
            )
            output_ms = _elapsed_ms(started)

            active_experts = 0.0
            dual_ratio = 0.0
            head = getattr(model, "panoptic_head", None)
            diagnostics = getattr(head, "latest_route_diagnostics", None)
            if isinstance(diagnostics, dict) and "active_experts" in diagnostics:
                active = diagnostics["active_experts"].detach().float()
                active_experts = float(active.mean().item())
                dual_ratio = float((active > 1).float().mean().item())
            output.append(
                {
                    "model": model_name,
                    "record_id": record.record_id,
                    "sequence_id": record.group_id,
                    "hardware_id": _hardware_id(torch),
                    "protocol_id": "frozen-b1-stage-sync-v1",
                    "batch_size": 1,
                    "input_ms": input_ms,
                    "preprocess_ms": preprocess_ms,
                    "forward_ms": forward_ms,
                    "postprocess_ms": postprocess_ms,
                    "output_ms": output_ms,
                    "peak_memory_mb": (
                        float(torch.cuda.max_memory_allocated() / (1024**2))
                        if torch.cuda.is_available()
                        else 0.0
                    ),
                    "parameters": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "active_experts": active_experts,
                    "dual_activation_ratio": dual_ratio,
                    "prediction_count": prediction_count,
                    "missed": prediction_count == 0,
                }
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return output
