from __future__ import annotations

import statistics
import time
from typing import Any, Callable


def profile_callable(
    operation: Callable[[], Any],
    *,
    warmup: int = 5,
    repeats: int = 30,
    synchronize: Callable[[], None] | None = None,
) -> dict[str, Any]:
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    for _ in range(warmup):
        operation()
    if synchronize is not None:
        synchronize()
    durations: list[float] = []
    for _ in range(repeats):
        if synchronize is not None:
            synchronize()
        started = time.perf_counter()
        operation()
        if synchronize is not None:
            synchronize()
        durations.append((time.perf_counter() - started) * 1000.0)
    return {
        "warmup": warmup,
        "repeats": repeats,
        "latency_ms_mean": statistics.mean(durations),
        "latency_ms_median": statistics.median(durations),
        "latency_ms_std": statistics.stdev(durations) if repeats > 1 else 0.0,
        "throughput_per_second": 1000.0 / statistics.mean(durations),
        "latency_ms_all": durations,
    }


def torch_resource_snapshot(model=None) -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "torch_available": False,
            "parameters": None,
            "peak_memory_allocated_bytes": None,
            "peak_memory_reserved_bytes": None,
        }
    parameters = (
        sum(parameter.numel() for parameter in model.parameters())
        if model is not None
        else None
    )
    cuda = torch.cuda.is_available()
    return {
        "torch_available": True,
        "cuda_available": cuda,
        "parameters": parameters,
        "peak_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if cuda else None
        ),
        "peak_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved()) if cuda else None
        ),
    }
