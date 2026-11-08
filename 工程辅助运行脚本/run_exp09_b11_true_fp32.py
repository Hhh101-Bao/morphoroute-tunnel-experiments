from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import torch

from morphoroute.runtime.pcgrad_fp32 import (
    POLICY_REVISION,
    apply_b11_true_fp32_config,
    install_hungarian_finite_guard,
    install_pcgrad_fp32_optim_wrapper,
)
from morphoroute.runtime.strict_fp32 import configure_torch_strict_fp32


def _variant_id(resolved) -> str | None:
    model_kind = str(getattr(resolved, "model_kind", ""))
    candidate = model_kind.rsplit("_", 1)[-1]
    return candidate if candidate.startswith("B") else None


def _get(container, key, default=None):
    if hasattr(container, "get"):
        return container.get(key, default)
    return getattr(container, key, default)


def build_configuration_wrapper(original_configuration):
    def fp32_configuration(resolved):
        configuration = original_configuration(resolved)
        variant = _variant_id(resolved)
        if variant != "B11":
            return configuration

        apply_b11_true_fp32_config(configuration, variant_id=variant)
        config_path = getattr(resolved, "config_path", None)
        if config_path is not None and hasattr(configuration, "dump"):
            configuration.dump(config_path)

        wrapper = _get(configuration, "optim_wrapper")
        wrapper_type = str(_get(wrapper, "type", ""))
        if wrapper_type != "PCGradFP32OptimWrapper":
            raise RuntimeError(
                "effective B11 optimizer wrapper is not true FP32: "
                f"{wrapper_type!r}"
            )
        batch_augments = _get(
            _get(_get(configuration, "model"), "data_preprocessor"),
            "batch_augments",
            (),
        )
        padding_types = [
            str(_get(augment, "type", "")) for augment in batch_augments
        ]
        if padding_types.count("MorphoRouteSafeBatchFixedSizePad") != 1:
            raise RuntimeError(
                "effective B11 configuration does not contain exactly one "
                f"safe batch pad: {padding_types}"
            )
        fusion_type = str(
            _get(
                _get(_get(configuration, "model"), "panoptic_fusion_head"),
                "type",
                "",
            )
        )
        if fusion_type != "MorphoRouteMemoryEfficientMaskFormerFusionHead":
            raise RuntimeError(
                "effective B11 configuration does not use the memory-"
                f"efficient fusion head: {fusion_type!r}"
            )
        print(
            "EXP09_B11_TRUE_FP32_POLICY "
            f"revision={POLICY_REVISION} variant=B11 "
            "optimizer_wrapper=PCGradFP32OptimWrapper "
            "amp=False grad_scaler=False tf32=False safe_batch_pad=True "
            "empty_morphology=exclude_route_only "
            "memory_efficient_instance_postprocess=True",
            flush=True,
        )
        return configuration

    return fp32_configuration


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) < 2:
        raise SystemExit(
            "usage: run_exp09_b11_true_fp32.py INNER_RUNNER ENTRY "
            "[entry arguments...]"
        )

    artifact_text = os.environ.get(
        "EXP09_B11_NUMERICAL_ARTIFACT_DIR", ""
    ).strip()
    if not artifact_text:
        raise RuntimeError("EXP09_B11_NUMERICAL_ARTIFACT_DIR is required")
    artifact_dir = Path(artifact_text).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    inner_runner = Path(arguments[0]).resolve()
    entry = Path(arguments[1]).resolve()
    if not inner_runner.is_file():
        raise FileNotFoundError(f"inner runner not found: {inner_runner}")
    if not entry.is_file():
        raise FileNotFoundError(f"entry not found: {entry}")

    torch_audit = configure_torch_strict_fp32(torch)
    if any(
        (
            torch_audit["cuda_matmul_allow_tf32"],
            torch_audit["cudnn_allow_tf32"],
        )
    ):
        raise RuntimeError(f"TF32 disable failed: {torch_audit}")

    install_pcgrad_fp32_optim_wrapper(artifact_dir)
    install_hungarian_finite_guard(artifact_dir)

    import morphoroute.mmdet.models  # noqa: F401
    from morphoroute.runtime.mmdet_adapter import OfficialMMDetBackend

    original_configuration = OfficialMMDetBackend._configuration
    OfficialMMDetBackend._configuration = staticmethod(
        build_configuration_wrapper(original_configuration)
    )

    print(
        "EXP09_B11_TRUE_FP32_RUNTIME "
        f"revision={POLICY_REVISION} "
        f"matmul_precision={torch_audit['matmul_precision']} "
        "cuda_matmul_tf32=False cudnn_tf32=False",
        flush=True,
    )

    sys.argv = [str(inner_runner), str(entry), *arguments[2:]]
    runpy.run_path(str(inner_runner), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
