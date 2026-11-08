from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import torch

from morphoroute.runtime.strict_fp32 import (
    POLICY_REVISION,
    apply_strict_fp32_config,
    configure_torch_strict_fp32,
    install_hungarian_finite_guard,
    install_strict_finite_optim_wrapper,
)


PRIOR_FAILURE_UPPER_BOUND = 55600


def validate_mode(mode: str, stop_iter: int | None) -> None:
    if mode not in {"formal", "probe"}:
        raise ValueError("EXP07_FP32_MODE must be formal or probe")
    if mode == "formal" and stop_iter is not None:
        raise ValueError("formal mode must not define a probe stop iteration")
    if mode == "probe":
        if stop_iter is None:
            raise ValueError("probe mode requires EXP07_FP32_STOP_ITER")
        if stop_iter <= PRIOR_FAILURE_UPPER_BOUND:
            raise ValueError(
                "probe stop iteration must cross both prior failure regions; "
                f"got {stop_iter}"
            )


def _variant_id(resolved) -> str:
    model_kind = str(getattr(resolved, "model_kind", ""))
    variant = model_kind.rsplit("_", 1)[-1]
    if variant not in {"B4", "B5", "B6"}:
        raise RuntimeError(
            "strict EXP-07 runner received unexpected model kind "
            f"{model_kind!r}"
        )
    return variant


def _mapping_get(container, key, default=None):
    if hasattr(container, "get"):
        return container.get(key, default)
    return getattr(container, key, default)


def _mapping_set(container, key, value) -> None:
    try:
        container[key] = value
    except TypeError:
        setattr(container, key, value)


def build_strict_configuration_wrapper(
    original_configuration,
    *,
    mode: str,
    stop_iter: int | None,
):
    validate_mode(mode, stop_iter)

    def strict_configuration(resolved):
        configuration = original_configuration(resolved)
        variant = _variant_id(resolved)
        if mode == "probe" and variant != "B4":
            raise RuntimeError("probe mode is restricted to B4")
        apply_strict_fp32_config(
            configuration,
            variant_id=variant,
            mode=mode,
        )
        if mode == "probe":
            train_cfg = _mapping_get(configuration, "train_cfg")
            configured_max = int(_mapping_get(train_cfg, "max_iters"))
            if stop_iter is None or stop_iter > configured_max:
                raise RuntimeError(
                    f"probe stop {stop_iter} exceeds configured max "
                    f"{configured_max}"
                )
            _mapping_set(train_cfg, "max_iters", int(stop_iter))
            default_hooks = _mapping_get(configuration, "default_hooks")
            if default_hooks is None:
                default_hooks = {}
                _mapping_set(configuration, "default_hooks", default_hooks)
            checkpoint_hook = _mapping_get(default_hooks, "checkpoint")
            if checkpoint_hook is None:
                checkpoint_hook = {"type": "CheckpointHook"}
                _mapping_set(default_hooks, "checkpoint", checkpoint_hook)
            _mapping_set(checkpoint_hook, "save_last", True)

        config_path = getattr(resolved, "config_path", None)
        if config_path is not None and hasattr(configuration, "dump"):
            configuration.dump(config_path)

        wrapper = _mapping_get(configuration, "optim_wrapper")
        wrapper_type = str(_mapping_get(wrapper, "type"))
        if wrapper_type != "StrictFiniteOptimWrapper":
            raise RuntimeError(
                f"effective optimizer wrapper is not strict FP32: {wrapper_type}"
            )
        print(
            "EXP07_STRICT_FP32_POLICY "
            f"revision={POLICY_REVISION} variant={variant} mode={mode} "
            f"optimizer_wrapper={wrapper_type} amp=False tf32=False",
            flush=True,
        )
        return configuration

    return strict_configuration


def _build_probe_train_wrapper(original_train, stop_iter: int):
    def probe_train(self, resolved, *, resume: bool):
        checkpoint = original_train(self, resolved, resume=resume)
        print(
            "EXP07_STRICT_FP32_PROBE_REACHED_LIMIT "
            f"max_iters={stop_iter} checkpoint={checkpoint}",
            flush=True,
        )
        raise SystemExit(0)

    return probe_train


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) < 2:
        raise SystemExit(
            "usage: run_exp07_strict_fp32.py INNER_RUNNER ENTRY "
            "[entry arguments...]"
        )

    mode = os.environ.get("EXP07_FP32_MODE", "").strip()
    stop_text = os.environ.get("EXP07_FP32_STOP_ITER", "").strip()
    stop_iter = int(stop_text) if stop_text else None
    validate_mode(mode, stop_iter)

    artifact_text = os.environ.get(
        "EXP07_NUMERICAL_ARTIFACT_DIR",
        "",
    ).strip()
    if not artifact_text:
        raise RuntimeError("EXP07_NUMERICAL_ARTIFACT_DIR is required")
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

    install_strict_finite_optim_wrapper(artifact_dir)
    install_hungarian_finite_guard(artifact_dir)

    import morphoroute.mmdet.models  # noqa: F401
    from morphoroute.runtime.mmdet_adapter import OfficialMMDetBackend

    original_configuration = OfficialMMDetBackend._configuration
    original_train = OfficialMMDetBackend.train
    OfficialMMDetBackend._configuration = staticmethod(
        build_strict_configuration_wrapper(
            original_configuration,
            mode=mode,
            stop_iter=stop_iter,
        )
    )
    if mode == "probe":
        if stop_iter is None:
            raise RuntimeError("validated probe has no stop iteration")
        OfficialMMDetBackend.train = _build_probe_train_wrapper(
            original_train,
            stop_iter,
        )

    print(
        "EXP07_STRICT_FP32_RUNTIME "
        f"revision={POLICY_REVISION} mode={mode} "
        f"matmul_precision={torch_audit['matmul_precision']} "
        "cuda_matmul_tf32=False cudnn_tf32=False",
        flush=True,
    )

    sys.argv = [str(inner_runner), str(entry), *arguments[2:]]
    runpy.run_path(str(inner_runner), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
