from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


class CheckpointVerificationError(RuntimeError):
    """Raised when a checkpoint cannot be used as strict-FP32 evidence."""


def _contains_key(value, expected: str) -> bool:
    if isinstance(value, dict):
        return any(
            str(key) == expected or _contains_key(item, expected)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_key(item, expected) for item in value)
    return False


def _nonfinite_tensors(value, prefix: str):
    failures = []
    tensor_count = 0
    if torch.is_tensor(value):
        tensor_count = 1
        if value.is_floating_point() or value.is_complex():
            if not bool(torch.isfinite(value).all().item()):
                failures.append(prefix)
        return tensor_count, failures
    if isinstance(value, dict):
        for key, item in value.items():
            count, child_failures = _nonfinite_tensors(
                item,
                f"{prefix}.{key}" if prefix else str(key),
            )
            tensor_count += count
            failures.extend(child_failures)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            count, child_failures = _nonfinite_tensors(
                item,
                f"{prefix}[{index}]",
            )
            tensor_count += count
            failures.extend(child_failures)
    return tensor_count, failures


def verify_checkpoint(
    path: str | Path,
    *,
    allow_amp_diagnostic: bool = False,
) -> dict[str, object]:
    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.is_file():
        raise CheckpointVerificationError(
            f"checkpoint does not exist: {checkpoint_path}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", {})
    optimizer = checkpoint.get("optimizer", {})
    state_count, state_bad = _nonfinite_tensors(state_dict, "state_dict")
    optimizer_count, optimizer_bad = _nonfinite_tensors(
        optimizer,
        "optimizer",
    )
    bad = state_bad + optimizer_bad
    contains_loss_scaler = _contains_key(optimizer, "loss_scaler")
    report = {
        "checkpoint": str(checkpoint_path),
        "top_level_keys": sorted(checkpoint.keys()),
        "model_tensor_count": state_count,
        "optimizer_tensor_count": optimizer_count,
        "nonfinite_tensor_count": len(bad),
        "nonfinite_tensors": bad[:50],
        "contains_loss_scaler": contains_loss_scaler,
        "amp_state_diagnostic_only": bool(
            contains_loss_scaler and allow_amp_diagnostic
        ),
    }
    if bad or (contains_loss_scaler and not allow_amp_diagnostic):
        raise CheckpointVerificationError(
            json.dumps(report, ensure_ascii=False, sort_keys=True)
        )
    return report


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    allow_amp_diagnostic = False
    if arguments and arguments[0] == "--allow-amp-diagnostic":
        allow_amp_diagnostic = True
        arguments.pop(0)
    if len(arguments) != 1:
        raise SystemExit(
            "usage: verify_exp07_strict_fp32_checkpoint.py "
            "[--allow-amp-diagnostic] CHECKPOINT"
        )
    try:
        report = verify_checkpoint(
            arguments[0],
            allow_amp_diagnostic=allow_amp_diagnostic,
        )
    except CheckpointVerificationError as error:
        print(f"EXP07_STRICT_FP32_CHECKPOINT=FAILED error={error}")
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if allow_amp_diagnostic:
        print("CHECKPOINT_NUMERICS=PASSED mode=amp_diagnostic_only")
    print("EXP07_STRICT_FP32_CHECKPOINT=PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
