from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


def _contains_key(value, expected: str) -> bool:
    if isinstance(value, dict):
        return any(
            str(key) == expected or _contains_key(item, expected)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_key(item, expected) for item in value)
    return False


def _scan(value, prefix: str):
    tensor_count = 0
    failures = []
    if torch.is_tensor(value):
        tensor_count = 1
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all().item()
        ):
            failures.append(prefix)
        return tensor_count, failures
    if isinstance(value, dict):
        for key, item in value.items():
            count, child = _scan(
                item,
                f"{prefix}.{key}" if prefix else str(key),
            )
            tensor_count += count
            failures.extend(child)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            count, child = _scan(item, f"{prefix}[{index}]")
            tensor_count += count
            failures.extend(child)
    return tensor_count, failures


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        raise SystemExit("usage: verify_exp09_b11_checkpoint.py CHECKPOINT")
    path = Path(arguments[0]).resolve()
    if not path.is_file():
        print(f"EXP09_B11_CHECKPOINT=FAILED missing={path}")
        return 2

    checkpoint = torch.load(path, map_location="cpu")
    model_count, model_bad = _scan(
        checkpoint.get("state_dict", {}), "state_dict"
    )
    optimizer = checkpoint.get("optimizer", {})
    optimizer_count, optimizer_bad = _scan(optimizer, "optimizer")
    failures = model_bad + optimizer_bad
    report = {
        "checkpoint": str(path),
        "model_tensor_count": model_count,
        "optimizer_tensor_count": optimizer_count,
        "nonfinite_tensor_count": len(failures),
        "nonfinite_tensors": failures[:50],
        "contains_amp_loss_scaler": _contains_key(
            optimizer, "loss_scaler"
        ),
        "amp_loss_scaler_action": "discard_on_true_fp32_resume",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        print("EXP09_B11_CHECKPOINT=FAILED")
        return 3
    print("EXP09_B11_CHECKPOINT=PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
