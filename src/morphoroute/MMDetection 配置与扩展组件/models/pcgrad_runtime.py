from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import torch

from morphoroute.models.gradient_probe import gradient_conflict_record
from morphoroute.models.morphology import FEATURE_NAMES, extract_morphology
from morphoroute.models.pcgrad import project_conflicting_pair


STRUCTURE_LOSS_TOKENS = ("route", "cldice", "boundary")


def split_pcgrad_losses(losses: Mapping[str, object]):
    primary = []
    structure = []
    for name, value in losses.items():
        if "loss" not in name:
            continue
        values = value if isinstance(value, (list, tuple)) else (value,)
        target = (
            structure
            if any(token in name.lower() for token in STRUCTURE_LOSS_TOKENS)
            else primary
        )
        target.extend(item.mean() for item in values if torch.is_tensor(item))
    if not primary or not structure:
        return None
    structure_loss = sum(structure)
    if float(structure_loss.detach().abs().item()) == 0.0:
        return None
    return sum(primary), structure_loss


try:
    from mmengine.optim import AmpOptimWrapper, OptimWrapper
    from mmengine.registry import OPTIM_WRAPPERS
    from mmdet.models.detectors import Mask2Former
    from mmdet.registry import MODELS
except ImportError:
    OptimWrapper = None
    AmpOptimWrapper = None
    OPTIM_WRAPPERS = None
    Mask2Former = None
    MODELS = None


if OPTIM_WRAPPERS is not None:

    @OPTIM_WRAPPERS.register_module(force=True)
    class PCGradAmpOptimWrapper(AmpOptimWrapper):
        """AMP MMEngine wrapper with effective-batch task-wise PCGrad."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._pcgrad_first = None
            self._pcgrad_second = None

        def update_params_from_loss_dict(
            self,
            losses,
            step_kwargs=None,
            zero_kwargs=None,
        ) -> None:
            pair = split_pcgrad_losses(losses)
            if pair is None:
                parsed = sum(
                    item.mean()
                    for name, value in losses.items()
                    if "loss" in name
                    for item in (
                        value if isinstance(value, (list, tuple)) else (value,)
                    )
                    if torch.is_tensor(item)
                )
                self.update_params(parsed, step_kwargs, zero_kwargs)
                return
            step_kwargs = step_kwargs or {}
            zero_kwargs = zero_kwargs or {}
            first_loss, second_loss = (
                self.loss_scaler.scale(self.scale_loss(loss))
                for loss in pair
            )
            parameters = [
                parameter
                for group in self.optimizer.param_groups
                for parameter in group["params"]
                if parameter.requires_grad
            ]
            first = torch.autograd.grad(
                first_loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            second = torch.autograd.grad(
                second_loss,
                parameters,
                allow_unused=True,
            )
            if self._pcgrad_first is None:
                self._pcgrad_first = [
                    None if gradient is None else gradient.detach().clone()
                    for gradient in first
                ]
                self._pcgrad_second = [
                    None if gradient is None else gradient.detach().clone()
                    for gradient in second
                ]
            else:
                for buffer, gradient in (
                    (self._pcgrad_first, first),
                    (self._pcgrad_second, second),
                ):
                    for index, value in enumerate(gradient):
                        if value is None:
                            continue
                        if buffer[index] is None:
                            buffer[index] = value.detach().clone()
                        else:
                            buffer[index].add_(value.detach())
            self._inner_count += 1
            if self.should_update():
                for parameter, left, right in zip(
                    parameters,
                    self._pcgrad_first,
                    self._pcgrad_second,
                ):
                    if left is None and right is None:
                        continue
                    if left is None:
                        gradient = right
                    elif right is None:
                        gradient = left
                    else:
                        projected_left, projected_right = (
                            project_conflicting_pair(left, right)
                        )
                        gradient = (projected_left + projected_right) / 2
                    parameter.grad = gradient
                self.step(**step_kwargs)
                self.zero_grad(**zero_kwargs)
                self._pcgrad_first = None
                self._pcgrad_second = None


if MODELS is not None:

    @MODELS.register_module(force=True)
    class MorphoRouteMask2Former(Mask2Former):
        """Mask2Former detector that exposes raw task losses to PCGrad."""

        def __init__(
            self,
            *args,
            diagnostic_path=None,
            diagnostic_updates=(),
            gradient_accumulation=1,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            if int(gradient_accumulation) <= 0:
                raise ValueError("gradient_accumulation must be positive")
            self.gradient_accumulation = int(gradient_accumulation)
            self.diagnostic_path = (
                Path(diagnostic_path) if diagnostic_path else None
            )
            self.diagnostic_updates = {
                int(update) for update in diagnostic_updates
            }
            self.register_buffer(
                "_training_update_tensor",
                torch.zeros((), dtype=torch.long),
                persistent=True,
            )
            self.register_buffer(
                "_training_micro_step_tensor",
                torch.zeros((), dtype=torch.long),
                persistent=True,
            )

        @staticmethod
        def _tensor_hash(tensors) -> str:
            digest = hashlib.sha256()
            for tensor in tensors:
                digest.update(
                    tensor.detach().float().cpu().contiguous().numpy().tobytes()
                )
            return digest.hexdigest()

        def _write_gradient_diagnostic(self, losses, data) -> None:
            current_update = int(self._training_update_tensor.item())
            current_micro_step = int(
                self._training_micro_step_tensor.item()
            )
            if (
                self.diagnostic_path is None
                or current_update not in self.diagnostic_updates
                or current_micro_step % self.gradient_accumulation != 0
            ):
                return
            line_terms = [
                value.mean()
                for name, raw in losses.items()
                if any(token in name.lower() for token in ("route", "cldice"))
                for value in (
                    raw if isinstance(raw, (list, tuple)) else (raw,)
                )
                if torch.is_tensor(value)
            ]
            area_terms = [
                value.mean()
                for name, raw in losses.items()
                if "boundary" in name.lower()
                for value in (
                    raw if isinstance(raw, (list, tuple)) else (raw,)
                )
                if torch.is_tensor(value)
            ]
            if not line_terms or not area_terms:
                return
            selected = [
                (name, parameter)
                for name, parameter in self.named_parameters()
                if parameter.requires_grad
                and (
                    "panoptic_head.transformer_decoder" in name
                    or "panoptic_head.pixel_decoder" in name
                )
            ]
            if not selected:
                return
            record = gradient_conflict_record(
                sum(line_terms),
                sum(area_terms),
                [parameter for _, parameter in selected],
            )
            samples = data.get("data_samples", []) if isinstance(data, dict) else []
            image_ids = [
                str(sample.metainfo.get("img_id", index))
                for index, sample in enumerate(samples)
            ]
            morphology = []
            class_labels = []
            areas = []
            for sample in samples:
                instances = sample.gt_instances
                masks = instances.masks
                labels = instances.labels.detach().cpu().tolist()
                for mask, label in zip(masks, labels):
                    array = mask.detach().cpu().numpy().astype(bool)
                    if not bool(array.any()):
                        continue
                    morphology.append(extract_morphology(array))
                    class_labels.append(int(label))
                    areas.append(float(array.sum()))
            if morphology:
                import numpy as np

                means = np.asarray(morphology, dtype=float).mean(axis=0)
                record.update(
                    {
                        name: float(value)
                        for name, value in zip(FEATURE_NAMES, means)
                    }
                )
                record["instance_area"] = float(
                    sum(areas) / max(len(areas), 1)
                )
                record["class_name"] = str(
                    max(set(class_labels), key=class_labels.count)
                )
            record.update(
                {
                    "group_id": "|".join(image_ids),
                    "seed": int(torch.initial_seed() % (2**31)),
                    "checkpoint_stage": str(current_update),
                    "parameter_subset": "shared_decoder_pixel_decoder",
                    "parameter_names": [name for name, _ in selected],
                    "input_size": (
                        int(data["inputs"].shape[-1])
                        if isinstance(data, dict) and "inputs" in data
                        else None
                    ),
                    "batch_hash": hashlib.sha256(
                        json.dumps(image_ids, sort_keys=True).encode("utf-8")
                    ).hexdigest(),
                    "checkpoint_hash": self._tensor_hash(
                        [parameter for _, parameter in selected]
                    ),
                }
            )
            self.diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
            with self.diagnostic_path.open(
                "a",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")

        def train_step(self, data, optim_wrapper):
            self.panoptic_head.training_update = int(
                self._training_update_tensor.item()
            )
            with optim_wrapper.optim_context(self):
                data = self.data_preprocessor(data, True)
                losses = self._run_forward(data, mode="loss")
            self._write_gradient_diagnostic(losses, data)
            _, log_vars = self.parse_losses(losses)
            update = getattr(
                optim_wrapper,
                "update_params_from_loss_dict",
                None,
            )
            if update is None:
                parsed_losses, _ = self.parse_losses(losses)
                optim_wrapper.update_params(parsed_losses)
            else:
                update(losses)
            self._training_micro_step_tensor.add_(1)
            if (
                int(self._training_micro_step_tensor.item())
                % self.gradient_accumulation
                == 0
            ):
                self._training_update_tensor.add_(1)
            return log_vars
