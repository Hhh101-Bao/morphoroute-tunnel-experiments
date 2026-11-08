from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from morphoroute.models.morphology import (
    MorphologyPrior,
    extract_observable_morphology,
)


class MorphologyTargetEncoder:
    def __init__(self, prior: MorphologyPrior) -> None:
        self.prior = prior

    def __call__(
        self,
        matched_masks: Sequence[np.ndarray | torch.Tensor],
        *,
        matched_query_indices: Sequence[int],
        query_count: int,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(matched_masks) != len(matched_query_indices):
            raise ValueError("matched masks and query indices must have equal length")
        if len(set(int(index) for index in matched_query_indices)) != len(
            matched_query_indices
        ):
            raise ValueError("a query may receive at most one morphology target")
        targets = torch.zeros(query_count, 2, dtype=torch.float32, device=device)
        weights = torch.zeros(query_count, dtype=torch.float32, device=device)
        if not matched_masks:
            return targets, weights
        query_indices = [int(index) for index in matched_query_indices]
        for index in query_indices:
            if not 0 <= index < query_count:
                raise IndexError(f"matched query index out of range: {index}")
        arrays = []
        for mask in matched_masks:
            if isinstance(mask, torch.Tensor):
                array = mask.detach().cpu().numpy()
            else:
                array = np.asarray(mask)
            arrays.append(array)
        features, observable_positions = extract_observable_morphology(arrays)
        if not len(observable_positions):
            return targets, weights
        probabilities = self.prior.soft_targets(features)
        for position, probability in zip(
            observable_positions,
            probabilities,
        ):
            index = query_indices[int(position)]
            targets[index] = torch.as_tensor(
                probability,
                dtype=torch.float32,
                device=device,
            )
            weights[index] = 1.0
        return targets, weights
