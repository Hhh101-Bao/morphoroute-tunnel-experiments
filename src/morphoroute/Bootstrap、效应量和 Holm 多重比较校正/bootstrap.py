from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def paired_group_bootstrap(
    records: pd.DataFrame,
    *,
    group_col: str,
    value_a: str,
    value_b: str,
    n_resamples: int = 10_000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    required = {group_col, value_a, value_b}
    if missing := required.difference(records.columns):
        raise ValueError(f"records lack columns: {sorted(missing)}")
    grouped = (
        records[[group_col, value_a, value_b]]
        .dropna()
        .groupby(group_col, sort=True)[[value_a, value_b]]
        .mean()
    )
    if grouped.empty:
        raise ValueError("no complete paired groups")
    groups = grouped.index.to_numpy()
    differences = (grouped[value_a] - grouped[value_b]).to_numpy(dtype=float)
    point_estimate = float(differences.mean())
    rng = np.random.default_rng(seed)
    distribution = np.empty(n_resamples, dtype=float)
    resample_groups: list[list[str]] = []
    for index in range(n_resamples):
        positions = rng.integers(0, len(groups), size=len(groups))
        distribution[index] = float(differences[positions].mean())
        resample_groups.append([str(group) for group in groups[positions]])
    alpha = 1.0 - confidence_level
    return {
        "point_estimate": point_estimate,
        "ci_low": float(np.quantile(distribution, alpha / 2)),
        "ci_high": float(np.quantile(distribution, 1 - alpha / 2)),
        "confidence_level": confidence_level,
        "n_groups": int(len(groups)),
        "n_resamples": n_resamples,
        "seed": seed,
        "distribution": distribution.tolist(),
        "resample_groups": resample_groups,
    }
