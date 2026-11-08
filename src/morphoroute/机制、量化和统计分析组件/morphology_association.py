from __future__ import annotations

import numpy as np
import pandas as pd

from morphoroute.models.morphology import FEATURE_NAMES


def fit_adjusted_association(
    table: pd.DataFrame,
    *,
    n_resamples: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    required = {
        "group_id",
        "gradient_cosine",
        "class_name",
        "instance_area",
        *FEATURE_NAMES,
    }
    if missing := required.difference(table.columns):
        raise ValueError(f"association table lacks: {sorted(missing)}")
    if (table["instance_area"] <= 0).any():
        raise ValueError("instance_area must be positive")
    class_dummies = pd.get_dummies(
        table["class_name"],
        prefix="class",
        drop_first=True,
    )
    predictors = pd.concat(
        [
            table[list(FEATURE_NAMES)].astype(float),
            np.log(table[["instance_area"]].astype(float)).rename(
                columns={"instance_area": "log_instance_area"}
            ),
            class_dummies.astype(float),
        ],
        axis=1,
    )
    design = np.column_stack(
        [np.ones(len(predictors)), predictors.to_numpy(float)]
    )
    response = table["gradient_cosine"].to_numpy(float)
    coefficients = np.linalg.lstsq(design, response, rcond=None)[0]
    groups = table["group_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    generator = np.random.default_rng(seed)
    distributions = []
    for _ in range(n_resamples):
        sampled_groups = generator.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )
        sampled_indices = np.concatenate(
            [np.flatnonzero(groups == group) for group in sampled_groups]
        )
        distributions.append(
            np.linalg.lstsq(
                design[sampled_indices],
                response[sampled_indices],
                rcond=None,
            )[0]
        )
    bootstrap = np.asarray(distributions)
    return pd.DataFrame(
        {
            "term": ["intercept", *predictors.columns],
            "coefficient": coefficients,
            "ci_low": np.quantile(bootstrap, 0.025, axis=0),
            "ci_high": np.quantile(bootstrap, 0.975, axis=0),
            "p_value": np.minimum(
                1.0,
                2
                * np.minimum(
                    np.mean(bootstrap <= 0, axis=0),
                    np.mean(bootstrap >= 0, axis=0),
                ),
            ),
            "model": (
                "OLS adjusted for class and log-area; "
                "group-cluster bootstrap confidence intervals"
            ),
            "bootstrap_groups": len(unique_groups),
            "bootstrap_resamples": n_resamples,
        }
    )
