from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Iterable, Mapping

from morphoroute.core.config import ExperimentConfig
from morphoroute.data.materialize import MaterializedDataset
from morphoroute.evaluation.pipeline import write_prediction_bundle
from morphoroute.evaluation.pipeline import evaluate_bundle
from morphoroute.evaluation.reliability_export import (
    reliability_rows_from_bundle,
)
from morphoroute.models.variants import VARIANT_IDS
from morphoroute.runtime.mmdet_adapter import (
    MMDetRunResult,
    train_validate_predict_mmdet,
)


def _variant_config(
    config: ExperimentConfig,
    variant_id: str,
    seed: int,
) -> ExperimentConfig:
    parameters = dict(config.parameters)
    promoted_variants = {
        str(value)
        for value in parameters.get("release_budget_gate_variants", ())
    }
    parameters.update(
        {
            "run_name": f"{variant_id}/seed_{seed}",
            "variant_id": variant_id,
            "release_budget_gate": bool(
                parameters.get("release_budget_gate", False)
                or variant_id in promoted_variants
            ),
        }
    )
    return replace(config, seed=seed, parameters=parameters)


def resolve_variant_seeds(
    config: ExperimentConfig,
    variants: Iterable[str],
    *,
    seeds: Iterable[int] | None = None,
) -> dict[str, tuple[int, ...]]:
    selected_variants = tuple(str(variant) for variant in variants)
    if seeds is not None:
        shared = tuple(dict.fromkeys(int(seed) for seed in seeds))
        if not shared:
            raise ValueError("seeds must not be empty")
        return {variant: shared for variant in selected_variants}

    explicit = config.parameters.get("seeds_by_variant")
    if explicit is not None:
        if not isinstance(explicit, Mapping):
            raise ValueError("seeds_by_variant must be an object")
        result = {}
        for variant in selected_variants:
            values = explicit.get(variant, (config.seed,))
            selected = tuple(dict.fromkeys(int(seed) for seed in values))
            if not selected:
                raise ValueError(f"seeds_by_variant[{variant}] must not be empty")
            result[variant] = selected
        return result

    confirmation_variants = {
        str(value)
        for value in config.parameters.get("confirmation_variants", ())
    }
    if confirmation_variants:
        screening = tuple(
            dict.fromkeys(
                int(seed)
                for seed in config.parameters.get(
                    "screening_seeds", (config.seed,)
                )
            )
        )
        confirmation = tuple(
            dict.fromkeys(
                int(seed)
                for seed in config.parameters.get(
                    "confirmation_seeds", (42, 43, 44, 45, 46)
                )
            )
        )
        if not screening or not confirmation:
            raise ValueError("screening and confirmation seeds must not be empty")
        return {
            variant: (
                confirmation if variant in confirmation_variants else screening
            )
            for variant in selected_variants
        }

    shared = tuple(
        dict.fromkeys(
            int(seed)
            for seed in config.parameters.get("seeds", (42, 43, 44, 45, 46))
        )
    )
    if not shared:
        raise ValueError("seeds must not be empty")
    return {variant: shared for variant in selected_variants}


def run_variant_matrix(
    config: ExperimentConfig,
    materialized: MaterializedDataset,
    *,
    morphology_prior_path: str | Path,
    variants: Iterable[str] = VARIANT_IDS,
    seeds: Iterable[int] | None = None,
) -> tuple[list[dict[str, object]], dict[tuple[str, int], MMDetRunResult]]:
    selected_variants = tuple(variants)
    unknown = set(selected_variants).difference(VARIANT_IDS)
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")
    seeds_by_variant = resolve_variant_seeds(
        config,
        selected_variants,
        seeds=seeds,
    )
    rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    results: dict[tuple[str, int], MMDetRunResult] = {}
    for variant_id in selected_variants:
        for seed in seeds_by_variant[variant_id]:
            variant_config = _variant_config(config, variant_id, seed)
            if variant_id == "B0":
                result = train_validate_predict_mmdet(
                    variant_config,
                    materialized,
                    model_kind="mask2former",
                )
            else:
                result = train_validate_predict_mmdet(
                    variant_config,
                    materialized,
                    model_kind="morphoroute",
                    variant_id=variant_id,
                    morphology_prior_path=morphology_prior_path,
                    tau=float(config.parameters.get("routing_tau", 0.75)),
                )
            results[(variant_id, seed)] = result
            prediction_path = (
                config.output_root
                / "experiments"
                / config.experiment_id
                / config.profile
                / f"seed_{config.seed}"
                / "runs"
                / variant_id
                / f"seed_{seed}"
                / "predictions_D3.json"
            )
            write_prediction_bundle(result.predictions, prediction_path)
            validation_prediction_path = prediction_path.with_name(
                "predictions_D2.json"
            )
            write_prediction_bundle(
                result.validation_predictions,
                validation_prediction_path,
            )
            validation_evaluation = evaluate_bundle(
                result.validation_predictions
            )
            threshold_artifact = (
                result.resolved_config.work_dir
                / "routing_threshold_selection_D2.json"
            )
            threshold_payload = (
                json.loads(threshold_artifact.read_text(encoding="utf-8"))
                if threshold_artifact.is_file()
                else {}
            )
            if variant_id == "B9":
                reliability_rows.extend(
                    reliability_rows_from_bundle(
                        result.validation_predictions,
                        model=variant_id,
                        seed=seed,
                    )
                )
                reliability_rows.extend(
                    reliability_rows_from_bundle(
                        result.predictions,
                        model=variant_id,
                        seed=seed,
                    )
                )
            rows.append(
                {
                    "variant": variant_id,
                    "seed": seed,
                    "evidence_tier": (
                        "confirmation"
                        if len(seeds_by_variant[variant_id]) > 1
                        else "screening"
                    ),
                    "mask_ap": result.evaluation.mask_ap,
                    "mask_ap50": result.evaluation.mask_ap50,
                    "d2_mask_ap": validation_evaluation.mask_ap,
                    "d2_mask_ap50": validation_evaluation.mask_ap50,
                    "crack_cldice": result.evaluation.crack_cldice,
                    "crack_break_rate": (
                        result.evaluation.crack_break_rate
                    ),
                    "crack_component_error": (
                        result.evaluation.crack_component_error
                    ),
                    "area_boundary_f1": result.evaluation.area_boundary_f1,
                    "area_hd95": result.evaluation.area_hd95,
                    "area_relative_error": (
                        result.evaluation.area_relative_error
                    ),
                    "routing_tau": threshold_payload.get(
                        "selected_threshold"
                    ),
                    "routing_threshold_artifact": (
                        str(threshold_artifact)
                        if threshold_artifact.is_file()
                        else None
                    ),
                    "best_checkpoint": str(result.best_checkpoint),
                    "resolved_config": str(
                        result.resolved_config.config_path
                    ),
                    "predictions_D2": str(validation_prediction_path),
                    "predictions": str(prediction_path),
                }
            )
    if reliability_rows:
        from morphoroute.experiments.common import write_csv

        experiment_directory = (
            config.output_root
            / "experiments"
            / config.experiment_id
            / config.profile
            / f"seed_{config.seed}"
        )
        write_csv(
            experiment_directory / "reliability_logits_D2_D3.csv",
            reliability_rows,
        )
    return rows, results
