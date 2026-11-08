from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Figure generation requires matplotlib; install ENV-STATS dependencies"
        ) from error
    return plt


def _save(fig, output_stem: str | Path) -> dict[str, Path]:
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    _pyplot().close(fig)
    return {"png": png, "pdf": pdf}


def plot_class_distribution(
    rows: Sequence[Mapping[str, object]],
    output_stem: str | Path,
) -> dict[str, Path]:
    plt = _pyplot()
    labels = [f"{row['split']} · {row['class_name']}" for row in rows]
    values = [float(row["positive_images"]) for row in rows]
    fig, axis = plt.subplots(figsize=(10, max(4, len(rows) * 0.24)))
    positions = np.arange(len(rows))
    axis.barh(positions, values, color="#2F6B8A")
    axis.set_yticks(positions, labels)
    axis.set_xlabel("Positive images")
    axis.set_title("Class distribution by frozen split")
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=0.25)
    return _save(fig, output_stem)


def plot_training_curves(
    steps: Sequence[int],
    series: Mapping[str, Sequence[float]],
    output_stem: str | Path,
) -> dict[str, Path]:
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for name, values in series.items():
        axis.plot(steps, values, label=name, linewidth=1.8)
    axis.set_xlabel("Update")
    axis.set_ylabel("Value")
    axis.set_title("Training and validation curves")
    axis.grid(alpha=0.25)
    axis.legend()
    return _save(fig, output_stem)


def plot_prediction_overlay(
    image: np.ndarray,
    masks: Sequence[np.ndarray],
    output_stem: str | Path,
) -> dict[str, Path]:
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(8, 6))
    axis.imshow(image)
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(masks))))
    for mask, color in zip(masks, colors):
        overlay = np.zeros((*mask.shape, 4), dtype=float)
        overlay[..., :3] = color[:3]
        overlay[..., 3] = np.asarray(mask, dtype=float) * 0.38
        axis.imshow(overlay)
    axis.axis("off")
    axis.set_title("Frozen prediction overlay")
    return _save(fig, output_stem)


def _bar_plot(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    ylabel: str,
    output_stem: str | Path,
) -> dict[str, Path]:
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(labels, values, color="#3A7D6C")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    return _save(fig, output_stem)


def plot_routing_loads(labels, values, output_stem):
    return _bar_plot(labels, values, "Expert load", "Queries", output_stem)


def plot_gradient_conflict(labels, values, output_stem):
    return _bar_plot(labels, values, "Gradient conflict", "Cosine", output_stem)


def plot_intervention_effects(labels, values, output_stem):
    return _bar_plot(labels, values, "Expert intervention effects", "Δ metric", output_stem)


def plot_ablation_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    output_stem: str | Path,
) -> dict[str, Path]:
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(9, 6))
    image = axis.imshow(matrix, cmap="RdYlBu", aspect="auto")
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    axis.set_xticks(np.arange(len(column_labels)), column_labels, rotation=35)
    axis.set_title("B0–B11 ablation")
    fig.colorbar(image, ax=axis)
    return _save(fig, output_stem)


def plot_morphology_association(x, y, output_stem):
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(x, y, alpha=0.45, color="#865A9C")
    axis.set_xlabel("Continuous morphology")
    axis.set_ylabel("Gradient conflict")
    axis.set_title("Morphology–conflict association")
    axis.grid(alpha=0.2)
    return _save(fig, output_stem)


def plot_risk_coverage(coverage, risk, output_stem):
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(coverage, risk, color="#B04A4A", linewidth=2)
    axis.set_xlabel("Coverage")
    axis.set_ylabel("Risk")
    axis.set_title("Risk–coverage")
    axis.grid(alpha=0.25)
    return _save(fig, output_stem)


def plot_bland_altman(reference, prediction, output_stem):
    plt = _pyplot()
    reference_array = np.asarray(reference, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    means = (reference_array + prediction_array) / 2
    differences = prediction_array - reference_array
    mean_difference = differences.mean()
    limit = 1.96 * differences.std(ddof=1) if len(differences) > 1 else 0.0
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(means, differences, alpha=0.55)
    axis.axhline(mean_difference, color="black")
    axis.axhline(mean_difference + limit, color="red", linestyle="--")
    axis.axhline(mean_difference - limit, color="red", linestyle="--")
    axis.set_xlabel("Mean measurement")
    axis.set_ylabel("Prediction − reference")
    axis.set_title("Bland–Altman agreement")
    return _save(fig, output_stem)


def plot_latency_stack(labels, stage_values, output_stem):
    plt = _pyplot()
    fig, axis = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(labels), dtype=float)
    for stage_name, values in stage_values.items():
        values_array = np.asarray(values, dtype=float)
        axis.bar(labels, values_array, bottom=bottom, label=stage_name)
        bottom += values_array
    axis.set_ylabel("Latency (ms)")
    axis.set_title("End-to-end stage latency")
    axis.legend()
    return _save(fig, output_stem)


def plot_effect_forest(labels, effects, lows, highs, output_stem):
    plt = _pyplot()
    positions = np.arange(len(labels))
    effects_array = np.asarray(effects, dtype=float)
    fig, axis = plt.subplots(figsize=(8, max(4, len(labels) * 0.45)))
    axis.errorbar(
        effects_array,
        positions,
        xerr=[effects_array - np.asarray(lows), np.asarray(highs) - effects_array],
        fmt="o",
        color="#2F6B8A",
        capsize=3,
    )
    axis.axvline(0, color="black", linewidth=1)
    axis.set_yticks(positions, labels)
    axis.set_xlabel("Effect with 95% CI")
    axis.set_title("Paired group effects")
    axis.grid(axis="x", alpha=0.25)
    return _save(fig, output_stem)
