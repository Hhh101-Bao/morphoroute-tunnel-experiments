#!/usr/bin/env python3
"""EXP-08A/08C supplement: fixed-checkpoint D1 loss-gradient probes.

This is a POST-HOC experiment, not a reconstruction of early training.
No Runner, optimizer, train_step, prediction shards, or D4 images are used.
The default cohort is 300 D1 images shared by B9 seeds 42, 43, 44.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REVISION = "fixed_checkpoint_D1_probe_v4"
FEATURES = ("log_aspect_ratio", "skeleton_length_over_area", "branch_density",
            "compactness", "area_fraction")
SEEDS = (42, 43, 44)
OBJECTIVES = {
    "gradient_cosine": "weighted loss_route + loss_cldice versus weighted loss_boundary",
    "primary_structure_cosine": "all primary segmentation losses versus all three auxiliary losses",
}
ACTIVE_OUTPUT = None


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def ensure_protocol(path, protocol):
    path = Path(path)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != protocol:
            raise ValueError("probe protocol changed; use a NEW output directory, do not mix records")
    else:
        atomic_json(path, protocol)


def select_cohort(rows, limit=300, sampling_seed=20261010):
    if limit < 1:
        raise ValueError("cohort limit must be positive")
    eligible, seen = [], set()
    for original in rows:
        if original.get("logical_split") not in {"D1", "train"}:
            continue
        row = dict(original)
        rid = str(row.get("record_id", "")).strip()
        if not rid or rid in seen:
            raise ValueError(f"empty or duplicate D1 record_id: {rid}")
        seen.add(rid)
        group = str(row.get("assignment_group_id") or row.get("group_key") or "").strip()
        if not group:
            raise ValueError(f"missing true manifest group for {rid}")
        row.update(record_id=rid, group_id=group)
        eligible.append(row)
    if not eligible:
        raise ValueError("no D1 manifest rows")
    def rank(value):
        return hashlib.sha256(f"{sampling_seed}:{value}".encode()).hexdigest()
    grouped = defaultdict(list)
    for row in eligible:
        grouped[row["group_id"]].append(row)
    for records in grouped.values():
        records.sort(key=lambda row: rank(row["record_id"]))
    groups = sorted(grouped, key=rank)
    selected, level = [], 0
    while len(selected) < min(limit, len(eligible)):
        for group in groups:
            if level < len(grouped[group]):
                selected.append(grouped[group][level])
                if len(selected) == min(limit, len(eligible)):
                    break
        level += 1
    return selected


def select_checkpoint(directory):
    candidates = sorted(Path(directory).glob("best_coco_segm_mAP_iter_*.pth"))
    candidates = [p for p in candidates if p.is_file() and p.stat().st_size > 0]
    if len(candidates) != 1:
        raise ValueError(f"missing or ambiguous D2-selected best checkpoint: {directory}")
    match = re.fullmatch(r"best_coco_segm_mAP_iter_(\d+)\.pth", candidates[0].name)
    return candidates[0], int(match.group(1))


def gradient_pair(left, right, shapes):
    if len(left) != len(right) or len(left) != len(shapes):
        raise ValueError("gradient parameter alignment mismatch")
    dot = left_norm2 = right_norm2 = 0.
    for a, b, shape in zip(left, right, shapes):
        a = np.zeros(shape) if a is None else np.asarray(a, dtype=np.float64)
        b = np.zeros(shape) if b is None else np.asarray(b, dtype=np.float64)
        if a.shape != tuple(shape) or b.shape != tuple(shape):
            raise ValueError("gradient shape mismatch")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("nonfinite gradient")
        dot += float(np.sum(a * b))
        left_norm2 += float(np.sum(a * a))
        right_norm2 += float(np.sum(b * b))
    a_norm, b_norm = math.sqrt(left_norm2), math.sqrt(right_norm2)
    if a_norm <= 1e-12 or b_norm <= 1e-12:
        return {"status": "zero_norm", "gradient_cosine": None, "norm_ratio": None,
                "line_norm": a_norm, "area_norm": b_norm, "negative_cosine": None}
    cosine = float(np.clip(dot / (a_norm * b_norm), -1., 1.))
    return {"status": "valid", "gradient_cosine": cosine, "norm_ratio": a_norm / b_norm,
            "line_norm": a_norm, "area_norm": b_norm, "negative_cosine": cosine < 0}


def load_records(path):
    path = Path(path)
    if not path.exists():
        return []
    rows, keys = [], set()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"partial record is damaged at line {number}; preserve it and inspect") from exc
            key = int(row["seed"]), str(row["record_id"])
            if key in keys:
                raise ValueError(f"duplicate gradient record: {key}")
            keys.add(key)
            rows.append(row)
    return rows


def validate_training_update(update, checkpoint_iteration):
    update = int(update)
    if update < 1 or update > int(checkpoint_iteration):
        raise ValueError("invalid or unloaded persisted training-update counter; inspect checkpoint")
    return update


def validate_observations(protocol, rows):
    cohort = {r["record_id"]: r for r in protocol["cohort"]}
    model_hashes = {}
    for row in rows:
        seed, rid = int(row["seed"]), str(row["record_id"])
        if seed not in SEEDS or rid not in cohort:
            raise ValueError("partial observation is outside the fixed cohort/seeds")
        expected = cohort[rid]
        if expected.get("preflight_status") == "excluded_background":
            raise ValueError("excluded background cannot be a gradient observation")
        checkpoint = protocol["checkpoints"][str(seed)]
        if row["group_id"] != expected["group_id"] or row["coco_image_id"] != expected["coco_image_id"]:
            raise ValueError("partial observation group/image identity changed")
        if row["checkpoint_hash"] != checkpoint["sha256"] or str(row["checkpoint_stage"]) != str(checkpoint["iteration"]):
            raise ValueError("partial observation checkpoint identity changed")
        digest = row.get("model_state_sha256", "")
        if row.get("model_state_verified_before_append") is not True or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("partial observation lacks verified immutable model state")
        if model_hashes.get(seed, digest) != digest:
            raise ValueError("partial observations have inconsistent model state")
        model_hashes[seed] = digest
        validate_training_update(row["training_update"], checkpoint["iteration"])
    return model_hashes


def summarize_groups(rows, expected_seeds=SEEDS):
    images = defaultdict(list)
    for row in rows:
        images[row["record_id"]].append(row)
    groups = defaultdict(list)
    for rid, members in images.items():
        if sorted(int(r["seed"]) for r in members) != sorted(expected_seeds):
            raise ValueError(f"unpaired seeds for image {rid}; refuse partial paired summary")
        if len({r["group_id"] for r in members}) != 1:
            raise ValueError(f"inconsistent group for {rid}")
        if not all(r["status"] == "valid" for r in members):
            continue  # Complete-case pairing; exclusions are separately reported.
        groups[members[0]["group_id"]].append({
            "cosine": float(np.mean([r["gradient_cosine"] for r in members])),
            "negative": float(np.mean([r["gradient_cosine"] < 0 for r in members])),
            "ratio": float(np.mean([r["norm_ratio"] for r in members])),
        })
    return [{"group_id": group, "image_count": len(values),
             "mean_cosine": float(np.mean([r["cosine"] for r in values])),
             "negative_observation_rate": float(np.mean([r["negative"] for r in values])),
             "mean_norm_ratio": float(np.mean([r["ratio"] for r in values]))}
            for group, values in sorted(groups.items())]


def cluster_ols(x, y, groups, bootstrap=1000, random_seed=20261010):
    x, y = np.asarray(x, float), np.asarray(y, float)
    groups = np.asarray(groups, str)
    if len(x) != len(y) or len(y) != len(groups) or bootstrap < 20:
        raise ValueError("invalid OLS inputs")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite regression inputs")
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("regression design rank deficient; cannot report significance")
    unique = np.unique(groups)
    if len(unique) < 20 or len(x) <= x.shape[1] * 5:
        raise ValueError("too few independent groups/observations for this exploratory model")
    coefficient = np.linalg.lstsq(x, y, rcond=None)[0]
    # Group cross-products make repeated cluster refits cheap and keep all seeds paired.
    xtx = np.asarray([x[groups == g].T @ x[groups == g] for g in unique])
    xty = np.asarray([x[groups == g].T @ y[groups == g] for g in unique])
    rng, estimates = np.random.default_rng(random_seed), []
    attempts = 0
    while len(estimates) < bootstrap and attempts < bootstrap * 3:
        attempts += 1
        counts = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        a = np.tensordot(counts, xtx, axes=(0, 0))
        b = np.tensordot(counts, xty, axes=(0, 0))
        if np.linalg.matrix_rank(a) != x.shape[1]:
            continue
        estimates.append(np.linalg.solve(a, b))
    if len(estimates) != bootstrap:
        raise ValueError("too many rank-deficient cluster bootstrap draws")
    draws = np.asarray(estimates)
    return {"coefficients": coefficient.tolist(),
            "ci_lower": np.percentile(draws, 2.5, axis=0).tolist(),
            "ci_upper": np.percentile(draws, 97.5, axis=0).tolist(),
            "group_count": len(unique), "observation_count": len(x),
            "bootstrap": bootstrap, "rank_rejected_draws": attempts - bootstrap,
            "inference": "exploratory group-cluster percentile CI; no causal claim or automatic M gate"}


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"refuse empty table: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})
    os.replace(temporary, path)


def dataset_definition(cfg):
    dataset = cfg.train_dataloader.dataset
    if dataset.get("type") != "CocoDataset":
        raise ValueError("unexpected dataset wrapper; inspect before collecting, no automatic fallback")
    return dataset


def validate_model_identity(configuration, seed, directory):
    """The project stores the variant in the head, not necessarily the detector."""
    model = configuration.get("model", {})
    head = model.get("panoptic_head", {})
    audit = {"model_type": model.get("type"), "top_variant": model.get("variant_id"),
             "head_type": head.get("type"), "head_variant": head.get("variant_id"),
             "seed": configuration.get("randomness", {}).get("seed"),
             "work_dir": configuration.get("work_dir")}
    errors = []
    if audit["model_type"] != "MorphoRouteMask2Former":
        errors.append("detector type is not MorphoRouteMask2Former")
    if audit["head_type"] != "MorphoRouteMask2FormerHead" or audit["head_variant"] != "B9":
        errors.append("head type/variant does not confirm B9")
    if audit["top_variant"] not in (None, "B9"):
        errors.append("optional detector variant contradicts B9")
    if audit["seed"] != seed:
        errors.append("randomness seed does not match checkpoint directory")
    if not audit["work_dir"] or Path(audit["work_dir"]).resolve() != Path(directory).resolve():
        errors.append("work_dir does not match checkpoint directory")
    if errors:
        raise ValueError("B9 model identity rejected: " + "; ".join(errors) +
                         "; actual=" + json.dumps(audit, ensure_ascii=False))
    return audit


def validate_image_fingerprint(row, actual_sha256, actual_size, annotation_sha256, coco_size, orientation):
    """Preserve source provenance; recognize only one byte-exact, audited EXIF copy."""
    expected = str(row.get("image_sha256", "")).strip().lower()
    expected_annotation = str(row.get("annotation_sha256", "")).strip().lower()
    actual_sha256 = str(actual_sha256).strip().lower()
    annotation_sha256 = str(annotation_sha256).strip().lower()
    if any(not re.fullmatch(r"[0-9a-f]{64}", h) for h in
           (expected, expected_annotation, actual_sha256, annotation_sha256)):
        raise ValueError(f"missing or malformed D1 fingerprint: {row['record_id']}")
    if row.get("logical_split") not in {"D1", "train"}:
        raise ValueError("image fingerprint is outside D1")
    if annotation_sha256 != expected_annotation:
        raise ValueError(f"D1 annotation hash mismatch: {row['record_id']}")
    if tuple(actual_size) != tuple(coco_size) or orientation not in (None, 1):
        raise ValueError(f"D1 image/COCO dimensions or orientation disagree: {row['record_id']}; "
                         f"image={actual_size} coco={coco_size} orientation={orientation}")
    audit = {"record_id": row["record_id"], "manifest_image_sha256": expected,
             "probe_image_sha256": actual_sha256, "probe_annotation_sha256": annotation_sha256,
             "image_size": list(actual_size), "coco_size": list(coco_size),
             "exif_orientation": orientation, "status": "manifest_match"}
    if actual_sha256 == expected:
        return audit
    known = (
        row["record_id"] == "DACL_train_dacl10k_v2_train_3520" and row.get("source") == "DACL"
        and row.get("destination_image_relative") == "D1_D3_internal/train/images/DACL__dacl10k_v2_train_3520.jpg"
        and (int(row.get("width", 0)), int(row.get("height", 0))) == (4032, 3024)
        and expected == "2a85c12efdce62f89afc93e93984e1cf558a2cb594985fa40d675973d3a31d8f"
        and actual_sha256 == "739ae79710e05687ea0e94e9930918102409e9a018626a42849327c0f7a6ad0f"
        and annotation_sha256 == "6f7c51692efae19289d7c837b1b3fde21c3d2bce168e98622ae6b1b6a080841f"
        and tuple(actual_size) == (3024, 4032)
    )
    if not known:
        raise ValueError(f"unrecognized D1 image hash mismatch: {row['record_id']}; "
                         f"manifest={expected} actual={actual_sha256} image_size={actual_size}")
    audit.update(status="audited_orientation_copy",
                 reason="Byte-exact local upright JPEG copy; raw source EXIF=6; manifest retains raw source hash. "
                        "Current copy matches annotation coordinate dimensions; JPEG pixels are not bit-identical.",
                 local_reference_checked="2026-10-10",
                 raw_source_size=[4032, 3024], raw_source_exif_orientation=6)
    return audit


def verify_probe_input(row, data):
    root = Path(data).resolve()
    for relative, field in (("destination_image_relative", "probe_image_sha256"),
                            ("destination_annotation_relative", "probe_annotation_sha256")):
        path = (root / row[relative]).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path) != row[field]:
            raise ValueError(f"frozen probe input changed: {row['record_id']} {relative}")


def audit_coco_cohort(cohort, coco, data, image_prefix=""):
    """Separate missing mappings from verified negatives without replacing the sample."""
    from PIL import Image
    root = Path(data).resolve()
    by_record, by_path, image_ids = {}, {}, set()
    for image in coco["images"]:
        image_id = int(image["id"])
        if image_id in image_ids:
            raise ValueError("duplicate COCO image id")
        image_ids.add(image_id)
        rid = str(image.get("record_id", ""))
        if rid:
            if rid in by_record:
                raise ValueError("duplicate COCO record_id")
            by_record[rid] = image
        filename = Path(str(image["file_name"]))
        if not filename.is_absolute():
            filename = Path(image_prefix) / filename
        key = str(filename.resolve())
        if key in by_path:
            raise ValueError("duplicate COCO image path")
        by_path[key] = image
    annotations = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations[int(annotation["image_id"])].append(annotation)
    audits = []
    for row in cohort:
        rid = row["record_id"]
        if row.get("logical_split") not in {"D1", "train"}:
            raise ValueError(f"cohort outside D1: {rid}")
        path = (root / row["destination_image_relative"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"D1 image missing or outside dataset root: {path}")
        image = by_record.get(rid) or by_path.get(str(path))
        if image is None:
            raise ValueError(f"selected D1 missing COCO image record: {rid}")
        if image.get("record_id") and str(image["record_id"]) != rid:
            raise ValueError(f"COCO record_id identity mismatch: manifest={rid} coco={image['record_id']}")
        filename = Path(str(image["file_name"]))
        if not filename.is_absolute():
            filename = Path(image_prefix) / filename
        if filename.resolve() != path or image.get("dataset_id", "D1") != "D1":
            raise ValueError(f"COCO/manifest path or split mismatch: {rid}")
        image_id = int(image["id"])
        all_annotations = annotations[image_id]
        foreground = [a for a in all_annotations if not a.get("iscrowd", 0)]
        try:
            instances = json.loads(row["class_instances_json"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"malformed manifest foreground declaration: {rid}") from exc
        declared_negative = (row.get("classes_present", "").strip() == "" and instances == {})
        audit = {"record_id": rid, "group_id": row["group_id"], "coco_image_id": image_id,
                 "coco_size": [int(image["width"]), int(image["height"])],
                 "foreground_annotation_count": len(foreground), "preflight_status": "eligible_foreground"}
        if foreground:
            if declared_negative:
                raise ValueError(f"COCO foreground contradicts negative manifest: {rid}")
        else:
            if all_annotations or not declared_negative:
                raise ValueError(f"empty/crowd COCO contradicts foreground declaration: {rid}")
            if row.get("source") != "TACK":
                raise ValueError(f"unverified empty annotation source; not a TACK negative: {rid}")
            annotation_path = (root / row["destination_annotation_relative"]).resolve()
            if not annotation_path.is_relative_to(root) or not annotation_path.is_file():
                raise ValueError(f"original negative annotation missing or outside root: {rid}")
            annotation_hash = sha256(annotation_path)
            if annotation_hash != row.get("annotation_sha256"):
                raise ValueError(f"original negative annotation hash mismatch: {rid}")
            with Image.open(annotation_path) as mask_image:
                mask = np.asarray(mask_image)
            if mask.ndim != 2 or tuple(mask.shape[::-1]) != tuple(audit["coco_size"]):
                raise ValueError(f"negative original mask dimensions/channels disagree: {rid}")
            values = np.unique(mask).tolist()
            if values != [0]:
                raise ValueError(f"empty COCO has nonzero original mask, not a verified negative: {rid}; values={values}")
            audit.update(preflight_status="excluded_background", raw_mask_values=values,
                         original_annotation_sha256=annotation_hash,
                         exclusion_reason="Verified original TACK all-zero mask; no foreground morphology or auxiliary gradient pair. "
                                          "Preserved in sampled cohort; no replacement image, no invented gradient.")
        audits.append(audit)
    return audits


def collection_cohort(protocol):
    rows = protocol["cohort"]
    if any(r.get("preflight_status") not in {"eligible_foreground", "excluded_background"} for r in rows):
        raise ValueError("cohort lacks explicit foreground eligibility audit")
    return [r for r in rows if r["preflight_status"] == "eligible_foreground"]


def pilot_cohort(protocol):
    target = collection_cohort(protocol)[:5]
    if len(target) != 5:
        raise ValueError("fewer than five positive cohort members; pilot cannot run")
    return target


def prepare(project, data, out, limit, bootstrap):
    from mmengine.config import Config
    from PIL import Image
    manifest = data / "metadata/split_manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    cohort = select_cohort(rows, limit)
    if len({r["group_id"] for r in cohort}) < 20:
        raise ValueError("fewer than 20 manifest groups; this association protocol needs redesign before GPU collection")
    checkpoints, annotation_hash, class_names = {}, None, None
    image_hash_cache, metadata_cache, input_audits, foreground_audits = {}, {}, {}, {}
    for seed in SEEDS:
        directory = project / f"outputs/experiments/EXP-09/local_6gb/seed_{seed}/runs/B9/seed_{seed}"
        checkpoint, iteration = select_checkpoint(directory)
        configuration = directory / "resolved_mmdet_config.py"
        cfg = Config.fromfile(str(configuration))
        identity = validate_model_identity(cfg.to_dict(), seed, directory)
        print("PROBE_MODEL_IDENTITY=PASSED " + json.dumps(identity, ensure_ascii=False), flush=True)
        definition = dataset_definition(cfg)
        names = list(definition.metainfo.classes)
        if len(names) != 5 or class_names not in (None, names):
            raise ValueError("class mapping differs from five-class protocol")
        class_names = names
        prior = cfg.model.panoptic_head.get("morphology_prior_path")
        if not prior or not Path(prior).is_file():
            raise ValueError(f"checkpoint morphology prior is missing: {prior}")
        annotation = Path(definition.ann_file)
        digest = sha256(annotation)
        if annotation_hash not in {None, digest}:
            raise ValueError("D1 annotation differs across checkpoints")
        annotation_hash = digest
        coco = json.loads(annotation.read_text(encoding="utf-8"))
        seed_audits = audit_coco_cohort(cohort, coco, data, definition.get("data_prefix", {}).get("img", ""))
        for row, foreground_audit in zip(cohort, seed_audits):
            path = (data / row["destination_image_relative"]).resolve()
            previous_audit = foreground_audits.get(row["record_id"])
            if previous_audit is not None and previous_audit != foreground_audit:
                raise ValueError("COCO foreground eligibility differs across seeds")
            foreground_audits[row["record_id"]] = foreground_audit
            if str(path) not in image_hash_cache:
                image_hash_cache[str(path)] = sha256(path)
                annotation_path = (data / row["destination_annotation_relative"]).resolve()
                if not annotation_path.is_relative_to(data.resolve()) or not annotation_path.is_file():
                    raise ValueError(f"D1 annotation missing or outside dataset root: {annotation_path}")
                with Image.open(path) as actual_image:
                    size, orientation = actual_image.size, actual_image.getexif().get(274)
                metadata_cache[str(path)] = (size, orientation, sha256(annotation_path))
            size, orientation, original_annotation_hash = metadata_cache[str(path)]
            audit = validate_image_fingerprint(row, image_hash_cache[str(path)], size,
                                              original_annotation_hash,
                                              foreground_audit["coco_size"], orientation)
            input_audits[row["record_id"]] = audit
            row["probe_image_sha256"] = audit["probe_image_sha256"]
            row["probe_annotation_sha256"] = audit["probe_annotation_sha256"]
            previous = row.get("coco_image_id")
            if previous is not None and previous != foreground_audit["coco_image_id"]:
                raise ValueError("COCO image mapping differs across seeds")
            row.update(coco_image_id=foreground_audit["coco_image_id"],
                       preflight_status=foreground_audit["preflight_status"])
        checkpoints[str(seed)] = {
            "checkpoint": str(checkpoint), "sha256": sha256(checkpoint),
            "iteration": iteration, "configuration": str(configuration),
            "configuration_sha256": sha256(configuration), "annotation": str(annotation),
            "prior": str(prior), "prior_sha256": sha256(prior),
            "model_identity": identity,
        }
    exclusions = [a for a in foreground_audits.values() if a["preflight_status"] == "excluded_background"]
    positive_count = len(cohort) - len(exclusions)
    if positive_count < 5 or len({r["group_id"] for r in cohort if r["preflight_status"] == "eligible_foreground"}) < 20:
        raise ValueError("too few positive images/groups after audited background exclusions")
    print(f"PROBE_D1_FOREGROUND_AUDIT=PASSED sampled={len(cohort)} positive={positive_count} "
          f"background_excluded={len(exclusions)} missing=0 max_gradient_records={positive_count*len(SEEDS)}", flush=True)
    audited_copies = [a for a in input_audits.values() if a["status"] == "audited_orientation_copy"]
    print(f"PROBE_D1_FINGERPRINTS=PASSED images={len(cohort)} "
          f"manifest_matches={len(cohort)-len(audited_copies)} audited_orientation_copies={len(audited_copies)}", flush=True)
    for audit in audited_copies:
        print("PROBE_D1_ORIENTATION_PROVENANCE=" + json.dumps(audit, ensure_ascii=False), flush=True)
    source_hash = hashlib.sha256()
    for path in sorted((project / "src/morphoroute").rglob("*.py")):
        source_hash.update(str(path.relative_to(project)).encode())
        source_hash.update(bytes.fromhex(sha256(path)))
    protocol = {
        "revision": REVISION, "script_sha256": sha256(__file__),
        "project_source_sha256": source_hash.hexdigest(),
        "posthoc": True, "early_training_reconstruction": False,
        "optimizer_updates": 0, "d4_images_accessed": False,
        "sample_split": "D1", "sampling_seed": 20261010,
        "dataset_root": str(data.resolve()),
        "input_fingerprint_audit": [input_audits[r["record_id"]] for r in cohort],
        "foreground_audit": [foreground_audits[r["record_id"]] for r in cohort],
        "preflight_exclusions": exclusions,
        "gradient_eligibility": "Only foreground members of the original fixed cohort; verified negatives logged separately, never replaced.",
        "sampling": "deterministic hash ordering; round-robin manifest groups; no outcome filtering",
        "cohort": cohort, "manifest_sha256": sha256(manifest),
        "annotation_sha256": annotation_hash, "checkpoints": checkpoints,
        "class_names": class_names,
        "input_size": 640, "amp": False, "tf32": False,
        "augmentation": "keep-ratio resize to 640 and fixed padding; no flip/random resize",
        "objectives": OBJECTIVES, "seeds": list(SEEDS), "bootstrap": bootstrap,
        "notes": ["Checkpoint stages vary by D2-selected best iteration; do not call this a stage-controlled study.",
                  "Morphology covariates are image-level instance means at the probe resolution, not original-image physical measurements.",
                  "Verified original background images have no defined instance morphology/auxiliary gradient pair; "
                  "excluded without replacement. Associations are conditional on foreground and valid gradients, not all D1 images.",
                  "This supplement does not change candidate selection, the sealed E gate, or any training result."],
    }
    ensure_protocol(out / "protocol.json", protocol)
    atomic_json(out / "input_fingerprint_audit.json", protocol["input_fingerprint_audit"])
    atomic_json(out / "foreground_eligibility_audit.json", protocol["foreground_audit"])
    if exclusions:
        write_csv(out / "sample_exclusions_D1.csv", exclusions)
    write_csv(out / "sample_manifest_D1.csv", cohort)
    return protocol


def disable_checkpointing(value):
    if isinstance(value, dict):
        for key in list(value):
            if key in {"with_cp", "use_checkpoint", "gradient_checkpointing"}:
                value[key] = False
            else:
                disable_checkpointing(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            disable_checkpointing(item)


def model_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        array = tensor.detach().cpu().contiguous().numpy()
        if array.dtype.kind in "fc" and not np.isfinite(array).all():
            raise ValueError(f"nonfinite model tensor: {name}")
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def loss_sum(losses, tokens=None, primary=False):
    import torch
    terms = []
    for name, raw in losses.items():
        if "loss" not in name:
            continue
        auxiliary = any(t in name.lower() for t in ("route", "cldice", "boundary"))
        include = (not auxiliary) if primary else any(t in name.lower() for t in tokens)
        if include:
            for value in raw if isinstance(raw, (list, tuple)) else (raw,):
                if torch.is_tensor(value):
                    terms.append(value.mean())
    if not terms:
        raise ValueError(f"missing loss terms: {tokens}, primary={primary}")
    result = sum(terms)
    if not result.requires_grad or not torch.isfinite(result):
        raise ValueError("objective missing gradient graph or has nonfinite loss")
    return result


def runtime_setup(project):
    sys.path.insert(0, str(project / "src"))
    import torch
    from mmdet.utils import register_all_modules
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no CPU/AMP fallback for this protocol")
    register_all_modules(init_default_scope=True)
    print(f"PROBE_RUNTIME torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}", flush=True)


def acquire_gpu_guard():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True)
    pids = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
    if any(pid != os.getpid() for pid in pids):
        raise RuntimeError(f"another GPU job exists; no process was stopped: {pids}")


def build_probe_model(protocol, seed):
    import torch
    from mmengine.config import Config
    from mmdet.apis import init_detector
    from mmdet.registry import DATASETS
    item = protocol["checkpoints"][str(seed)]
    cfg = Config.fromfile(item["configuration"])
    disable_checkpointing(cfg.model)
    if "backbone" in cfg.model:
        cfg.model.backbone.init_cfg = None
    model = init_detector(cfg, item["checkpoint"], device="cuda:0")
    if not hasattr(model, "_training_update_tensor"):
        raise ValueError("checkpoint model lacks persisted training-update counter; inspect before probing")
    update = validate_training_update(model._training_update_tensor.item(), item["iteration"])
    model.train()
    for module in model.modules():
        if isinstance(module, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.modules.dropout._DropoutNd)):
            module.eval()  # BN buffers remain unchanged; dropout not an extra source of noise.
        for flag in ("with_cp", "use_checkpoint"):
            if hasattr(module, flag):
                setattr(module, flag, False)
    model.panoptic_head.training_update = update
    model.diagnostic_path = None
    definition = dict(dataset_definition(cfg))
    definition.update(test_mode=False, filter_cfg={"filter_empty_gt": False, "min_size": 0},
                      pipeline=[
                          {"type": "LoadImageFromFile", "backend_args": None},
                          {"type": "LoadAnnotations", "with_bbox": True, "with_mask": True},
                          {"type": "Resize", "scale": (640, 640), "keep_ratio": True},
                          {"type": "PackDetInputs", "meta_keys": ("img_id", "img_path", "ori_shape", "img_shape", "scale_factor")},
                      ])
    dataset = DATASETS.build(definition)
    index = {int(dataset.get_data_info(i)["img_id"]): i for i in range(len(dataset))}
    selected = [(name, p) for name, p in model.named_parameters()
                if p.requires_grad and ("panoptic_head.transformer_decoder" in name or
                                        "panoptic_head.pixel_decoder" in name)]
    if not selected:
        raise ValueError("shared decoder parameter subset is empty")
    return model, dataset, index, selected, update


def collect_one(model, dataset, index, selected, row, seed, checkpoint_info, update):
    if row.get("preflight_status") != "eligible_foreground":
        raise ValueError("excluded background is not a gradient target")
    import torch
    from morphoroute.models.morphology import extract_morphology
    # Same point-sampling RNG for the same image in every seed model.
    probe_seed = int(hashlib.sha256(row["record_id"].encode()).hexdigest()[:8], 16)
    torch.manual_seed(probe_seed)
    torch.cuda.manual_seed_all(probe_seed)
    data = dataset[index[row["coco_image_id"]]]
    if data is None:
        raise ValueError("deterministic pipeline unexpectedly rejected a selected image")
    data = model.data_preprocessor({"inputs": [data["inputs"]], "data_samples": [data["data_samples"]]}, training=True)
    if tuple(data["inputs"].shape[-2:]) != (640, 640):
        raise ValueError(f"probe input size is not 640 square: {tuple(data['inputs'].shape)}")
    sample = data["data_samples"][0]
    if int(sample.metainfo["img_id"]) != row["coco_image_id"]:
        raise ValueError("selected image identity changed in pipeline")
    base = {"seed": seed, "record_id": row["record_id"], "group_id": row["group_id"],
            "coco_image_id": row["coco_image_id"], "input_size": 640,
            "checkpoint_stage": str(checkpoint_info["iteration"]), "training_update": update,
            "checkpoint_hash": checkpoint_info["sha256"],
            "parameter_subset": "shared_decoder_pixel_decoder", "sampling_rng": probe_seed,
            "objective_scope": "auxiliary route+cldice versus boundary; not class-specific primary losses"}
    masks = sample.gt_instances.masks
    arrays = masks.detach().cpu().numpy().astype(bool) if torch.is_tensor(masks) else masks.to_ndarray().astype(bool)
    labels = sample.gt_instances.labels.detach().cpu().numpy()
    nonempty = [i for i, mask in enumerate(arrays) if mask.any()]
    if not nonempty:
        return {**base, "status": "empty_after_resize", "gradient_cosine": None, "norm_ratio": None}
    features = np.asarray([extract_morphology(arrays[i]) for i in nonempty]).mean(axis=0)
    counts = np.bincount(labels[nonempty], minlength=5)
    base.update(dict(zip(FEATURES, features.tolist())))
    base.update(instance_area=float(np.mean([arrays[i].sum() for i in nonempty])),
                instance_count=len(nonempty), empty_instance_count=len(arrays) - len(nonempty),
                class_name=str(int(np.argmax(counts))), class_fractions=(counts / counts.sum()).tolist())
    losses = model(**data, mode="loss")
    objectives = [loss_sum(losses, ("route", "cldice")), loss_sum(losses, ("boundary",)),
                  loss_sum(losses, primary=True)]
    parameters = [p for _, p in selected]
    gradients = []
    for i, objective in enumerate(objectives):
        raw = torch.autograd.grad(objective, parameters, retain_graph=i < 2, allow_unused=True)
        gradients.append([None if g is None else g.detach().float().cpu().numpy().copy() for g in raw])
        del raw
    shapes = [tuple(p.shape) for p in parameters]
    record = gradient_pair(gradients[0], gradients[1], shapes)
    structure = [None if a is None and b is None else
                 (np.zeros(shape, np.float32) if a is None else a) +
                 (np.zeros(shape, np.float32) if b is None else b)
                 for a, b, shape in zip(gradients[0], gradients[1], shapes)]
    second = gradient_pair(gradients[2], structure, shapes)
    base.update(record)
    base.update(primary_structure_cosine=second["gradient_cosine"],
                primary_structure_status=second["status"],
                primary_structure_norm_ratio=second["norm_ratio"],
                objective_values=[float(v.detach().cpu()) for v in objectives])
    del losses, objectives, gradients, structure, data
    return base


def collect(protocol, out, pilot_only=False):
    import torch
    rows_path = out / "gradient_records.jsonl"
    rows = load_records(rows_path)
    verified_hashes = validate_observations(protocol, rows)
    existing = {(int(r["seed"]), r["record_id"]) for r in rows}
    positive_cohort = collection_cohort(protocol)
    expected = {(seed, row["record_id"]) for seed in SEEDS for row in positive_cohort}
    if not existing <= expected:
        raise ValueError("partial records do not belong to the frozen cohort")
    timing, audit_path = [], out / "weight_integrity.json"
    integrity = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    seeds = SEEDS[:1] if pilot_only else SEEDS
    for seed in seeds:
        target = pilot_cohort(protocol) if pilot_only else positive_cohort
        pending = [row for row in target if (seed, row["record_id"]) not in existing]
        if not pending:
            if str(seed) not in integrity:
                # A crash may occur after the last fsynced, individually verified row,
                # but before the summary audit write. Never label unverified rows passed.
                digest = verified_hashes[seed]
                integrity[str(seed)] = {"before": digest, "after": digest, "unchanged": True,
                                        "checked_at": now(), "evidence": "per-record state hash verified before durable append"}
                atomic_json(audit_path, integrity)
            continue
        print(f"PROBE_SEED_START seed={seed} pending={len(pending)} checkpoint={protocol['checkpoints'][str(seed)]['checkpoint']}", flush=True)
        model, dataset, index, selected, update = build_probe_model(protocol, seed)
        before = model_hash(model)
        if verified_hashes.get(seed, before) != before:
            raise RuntimeError("resumed model state differs from verified earlier observations")
        start = time.monotonic()
        for position, row in enumerate(pending, 1):
            tick = time.monotonic()
            verify_probe_input(row, protocol["dataset_root"])
            record = collect_one(model, dataset, index, selected, row, seed,
                                 protocol["checkpoints"][str(seed)], update)
            verify_probe_input(row, protocol["dataset_root"])
            if model_hash(model) != before:
                raise RuntimeError("model state changed during probe; observation not appended")
            record.update(model_state_sha256=before, model_state_verified_before_append=True)
            elapsed = time.monotonic() - tick
            with rows_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            rows.append(record)
            existing.add((seed, row["record_id"]))
            timing.append(elapsed)
            if position <= 5 or position % 10 == 0:
                print(f"PROBE_PROGRESS seed={seed} {position}/{len(pending)} status={record['status']} "
                      f"seconds_per_image={(time.monotonic()-start)/position:.2f} "
                      f"remaining_min={(len(pending)-position)*(time.monotonic()-start)/position/60:.1f}", flush=True)
        after = model_hash(model)
        if before != after or sha256(protocol["checkpoints"][str(seed)]["checkpoint"]) != protocol["checkpoints"][str(seed)]["sha256"]:
            raise RuntimeError("model/checkpoint integrity changed; do not use supplement observations")
        integrity[str(seed)] = {"before": before, "after": after, "unchanged": True,
                                "checked_at": now(), "training_update": update,
                                "parameter_names": [n for n, _ in selected]}
        atomic_json(audit_path, integrity)
        del model, dataset, index, selected
        gc.collect()
        torch.cuda.empty_cache()
    if pilot_only:
        pilot = [r for r in rows if r["seed"] == SEEDS[0] and r["record_id"] in {x["record_id"] for x in pilot_cohort(protocol)}]
        if len(pilot) != 5 or not all(r["status"] == "valid" for r in pilot):
            raise RuntimeError("pilot has empty/zero-gradient observations; inspect exclusions before long collection")
        report = {"status": "pilot_passed", "records": len(pilot), "weights_unchanged": True,
                  "median_seconds_per_image": float(np.median(timing)) if timing else None,
                  "sampled_images": len(protocol["cohort"]), "positive_images": len(positive_cohort),
                  "preflight_background_exclusions": len(protocol["cohort"]) - len(positive_cohort),
                  "collection_hours_at_pilot_speed": float(np.median(timing))*len(positive_cohort)*len(SEEDS)/3600 if timing else None,
                  "caveat": "Pilot estimate covers collection only, not regression/resource/final-statistics stages."}
        atomic_json(out / "pilot_report.json", report)
        print("PROBE_PILOT=PASSED " + json.dumps(report, ensure_ascii=False), flush=True)
    else:
        if existing != expected or not all(str(s) in integrity for s in SEEDS):
            raise RuntimeError("collection or immutable-weight audit is incomplete")
    for seed in seeds:
        checkpoint = protocol["checkpoints"][str(seed)]
        if sha256(checkpoint["checkpoint"]) != checkpoint["sha256"]:
            raise RuntimeError("checkpoint file changed during collection")
    validate_observations(protocol, rows)
    return rows


def analysis(protocol, rows, out):
    groups = summarize_groups(rows)
    write_csv(out / "M1_group_summary.csv", groups)
    write_csv(out / "gradient_conflict_records.csv", rows)
    complete_ids = {rid for rid in {r["record_id"] for r in rows}
                    if all(r["status"] == "valid" for r in rows if r["record_id"] == rid)}
    eligible = [r for r in rows if r["record_id"] in complete_ids]
    counts = {status: sum(r["status"] == status for r in rows) for status in sorted({r["status"] for r in rows})}
    if not eligible:
        raise ValueError("no complete paired images for association analysis")
    raw = np.asarray([[r[k] for k in FEATURES] + [math.log(max(r["instance_area"], 1.))] for r in eligible])
    scales = raw.std(axis=0)
    if np.any(scales < 1e-12):
        raise ValueError("morphology covariate has zero variance; inspect instead of fabricating M3")
    standard = (raw - raw.mean(axis=0)) / scales
    fractions = np.asarray([r["class_fractions"] for r in eligible])
    # Omit a PRESENT reference class, even when class 4 is absent in the cohort.
    present_classes = [i for i in range(5) if fractions[:, i].mean() > 1e-12]
    class_columns = [i for i in present_classes[:-1] if fractions[:, i].std() > 1e-12]
    x = np.column_stack([np.ones(len(eligible)), standard, fractions[:, class_columns],
                         [r["seed"] == 43 for r in eligible], [r["seed"] == 44 for r in eligible]])
    names = ["intercept", *FEATURES, "log_instance_area", *[f"class_fraction_{i}" for i in class_columns],
             "seed43_effect", "seed44_effect"]
    result = cluster_ols(x, [r["gradient_cosine"] for r in eligible],
                         [r["group_id"] for r in eligible], bootstrap=protocol["bootstrap"])
    coefficient_rows = [{"term": name, "coefficient": value, "ci_lower": lower, "ci_upper": upper,
                         "outcome": OBJECTIVES["gradient_cosine"], "posthoc": True}
                        for name, value, lower, upper in zip(names, result["coefficients"], result["ci_lower"], result["ci_upper"])]
    write_csv(out / "M3_association_coefficients.csv", coefficient_rows)
    write_csv(out / "M3_analysis_table.csv", eligible)
    result.update(term_names=names, covariate_mean=raw.mean(axis=0).tolist(), covariate_scale=scales.tolist(),
                  complete_image_count=len(complete_ids), excluded_image_count=len(protocol["cohort"]) - len(complete_ids),
                  observation_status_counts=counts, caveats=protocol["notes"],
                  sampled_image_count=len(protocol["cohort"]),
                  preflight_background_excluded_image_count=len(protocol.get("preflight_exclusions", [])),
                  preflight_excluded_record_ids=[r["record_id"] for r in protocol.get("preflight_exclusions", [])],
                  actual_gradient_record_count=len(rows),
                  estimand="Foreground members of the fixed D1 cohort with valid paired auxiliary gradients; not whole-D1 prevalence.",
                  class_reference=present_classes[-1] if present_classes else None,
                  multiplicity="Exploratory CIs, no multiplicity-adjusted significance claims or mechanism gate.")
    atomic_json(out / "M3_analysis_audit.json", result)
    atomic_json(out / "run_manifest.json", {
        "revision": REVISION, "status": "completed_posthoc_probe", "finished_at_utc": now(),
        "experiment_scope": "EXP-08A/08C fixed-checkpoint supplement only", "sample_split": "D1",
        "seeds": list(SEEDS), "checkpoint_stages_reconstructed": False,
        "sampled_images": len(protocol["cohort"]), "complete_paired_images": len(complete_ids),
        "preflight_background_excluded_images": len(protocol.get("preflight_exclusions", [])),
        "weights_unchanged": True, "optimizer_updates": 0, "D4_access": False,
        "EXP08B_resources_completed_by_this_script": False, "EXP15_completed_by_this_script": False,
        "produced_outputs": ["gradient_conflict_records.csv", "M1_group_summary.csv",
                             "M3_association_coefficients.csv", "M3_analysis_table.csv",
                             "M3_analysis_audit.json", "weight_integrity.json", "protocol.json",
                             "input_fingerprint_audit.json", "foreground_eligibility_audit.json",
                             *(["sample_exclusions_D1.csv"] if protocol.get("preflight_exclusions") else [])],
    })
    print("EXP08A_EXP08C_POSTHOC_SUPPLEMENT=COMPLETED", flush=True)


def main():
    global ACTIVE_OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--images", type=int, default=300)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--stage", choices=("preflight", "pilot", "run"), default="pilot")
    args = parser.parse_args()
    project, data = args.project_root.resolve(), args.dataset_root.resolve()
    out = (args.output or project / "outputs/supplements/EXP08AC_fixed_checkpoint_D1_v4").resolve()
    if not out.is_relative_to((project / "outputs/supplements").resolve()):
        raise ValueError("output must be inside project outputs/supplements; canonical experiments stay untouched")
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / "process.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another supplement process holds this output lock") from exc
        ACTIVE_OUTPUT = out
        atomic_json(out / "state.json", {"status": "preflight", "pid": os.getpid(), "time": now()})
        protocol = prepare(project, data, out, args.images, args.bootstrap)
        print(f"PROBE_PREFLIGHT=PASSED images={len(protocol['cohort'])} "
              f"groups={len({r['group_id'] for r in protocol['cohort']})} seeds={list(SEEDS)} output={out}", flush=True)
        if args.stage == "preflight":
            atomic_json(out / "state.json", {"status": "preflight_passed", "time": now()})
            return
        acquire_gpu_guard()
        runtime_setup(project)
        atomic_json(out / "state.json", {"status": "pilot", "pid": os.getpid(), "time": now()})
        collect(protocol, out, pilot_only=True)
        if args.stage == "pilot":
            atomic_json(out / "state.json", {"status": "pilot_passed", "time": now()})
            return
        atomic_json(out / "state.json", {"status": "collecting", "pid": os.getpid(), "time": now()})
        rows = collect(protocol, out)
        atomic_json(out / "state.json", {"status": "association_analysis", "pid": os.getpid(), "time": now()})
        analysis(protocol, rows, out)
        atomic_json(out / "state.json", {"status": "completed_posthoc_probe", "time": now()})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        import traceback
        traceback.print_exc()
        print(f"SUPPLEMENT_FAILED: {error}", flush=True)
        if ACTIVE_OUTPUT is not None:
            try:
                atomic_json(ACTIVE_OUTPUT / "state.json", {
                    "status": "failed", "reason": str(error), "time": now(), "pid": os.getpid(),
                    "completed_training_untouched": True,
                })
            except OSError:
                pass
        # Do not mask the failure or change existing experiment manifests.
        raise SystemExit(2)
