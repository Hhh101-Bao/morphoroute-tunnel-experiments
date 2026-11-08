"""Reuse the installed v4 fusion head for EXP-08D, then run EXP-12 external.

Run on AutoDL with Python 3.10. No packages are installed and no weights are
updated. The existing EXP-08D D1 products and old prediction shards are kept.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime


FUSION = "MorphoRouteMemoryEfficientMaskFormerFusionHead"
NAMESPACE = "prediction_shards_memory_efficient_v1"
MARKER = "# EXP08D_MEMORY_EFFICIENT_INFERENCE_V1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    if path.exists():
        shutil.copymode(path, temporary)
    os.replace(temporary, path)


def patch_text(source: str) -> str:
    new_root = (
        'shard_root=prediction_directory / "' + NAMESPACE
        + '" / mode / "D3",'
    )
    if MARKER in source:
        if new_root not in source:
            raise RuntimeError("partial EXP-08D patch detected; nothing changed")
        compile(source, "exp08d.py", "exec")
        return source
    anchor = "        bundle = backend.predict(\n"
    old_root = (
        'shard_root=prediction_directory / "prediction_shards" / mode / "D3",'
    )
    if source.count(anchor) != 1 or source.count(old_root) != 1:
        raise RuntimeError("EXP-08D source differs from the inspected version")
    block = '''        # EXP08D_MEMORY_EFFICIENT_INFERENCE_V1
        from morphoroute.mmdet.models.memory_efficient_fusion import (
            MorphoRouteMemoryEfficientMaskFormerFusionHead,
        )
        configuration = backend._configuration(resolved)
        fusion_before = str(configuration.model.panoptic_fusion_head.type)
        if fusion_before not in {
            "MaskFormerFusionHead",
            "mmdet.MaskFormerFusionHead",
            "MorphoRouteMemoryEfficientMaskFormerFusionHead",
        }:
            raise RuntimeError(f"unexpected fusion head: {fusion_before!r}")
        configuration.model.panoptic_fusion_head.type = (
            MorphoRouteMemoryEfficientMaskFormerFusionHead.__name__
        )
        configuration.dump(resolved.config_path)
        print(
            f"EXP08D_MEMORY_EFFICIENT_INFERENCE mode={mode} "
            f"fusion_head={configuration.model.panoptic_fusion_head.type}",
            flush=True,
        )
'''
    updated = source.replace(anchor, block + anchor, 1).replace(
        old_root, new_root, 1
    )
    compile(updated, "exp08d.py", "exec")
    return updated


def compare_postprocess(project: Path) -> dict:
    sys.path.insert(0, str(project / "src"))
    import inspect
    import torch
    from mmdet.models.seg_heads.panoptic_fusion_heads import MaskFormerFusionHead
    from morphoroute.mmdet.models.memory_efficient_fusion import (
        MorphoRouteMemoryEfficientMaskFormerFusionHead,
    )
    from morphoroute.runtime.mmdet_adapter import OfficialMMDetBackend

    if "shard_root" not in inspect.signature(OfficialMMDetBackend.predict).parameters:
        raise RuntimeError("the earlier shard isolation patch is not installed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; recovery was not started")
    cfg = dict(instance_on=True, max_per_image=12,
               panoptic_on=False, semantic_on=False)
    original = MaskFormerFusionHead(
        num_things_classes=3, num_stuff_classes=0, test_cfg=cfg
    )
    efficient = MorphoRouteMemoryEfficientMaskFormerFusionHead(
        num_things_classes=3, num_stuff_classes=0, test_cfg=cfg
    )
    cases = 0
    with torch.inference_mode():
        for device in ("cpu", "cuda:0"):
            for seed in (42, 43, 44):
                generator = torch.Generator().manual_seed(seed)
                cls = torch.randn(7, 4, generator=generator).to(device)
                masks = torch.randn(7, 23, 31, generator=generator).to(device)
                expected = original.instance_postprocess(cls, masks)
                actual = efficient.instance_postprocess(cls, masks)
                for field in ("labels", "masks", "bboxes"):
                    if not torch.equal(getattr(actual, field), getattr(expected, field)):
                        raise RuntimeError(f"postprocess equivalence failed: {device}/{seed}/{field}")
                if not torch.allclose(actual.scores, expected.scores, atol=1e-6, rtol=0):
                    raise RuntimeError(f"postprocess score equivalence failed: {device}/{seed}")
                cases += 1
    torch.cuda.empty_cache()
    path = Path(inspect.getfile(MorphoRouteMemoryEfficientMaskFormerFusionHead))
    print(f"POSTPROCESS_EQUIVALENCE=PASSED cases={cases}", flush=True)
    return {"cases": cases, "score_atol": 1e-6,
            "fusion_source": str(path), "fusion_sha256": sha256(path)}


def nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"required file is missing or empty: {path}")


def verify_completed(project: Path) -> None:
    output = project / "outputs/experiments"
    directory = output / "EXP-08D/local_6gb/seed_42"
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise RuntimeError("EXP-08D manifest is not completed")
    from morphoroute.models.interventions import INTERVENTIONS
    for mode in INTERVENTIONS:
        nonempty(directory / "expert_response_figures" / f"{mode}_predictions_D3.json")
    for name in ("M4_paired_effects.csv", "expert_load.csv"):
        path = directory / name
        nonempty(path)
        with path.open(encoding="utf-8", newline="") as handle:
            if not list(csv.DictReader(handle)):
                raise RuntimeError(f"table has no data rows: {path}")
    external = json.loads((output / "EXP-12/external_manifest.json").read_text(encoding="utf-8"))
    if external.get("status") != "completed" or external.get("fit_calls") != 0:
        raise RuntimeError("EXP-12 external must be completed with fit_calls=0")
    nonempty(output / "EXP-12/local_6gb/seed_42/calibration_D4.csv")
    print("EXP08D_EXP12_OUTPUT_VERIFICATION=PASSED", flush=True)


CHAIN = r'''set -eu
cd "$PROJECT_ROOT"
stage="EXP-08D"
trap 'rc=$?; if [ "$rc" -ne 0 ]; then printf "FAILED stage=%s exit_code=%s time=%s\n" "$stage" "$rc" "$(date -Iseconds)" > "$STATE_FILE"; fi' EXIT
printf 'RUNNING stage=%s time=%s\n' "$stage" "$(date -Iseconds)" > "$STATE_FILE"
echo "EXP-08D memory-efficient recovery starting"
"$TASK_PYTHON" "$RUNNER" "$ENTRY08D" --profile local_6gb --dataset-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --no-smoke-test --no-auto-run-prerequisites --resume
stage="EXP-12_external"
printf 'RUNNING stage=%s time=%s\n' "$stage" "$(date -Iseconds)" > "$STATE_FILE"
echo "EXP-08D completed; EXP-12 external starting"
"$TASK_PYTHON" "$ENTRY12" --profile local_6gb --dataset-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT" --stage external --no-smoke-test --no-auto-run-prerequisites --resume
stage="output_verification"
"$TASK_PYTHON" "$RECOVERY_SCRIPT" --project-root "$PROJECT_ROOT" --verify-completed
printf 'COMPLETED time=%s\n' "$(date -Iseconds)" > "$STATE_FILE"
echo "EXP08D_EXP12_CHAIN=COMPLETED"
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default="/root/autodl-tmp/railway_tunnel_experiments")
    parser.add_argument("--dataset-root", default="/root/autodl-tmp/railway_dataset_v1/铁路隧道多病害实例分割_原始影像严格划分_v1")
    parser.add_argument("--verify-completed", action="store_true")
    args = parser.parse_args()
    project = Path(args.project_root).resolve()
    sys.path.insert(0, str(project / "src"))
    if args.verify_completed:
        verify_completed(project)
        return 0

    output = project / "outputs"
    logs = output / "autodl_logs"
    source = project / "src/morphoroute/experiments/exp08d.py"
    d1 = output / "experiments/EXP-08D/local_6gb/seed_42"
    data_root = Path(args.dataset_root).resolve()
    runner = logs / "run_exp07_efficient.py"
    entry08d = project / "19 个独立实验运行入口/exp08d_expert_intervention.py"
    entry12 = project / "19 个独立实验运行入口/exp12_reliability.py"
    for path in (source, runner, entry08d, entry12,
                 d1 / "morphology_features_D1.csv", d1 / "gmm_D1.json",
                 data_root / "metadata/split_manifest.csv"):
        nonempty(path)
    prefreeze = output / "experiments/EXP-12/prefreeze_manifest.json"
    nonempty(prefreeze)
    payload = json.loads(prefreeze.read_text(encoding="utf-8"))
    nonempty(Path(payload["calibrator_path"]))
    if not any(p.stat().st_size for p in (output / "experiments/EXP-11").rglob("reliability_logits_D4.csv")):
        raise RuntimeError("existing EXP-11 D4 logits were not found")

    active = subprocess.run(
        ["pgrep", "-af", "[e]xp08d_expert_intervention.py|[e]xp12_reliability.py|[r]un_exp07_efficient.py"],
        text=True, capture_output=True,
    )
    if active.returncode == 0:
        raise RuntimeError("experiment processes already exist:\n" + active.stdout)
    if active.returncode != 1:
        raise RuntimeError("could not check experiment processes: " + active.stderr)
    gpu = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader"],
        text=True, capture_output=True, check=True,
    )
    if gpu.stdout.strip():
        raise RuntimeError("GPU already has a compute process:\n" + gpu.stdout)
    subprocess.run(["bash", "-n"], input=CHAIN, text=True, check=True)
    original = source.read_text(encoding="utf-8")
    updated = patch_text(original)
    equivalent = compare_postprocess(project)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = output / "source_backups" / ("exp08d_memory_efficient_" + stamp)
    backup.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source, backup / "exp08d.py")
    configs = list((output / "experiments/EXP-08D").rglob("resolved_mmdet_config.py"))
    for cfg in configs:
        destination = backup / cfg.relative_to(output / "experiments/EXP-08D")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cfg, destination)
    old_shards = list((d1 / "expert_response_figures/prediction_shards").rglob("part-*.json"))
    d1_hashes = {name: sha256(d1 / name) for name in ("morphology_features_D1.csv", "gmm_D1.json")}
    if updated != original:
        atomic_write(source, updated)
    if source.read_text(encoding="utf-8") != updated:
        raise RuntimeError("source verification failed after write")
    audit = dict(revision="exp08d_memory_efficient_inference_v1",
                 time=datetime.now().astimezone().isoformat(),
                 source_backup=str(backup),
                 source_sha256_before=hashlib.sha256(original.encode()).hexdigest(),
                 source_sha256_after=sha256(source),
                 old_shards_preserved=len(old_shards),
                 new_shard_namespace=NAMESPACE,
                 d1_sha256=d1_hashes, equivalence=equivalent)
    atomic_write(logs / f"EXP08D_MEMORY_REPAIR_{stamp}.json", json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print("EXP08D_MEMORY_PATCH=APPLIED", flush=True)
    print(f"SOURCE_BACKUP={backup}", flush=True)
    print(f"OLD_SHARDS_PRESERVED={len(old_shards)}", flush=True)

    state = logs / "EXP08D_EXP12.state"
    logfile = logs / f"EXP08D_EXP12_memory_recovery_{stamp}.log"
    env = dict(os.environ)
    env.update(PROJECT_ROOT=str(project), DATA_ROOT=str(data_root),
               OUTPUT_ROOT=str(output), RUNNER=str(runner), ENTRY08D=str(entry08d),
               ENTRY12=str(entry12), STATE_FILE=str(state),
               RECOVERY_SCRIPT=str(Path(__file__).resolve()), TASK_PYTHON=sys.executable,
               PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="0",
               PYTHONPATH=str(project / "src") + os.pathsep + env.get("PYTHONPATH", ""))
    atomic_write(state, "STARTING_MEMORY_EFFICIENT_RECOVERY\n")
    with logfile.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            ["bash", "-c", CHAIN], cwd=project, env=env,
            stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    atomic_write(logs / "EXP08D_EXP12.pid", str(process.pid) + "\n")
    atomic_write(logs / "EXP08D_EXP12.current_log", str(logfile) + "\n")
    print(f"RECOVERY_CHAIN_PID={process.pid}", flush=True)
    print(f"RECOVERY_LOG={logfile}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        raise SystemExit(2)
