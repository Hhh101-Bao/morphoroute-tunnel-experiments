#!/usr/bin/env python3
"""Start ONLY the approved EXP-08A/08C supplement on AutoDL (Linux)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ACTIVE_LOG = None


def run_logged(command, log, *, env=None, cwd=None):
    """Persist real preflight diagnostics while still showing them in the terminal."""
    with subprocess.Popen(command, env=env, cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                          errors="replace", bufsize=1) as process:
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def write_pointer(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(str(value) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def command_for(script, project, data, mode):
    return [sys.executable, "-u", str(script), "--project-root", str(project),
            "--dataset-root", str(data), "--stage", mode,
            "--images", "300", "--bootstrap", "1000"]


def main():
    global ACTIVE_LOG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path,
                        default=Path("/root/autodl-tmp/railway_tunnel_experiments"))
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("/root/autodl-tmp/railway_dataset_v1/铁路隧道多病害实例分割_原始影像严格划分_v1"))
    parser.add_argument("--mode", choices=("pilot", "run"), default="pilot")
    args = parser.parse_args()
    if sys.platform != "linux":
        raise RuntimeError("This launcher is for the AutoDL Linux instance, not local Windows")
    import fcntl
    project, data = args.project_root.resolve(), args.dataset_root.resolve()
    package = Path(__file__).resolve().parent
    script, tests = package / "fixed_checkpoint_probe.py", package / "test_fixed_checkpoint_probe.py"
    if not project.is_dir() or not (data / "metadata/split_manifest.csv").is_file():
        raise FileNotFoundError("project root or dataset split manifest is missing")
    if not script.is_file() or not tests.is_file():
        raise FileNotFoundError("upload the complete package; collector or tests missing")
    logs = project / "outputs/autodl_logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "EXP08AC_FIXED_PROBE.launch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A supplement launcher/worker is already active; nothing was stopped") from exc
        environment = os.environ.copy()
        environment.update(PYTHONPATH=str(project / "src") + os.pathsep + environment.get("PYTHONPATH", ""),
                           PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="0", CUDA_MODULE_LOADING="LAZY",
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                           NUMEXPR_NUM_THREADS="1")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = logs / f"EXP08AC_FIXED_PROBE_v4_{args.mode}_{stamp}_{os.getpid()}.log"
        with log_path.open("x", encoding="utf-8") as log:
            ACTIVE_LOG = log_path
            write_pointer(logs / "EXP08AC_FIXED_PROBE.current_log", log_path)
            print(f"STARTUP_LOG={log_path}", flush=True)
            print("===== Run focused supplement tests (must include the Torch test on AutoDL) =====", flush=True)
            run_logged([sys.executable, "-c", "import torch, mmdet, mmengine; assert torch.cuda.is_available(); print('DEPENDENCIES=PASSED')"],
                       log, env=environment)
            run_logged([sys.executable, str(tests)], log, env=environment, cwd=package)
            print("===== Validate D1 cohort, checkpoints, hashes and mappings; no GPU collection yet =====", flush=True)
            run_logged(command_for(script, project, data, "preflight"), log, env=environment)
            gpu = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, check=True)
            active = [line.strip() for line in gpu.stdout.splitlines() if line.strip().isdigit()]
            if active:
                raise RuntimeError(f"GPU processes exist {active}; launch refused, no process was killed")
            log.write("SUPPLEMENT_PREFLIGHT_COMPLETED; starting guarded worker\n")
            log.flush()
            child = subprocess.Popen(command_for(script, project, data, args.mode),
                                     env=environment, cwd=project, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                     pass_fds=(lock.fileno(),))
        write_pointer(logs / "EXP08AC_FIXED_PROBE.pid", child.pid)
        write_pointer(logs / "EXP08AC_FIXED_PROBE.current_log", log_path)
        print(f"SUPPLEMENT_LAUNCHED pid={child.pid} mode={args.mode}", flush=True)
        print(f"log={log_path}", flush=True)
        print(f"state={project / 'outputs/supplements/EXP08AC_fixed_checkpoint_D1_v4/state.json'}", flush=True)
        print("Scope: 08A/08C post-hoc supplement only. 08B resources and EXP15 are NOT launched.", flush=True)
        print("In run mode, a five-image immutable-weight pilot must pass before full collection.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if ACTIVE_LOG is not None:
            with ACTIVE_LOG.open("a", encoding="utf-8") as log:
                log.write(f"LAUNCH_REFUSED: {error}\n")
        print(f"LAUNCH_REFUSED: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
