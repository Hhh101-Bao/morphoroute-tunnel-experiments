"""Fail-stop, single-controller AutoDL launcher. Does not kill existing jobs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import unittest
from datetime import datetime
from pathlib import Path

from supplement_common import atomic_json, read_json, sha256

PROJECT="/root/autodl-tmp/railway_tunnel_experiments"
DATA="/root/autodl-tmp/railway_dataset_v1/铁路隧道多病害实例分割_原始影像严格划分_v1"


def assert_idle_gpu(output):
    if output.strip():raise RuntimeError(f"GPU has an existing compute process; no job will be stopped: {output.strip()}")


def verify_completion(out):
    out=Path(out).resolve();path=out/"run_manifest.json"
    if not path.exists():return False
    manifest=read_json(path)
    if manifest.get("status")!="completed_resources_and_statistics_supplement":return False
    if manifest.get("protocol_sha256")!=sha256(out/"protocol.json"):return False
    if not manifest.get("outputs"):return False
    for name,digest in manifest["outputs"].items():
        if Path(name).name!=name:raise ValueError("completion manifest has an invalid output name")
        path=out/name
        if not path.is_file() or sha256(path)!=digest:return False
    return True


def test_package(project):
    os.environ["SUPPLEMENT_TEST_PROJECT"]=str(project)
    package=Path(__file__).parent
    for name,digest in read_json(package/"PACKAGE_SHA256.json").items():
        if Path(name).name!=name or sha256(package/name)!=digest:raise ValueError(f"package verification failed: {name}")
    import torch,mmengine,mmdet  # mandatory, not an optional skipped GPU test
    if not torch.cuda.is_available():raise RuntimeError("CUDA is unavailable; activate the existing training environment")
    suite=unittest.defaultTestLoader.discover(str(package),pattern="test_supplement.py")
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:raise RuntimeError("focused tests failed or were skipped on AutoDL")


def main():
    parser=argparse.ArgumentParser(description="Sequential EXP08B resources and EXP15 statistics supplement, no retraining/D4 inference")
    parser.add_argument("--project-root",default=PROJECT)
    parser.add_argument("--dataset-root",default=DATA)
    parser.add_argument("--mode",choices=("run","preflight"),default="run")
    args=parser.parse_args()
    if sys.platform!="linux":raise RuntimeError("Run this launcher in the AutoDL Linux training environment")
    project=Path(args.project_root).resolve();data=Path(args.dataset_root).resolve()
    out=project/"outputs/supplements/EXP08B_EXP15_20261010_v1"
    logs=project/"outputs/autodl_logs";logs.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=open(logs/"EXP08B_EXP15.lock","a+")
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError("supplement controller is already active; refusing a duplicate")
    test_package(project)
    from final_statistics import prepare
    prepare(project,data,out)
    if args.mode=="preflight":return
    if verify_completion(out):
        print(f"SUPPLEMENT_ALREADY_COMPLETED output={out}");return
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile=logs/f"EXP08B_EXP15_{stamp}.log"
    env=os.environ.copy()
    env.update(PYTHONUNBUFFERED="1",CUDA_VISIBLE_DEVICES="0",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",PYTHONPATH=str(project/"src")+os.pathsep+env.get("PYTHONPATH",""))
    cmd=[sys.executable,"-B","-u",str(Path(__file__).parent/"run_supplement.py"),"--stage","run",
         "--project-root",str(project),"--out",str(out),"--lock-fd",str(lock.fileno())]
    with logfile.open("wb") as f:
        child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,
                               env=env,start_new_session=True,close_fds=True,pass_fds=(lock.fileno(),))
    (logs/"EXP08B_EXP15.pid").write_text(str(child.pid)+"\n",encoding="utf-8")
    (logs/"EXP08B_EXP15.current_log").write_text(str(logfile)+"\n",encoding="utf-8")
    print(f"SUPPLEMENT_STARTED pid={child.pid}\nlog={logfile}\noutput={out}",flush=True)
    # The child inherits the very same locked descriptor. Parent exit does not unlock it.


if __name__=="__main__":
    try:main()
    except Exception as error:
        import traceback
        traceback.print_exc();print(f"LAUNCH_REFUSED: {error}",flush=True);sys.exit(2)
