"""Serial coordinator: each resource model gets a fresh process, then cached statistics."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from supplement_common import RESOURCE_MODELS, atomic_json, now, read_json, sha256, write_csv, verify_cache_row


def state(out,status,**extra):
    value=dict(status=status,time=now(),pid=os.getpid(),**extra)
    atomic_json(out/"state.json",value)
    logs=Path(read_json(out/"protocol.json")["project_root"])/"outputs/autodl_logs"
    (logs/"EXP08B_EXP15.state").write_text(json.dumps(value,ensure_ascii=False)+"\n",encoding="utf-8")
    print("SUPPLEMENT_STATE "+json.dumps(value,ensure_ascii=False),flush=True)


def resource_table(all_results):
    table=[]
    for r in all_results:
        table.append(dict(model=r["model"],seed=r["seed"],parameters=r["parameters"],
            counted_GFLOPs_mean=r["counted_flops_mmengine_mean"]/1e9,counted_GFLOPs_max=r["counted_flops_mmengine_max"]/1e9,
            operator_coverage_complete=r["flops_complete_operator_coverage"],unsupported_ops=json.dumps(r["unsupported_ops"],sort_keys=True),
            peak_allocated_MiB=r["peak_memory_allocated_bytes"]/1024**2,peak_reserved_MiB=r["peak_memory_reserved_bytes"]/1024**2,
            latency_mean_ms=r["latency_ms_mean"],latency_median_ms=r["latency_ms_median"],latency_p95_ms=r["latency_ms_p95"],
            throughput_images_per_second=r["throughput_images_per_second"],batch_size=1,input_size=640,precision="FP32",TF32=False,
            timing_scope=r["timing_scope"],weights_unchanged=r["weights_unchanged"],source_checkpoint=r["identity"]["checkpoint_sha256"],
            GPU=r["hardware"]["gpu"],FLOPs_convention=r["flops_convention"]))
    return table


def summarize_resources(protocol,out):
    all_results=[verify_cache_row(read_json(out/"resources"/f"{m}.json")) for m in RESOURCE_MODELS]
    write_csv(out/"EXP08B_resources_640_FP32.csv",resource_table(all_results))
    atomic_json(out/"EXP08B_resources_audit.json",dict(status="completed_fixed_input_resources_supplement",
        unsupported_ops_models=[r["model"] for r in all_results if not r["flops_complete_operator_coverage"]],
        flops_caveat="Counted supported operators are not exact full-network FLOPs if unsupported_ops is nonempty; do not hide omissions",
        measurements=all_results,training=False,optimizer_updates=0,D4_images_accessed=False))


def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",choices=("run","resource","statistics"),required=True)
    p.add_argument("--project-root",required=True);p.add_argument("--out",required=True)
    p.add_argument("--model",choices=RESOURCE_MODELS);p.add_argument("--lock-fd",type=int)
    args=p.parse_args();out=Path(args.out).resolve();project=Path(args.project_root).resolve()
    if args.lock_fd is None:raise RuntimeError("all workers must inherit the exclusive launcher lock")
    stat=os.fstat(args.lock_fd);expected=os.stat(project/"outputs/autodl_logs/EXP08B_EXP15.lock")
    if (stat.st_dev,stat.st_ino)!=(expected.st_dev,expected.st_ino):raise RuntimeError("worker has an unrelated lock descriptor")
    import fcntl
    fcntl.flock(args.lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    protocol=read_json(out/"protocol.json")
    if Path(protocol["project_root"]).resolve()!=project or Path(protocol["output_dir"]).resolve()!=out:
        raise ValueError("worker protocol path mismatch")
    for name,digest in protocol["package_hashes"].items():
        if sha256(Path(__file__).parent/name)!=digest:raise ValueError("package changed after preflight")
    if args.stage=="resource":
        if not args.model:raise ValueError("resource worker requires model")
        from resource_measurement import measure_one
        measure_one(project,protocol,out,args.model);return
    if args.stage=="statistics":
        from final_statistics import analyze
        analyze(project,protocol,out);return
    from final_statistics import verify_sources
    verify_sources(protocol)
    try:
        for model in RESOURCE_MODELS:
            state(out,"RUNNING",stage="EXP08B_resources",model=model)
            subprocess.run([sys.executable,"-B","-u",__file__,"--stage","resource","--project-root",str(project),"--out",str(out),"--model",model,
                            "--lock-fd",str(args.lock_fd)],check=True,pass_fds=(args.lock_fd,))
        summarize_resources(protocol,out)
        state(out,"RUNNING",stage="EXP15_statistics")
        subprocess.run([sys.executable,"-B","-u",__file__,"--stage","statistics","--project-root",str(project),"--out",str(out),
                        "--lock-fd",str(args.lock_fd)],check=True,pass_fds=(args.lock_fd,))
        verify_sources(protocol)
        products=[p for p in out.iterdir() if p.is_file() and p.name.startswith(("EXP08B_","EXP15_"))]
        atomic_json(out/"run_manifest.json",dict(status="completed_resources_and_statistics_supplement",finished_at_utc=now(),
            output_dir=str(out),outputs={p.name:sha256(p) for p in products},protocol_sha256=sha256(out/"protocol.json"),
            original_results_unchanged=True,training=False,optimizer_updates=0,D4_images_or_predictions_accessed=False,
            original_full_EXP08_or_EXP15_SOP_reconstructed=False))
        state(out,"COMPLETED",stage="EXP08B_EXP15_supplement")
    except Exception as e:
        state(out,"FAILED",reason=str(e));raise


if __name__=="__main__":
    try:main()
    except Exception as e:
        import traceback
        traceback.print_exc();print(f"SUPPLEMENT_FAILED: {e}",flush=True);sys.exit(2)
