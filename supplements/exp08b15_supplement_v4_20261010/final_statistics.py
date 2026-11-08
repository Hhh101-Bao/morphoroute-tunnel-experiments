"""Independent supplement preflight and cached D3 statistics. Never reads D4 images."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from supplement_common import (REVISION, METRICS, STRUCTURE, PRIMARY_SEEDS,
    atomic_json, checked_source, finite, group_comparison, holm, now, paired_seeds,
    read_csv, read_json, seed_summaries, select_resources, sha256, stream_bundle,
    unique_runs, validate_bundle_header, write_csv, authenticated_cache_row, verify_cache_row, payload_hash)


def d3_mapping(rows):
    result={}
    for row in rows:
        if row.get("logical_split") not in {"D3","test_internal"}:continue
        rid=str(row["record_id"]); group=str(row.get("assignment_group_id") or row.get("group_key") or "").strip()
        if not rid or not group or rid in result:raise ValueError("D3 missing/duplicate identity or true group")
        result[rid]=group
    if not result:raise ValueError("no D3 manifest records")
    return result


def resource_inputs(cohort,data):
    result=[];data=Path(data).resolve()
    if len(cohort)<8:raise ValueError("eight audited D1 inputs required")
    for row in cohort[:8]:
        if row.get("logical_split") not in {"D1","train"}:raise ValueError("resource images must be audited D1, never D4")
        path=(data/row["destination_image_relative"]).resolve()
        if not path.is_relative_to(data) or not path.is_file() or sha256(path)!=row["probe_image_sha256"]:
            raise ValueError(f"audited D1 resource input changed/missing: {path}")
        result.append(dict(record_id=row["record_id"],image_path=str(path),sha256=row["probe_image_sha256"]))
    return result


def seed_statistics(rows):
    primary=[paired_seeds(rows,"B9","B0",k,PRIMARY_SEEDS) for k in METRICS]
    valid=[r for r in primary if r["exact_signflip_p"] is not None]
    for row,p in zip(valid,holm([r["exact_signflip_p"] for r in valid])):
        row.update(holm_p=p,family="Eight B9 vs B0 D3 metrics; undefined metrics not tested",significant_holm_005=p<.05)
    subset=[r for r in rows if int(r["seed"]) in (42,43,44) and r["variant"] in ("B11","B0")]
    secondary=[paired_seeds(subset,"B11","B0",k,(42,43,44)) for k in METRICS]
    for row in secondary:row.update(scope="Exploratory three-seed comparator, not primary confirmatory evidence")
    return primary,secondary


def recover_primary_rows(rows,output):
    """Slim table may omit old seeds45–46. Recover only authenticated completed runs."""
    output=Path(output).resolve();runs=unique_runs(rows);sources={}
    missing=sorted((m,s) for m in ("B0","B9") for s in PRIMARY_SEEDS if (m,s) not in runs)
    candidates={key:[] for key in missing}
    for path in sorted((output/"training_cache/mmdet").glob("*/completed_run.json")):
        record=read_json(path)
        if record.get("status")!="completed":continue
        for m,s in missing:
            expected=output/f"experiments/EXP-09/local_6gb/seed_{s}/runs/{m}/seed_{s}"
            if Path(str(record.get("source_work_dir",""))).resolve()!=expected:continue
            evaluation=record.get("evaluation",{})
            if evaluation.get("split")!="D3" or evaluation.get("experiment_id")!="EXP-09":continue
            checkpoint=Path(record["best_checkpoint"]).resolve();configuration=Path(record["resolved_config_path"]).resolve()
            if checkpoint.parent!=expected or configuration.parent!=expected:raise ValueError("legacy cache links a different run")
            row=dict(variant=m,seed=s,evidence_tier="legacy_completed_confirmation",best_checkpoint=str(checkpoint),
                resolved_config=str(configuration),predictions=str(path.parent/"predictions_D3.json"),
                evidence_source=str(path),**{k:evaluation.get(k) for k in METRICS})
            candidates[m,s].append((row,path))
    for key,options in candidates.items():
        if not options:raise ValueError(f"missing completed primary seed {key}; no training will be started")
        if len(options)>1:raise ValueError(f"ambiguous completed legacy cache for {key}; refuse newest-by-time selection")
        row,path=options[0];runs[key]=unique_runs([row])[key];sources[str(path)]=sha256(path)
    return [runs[key] for key in sorted(runs)],sources


def source_code_hash(project):
    import hashlib
    h=hashlib.sha256()
    for path in sorted((Path(project)/"src/morphoroute").rglob("*.py")):
        h.update(str(path.relative_to(project)).replace("\\","/").encode());h.update(bytes.fromhex(sha256(path)))
    return h.hexdigest()


def validate_run_paths(row,output,table,legacy_sources):
    """Model files use per-seed work_dir; matrix exports use the parent table seed."""
    output=Path(output).resolve();table=Path(table).resolve()
    m,s=str(row["variant"]),int(row["seed"])
    directory=output/f"experiments/EXP-09/local_6gb/seed_{s}/runs/{m}/seed_{s}"
    exported=table.parent/"runs"/m/f"seed_{s}"/"predictions_D3.json"
    paths={}
    for field in ("best_checkpoint","resolved_config","predictions"):
        path=checked_source(row[field],output)
        if field=="predictions":
            source=row.get("evidence_source")
            authenticated_legacy=(source in legacy_sources and
                path==Path(source).resolve().parent/"predictions_D3.json")
            valid=path in {directory/"predictions_D3.json",exported} or authenticated_legacy
        else:
            valid=path.parent==directory
        if not valid:raise ValueError(f"{m}/{s} file belongs to a different run: {path}")
        paths[field]=path
    return paths


def _bundle_images(path,header):
    for key,value in stream_bundle(path):
        if key=="image":yield value
        else:header[key]=value


def legacy_prediction_provenance(path,row,header,output,classes,groups):
    """Bind a missing historical hash through completed cache and exact image payloads.

    This proves cache/export consistency, NOT a historical cryptographic weight hash.
    """
    from itertools import zip_longest
    output=Path(output).resolve();path=checked_source(path,output)
    checkpoint=checked_source(row["best_checkpoint"],output)
    configuration=checked_source(row["resolved_config"],output)
    metadata=header.get("metadata",{});signature=metadata.get("training_cache_signature")
    if signature is not None and (not isinstance(signature,str) or len(signature)!=64
        or any(c not in "0123456789abcdef" for c in signature)):
        raise ValueError("invalid legacy training cache signature")
    cache_root=output/"training_cache/mmdet"
    if signature is not None:
        checked_source(cache_root/signature/"completed_run.json",output)
    evidence_source=(checked_source(row["evidence_source"],output)
                     if row.get("evidence_source") else None)
    # Metadata must identify the sole exact binding, not hide competing records.
    candidates=sorted(cache_root.glob("*/completed_run.json"))
    options=[]
    for candidate in candidates:
        candidate=checked_source(candidate,output);record=read_json(candidate)
        if (record.get("status")!="completed"
            or Path(str(record.get("source_work_dir",""))).resolve()!=checkpoint.parent
            or Path(str(record.get("best_checkpoint",""))).resolve()!=checkpoint
            or Path(str(record.get("resolved_config_path",""))).resolve()!=configuration):continue
        options.append((candidate,record))
    if len(options)!=1:raise ValueError(f"legacy completed cache binding missing/ambiguous for {row['variant']}/{row['seed']}")
    record_path,record=options[0]
    if signature is not None and record_path.parent.name!=signature:
        raise ValueError("exported cache signature disagrees with unique completed binding")
    if evidence_source is not None and record_path!=evidence_source:
        raise ValueError("evidence source disagrees with unique completed binding")
    signature=record_path.parent.name
    if record.get("signature")!=signature:raise ValueError("legacy completed record signature mismatch")
    if metadata.get("training_cache_source") is not None and Path(metadata["training_cache_source"]).resolve()!=checkpoint.parent:
        raise ValueError("legacy exported cache source disagrees with selected checkpoint")
    evaluation=record.get("evaluation",{})
    if evaluation.get("split")!="D3" or evaluation.get("experiment_id")!="EXP-09":
        raise ValueError("legacy completed cache is not EXP-09 D3")
    for metric in METRICS:
        a,b=evaluation.get(metric),row.get(metric)
        if a in (None,"") or b in (None,""):
            if a not in (None,"") or b not in (None,""):raise ValueError(f"legacy cache metric mismatch: {metric}")
        elif not np.isclose(finite(a),finite(b),rtol=1e-12,atol=1e-12):
            raise ValueError(f"legacy cache metric mismatch: {metric}")
    cached=checked_source(record_path.parent/"predictions_D3.json",output)
    sources={str(record_path):sha256(record_path),str(cached):sha256(cached)}
    cached_header={};exported_header={};sentinel=object();seen=set()
    for exported_image,cached_image in zip_longest(_bundle_images(path,exported_header),
                                                   _bundle_images(cached,cached_header),fillvalue=sentinel):
        if (exported_image is sentinel or cached_image is sentinel
            or payload_hash(exported_image)!=payload_hash(cached_image)):
            raise ValueError(f"legacy cache/export image payload mismatch at image {len(seen)+1}")
        rid=str(exported_image["image_id"])
        if rid in seen or rid not in groups or str(exported_image["group_id"])!=groups[rid]:
            raise ValueError("legacy cache D3 image/group coverage mismatch")
        seen.add(rid)
        if len(seen)%250==0:print(f"LEGACY_CACHE_COMPARE {row['variant']}/seed{row['seed']} images={len(seen)}/{len(groups)}",flush=True)
    if not seen or seen!=set(groups):raise ValueError("legacy cache D3 image coverage incomplete")
    if exported_header!=header:raise ValueError("exported bundle changed during provenance audit")
    # Check the cache header independently; only absent fingerprints are accepted as legacy.
    recorded=cached_header.get("checkpoint_sha256")
    if recorded is not None and recorded!=row["checkpoint_sha256"]:raise ValueError("legacy cache checkpoint hash conflict")
    validate_bundle_header(cached_header,row["variant"],int(row["seed"]),recorded)
    if (set(cached_header["class_names"])!=set(classes)
        or cached_header.get("framework")!=header.get("framework")):
        raise ValueError("legacy cache/export framework or class mapping mismatch")
    for source,digest in sources.items():
        if sha256(source)!=digest:raise ValueError("legacy cache changed during provenance audit")
    if sha256(path)!=row["prediction_sha256"]:raise ValueError("exported prediction changed during provenance audit")
    return dict(mode="legacy_completed_cache_semantic_binding",model=row["variant"],seed=int(row["seed"]),
        historical_checkpoint_bytes_verified=False,recorded_checkpoint_sha256=None,
        current_checkpoint_sha256=row["checkpoint_sha256"],exported_header_sha256=payload_hash(header),
        exported_prediction_sha256=row["prediction_sha256"],source_files=sources,
        cache_signature=signature,matching_images=len(seen),all_eight_metrics_agree=True,
        caveat="Historical checkpoint hash was not recorded. Completed-run paths and identical per-image payloads bind the cache; current file fingerprint is not historical cryptographic proof.")


def validate_prediction_identity(path,row,classes,*,output=None,groups=None):
    """Read exported bundle metadata, never masks/images from D4 or run inference."""
    header={};count=0
    for key,value in stream_bundle(path):
        if key=="image":
            count+=1
            if count%250==0:print(f"PREFLIGHT_BUNDLE {row['variant']}/seed{row['seed']} images={count}",flush=True)
        else:header[key]=value
    proof=None
    if header.get("checkpoint_sha256") is None and output is not None and groups:
        proof=legacy_prediction_provenance(path,row,header,output,classes,groups)
        row["prediction_identity"]=proof
        print(f"LEGACY_PREDICTION_PROVENANCE=PASSED {row['variant']}/seed{row['seed']} historical_hash=NOT_RECORDED payloads=IDENTICAL",flush=True)
    validate_bundle_header(header,row["variant"],int(row["seed"]),row["checkpoint_sha256"],legacy_provenance=proof)
    if set(header["class_names"])!=set(classes):raise ValueError("bundle class mapping disagrees with audited protocol")
    if proof is None:
        row["prediction_identity"]=dict(mode="recorded_checkpoint_fingerprint",model=row["variant"],seed=int(row["seed"]),
            recorded_checkpoint_sha256=header["checkpoint_sha256"],current_checkpoint_sha256=row["checkpoint_sha256"],source_files={})
    return header


def prepare(project,data,out):
    """Validate exact table-selected identities, never select by file modification time."""
    project=Path(project).resolve();data=Path(data).resolve();out=Path(out).resolve();output=project/"outputs"
    if not out.is_relative_to((output/"supplements").resolve()):raise ValueError("output must be an independent outputs/supplements directory")
    if out in ((output/"supplements/EXP08AC_fixed_checkpoint_D1_v4").resolve(),):raise ValueError("cannot overwrite the previous probe")
    sys.path.insert(0,str(project/"src"))
    from mmengine.config import Config
    import torch,mmdet,mmengine
    if not torch.cuda.is_available():raise RuntimeError("CUDA/ENV-MAIN unavailable")
    from launch_supplement import assert_idle_gpu
    gpu_processes=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader,nounits"],text=True)
    assert_idle_gpu(gpu_processes)
    hardware=dict(gpu=torch.cuda.get_device_name(0),gpu_uuid=subprocess.check_output(
        ["nvidia-smi","--query-gpu=uuid,driver_version","--format=csv,noheader"],text=True).strip(),
        torch=torch.__version__,mmdet=mmdet.__version__,mmengine=mmengine.__version__,cuda=torch.version.cuda)
    table=checked_source(output/"experiments/EXP-09/local_6gb/seed_42/ablation_D3.csv",output)
    rows,legacy_sources=recover_primary_rows(read_csv(table),output);seed_statistics(rows)
    if {r["seed"] for r in rows if r["variant"]=="B11"}!={42,43,44}:
        raise ValueError("B11 must have exactly the approved three completed seeds 42–44")
    probe=output/"supplements/EXP08AC_fixed_checkpoint_D1_v4"
    probe_manifest=read_json(checked_source(probe/"run_manifest.json",output))
    if (probe_manifest.get("status")!="completed_posthoc_probe" or probe_manifest.get("weights_unchanged") is not True
        or probe_manifest.get("optimizer_updates")!=0 or probe_manifest.get("D4_access") is not False):
        raise ValueError("EXP08AC posthoc probe is not confirmed complete/read-only")
    probe_path=checked_source(probe/"protocol.json",output);pp=read_json(probe_path)
    manifest=data/"metadata/split_manifest.csv"
    if sha256(manifest)!=pp["manifest_sha256"] or Path(pp["dataset_root"]).resolve()!=data:
        raise ValueError("dataset manifest/root changed since the audited D1 probe")
    groups=d3_mapping(read_csv(manifest))
    if len(groups)!=1160:raise ValueError(f"D3 coverage differs from completed protocol: {len(groups)}")
    inputs=resource_inputs(pp["cohort"],data)
    if len({r["record_id"] for r in inputs})!=8:raise ValueError("duplicate resource input image")
    canonical_classes=("crack","water_leakage","spalling","efflorescence","exposed_rebar")
    if set(pp["class_names"])!=set(canonical_classes):raise ValueError("unexpected five-class mapping")
    selected=select_resources(rows)
    needed={(r["variant"],r["seed"]) for r in selected}
    needed.update((m,s) for m in ("B0","B9") for s in PRIMARY_SEEDS)
    inventory=[];files={str(table):sha256(table),str(manifest):sha256(manifest),str(probe_path):sha256(probe_path),**legacy_sources}
    for row in rows:
        if (row["variant"],row["seed"]) not in needed:continue
        row=dict(row);m,s=row["variant"],row["seed"]
        directory=output/f"experiments/EXP-09/local_6gb/seed_{s}/runs/{m}/seed_{s}"
        paths=validate_run_paths(row,output,table,legacy_sources)
        for field in ("best_checkpoint","resolved_config","predictions"):
            path=paths[field]
            print(f"PREFLIGHT_HASH {m}/seed{s} {field}",flush=True)
            row[field]=str(path);files[str(path)]=sha256(path)
        cfg=Config.fromfile(row["resolved_config"])
        if int(cfg.randomness.seed)!=s:raise ValueError("configuration seed disagrees with result table")
        if Path(cfg.work_dir).resolve()!=directory.resolve():raise ValueError("configuration work_dir disagrees with result table")
        variants=[]
        def inspect(value):
            if isinstance(value,dict):
                for key,item in value.items():
                    if key=="variant_id":variants.append(str(item))
                    inspect(item)
            elif isinstance(value,(list,tuple)):
                for item in value:inspect(item)
        inspect(cfg.model)
        if m!="B0" and (not variants or any(v!=m for v in variants)):raise ValueError(f"model identity not verified: {m}/{s}")
        if m=="B0" and any(v!="B0" for v in variants):raise ValueError("baseline configuration is not B0")
        row.update(checkpoint_sha256=files[row["best_checkpoint"]],configuration_sha256=files[row["resolved_config"]],
                   prediction_sha256=files[row["predictions"]])
        validate_prediction_identity(row["predictions"],row,canonical_classes,output=output,groups=groups)
        if row.get("prediction_identity"):files.update(row["prediction_identity"]["source_files"])
        inventory.append(row)
    for name in ("run_manifest.json","weight_integrity.json","M3_association_coefficients.csv","M3_analysis_audit.json"):
        p=checked_source(probe/name,output);files[str(p)]=sha256(p)
    epath=checked_source(output/"experiments/EXP-11/local_6gb/seed_42/E_gate_decision.json",output)
    sealed_e=read_json(epath)
    if sealed_e.get("candidate")!="B9" or sealed_e.get("strong_baseline")!="B0":raise ValueError("sealed E decision is for another comparison")
    files[str(epath)]=sha256(epath)
    package_hashes={p.name:sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    protocol=dict(revision=REVISION,project_root=str(project),dataset_root=str(data),output_dir=str(out),
        hardware=hardware,source_code_sha256=source_code_hash(project),package_hashes=package_hashes,
        source_files=files,result_rows=rows,resource_runs=[r for m in ("B0","B4","B5","B11","B9") for r in inventory if r["variant"]==m and r["seed"]==42],
        statistical_runs=[r for r in inventory if r["variant"] in ("B0","B9")],resource_inputs=inputs,
        d3_groups=groups,class_names=list(pp["class_names"]),probe_directory=str(probe),sealed_e_decision=sealed_e,
        primary_seeds=list(PRIMARY_SEEDS),secondary_seeds=[42,43,44],bootstrap=2000,random_seed=20261010,
        training=False,optimizer_updates=0,D4_image_or_prediction_access=False,
        resource_protocol=dict(input_size=640,batch_size=1,FP32=True,TF32=False,timed_images=30,
            sampling="First eight original audited D1 probe cohort entries, fixed before profiling, no performance filtering",
            timing="GPU model.predict including fusion, excludes disk/CPU preprocessing/output; distinct from EXP14 end-to-end",
            flops="Supported operators only, MMEngine one FMA=one FLOP; eight real D1 graph traces; unsupported ops disclosed"),
        statistics_protocol=dict(primary="B9-B0 five paired seeds, exact two-sided signflip, Holm across eight D3 metrics",
            AP="Use completed whole-dataset D3 AP; never average per-image AP",
            structure="Joint group bootstrap across paired seeds using IoU>=0.5 matched-instance sufficient statistics; exploratory conditional estimand",
            mechanism="08A/C remains posthoc, no mechanism gate inferred",E="Existing sealed decision retained unchanged"))
    path=out/"protocol.json"
    if path.exists() and read_json(path)!=protocol:raise ValueError("supplement protocol/input/runtime changed; refuse mixed resume")
    if not path.exists():atomic_json(path,protocol)
    print(f"SUPPLEMENT_PREFLIGHT=PASSED D3={len(groups)} models=5 primary_seeds=5 output={out}",flush=True)
    return protocol


def verify_sources(protocol):
    for name,expected in protocol["source_files"].items():
        if sha256(name)!=expected:raise ValueError(f"source changed during supplement: {name}")
    if source_code_hash(Path(protocol["project_root"]))!=protocol["source_code_sha256"]:
        raise ValueError("project source changed during supplement")


def _coco_counts(encoded,pixels):
    """Lossless COCO signed 5-bit/delta RLE; no image-size mask allocation.

    Wire format: cocodataset/cocoapi common/maskApi.c rleFrString.
    The two-back delta starts at the FOURTH count (m > 2), not third.
    """
    values=[];position=0;total=0
    while position<len(encoded):
        value=0;chunks=0
        while True:
            if position>=len(encoded):raise ValueError('truncated COCO compressed RLE')
            code=ord(encoded[position])-48;position+=1
            if not 0<=code<=63:raise ValueError('invalid COCO compressed RLE character')
            value|=(code & 31) << (5*chunks);chunks+=1
            if chunks>13:raise ValueError('COCO compressed RLE count overflow')
            if not code & 32:
                if code & 16:value|=-1 << (5*chunks)
                break
        if len(values)>2:value+=values[-2]
        if value<0 or value>pixels:raise ValueError('invalid COCO compressed RLE count')
        total+=value
        if total>pixels:raise ValueError('RLE length exceeds image size')
        values.append(value)
    return values


def _rle(instance,height,width):
    if any(not isinstance(n,int) or isinstance(n,bool) or n<=0 for n in (height,width)):
        raise ValueError('invalid RLE image dimensions')
    pixels=height*width
    if pixels>np.iinfo(np.int64).max:raise ValueError('RLE image size overflow')
    rle=instance["mask_rle"]
    if not isinstance(rle.get('size'),list) or any(not isinstance(n,int) or isinstance(n,bool) for n in rle['size']):
        raise ValueError('invalid RLE size')
    if tuple(rle["size"])!=(height,width):raise ValueError("RLE/image shape mismatch")
    values=rle["counts"]
    if isinstance(values,str):values=_coco_counts(values,pixels)
    if not isinstance(values,list) or any(not isinstance(n,int) or isinstance(n,bool) or n<0 for n in values):
        raise ValueError("RLE counts must be COCO compressed string or nonnegative integer list")
    if not values or sum(values)!=pixels:raise ValueError('RLE length mismatch')
    ends=np.cumsum(values,dtype=np.int64)
    if not len(ends) or int(ends[-1])!=height*width:raise ValueError("RLE length mismatch")
    starts=np.r_[0,ends[:-1]]
    foreground=np.array([(a,b) for i,(a,b) in enumerate(zip(starts,ends)) if i%2 and b>a],dtype=np.int64).reshape(-1,2)
    score=finite(instance.get("score",1))
    if not 0<=score<=1:raise ValueError("prediction score outside [0,1]")
    return dict(name=instance["class_name"],score=score,runs=foreground,area=int(sum(b-a for a,b in foreground)))


def _rle_iou(a,b):
    aa,bb=a["runs"],b["runs"];i=j=0;intersection=0
    while i<len(aa) and j<len(bb):
        left=max(aa[i,0],bb[j,0]);right=min(aa[i,1],bb[j,1]);intersection+=max(0,int(right-left))
        if aa[i,1]<bb[j,1]:i+=1
        else:j+=1
    union=a["area"]+b["area"]-intersection
    return intersection/union if union else 1.


def _rle_pair_crop(a,b,height):
    """Decode only the joint foreground bounding box, never resize metric masks."""
    allruns=np.concatenate((a["runs"],b["runs"]),axis=0)
    if not len(allruns):return np.zeros((5,5),bool),np.zeros((5,5),bool)
    x0=int(allruns[:,0].min()//height);x1=int((allruns[:,1]-1).max()//height)
    crosses=allruns[:,0]//height!=(allruns[:,1]-1)//height
    y0=0 if crosses.any() else int((allruns[:,0]%height).min())
    y1=height-1 if crosses.any() else int(((allruns[:,1]-1)%height).max())
    masks=[]
    for item in (a,b):
        mask=np.zeros((y1-y0+5,x1-x0+5),bool)
        for left,right in item["runs"]:
            for x in range(int(left//height),int((right-1)//height)+1):
                lo=max(int(left),x*height)-x*height;hi=min(int(right),(x+1)*height)-x*height
                mask[lo-y0+2:hi-y0+2,x-x0+2]=True
        masks.append(mask)
    return tuple(masks)


def image_statistics(payload,classes):
    from morphoroute.metrics import structure as st
    h,w=int(payload["height"]),int(payload["width"])
    if h<=0 or w<=0:raise ValueError("invalid prediction image shape")
    pred=[_rle(r,h,w) for r in payload["predictions"]];gt=[_rle(r,h,w) for r in payload["targets"]]
    if any(r["name"] not in classes for r in pred+gt):raise ValueError("unknown class in D3 bundle")
    if any(r["area"]==0 for r in gt):raise ValueError("empty foreground GT instance is not a valid matched-instance target")
    total={k:0. for k in STRUCTURE};count={k:0 for k in STRUCTURE};fp=fn=0
    functions={"crack_cldice":st.cldice,"crack_break_rate":st.crack_break_rate,"crack_component_error":st.component_count_error,
        "area_boundary_f1":st.boundary_f1,"area_hd95":st.hd95,"area_relative_error":st.area_relative_error}
    for name in classes:
        candidates=[r for r in gt if r["name"]==name];unmatched=set(range(len(candidates)))
        for prediction in sorted((r for r in pred if r["name"]==name),key=lambda r:r["score"],reverse=True):
            iou,index=max(((_rle_iou(prediction,candidates[i]),i) for i in unmatched),default=(0.,-1))
            if index<0 or iou<.5:fp+=1;continue
            unmatched.remove(index);a,b=_rle_pair_crop(prediction,candidates[index],h)
            for key in (STRUCTURE[:3] if name=="crack" else STRUCTURE[3:]):
                value=finite(functions[key](a,b));total[key]+=value;count[key]+=1
        fn+=len(unmatched)
    return dict(totals=total,counts=count,false_positives=fp,false_negatives=fn,ground_truth_count=len(gt),prediction_count=len(pred))


def cached_structure(protocol,item,out):
    """Crash-safe per-image cache tied to frozen bundle hash; partial records not final evidence."""
    model,seed=item["variant"],item["seed"];groups=protocol["d3_groups"]
    base=Path(out)/"structure_cache"/f"{model}_seed{seed}";base.mkdir(parents=True,exist_ok=True)
    identity=dict(prediction_sha256=item["prediction_sha256"],checkpoint_sha256=item["checkpoint_sha256"],protocol_sha256=sha256(Path(out)/"protocol.json"))
    idpath=base/"identity.json"
    if idpath.exists() and read_json(idpath)!=identity:raise ValueError("structure partial cache identity mismatch")
    if not idpath.exists():atomic_json(idpath,identity)
    cache=base/"images.jsonl";existing={}
    if cache.exists():
        with cache.open("r+b") as f:
            offset=0
            for raw in iter(f.readline,b""):
                if not raw.endswith(b"\n"):
                    f.truncate(offset);break  # only an incomplete final write is recoverable
                from supplement_common import _no_duplicate_fields,_reject_constant
                row=verify_cache_row(json.loads(raw,object_pairs_hook=_no_duplicate_fields,parse_constant=_reject_constant))
                rid=row["image_id"]
                if rid in existing or groups.get(rid)!=row["group_id"]:raise ValueError("duplicate/invalid structure partial record")
                existing[rid]=row;offset=f.tell()
    started=time.monotonic();seen=set();header={};fresh=0
    with cache.open("a",encoding="utf-8",newline="\n") as f:
        for key,payload in stream_bundle(item["predictions"]):
            if key!="image":header[key]=payload;continue
            rid=str(payload["image_id"])
            if rid in seen or rid not in groups or str(payload["group_id"])!=groups[rid]:raise ValueError("D3 image/group coverage mismatch")
            seen.add(rid)
            phash=payload_hash(payload)
            if rid not in existing:
                print(f"STRUCTURE_IMAGE_START model={model} seed={seed} image={len(seen)}/{len(groups)} id={rid}",flush=True)
                row=authenticated_cache_row(dict(image_id=rid,group_id=groups[rid],payload_sha256=phash,
                                                **image_statistics(payload,protocol["class_names"])))
                f.write(json.dumps(row,allow_nan=False,separators=(",",":"))+"\n");f.flush();os.fsync(f.fileno())
                existing[rid]=row;fresh+=1
            elif existing[rid].get("payload_sha256")!=phash:raise ValueError("cached image payload identity mismatch")
            if len(seen)%25==0:
                print(f"STRUCTURE_CACHE model={model} seed={seed} images={len(seen)}/{len(groups)} new={fresh} elapsed_min={(time.monotonic()-started)/60:.1f}",flush=True)
    validate_bundle_header(header,model,seed,item["checkpoint_sha256"],legacy_provenance=item.get("prediction_identity"))
    if set(header["class_names"])!=set(protocol["class_names"]) or seen!=set(groups) or set(existing)!=seen:
        raise ValueError("cached D3 bundle does not fully match the fixed dataset/classes")
    results=[]
    for rid,row in existing.items():
        for metric in STRUCTURE:
            results.append(dict(model=model,seed=seed,image_id=rid,group_id=groups[rid],metric=metric,
                                total=finite(row["totals"][metric]),count=int(row["counts"][metric])))
    checks=[]
    for metric in STRUCTURE:
        numerator=sum(r["total"] for r in results if r["metric"]==metric)
        denominator=sum(r["count"] for r in results if r["metric"]==metric)
        value=numerator/denominator if denominator else None;reported=item[metric]
        if ((value is None)!=(reported is None)) or (value is not None and not np.isclose(value,reported,rtol=1e-5,atol=1e-6)):
            raise ValueError(f"D3 structure reconciliation mismatch {model}/{seed} {metric}: cache={value}, original={reported}")
        checks.append(dict(model=model,seed=seed,metric=metric,recomputed=value,reported=reported,count=denominator))
    atomic_json(base/"completed.json",dict(identity=identity,image_count=len(seen),reconciliation=checks))
    return results,checks


def analyze(project,protocol,out):
    sys.path.insert(0,str(Path(project)/"src"));out=Path(out)
    verify_sources(protocol)
    rows=protocol["result_rows"];primary,secondary=seed_statistics(rows)
    runs={(r["variant"],r["seed"]):r for r in protocol["resource_runs"]+protocol["statistical_runs"]}
    atomic_json(out/"EXP15_prediction_provenance.json",dict(runs=[dict(model=m,seed=s,**{"audit":r["prediction_identity"]})
        for (m,s),r in sorted(runs.items())],
        scope="Legacy absent hashes are not filled in. Completed cache/export payload binding does not reconstruct a historical checkpoint fingerprint."))
    write_csv(out/"EXP15_seed_summary_D3.csv",seed_summaries(rows))
    atomic_json(out/"EXP15_paired_seed_tests.json",dict(primary=primary,secondary=secondary))
    write_csv(out/"EXP15_paired_seed_tests.csv",[dict(r,seeds=";".join(map(str,r.get("seeds",[])))) for r in primary+secondary])
    sufficient=[];reconciliation=[]
    for item in protocol["statistical_runs"]:
        print(f"STRUCTURE_START model={item['variant']} seed={item['seed']}",flush=True)
        r,c=cached_structure(protocol,item,out);sufficient.extend(r);reconciliation.extend(c)
    write_csv(out/"EXP15_structure_sufficient_statistics.csv",sufficient)
    write_csv(out/"EXP15_structure_reconciliation.csv",reconciliation)
    comparisons=[group_comparison(sufficient,k,PRIMARY_SEEDS,protocol["bootstrap"],protocol["random_seed"]+i)
                 for i,k in enumerate(STRUCTURE)]
    atomic_json(out/"EXP15_group_bootstrap_D3.json",comparisons)
    write_csv(out/"EXP15_group_bootstrap_D3.csv",[dict(r,matched_instance_counts=json.dumps(r["matched_instance_counts"])) for r in comparisons])
    atomic_json(out/"EXP15_evidence_scope.json",dict(status="completed_statistics_supplement",sealed_E=protocol["sealed_e_decision"],
        mechanism_gate="not_established_by_this_supplement",posthoc_probe_reference=protocol["probe_directory"],
        group_bootstrap_scope="Matched-instance structure only; no group AP significance test",
        seed_inference_scope="Completed fixed-D3 runs, five paired primary seeds; not population-wide/domain-wide causal evidence",
        prediction_provenance="See EXP15_prediction_provenance.json. A legacy cache with no recorded checkpoint hash has completed-record and semantic payload provenance, not historical cryptographic weight verification.",
        D4_images_accessed=False,training=False,optimizer_updates=0,not_full_original_EXP15_SOP=True))
    verify_sources(protocol)
    print("EXP15_STATISTICS_SUPPLEMENT=COMPLETED",flush=True)
