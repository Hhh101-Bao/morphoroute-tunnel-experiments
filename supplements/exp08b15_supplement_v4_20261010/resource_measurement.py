"""Frozen 640px GPU model-only profiling. No dataset inference export/training."""
from __future__ import annotations

import gc
import hashlib
import time
from pathlib import Path

import numpy as np

from supplement_common import atomic_json, now, read_json, sha256, authenticated_cache_row, verify_cache_row


def crop_pair(a, b):
    """Translation-invariant joint foreground crop, with 2px zero margin."""
    a=np.asarray(a,bool);b=np.asarray(b,bool)
    if a.shape!=b.shape or a.ndim!=2:
        raise ValueError("masks must have matching 2D shapes")
    yy,xx=np.where(a|b)
    if not len(yy):return np.zeros((5,5),bool),np.zeros((5,5),bool)
    sl=np.s_[yy.min():yy.max()+1,xx.min():xx.max()+1]
    return np.pad(a[sl],2),np.pad(b[sl],2)


def model_digest(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        value=tensor.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def runtime(project):
    import sys
    sys.path.insert(0,str(Path(project)/"src"))
    import torch
    from mmdet.utils import register_all_modules
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():raise RuntimeError("CUDA required; no CPU/AMP fallback")
    register_all_modules(init_default_scope=True)
    return torch


def _disable_checkpointing(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if key in ("with_cp","use_checkpoint","gradient_checkpointing") and isinstance(item,bool):value[key]=False
            else:_disable_checkpointing(item)
    elif isinstance(value,(tuple,list)):
        for item in value:_disable_checkpointing(item)


def load_frozen(item):
    import torch
    from mmengine.config import Config
    from mmdet.apis import init_detector
    cfg=Config.fromfile(item["resolved_config"])
    _disable_checkpointing(cfg.model)
    if "backbone" in cfg.model:cfg.model.backbone.init_cfg=None
    model=init_detector(cfg,item["best_checkpoint"],device="cuda:0")
    model.eval();model.diagnostic_path=None
    for module in model.modules():
        for flag in ("with_cp","use_checkpoint"):
            if hasattr(module,flag):setattr(module,flag,False)
    if any(p.dtype!=torch.float32 for p in model.parameters() if p.is_floating_point()):
        raise ValueError("loaded model is not FP32")
    checkpoint=torch.load(item["best_checkpoint"],map_location="cpu")
    state=checkpoint["state_dict"]
    actual=model.state_dict()
    if set(actual)!=set(state):
        raise ValueError(f"checkpoint/model state keys differ: missing={sorted(set(actual)-set(state))[:8]}, unexpected={sorted(set(state)-set(actual))[:8]}")
    for key,tensor in actual.items():
        saved=state[key]
        if tensor.shape!=saved.shape or not torch.equal(tensor.detach().cpu(),saved):
            raise ValueError(f"checkpoint tensor not faithfully loaded: {key}")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"nonfinite model tensor: {key}")
    del state,checkpoint,actual;gc.collect()
    return model


def make_batch(model, entry):
    import torch
    import cv2
    from PIL import Image,ImageOps
    from mmdet.structures import DetDataSample
    if sha256(entry["image_path"])!=entry["sha256"]:
        raise ValueError("resource input image changed")
    with Image.open(entry["image_path"]) as image:
        rgb=np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    h,w=rgb.shape[:2];scale=min(640/h,640/w)
    hh,ww=max(1,int(round(h*scale))),max(1,int(round(w*scale)))
    bgr=np.ascontiguousarray(cv2.resize(rgb,(ww,hh),interpolation=cv2.INTER_LINEAR)[:,:,::-1])
    canvas=np.zeros((640,640,3),np.uint8);canvas[:hh,:ww]=bgr
    tensor=torch.from_numpy(canvas.transpose(2,0,1).copy())
    sample=DetDataSample()
    sample.set_metainfo(dict(img_id=entry["record_id"],img_shape=(hh,ww),ori_shape=(hh,ww),
                             batch_input_shape=(640,640),pad_shape=(640,640),scale_factor=(1.,1.)))
    batch=model.data_preprocessor(dict(inputs=[tensor],data_samples=[sample]),training=False)
    if tuple(batch["inputs"].shape)!=(1,3,640,640):raise ValueError("profile input is not 1x3x640x640")
    # Keep the input cohort on CPU. Retaining predict-mutated DataSamples would
    # retain eight sets of GPU masks and inflate the batch1 peak measurement.
    return dict(inputs=batch["inputs"].cpu(),metadata=dict(batch["data_samples"][0].metainfo))


def fresh_batch(template):
    from mmdet.structures import DetDataSample
    sample=DetDataSample();sample.set_metainfo(dict(template["metadata"]))
    return dict(inputs=template["inputs"].to("cuda:0"),data_samples=[sample])


def _flops(model,batch):
    import torch
    from mmengine.analysis import FlopAnalyzer

    class TensorForward(torch.nn.Module):
        def __init__(self,detector,samples):
            super().__init__();self.detector=detector;self.samples=samples
        def forward(self,x):
            return self.detector(x,self.samples,mode="tensor")

    wrapper=TensorForward(model,batch["data_samples"])
    analyzer=FlopAnalyzer(wrapper,(batch["inputs"],))
    analyzer.unsupported_ops_warnings(False);analyzer.uncalled_modules_warnings(False);analyzer.tracer_warnings("none")
    count=float(analyzer.total())
    if not np.isfinite(count) or count<=0:raise ValueError("FLOPs trace produced no finite positive count")
    return dict(counted_flops=count,unsupported_ops=dict(analyzer.unsupported_ops()),
                by_operator=dict(analyzer.by_operator()))


def measure_one(project,protocol,out,model_name):
    torch=runtime(project)
    item=next(r for r in protocol["resource_runs"] if r["variant"]==model_name)
    out=Path(out); destination=out/"resources"/f"{model_name}.json"
    identity=dict(checkpoint_sha256=item["checkpoint_sha256"],configuration_sha256=item["configuration_sha256"],
                  protocol_sha256=sha256(out/"protocol.json"))
    if destination.exists():
        previous=verify_cache_row(read_json(destination))
        if (previous.get("identity")!=identity or previous.get("weights_unchanged") is not True
            or previous.get("state_sha256_before")!=previous.get("state_sha256_after")
            or previous.get("model")!=model_name or previous.get("hardware")!=protocol["hardware"]
            or previous.get("batch_size")!=1 or previous.get("input_size")!=640
            or previous.get("precision")!="FP32" or previous.get("tf32") is not False):
            raise ValueError("resource cache identity changed; do not mix measurements")
        for name in ("latency_ms_mean","latency_ms_median","latency_ms_p95","parameters","counted_flops_mmengine_mean",
                     "peak_memory_allocated_bytes","peak_memory_reserved_bytes","throughput_images_per_second"):
            if not np.isfinite(previous[name]) or previous[name]<=0:raise ValueError("invalid cached resource measurement")
        print(f"RESOURCE_REUSED model={model_name}",flush=True);return
    print(f"RESOURCE_START model={model_name} checkpoint={item['best_checkpoint']}",flush=True)
    model=load_frozen(item);before=model_digest(model)
    batches=[make_batch(model,e) for e in protocol["resource_inputs"]]
    durations=[];traces=[]
    with torch.inference_mode():
        for template in batches:
            batch=fresh_batch(template)
            result=model.predict(batch["inputs"],batch["data_samples"],rescale=False);del result,batch
        torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        for repeat in range(30):
            batch=fresh_batch(batches[repeat%len(batches)])
            torch.cuda.synchronize();started=time.perf_counter()
            result=model.predict(batch["inputs"],batch["data_samples"],rescale=False)
            torch.cuda.synchronize();durations.append((time.perf_counter()-started)*1000);del result,batch
        allocated=int(torch.cuda.max_memory_allocated());reserved=int(torch.cuda.max_memory_reserved())
        for index,template in enumerate(batches):
            batch=fresh_batch(template)
            print(f"RESOURCE_FLOPS model={model_name} input={index+1}/{len(batches)}",flush=True)
            traces.append(_flops(model,batch))
            del batch
    after=model_digest(model)
    if before!=after:raise ValueError("profiling changed a model tensor; result discarded")
    if sha256(item["best_checkpoint"])!=item["checkpoint_sha256"]:
        raise ValueError("checkpoint file changed during profiling")
    if not np.isfinite(durations).all() or min(durations)<=0:raise ValueError("invalid timing")
    counts=[t["counted_flops"] for t in traces]
    unsupported={k: max(t["unsupported_ops"].get(k,0) for t in traces) for k in set().union(*(t["unsupported_ops"] for t in traces))}
    result=dict(model=model_name,seed=42,identity=identity,finished_at_utc=now(),weights_unchanged=True,
                optimizer_updates=0,batch_size=1,input_size=640,precision="FP32",tf32=False,
                parameters=sum(p.numel() for p in model.parameters()),
                latency_ms_mean=float(np.mean(durations)),latency_ms_median=float(np.median(durations)),
                latency_ms_p95=float(np.percentile(durations,95)),throughput_images_per_second=1000/float(np.mean(durations)),
                peak_memory_allocated_bytes=allocated,peak_memory_reserved_bytes=reserved,
                counted_flops_mmengine_mean=float(np.mean(counts)),counted_flops_mmengine_max=max(counts),
                flops_complete_operator_coverage=not bool(unsupported),unsupported_ops=unsupported,
                flops_convention="MMEngine: one fused multiply-add = one FLOP; counted supported graph operators only",
                timing_scope="GPU model.predict including fusion; excludes disk read, resize, CPU input preprocessing and file output",
                warmup_images=len(batches),timed_images=30,latencies_ms=durations,flop_traces=traces,
                hardware=protocol["hardware"],state_sha256_before=before,state_sha256_after=after)
    atomic_json(destination,authenticated_cache_row(result))
    print(f"RESOURCE_COMPLETED model={model_name} latency_ms={result['latency_ms_mean']:.3f} parameters={result['parameters']} unsupported_ops={len(unsupported)}",flush=True)
    del model,batches;gc.collect();torch.cuda.empty_cache()
