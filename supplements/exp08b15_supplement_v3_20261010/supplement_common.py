"""Strict input contracts and seed/group statistics; never trains a model."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REVISION = "exp08b_resources_exp15_statistics_v3_legacy_provenance"
METRICS = ("mask_ap", "mask_ap50", "crack_cldice", "crack_break_rate",
           "crack_component_error", "area_boundary_f1", "area_hd95", "area_relative_error")
STRUCTURE = METRICS[2:]
LOWER_BETTER = {"crack_break_rate", "crack_component_error", "area_hd95", "area_relative_error"}
PRIMARY_SEEDS = (42, 43, 44, 45, 46)
RESOURCE_MODELS = ("B0", "B4", "B5", "B11", "B9")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def payload_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")).hexdigest()


def authenticated_cache_row(row):
    value=dict(row);value.pop("cache_sha256",None)
    return dict(value,cache_sha256=payload_hash(value))


def verify_cache_row(row):
    if authenticated_cache_row(row)["cache_sha256"]!=row.get("cache_sha256"):
        raise ValueError("structure partial record checksum mismatch")
    return row


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=_no_duplicate_fields,
                      parse_constant=_reject_constant)


def checked_source(path, output_root):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(output_root).resolve()) or not path.is_file():
        raise ValueError(f"source missing/outside output root: {path}")
    return path


def finite(value):
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"nonfinite result: {value}")
    return x


def unique_runs(rows):
    runs = {}
    for original in rows:
        row = dict(original)
        model, seed = str(row["variant"]), int(row["seed"])
        if not model.startswith("B") or not model[1:].isdigit():
            raise ValueError(f"invalid model identity: {model}")
        key = (model, seed)
        if key in runs:
            raise ValueError(f"duplicate model/seed: {key}")
        for metric in METRICS:
            value = row.get(metric)
            row[metric] = None if value in (None, "") else finite(value)
        row.update(variant=model, seed=seed)
        runs[key] = row
    if not runs:
        raise ValueError("empty run table")
    return runs


def seed_summaries(rows, metrics=METRICS):
    runs = unique_runs(rows)
    result = []
    for model in sorted({k[0] for k in runs}):
        selected = [r for (m, _), r in runs.items() if m == model]
        for metric in metrics:
            good = [r for r in selected if r.get(metric) is not None]
            values = [r[metric] for r in good]
            result.append(dict(model=model, metric=metric, n_seeds=len(values),
                               seeds=";".join(str(r["seed"]) for r in sorted(good, key=lambda r: r["seed"])),
                               mean=float(np.mean(values)) if values else None,
                               std=float(np.std(values, ddof=1)) if len(values)>1 else None,
                               missing_metric_seeds=";".join(str(r["seed"]) for r in selected if r.get(metric) is None),
                               scope="D3 whole-dataset result per seed; not mean image AP"))
    return result


def select_resources(rows):
    runs=unique_runs(rows)
    if any((m,42) not in runs for m in RESOURCE_MODELS):
        raise ValueError("missing seed42 resource checkpoint row")
    return [runs[m,42] for m in RESOURCE_MODELS]


def paired_seeds(rows, left, right, metric, expected_seeds):
    runs = unique_runs(rows)
    for model in (left, right):
        if {s for m,s in runs if m==model} != set(expected_seeds):
            raise ValueError(f"unexpected/missing seed set for {model}; expected {expected_seeds}")
    a = [runs[left, s].get(metric) for s in expected_seeds]
    b = [runs[right, s].get(metric) for s in expected_seeds]
    if any(v is None for v in a+b):
        return dict(left=left, right=right, metric=metric, n_pairs=0, status="undefined_missing_metric",
                    difference=None, exact_signflip_p=None)
    difference = np.array(a, float)-np.array(b, float)
    if len(difference)>16:
        raise ValueError("exact signflip limited to 16 seed pairs")
    point = float(difference.mean())
    statistics = [abs(float(np.mean(difference*np.array(sign))))
                  for sign in itertools.product((-1,1), repeat=len(difference))]
    p = sum(x >= abs(point)-1e-12 for x in statistics)/len(statistics)
    return dict(left=left, right=right, metric=metric, seeds=list(expected_seeds), n_pairs=len(difference),
                difference=point, improvement=(-point if metric in LOWER_BETTER else point),
                exact_signflip_p=p, status="completed", inference_unit="paired training seed; fixed D3 dataset",
                interpretation="Seed-level exact two-sided sign-flip under exchangeability; few seeds limit resolution")


def holm(p_values):
    p = np.asarray(p_values, float)
    if not np.isfinite(p).all() or np.any(p<0) or np.any(p>1):
        raise ValueError("invalid p-values")
    order = np.argsort(p)
    result = np.zeros(len(p)); running=0.
    for rank, i in enumerate(order):
        running=max(running, min(1., float(p[i])*(len(p)-rank)))
        result[i]=running
    return result.tolist()


def group_comparison(rows, metric, seeds, bootstrap, rng_seed):
    """Resample manifest groups together across every paired seed and model.

    Sums/counts preserve the original mean over IoU-matched instances. Missing
    matches are zero denominators, never artificial zero metric observations.
    """
    indexed={}
    for original in rows:
        if original["metric"]!=metric:
            continue
        row=dict(original)
        row["seed"]=int(row["seed"])
        key=(row["model"],row["seed"],row["image_id"])
        if key in indexed:
            raise ValueError(f"duplicate image/seed/metric: {key}")
        row["total"]=finite(row["total"]); row["count"]=int(row["count"])
        if row["count"]<0 or (row["count"]==0 and row["total"]!=0):
            raise ValueError("invalid sufficient statistics")
        indexed[key]=row
    ids = {m:{(s,i) for mm,s,i in indexed if mm==m} for m in ("B9","B0")}
    if not ids["B9"] or ids["B9"] != ids["B0"]:
        raise ValueError("unmatched image/seed pairs")
    if {s for s,_ in ids["B9"]} != set(seeds):
        raise ValueError("group statistics lack expected seed pairs")
    per_seed={s:{i for ss,i in ids["B9"] if ss==s} for s in seeds}
    if any(v!=per_seed[seeds[0]] for v in per_seed.values()):
        raise ValueError("image coverage differs across paired seeds")
    groups=set()
    for s,i in ids["B9"]:
        a,b=indexed["B9",s,i],indexed["B0",s,i]
        if not a["group_id"] or a["group_id"]!=b["group_id"]:
            raise ValueError("pair group mismatch")
        groups.add(a["group_id"])
    groups=sorted(groups); positions={g:i for i,g in enumerate(groups)}
    totals=np.zeros((2,len(seeds),len(groups))); counts=np.zeros_like(totals)
    for mi,m in enumerate(("B9","B0")):
        for si,s in enumerate(seeds):
            for image in per_seed[s]:
                r=indexed[m,s,image]; gi=positions[r["group_id"]]
                totals[mi,si,gi]+=r["total"];counts[mi,si,gi]+=r["count"]
    base=dict(metric=metric,left="B9",right="B0",n_groups=len(groups),n_seeds=len(seeds),
              n_image_seed_pairs=len(ids["B9"]),matched_instance_counts=counts.sum(axis=2).tolist(),
              estimand="Equal-seed mean difference of means over IoU>=0.5 matched instances; NOT unconditional recall",
              inference_unit="manifest group, jointly resampled for both models and all seeds",
              difference=None,ci_low=None,ci_high=None)
    if np.any(counts.sum(axis=2)==0):
        return dict(base,status="undefined_no_matches")
    ratios=totals.sum(axis=2)/counts.sum(axis=2)
    point=float((ratios[0]-ratios[1]).mean())
    base.update(difference=point,improvement=-point if metric in LOWER_BETTER else point)
    if len(groups)<2:
        return dict(base,status="insufficient_clusters",bootstrap_valid=0)
    if bootstrap<100:
        raise ValueError("at least 100 bootstrap replicates required")
    rng=np.random.default_rng(rng_seed); distribution=[]; rejected=0
    for _ in range(bootstrap):
        sample=rng.integers(0,len(groups),len(groups))
        denom=counts[:,:,sample].sum(axis=2)
        if np.any(denom==0):
            rejected+=1;continue
        means=totals[:,:,sample].sum(axis=2)/denom
        distribution.append(float((means[0]-means[1]).mean()))
    if len(distribution)<.95*bootstrap:
        return dict(base,status="insufficient_valid_bootstrap",bootstrap_valid=len(distribution),bootstrap_rejected=rejected)
    low,high=np.percentile(distribution,[2.5,97.5])
    return dict(base,status="completed_exploratory",ci_low=float(low),ci_high=float(high),
                bootstrap_valid=len(distribution),bootstrap_rejected=rejected,
                multiplicity="Exploratory pointwise percentile CI; no significance gate or causal claim")


def _no_duplicate_fields(pairs):
    result={}
    for k,v in pairs:
        if k in result:
            raise ValueError(f"duplicate JSON field: {k}")
        result[k]=v
    return result


def _reject_constant(value):
    raise ValueError(f"nonstandard/nonfinite JSON value: {value}")


class _JSONStream:
    def __init__(self, stream, chunk_size):
        self.stream=stream;self.chunk=chunk_size;self.buffer="";self.pos=0;self.eof=False
        self.decoder=json.JSONDecoder(object_pairs_hook=_no_duplicate_fields,parse_constant=_reject_constant)

    def fill(self):
        self.buffer=self.buffer[self.pos:];self.pos=0
        text=self.stream.read(self.chunk)
        if not text:self.eof=True
        self.buffer+=text

    def peek(self):
        while True:
            while self.pos<len(self.buffer) and self.buffer[self.pos].isspace():self.pos+=1
            if self.pos<len(self.buffer):return self.buffer[self.pos]
            if self.eof:return ""
            self.fill()

    def expect(self, value):
        if self.peek()!=value:raise ValueError(f"expected JSON {value!r}, got {self.peek()!r}")
        self.pos+=1

    def value(self):
        self.peek()
        while True:
            try:
                value,end=self.decoder.raw_decode(self.buffer,self.pos)
                # A bare number may continue in the next chunk. Strings/objects
                # are delimited; hold all values ending at the current boundary.
                if end==len(self.buffer) and not self.eof:
                    self.fill();continue
                self.pos=end;return value
            except json.JSONDecodeError as e:
                if self.eof:raise ValueError("truncated/malformed JSON") from e
                self.fill()


def stream_bundle(path, chunk_size=65536):
    """Yield top-level fields and one image at a time, independent of key order."""
    with Path(path).open(encoding="utf-8-sig") as stream:
        r=_JSONStream(stream,chunk_size);r.expect("{");seen=set();first=True
        while r.peek()!="}":
            if not first:r.expect(",")
            first=False
            key=r.value()
            if not isinstance(key,str) or key in seen:raise ValueError("duplicate/invalid bundle key")
            seen.add(key);r.expect(":")
            if key=="images":
                r.expect("[");image_first=True
                while r.peek()!="]":
                    if not image_first:r.expect(",")
                    image_first=False
                    image=r.value()
                    if not isinstance(image,dict):raise ValueError("image must be object")
                    yield "image",image
                r.expect("]")
            else:yield key,r.value()
        r.expect("}")
        if r.peek():raise ValueError("trailing JSON data")
        if "images" not in seen:raise ValueError("bundle lacks images")


def validate_bundle_header(header, model, seed, checkpoint_hash, *, legacy_provenance=None):
    if header.get("split")!="D3" or header.get("experiment_id")!="EXP-09":
        raise ValueError("only completed EXP-09 D3 prediction bundles allowed")
    if header.get("checkpoint_sha256")!=checkpoint_hash:
        proof=legacy_provenance or {}
        if (header.get("checkpoint_sha256") is not None
            or proof.get("mode")!="legacy_completed_cache_semantic_binding"
            or proof.get("model")!=model or proof.get("seed")!=seed
            or proof.get("current_checkpoint_sha256")!=checkpoint_hash
            or proof.get("exported_header_sha256")!=payload_hash(header)
            or proof.get("historical_checkpoint_bytes_verified") is not False
            or proof.get("matching_images",0)<=0
            or not proof.get("source_files")):
            raise ValueError("bundle checkpoint hash mismatch (or missing legacy provenance)")
    meta=header.get("metadata",{})
    if meta.get("smoke_test") or meta.get("synthetic"):
        raise ValueError("synthetic predictions are not final evidence")
    if meta.get("variant_id") not in (None,model) or ("seed" in meta and int(meta["seed"])!=seed):
        raise ValueError("bundle model/seed mismatch")
    if len(header.get("class_names",[]))!=5:
        raise ValueError("bundle lacks five-class mapping")
