"""Audited v3 -> v4 reader recovery. Retain v3 protocol and measured resources.

This entry never invokes resource_measurement, model inference or training.
Original protocol package hashes continue to refer to the unchanged v3 package;
the separately hashed recovery audit records the new parser and new code.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import unittest
from collections import Counter
from datetime import datetime
from pathlib import Path

from supplement_common import (RESOURCE_MODELS, atomic_json, finite, now,
    read_csv, read_json, sha256, stream_bundle, verify_cache_row, validate_bundle_header)
from final_statistics import _rle, verify_sources
from run_supplement import resource_table, state
from launch_supplement import PROJECT, verify_completion

OLD_REVISION='exp08b_resources_exp15_statistics_v3_legacy_provenance'
RECOVERY_REVISION='exp08b15_v4_lossless_coco_rle_statistics_only'


def verify_package(package,expected=None):
    package=Path(package)
    hashes=read_json(package/'PACKAGE_SHA256.json')
    for name,digest in hashes.items():
        if Path(name).name!=name or sha256(package/name)!=digest:
            raise ValueError(f'package identity mismatch: {package}/{name}')
    if expected is not None:
        for name,digest in expected.items():
            if not name.endswith('.py') or hashes.get(name)!=digest:
                raise ValueError(f'original protocol/v3 package mismatch: {name}')
    return hashes


def verify_resources(protocol,out):
    """Read-only validation; never rewrite caches, resource CSV or audit."""
    out=Path(out);protected={'protocol.json':sha256(out/'protocol.json')};results=[]
    runs={r['variant']:r for r in protocol['resource_runs']}
    if set(runs)!=set(RESOURCE_MODELS) or len(protocol['resource_runs'])!=5:
        raise ValueError('resource inventory must contain exactly five fixed models')
    for model in RESOURCE_MODELS:
        item=runs[model];path=out/'resources'/f'{model}.json'
        r=verify_cache_row(read_json(path))
        expected=dict(checkpoint_sha256=item['checkpoint_sha256'],configuration_sha256=item['configuration_sha256'],
                      protocol_sha256=protected['protocol.json'])
        if (r.get('identity')!=expected or r.get('model')!=model or r.get('seed')!=42 or item.get('seed')!=42
            or r.get('hardware')!=protocol['hardware'] or r.get('weights_unchanged') is not True
            or not r.get('state_sha256_before') or r.get('state_sha256_before')!=r.get('state_sha256_after')
            or r.get('batch_size')!=1 or r.get('input_size')!=640 or r.get('precision')!='FP32' or r.get('tf32') is not False):
            raise ValueError(f'resource cache binding invalid: {model}')
        for name in ('parameters','counted_flops_mmengine_mean','counted_flops_mmengine_max','peak_memory_allocated_bytes',
                     'peak_memory_reserved_bytes','latency_ms_mean','latency_ms_median','latency_ms_p95','throughput_images_per_second'):
            if finite(r[name])<=0:raise ValueError(f'invalid resource measurement: {model}/{name}')
        results.append(r);protected[str(path.relative_to(out))]=sha256(path)
    table_path=out/'EXP08B_resources_640_FP32.csv';audit_path=out/'EXP08B_resources_audit.json'
    expected_rows=[{k:str(v) for k,v in r.items()} for r in resource_table(results)]
    if read_csv(table_path)!=expected_rows:raise ValueError('resource CSV differs from authenticated measurements')
    audit=read_json(audit_path)
    if (audit.get('measurements')!=results or audit.get('status')!='completed_fixed_input_resources_supplement'
        or audit.get('training') is not False or audit.get('optimizer_updates')!=0 or audit.get('D4_images_accessed') is not False):
        raise ValueError('resource audit differs from authenticated measurements')
    protected[table_path.name]=sha256(table_path);protected[audit_path.name]=sha256(audit_path)
    return protected


def verify_protected(out,protected):
    for name,digest in protected.items():
        if sha256(Path(out)/name)!=digest:raise ValueError(f'preserved resource/protocol changed: {name}')


def scan_rles(protocol):
    reports=[];classes=set(protocol['class_names']);groups=protocol['d3_groups']
    for run in protocol['statistical_runs']:
        seen=set();encodings=Counter();header={}
        for key,image in stream_bundle(run['predictions']):
            if key!='image':header[key]=image;continue
            rid=str(image['image_id'])
            if rid in seen or rid not in groups:raise ValueError(f'D3 duplicate/unknown image: {rid}')
            if str(image.get('group_id',''))!=groups[rid]:raise ValueError(f'D3 missing/mismatched group: {rid}')
            if any(not isinstance(image.get(k),int) or isinstance(image[k],bool) or image[k]<=0 for k in ('height','width')):
                raise ValueError(f'invalid D3 image dimensions: {rid}')
            seen.add(rid)
            for section in ('predictions','targets'):
                for index,instance in enumerate(image[section]):
                    try:
                        parsed=_rle(instance,image['height'],image['width'])
                        if parsed['name'] not in classes:raise ValueError('unknown class')
                        if section=='targets' and not parsed['area']:raise ValueError('empty GT instance')
                        encodings['compressed' if isinstance(instance['mask_rle']['counts'],str) else 'uncompressed']+=1
                    except Exception as error:
                        raise ValueError(f"RLE preflight {run['variant']}/{run['seed']} {rid}/{section}/{index}: {error}") from error
            if len(seen)%100==0:print(f"RLE_PREFLIGHT model={run['variant']} seed={run['seed']} images={len(seen)}/{len(groups)}",flush=True)
        if seen!=set(groups):raise ValueError(f"incomplete D3 coverage: {run['variant']}/{run['seed']}")
        if 'checkpoint_sha256' in run:
            validate_bundle_header(header,run['variant'],run['seed'],run['checkpoint_sha256'],legacy_provenance=run.get('prediction_identity'))
            if set(header.get('class_names',[]))!=classes:raise ValueError('D3 class inventory mismatch')
        report=dict(model=run['variant'],seed=run['seed'],images=len(seen),encodings=dict(encodings));reports.append(report)
        print('RLE_BUNDLE_VALIDATED '+str(report),flush=True)
    return reports


def preflight(project,out,old_package):
    protocol=read_json(out/'protocol.json')
    if protocol.get('revision')!=OLD_REVISION:raise ValueError('this recovery requires the original v3 protocol')
    if Path(protocol['project_root']).resolve()!=project or Path(protocol['output_dir']).resolve()!=out:
        raise ValueError('recovery/protocol scope mismatch')
    if protocol.get('training') is not False or protocol.get('optimizer_updates')!=0 or protocol.get('D4_image_or_prediction_access') is not False:
        raise ValueError('original protocol is not the read-only supplement')
    if sorted((r['variant'],int(r['seed'])) for r in protocol['statistical_runs'])!=[(m,s) for m in ('B0','B9') for s in range(42,47)]:
        raise ValueError('statistics require exactly ten B0/B9 D3 bundles')
    old_hashes=verify_package(old_package,protocol['package_hashes'])
    new_hashes=verify_package(Path(__file__).parent)
    print('RECOVERY_VERIFY_SOURCES start (hashing frozen checkpoints/predictions; no training)',flush=True)
    verify_sources(protocol);protected=verify_resources(protocol,out)
    print('RESOURCE_PRESERVATION=VERIFIED models=5 (no remeasurement)',flush=True)
    report=scan_rles(protocol)
    print('RECOVERY_VERIFY_SOURCES final preflight consistency check',flush=True)
    verify_sources(protocol);verify_protected(out,protected)
    return protocol,dict(revision=RECOVERY_REVISION,original_protocol_sha256=protected['protocol.json'],
        preserved_files=protected,old_package=str(old_package),old_package_hashes=old_hashes,new_package_hashes=new_hashes,
        validated_RLE=report,training=False,resource_models_remeasured=False,D4_images_or_predictions_accessed=False,
        scope='Parser-only recovery, original protocol and resource measurements are retained byte-for-byte')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('run','preflight','worker'),default='run')
    parser.add_argument('--project-root',default=PROJECT)
    parser.add_argument('--old-package',default=str(Path(__file__).parent.parent/'exp08b15_supplement_v3_20261010'))
    parser.add_argument('--lock-fd',type=int)
    args=parser.parse_args()
    if sys.platform!='linux':raise RuntimeError('Run this recovery on AutoDL Linux')
    project=Path(args.project_root).resolve();out=project/'outputs/supplements/EXP08B_EXP15_20261010_v1'
    old_package=Path(args.old_package).resolve();logs=project/'outputs/autodl_logs'
    import fcntl
    lock_path=logs/'EXP08B_EXP15.lock'
    if args.mode=='worker':
        if args.lock_fd is None:raise RuntimeError('worker requires inherited exclusive lock')
        actual=os.fstat(args.lock_fd);expected=os.stat(lock_path)
        if (actual.st_dev,actual.st_ino)!=(expected.st_dev,expected.st_ino):raise RuntimeError('unrelated recovery lock')
        fcntl.flock(args.lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            audit=read_json(out/'EXP15_RLE_recovery_audit.json');protocol=read_json(out/'protocol.json')
            if audit.get('revision')!=RECOVERY_REVISION:raise ValueError('recovery revision mismatch')
            if verify_package(Path(__file__).parent)!=audit['new_package_hashes']:raise ValueError('recovery code changed')
            if verify_package(old_package,protocol['package_hashes'])!=audit['old_package_hashes']:raise ValueError('original code changed')
            if verify_resources(protocol,out)!=audit['preserved_files']:raise ValueError('resources changed since preflight')
            verify_sources(protocol);state(out,'RUNNING',stage='EXP15_statistics_recovery_resources_retained')
            from final_statistics import analyze
            analyze(project,protocol,out)
            verify_protected(out,audit['preserved_files']);verify_sources(protocol)
            products=[p for p in out.iterdir() if p.is_file() and p.name.startswith(('EXP08B_','EXP15_'))]
            atomic_json(out/'run_manifest.json',dict(status='completed_resources_and_statistics_supplement',finished_at_utc=now(),
                output_dir=str(out),outputs={p.name:sha256(p) for p in products},protocol_sha256=sha256(out/'protocol.json'),
                recovery_revision=RECOVERY_REVISION,resources_retained_byte_for_byte=True,original_results_unchanged=True,
                training=False,optimizer_updates=0,D4_images_or_predictions_accessed=False,original_full_EXP08_or_EXP15_SOP_reconstructed=False))
            state(out,'COMPLETED',stage='EXP08B_EXP15_supplement')
        except Exception as error:state(out,'FAILED',reason=str(error));raise
        return
    lock=open(lock_path,'a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('supplement already running; no process will be stopped')
    verify_package(Path(__file__).parent)
    os.environ['SUPPLEMENT_TEST_PROJECT']=str(project)
    # No device allocation and no CUDA runtime needed for statistics-only recovery.
    import torch,mmengine,mmdet,pycocotools.mask
    suite=unittest.defaultTestLoader.discover(str(Path(__file__).parent),pattern='test*.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:raise RuntimeError('recovery tests failed or skipped')
    protocol,audit=preflight(project,out,old_package)
    if verify_completion(out):print('SUPPLEMENT_ALREADY_COMPLETED');return
    print('STATISTICS_ONLY_PREFLIGHT=PASSED resources_retained=5',flush=True)
    if args.mode=='preflight':return
    atomic_json(out/'EXP15_RLE_recovery_audit.json',audit)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S');logfile=logs/f'EXP08B_EXP15_v4_statistics_{stamp}.log'
    env=os.environ.copy();env.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        PYTHONPATH=str(project/'src')+os.pathsep+env.get('PYTHONPATH',''))
    command=[sys.executable,'-B','-u',__file__,'--mode','worker','--project-root',str(project),'--old-package',str(old_package),'--lock-fd',str(lock.fileno())]
    with logfile.open('wb') as handle:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,env=env,
            start_new_session=True,close_fds=True,pass_fds=(lock.fileno(),))
    (logs/'EXP08B_EXP15.pid').write_text(str(child.pid)+'\n',encoding='utf-8')
    (logs/'EXP08B_EXP15.current_log').write_text(str(logfile)+'\n',encoding='utf-8')
    print(f'STATISTICS_RECOVERY_STARTED pid={child.pid}\nlog={logfile}\nresource_remeasurement=False',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        import traceback
        traceback.print_exc();print(f'RECOVERY_REFUSED: {error}',flush=True);sys.exit(2)
