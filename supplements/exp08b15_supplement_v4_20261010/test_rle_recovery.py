import tempfile
import unittest
import os
import sys
from pathlib import Path

import numpy as np

from final_statistics import _rle, _rle_iou
import supplement_common as common


def encoded(counts):
    # Independent fixture encoder following the official COCO wire format.
    chars=[]
    for i,value in enumerate(counts):
        x=value-(counts[i-2] if i>2 else 0)
        while True:
            c=x & 31;x >>= 5
            more=x!=-1 if c & 16 else x!=0
            chars.append(chr(c+48+(32 if more else 0)))
            if not more:break
    return ''.join(chars)


def instance(counts,size=(4,5)):
    return dict(class_name='crack',score=.8,mask_rle=dict(size=list(size),counts=counts))


class RLEContracts(unittest.TestCase):
    def test_literal_compressed_fixtures(self):
        for counts,wire in [([20],'d0'),([0,20],'0d0'),([2,3,4,1,10],'234N6')]:
            a=_rle(instance(wire),4,5);b=_rle(instance(counts),4,5)
            np.testing.assert_array_equal(a['runs'],b['runs'])
            self.assertEqual(a['area'],b['area'])

    def test_signed_deltas_zero_runs_and_large_counts(self):
        for counts in ([0,2,0,1,17],[2,3,4,1,10],[1000,2000,3000,1000,2000]):
            size=(1,sum(counts));a=_rle(instance(encoded(counts),size),*size)
            b=_rle(instance(list(counts),size),*size)
            np.testing.assert_array_equal(a['runs'],b['runs'])
            self.assertEqual(_rle_iou(a,b),1.)

    def test_invalid_counts_fail_closed_before_numpy(self):
        for counts in ('','P','/','p','O','d1','000000000000000000000P', [True,19],[1.0,19],[-1,21],[2**64]):
            with self.subTest(counts=counts),self.assertRaises(ValueError):
                _rle(instance(counts),4,5)

    def test_official_coco_randomized_oracle(self):
        try:from pycocotools import mask as coco
        except ImportError:self.skipTest('pycocotools required on AutoDL')
        rng=np.random.default_rng(531)
        for shape in ((1,1),(4,5),(24,31),(480,640)):
            for probability in (0.,.01,.5,1.):
                mask=np.asfortranarray((rng.random(shape)<probability).astype(np.uint8))
                rle=coco.encode(mask);wire=rle['counts'].decode('ascii')
                item=_rle(instance(wire,shape),*shape)
                actual=np.zeros(mask.size,np.uint8)
                for start,end in item['runs']:actual[start:end]=1
                np.testing.assert_array_equal(actual.reshape(shape,order='F'),mask)
                self.assertEqual(item['area'],int(coco.area(rle)))

    def test_mixed_compressed_masks_real_metrics_and_cache_resume(self):
        from final_statistics import image_statistics,cached_structure
        project=Path(os.environ.get('SUPPLEMENT_TEST_PROJECT','/root/autodl-tmp/railway_tunnel_experiments'))
        if not (project/'src').is_dir():self.skipTest('real project required on AutoDL')
        sys.path.insert(0,str(project/'src'))
        from morphoroute.data.annotations import encode_uncompressed_rle
        classes=['crack','water_leakage','spalling','efflorescence','exposed_rebar']
        predictions=[];targets=[]
        for name in ('crack','spalling'):
            a=np.zeros((24,31),bool);a[2:15,3:10]=True
            b=a.copy();b[4:6,9:11]=True
            pa=dict(class_name=name,score=.9,mask_rle=encode_uncompressed_rle(a))
            gb=dict(class_name=name,score=1.,mask_rle=encode_uncompressed_rle(b))
            predictions.append(pa);targets.append(gb)
        image=dict(image_id='x',group_id='g',height=24,width=31,predictions=predictions,targets=targets)
        expected=image_statistics(image,classes)
        for item in predictions:item['mask_rle']['counts']=encoded(item['mask_rle']['counts'])
        self.assertEqual(image_statistics(image,classes),expected)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            out=Path(temporary);bundle=out/'source/bundle.json'
            common.atomic_json(bundle,dict(experiment_id='EXP-09',split='D3',checkpoint_sha256='a'*64,
                class_names=classes,metadata=dict(variant_id=None,seed=42),images=[image]))
            protocol=dict(d3_groups={'x':'g'},class_names=classes)
            common.atomic_json(out/'protocol.json',protocol)
            item=dict(variant='B0',seed=42,predictions=str(bundle),prediction_sha256=common.sha256(bundle),checkpoint_sha256='a'*64,
                **{k:expected['totals'][k]/expected['counts'][k] for k in common.STRUCTURE})
            protected={str(p):common.sha256(p) for p in (bundle,out/'protocol.json')}
            first=cached_structure(protocol,item,out);second=cached_structure(protocol,item,out)
            self.assertEqual(first,second)
            self.assertEqual(protected,{name:common.sha256(name) for name in protected})


class RecoveryContracts(unittest.TestCase):
    def test_ten_compressed_bundles_complete_seed_and_group_statistics(self):
        from final_statistics import analyze,image_statistics,source_code_hash
        project=Path(os.environ.get('SUPPLEMENT_TEST_PROJECT','/root/autodl-tmp/railway_tunnel_experiments'))
        if not (project/'src').is_dir():self.skipTest('real project required on AutoDL')
        sys.path.insert(0,str(project/'src'))
        from morphoroute.data.annotations import encode_uncompressed_rle
        classes=['crack','water_leakage','spalling','efflorescence','exposed_rebar']
        mask=np.zeros((24,31),bool);mask[2:15,3:10]=True
        gt=[dict(class_name=name,score=1.,mask_rle=encode_uncompressed_rle(mask)) for name in ('crack','spalling')]
        pred=[dict(class_name=r['class_name'],score=.9,mask_rle=dict(size=[24,31],counts=encoded(r['mask_rle']['counts']))) for r in gt]
        images=[dict(image_id=rid,group_id=group,height=24,width=31,predictions=pred,targets=gt) for rid,group in (('x','g1'),('y','g2'))]
        totals=image_statistics(images[0],classes)
        metrics={k:totals['totals'][k]/totals['counts'][k] for k in common.STRUCTURE}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root=Path(temporary);out=root/'statistics';runs=[];sources={};rows=[]
            for model in ('B0','B9'):
                for seed in range(42,47):
                    path=root/'originals'/f'{model}_{seed}.json'
                    common.atomic_json(path,dict(experiment_id='EXP-09',split='D3',checkpoint_sha256='a'*64,class_names=classes,
                        metadata=dict(seed=seed,variant_id=None if model=='B0' else model),images=images))
                    sources[str(path)]=common.sha256(path)
                    run=dict(variant=model,seed=seed,mask_ap=.1,mask_ap50=.2,predictions=str(path),
                        prediction_sha256=common.sha256(path),checkpoint_sha256='a'*64,prediction_identity=dict(mode='test_recorded_hash'),**metrics)
                    runs.append(run);rows.append(run)
            rows.extend(dict(variant='B11',seed=seed,mask_ap=.1,mask_ap50=.2,**metrics) for seed in (42,43,44))
            protocol=dict(project_root=str(project),source_code_sha256=source_code_hash(project),source_files=sources,
                result_rows=rows,resource_runs=[],statistical_runs=runs,d3_groups={'x':'g1','y':'g2'},class_names=classes,
                bootstrap=100,random_seed=34,sealed_e_decision={},probe_directory='fixture')
            common.atomic_json(out/'protocol.json',protocol)
            frozen=dict(sources);frozen[str(out/'protocol.json')]=common.sha256(out/'protocol.json')
            analyze(project,protocol,out)
            checks=common.read_csv(out/'EXP15_structure_reconciliation.csv')
            self.assertEqual(len(checks),60)
            self.assertEqual(common.read_json(out/'EXP15_evidence_scope.json')['status'],'completed_statistics_supplement')
            self.assertEqual(len(common.read_json(out/'EXP15_group_bootstrap_D3.json')),6)
            analyze(project,protocol,out)  # genuine per-image resumability, same source bytes
            self.assertEqual(frozen,{name:common.sha256(name) for name in frozen})

    def test_old_package_and_protocol_hashes_must_both_match(self):
        from recover_statistics import verify_package
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            package=Path(temporary);source=package/'old.py';source.write_bytes(b'old version\n')
            hashes={'old.py':common.sha256(source)};common.atomic_json(package/'PACKAGE_SHA256.json',hashes)
            self.assertEqual(verify_package(package,hashes),hashes)
            with self.assertRaisesRegex(ValueError,'protocol'):verify_package(package,{'old.py':'f'*64})
            source.write_bytes(b'changed\n')
            with self.assertRaisesRegex(ValueError,'identity'):verify_package(package,hashes)

    def test_preserved_resources_bound_to_original_protocol(self):
        from recover_statistics import verify_resources
        from run_supplement import summarize_resources
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            out=Path(temporary);common.atomic_json(out/'protocol.json',{'frozen':True})
            hardware={'gpu':'fixture'};runs=[]
            for model in common.RESOURCE_MODELS:
                item=dict(variant=model,seed=42,checkpoint_sha256='a'*64,configuration_sha256='b'*64)
                runs.append(item)
                result=dict(model=model,seed=42,hardware=hardware,
                    identity=dict(checkpoint_sha256=item['checkpoint_sha256'],configuration_sha256=item['configuration_sha256'],protocol_sha256=common.sha256(out/'protocol.json')),
                    weights_unchanged=True,state_sha256_before='c'*64,state_sha256_after='c'*64,
                    batch_size=1,input_size=640,precision='FP32',tf32=False,
                    parameters=10,counted_flops_mmengine_mean=20,counted_flops_mmengine_max=30,
                    flops_complete_operator_coverage=True,unsupported_ops={},
                    peak_memory_allocated_bytes=100,peak_memory_reserved_bytes=120,
                    latency_ms_mean=2,latency_ms_median=2,latency_ms_p95=3,throughput_images_per_second=500,
                    timing_scope='model',flops_convention='one multiply add one FLOP')
                common.atomic_json(out/'resources'/f'{model}.json',common.authenticated_cache_row(result))
            protocol=dict(resource_runs=runs,hardware=hardware)
            summarize_resources(protocol,out)
            before={str(p):common.sha256(p) for p in out.rglob('*') if p.is_file()}
            protected=verify_resources(protocol,out)
            self.assertEqual(before,{str(p):common.sha256(p) for p in out.rglob('*') if p.is_file()})
            self.assertEqual(len(protected),8)
            path=out/'resources/B9.json';r=common.read_json(path);r['identity']['protocol_sha256']='d'*64
            common.atomic_json(path,common.authenticated_cache_row(r))
            with self.assertRaisesRegex(ValueError,'resource'):verify_resources(protocol,out)

    def test_preflight_checks_every_image_and_preserves_bundles(self):
        from recover_statistics import scan_rles
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            out=Path(temporary);path=out/'bundle.json'
            image=dict(image_id='x',group_id='g',height=4,width=5,predictions=[instance('234N6')],targets=[instance([2,3,4,1,10])])
            common.atomic_json(path,dict(images=[image,dict(image,image_id='y')]))
            protocol=dict(statistical_runs=[dict(variant='B0',seed=42,predictions=str(path))],d3_groups={'x':'g','y':'g'},class_names=['crack'])
            before=common.sha256(path);report=scan_rles(protocol)
            self.assertEqual(common.sha256(path),before)
            self.assertEqual(report[0]['images'],2)
            bad=dict(image,image_id='y',predictions=[instance('P')])
            common.atomic_json(path,dict(images=[image,bad]))
            with self.assertRaisesRegex(ValueError,'y/predictions/0'):scan_rles(protocol)

    def test_preflight_rejects_missing_group_and_invalid_empty_image_dimensions(self):
        from recover_statistics import scan_rles
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            path=Path(temporary)/'bundle.json'
            protocol=dict(statistical_runs=[dict(variant='B0',seed=42,predictions=str(path))],d3_groups={'x':'g'},class_names=['crack'])
            for image in (dict(image_id='x',height=4,width=5,predictions=[],targets=[]),
                          dict(image_id='x',group_id='g',height=0,width=5,predictions=[],targets=[])):
                common.atomic_json(path,dict(images=[image]))
                with self.assertRaises(ValueError):scan_rles(protocol)


if __name__=='__main__':unittest.main(verbosity=2)
