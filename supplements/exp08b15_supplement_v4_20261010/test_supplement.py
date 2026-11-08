import importlib
import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import supplement_common as common
except ModuleNotFoundError:
    common = None


def legacy_fixture(output):
    classes=["crack","water_leakage","spalling","efflorescence","exposed_rebar"]
    run=output/"experiments/EXP-09/local_6gb/seed_42/runs/B0/seed_42"
    run.mkdir(parents=True);checkpoint=run/"best.pth";checkpoint.write_bytes(b"weights")
    config=run/"resolved_mmdet_config.py";config.write_bytes(b"configuration")
    cache=output/"training_cache/mmdet"/('a'*64);cache.mkdir(parents=True)
    image=dict(image_id="x",group_id="g",height=2,width=2,predictions=[],targets=[])
    bundle=dict(experiment_id="EXP-09",split="D3",checkpoint_sha256=None,class_names=classes,
                framework="mmdetection",metadata={"seed":42,"variant_id":None},images=[image])
    common.atomic_json(cache/"predictions_D3.json",bundle)
    exported=dict(bundle,metadata=dict(bundle["metadata"],training_cache_signature=cache.name,
        training_cache_source=str(run),reused_training_cache=True))
    common.atomic_json(run/"predictions_D3.json",exported)
    record=dict(status="completed",signature=cache.name,source_work_dir=str(run),
        best_checkpoint=str(checkpoint),resolved_config_path=str(config),
        evaluation=dict(experiment_id="EXP-09",split="D3",**{k:.1 for k in common.METRICS}))
    common.atomic_json(cache/"completed_run.json",record)
    row=dict(variant="B0",seed=42,best_checkpoint=str(checkpoint),resolved_config=str(config),
        predictions=str(run/"predictions_D3.json"),checkpoint_sha256=common.sha256(checkpoint),
        prediction_sha256=common.sha256(run/"predictions_D3.json"),**{k:.1 for k in common.METRICS})
    return row,classes,cache,exported,record


class Contracts(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(common, "supplement implementation is not present yet")

    def test_duplicate_model_seed_is_rejected(self):
        row = dict(variant="B9", seed="42", mask_ap="0.2")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            common.unique_runs([row, row])

    def test_nonfinite_metric_is_not_a_valid_result(self):
        with self.assertRaises(ValueError):
            common.unique_runs([dict(variant="B9", seed=42, mask_ap="nan")])

    def test_seed_summary_uses_one_dataset_ap_per_seed(self):
        rows = [dict(variant="B9", seed=s, mask_ap=v) for s, v in [(42, .1), (43, .3)]]
        summary = common.seed_summaries(rows, ["mask_ap"])
        self.assertAlmostEqual(summary[0]["mean"], .2)
        self.assertEqual(summary[0]["n_seeds"], 2)
        self.assertAlmostEqual(summary[0]["std"], .1414213562373095)

    def test_seed_pairing_does_not_cartesian_product(self):
        rows = []
        for s, a, b in [(42, .4, .1), (43, .2, .1)]:
            rows.extend([dict(variant="B9", seed=s, mask_ap=a), dict(variant="B0", seed=s, mask_ap=b)])
        result = common.paired_seeds(rows, "B9", "B0", "mask_ap", [42, 43])
        self.assertEqual(result["n_pairs"], 2)
        self.assertAlmostEqual(result["difference"], .2)
        self.assertEqual(result["exact_signflip_p"], .5)

    def test_missing_seed_pair_fails_closed(self):
        rows = [dict(variant="B9", seed=42, mask_ap=.1), dict(variant="B0", seed=43, mask_ap=.2)]
        with self.assertRaisesRegex(ValueError, "seed"):
            common.paired_seeds(rows, "B9", "B0", "mask_ap", [42, 43])

    def test_five_consistent_seed_effects_do_not_get_zero_p(self):
        rows = []
        for s in range(42, 47):
            rows.extend([dict(variant="B9", seed=s, mask_ap=.2), dict(variant="B0", seed=s, mask_ap=.1)])
        result = common.paired_seeds(rows, "B9", "B0", "mask_ap", list(range(42, 47)))
        self.assertEqual(result["exact_signflip_p"], .0625)

    def test_holm_correction_is_monotone_and_capped(self):
        np.testing.assert_allclose(common.holm([.04, .01, .2]), [.08, .03, .2])

    def test_group_pairing_preserves_seed_and_image(self):
        rows = []
        for seed in [42, 43]:
            for model in ["B9", "B0"]:
                for image in ["x", "y"]:
                    rows.append(dict(model=model, seed=seed, image_id=image, group_id="g", metric="m", total=2 if model=="B9" else 1, count=1))
        result = common.group_comparison(rows, "m", [42, 43], 100, 1)
        self.assertEqual(result["n_image_seed_pairs"], 4)
        self.assertAlmostEqual(result["difference"], 1.)

    def test_duplicate_image_seed_metric_is_rejected(self):
        row = dict(model="B9", seed=42, image_id="x", group_id="g", metric="m", total=1, count=1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            common.group_comparison([row, row], "m", [42], 10, 1)

    def test_group_comparison_rejects_unmatched_images(self):
        row = dict(model="B9", seed=42, image_id="x", group_id="g", metric="m", total=1, count=1)
        with self.assertRaisesRegex(ValueError, "pair"):
            common.group_comparison([row], "m", [42], 10, 1)

    def test_zero_matches_remain_undefined_not_zero_scores(self):
        rows = [dict(model=m, seed=42, image_id="x", group_id="g", metric="m", total=0, count=0) for m in ["B9", "B0"]]
        result = common.group_comparison(rows, "m", [42], 10, 1)
        self.assertEqual(result["status"], "undefined_no_matches")
        self.assertIsNone(result["difference"])

    def test_cluster_bootstrap_resamples_same_groups_for_all_seeds(self):
        rows=[]
        for seed in [42, 43]:
            for group, a, b in [("g1", 4, 1), ("g2", 0, 1)]:
                for model, value in [("B9", a), ("B0", b)]:
                    rows.append(dict(model=model, seed=seed, image_id=group, group_id=group, metric="m", total=value, count=1))
        result=common.group_comparison(rows, "m", [42,43], 200, 0)
        self.assertEqual(result["n_groups"], 2)
        self.assertAlmostEqual(result["difference"], 1.)
        self.assertAlmostEqual(result["ci_low"], -1.)
        self.assertAlmostEqual(result["ci_high"], 3.)

    def test_streaming_bundle_handles_images_before_metadata(self):
        payload=dict(images=[{"image_id":"a","text":"中文"},{"image_id":"b"}], metadata={"seed":42}, split="D3")
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            path=Path(td)/"bundle.json"
            path.write_text(json.dumps(payload, ensure_ascii=False),encoding="utf-8")
            events=list(common.stream_bundle(path, chunk_size=7))
        self.assertEqual([v["image_id"] for k,v in events if k=="image"], ["a","b"])
        self.assertEqual(dict((k,v) for k,v in events if k!="image")["metadata"]["seed"],42)

    def test_streaming_rejects_duplicate_fields_and_trailing_garbage(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            path=Path(td)/"bundle.json"
            for bad in ['{"images":[],"split":"D3","split":"D4"}', '{"images":[]}bad', '{"images":[{"a":1,"a":2}]}']:
                path.write_text(bad,encoding="utf-8")
                with self.assertRaises(ValueError):
                    list(common.stream_bundle(path,chunk_size=3))

    def test_data4_bundle_is_refused(self):
        with self.assertRaisesRegex(ValueError,"D3"):
            common.validate_bundle_header(dict(split="D4",experiment_id="EXP-09",metadata={}), "B9",42,"abc")

    def test_bundle_checkpoint_mismatch_is_refused(self):
        with self.assertRaisesRegex(ValueError,"checkpoint"):
            common.validate_bundle_header(dict(split="D3",experiment_id="EXP-09",checkpoint_sha256="wrong",metadata={"variant_id":"B9","seed":42}),"B9",42,"abc")

    def test_path_outside_output_root_is_refused(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            root=Path(td)/"out";root.mkdir()
            outside=Path(td)/"data.pth";outside.write_bytes(b"x")
            with self.assertRaises(ValueError):
                common.checked_source(outside, root)

    def test_resource_selection_uses_exact_frozen_seed42_rows(self):
        rows=[dict(variant=m,seed=s,mask_ap=.1,best_checkpoint=f"{m}-{s}")
              for m in common.RESOURCE_MODELS for s in [42,43]]
        chosen=common.select_resources(rows)
        self.assertEqual([r["best_checkpoint"] for r in chosen], [f"{m}-42" for m in common.RESOURCE_MODELS])

    def test_crop_preserves_the_complete_union_and_blank_margin(self):
        import resource_measurement as resource
        a=np.zeros((20,30),bool);b=a.copy()
        a[5:8,9:11]=1;b[6:10,10:14]=1
        ca,cb=resource.crop_pair(a,b)
        self.assertEqual(ca.sum(),6);self.assertEqual(cb.sum(),16)
        self.assertEqual(ca.shape,(9,9))
        self.assertFalse(ca[:2].any());self.assertFalse(cb[-2:].any())

    def test_checkpoint_model_integrity_detects_tensor_mutation(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch is unavailable locally; must run this test on AutoDL")
        import resource_measurement as resource
        model=torch.nn.Linear(2,2)
        digest=resource.model_digest(model)
        with torch.no_grad():
            model.weight[0,0]+=1
        self.assertNotEqual(resource.model_digest(model),digest)

    def test_joint_crop_structure_metrics_equal_full_canvas(self):
        import resource_measurement as resource
        import os,sys
        project=Path(os.environ.get("SUPPLEMENT_TEST_PROJECT", "/root/autodl-tmp/railway_tunnel_experiments"))
        if not (project/"src").is_dir():
            self.skipTest("real project structure metric implementation unavailable")
        sys.path.insert(0,str(project/"src"))
        from morphoroute.metrics import structure
        a=np.zeros((32,48),bool);b=a.copy()
        a[0:15,7:11]=1;b[1:14,8:12]=1
        ca,cb=resource.crop_pair(a,b)
        for name in ["cldice","boundary_f1","hd95","component_count_error","crack_break_rate","area_relative_error"]:
            with self.subTest(metric=name):
                self.assertAlmostEqual(getattr(structure,name)(a,b),getattr(structure,name)(ca,cb),places=7)

    def test_d3_mapping_uses_true_manifest_groups_and_rejects_duplicates(self):
        from final_statistics import d3_mapping
        row=dict(record_id="x",logical_split="test_internal",assignment_group_id="real",group_key="wrong")
        self.assertEqual(d3_mapping([row])["x"],"real")
        with self.assertRaises(ValueError):d3_mapping([row,row])
        with self.assertRaises(ValueError):d3_mapping([dict(record_id="x",logical_split="D3")])

    def test_resource_cohort_cannot_access_d4_or_change_audited_hash(self):
        from final_statistics import resource_inputs
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            root=Path(td);image=root/"x.jpg";image.write_bytes(b"x")
            row=dict(record_id="x",logical_split="D1",destination_image_relative="x.jpg",probe_image_sha256=common.sha256(image))
            self.assertEqual(resource_inputs([row]*8,root)[0]["sha256"],row["probe_image_sha256"])
            with self.assertRaises(ValueError):resource_inputs([dict(row,logical_split="D4")]*8,root)
            with self.assertRaises(ValueError):resource_inputs([dict(row,probe_image_sha256="wrong")]*8,root)

    def test_statistical_family_does_not_turn_five_seeds_into_significance(self):
        from final_statistics import seed_statistics
        rows=[]
        for m,seeds,v in [("B0",range(42,47),.1),("B9",range(42,47),.2),("B11",range(42,45),.3)]:
            for s in seeds:rows.append(dict(variant=m,seed=s,**{k:v for k in common.METRICS}))
        primary,secondary=seed_statistics(rows)
        self.assertEqual(len(primary),8);self.assertTrue(all(r["holm_p"]==.5 for r in primary))
        self.assertTrue(all(r["n_pairs"]==3 for r in secondary))

    def test_gpu_busy_is_refused_instead_of_killing_other_experiments(self):
        from launch_supplement import assert_idle_gpu
        with self.assertRaisesRegex(RuntimeError,"GPU"):
            assert_idle_gpu("1024, python, 1050\n")
        assert_idle_gpu("")

    def test_primary_legacy_seed_recovery_requires_exact_completed_cache_identity(self):
        from final_statistics import recover_primary_rows
        rows=[dict(variant=m,seed=s,**{k:.1 for k in common.METRICS}) for m in ("B0","B9") for s in (42,43,44)]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            output=Path(td)
            for m in ("B0","B9"):
                for s in (45,46):
                    run=output/f"experiments/EXP-09/local_6gb/seed_{s}/runs/{m}/seed_{s}";run.mkdir(parents=True)
                    cache=output/f"training_cache/mmdet/{m}{s}";cache.mkdir(parents=True)
                    common.atomic_json(cache/"completed_run.json",dict(status="completed",source_work_dir=str(run),
                        best_checkpoint=str(run/"best.pth"),resolved_config_path=str(run/"resolved_mmdet_config.py"),
                        evaluation=dict(split="D3",experiment_id="EXP-09",**{k:.2 for k in common.METRICS})))
            restored,sources=recover_primary_rows(rows,output)
            self.assertEqual(len(restored),10);self.assertEqual(len(sources),4)
            keys=[(r["variant"],r["seed"]) for r in restored]
            self.assertEqual(keys,sorted(keys),"frozen protocol order must not depend on Python hash randomization")
            self.assertEqual([r["mask_ap"] for r in restored if r["seed"]==46],[.2,.2])
            common.atomic_json(cache/"completed_run.json",dict(status="running",source_work_dir=str(run)))
            with self.assertRaises(ValueError):recover_primary_rows(rows,output)

    def test_structure_cache_rejects_offsetting_edits_even_if_run_mean_is_unchanged(self):
        rows=[common.authenticated_cache_row(dict(image_id=i,total=x)) for i,x in [("a",2),("b",4)]]
        self.assertEqual(common.verify_cache_row(rows[0])["total"],2)
        rows[0]["total"]+=1;rows[1]["total"]-=1
        with self.assertRaisesRegex(ValueError,"checksum"):common.verify_cache_row(rows[0])
        with self.assertRaisesRegex(ValueError,"checksum"):common.verify_cache_row(rows[1])

    def test_completed_shortcut_verifies_files_not_just_status(self):
        from launch_supplement import verify_completion
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            out=Path(td);common.atomic_json(out/"protocol.json",dict(test=1))
            result=out/"EXP15_test.csv";result.write_bytes(b"hello")
            manifest=dict(status="completed_resources_and_statistics_supplement",protocol_sha256=common.sha256(out/"protocol.json"),outputs={result.name:common.sha256(result)})
            common.atomic_json(out/"run_manifest.json",manifest)
            self.assertTrue(verify_completion(out))
            result.write_bytes(b"broken")
            self.assertFalse(verify_completion(out))

    def test_matrix_prediction_export_accepts_parent_seed42_but_keeps_run_identity(self):
        import final_statistics as statistics
        self.assertTrue(hasattr(statistics,"validate_run_paths"),
                        "preflight must distinguish matrix exports from seed-specific model files")
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            output=Path(td).resolve()
            table=output/"experiments/EXP-09/local_6gb/seed_42/ablation_D3.csv"
            run=output/"experiments/EXP-09/local_6gb/seed_43/runs/B0/seed_43"
            exported=table.parent/"runs/B0/seed_43/predictions_D3.json"
            row=dict(variant="B0",seed=43,best_checkpoint=str(run/"best.pth"),
                     resolved_config=str(run/"resolved_mmdet_config.py"),predictions=str(exported))
            for path in [Path(row["best_checkpoint"]),Path(row["resolved_config"]),exported]:
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b"test")
            paths=statistics.validate_run_paths(row,output,table,{})
            self.assertEqual(paths["predictions"],exported)
            for field in ("best_checkpoint","resolved_config","predictions"):
                wrong=table.parent/"runs/B0/seed_44"/Path(row[field]).name
                wrong.parent.mkdir(parents=True,exist_ok=True);wrong.write_bytes(b"test")
                with self.subTest(field=field),self.assertRaisesRegex(ValueError,"different run"):
                    statistics.validate_run_paths(dict(row,**{field:str(wrong)}),output,table,{})

    def test_exported_prediction_header_must_match_selected_checkpoint_before_profiling(self):
        import final_statistics as statistics
        self.assertTrue(hasattr(statistics,"validate_prediction_identity"),
                        "resource stage cannot start before exported prediction identity is checked")
        classes=["crack","water_leakage","spalling","efflorescence","exposed_rebar"]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            path=Path(td)/"predictions_D3.json"
            row=dict(variant="B0",seed=43,checkpoint_sha256="selected")
            good=dict(experiment_id="EXP-09",split="D3",checkpoint_sha256="selected",
                      class_names=classes,metadata={"seed":43},images=[])
            common.atomic_json(path,good)
            statistics.validate_prediction_identity(path,row,classes)
            # Key order does not excuse a wrong late header field.
            for bad in [dict(good,checkpoint_sha256="another"),dict(good,split="D4"),
                        dict(good,metadata={"seed":44}),dict(good,class_names=["wrong"]*5)]:
                common.atomic_json(path,dict(images=[],**{k:v for k,v in bad.items() if k!="images"}))
                with self.assertRaises(ValueError):statistics.validate_prediction_identity(path,row,classes)

    def test_legacy_null_hash_requires_matching_completed_record_and_identical_image_payloads(self):
        import final_statistics as statistics
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            output=Path(td).resolve();row,classes,cache,exported,record=legacy_fixture(output)
            self.assertNotEqual(common.sha256(row["predictions"]),common.sha256(cache/"predictions_D3.json"))
            try:
                header=statistics.validate_prediction_identity(row["predictions"],row,classes,output=output,groups={"x":"g"})
            except TypeError as error:
                self.fail(f"legacy completed-cache provenance audit is missing: {error}")
            self.assertIsNone(header["checkpoint_sha256"],"historical missing hash must never be fabricated")
            with self.assertRaisesRegex(ValueError,"checkpoint"):
                common.validate_bundle_header(header,"B0",42,row["checkpoint_sha256"])
            audit=row["prediction_identity"]
            self.assertEqual(audit["mode"],"legacy_completed_cache_semantic_binding")
            self.assertEqual(audit["matching_images"],1)
            self.assertFalse(audit["historical_checkpoint_bytes_verified"])
            self.assertEqual(audit["current_checkpoint_sha256"],row["checkpoint_sha256"])
            self.assertEqual(common.read_json(row["predictions"])["checkpoint_sha256"],None)
            common.validate_bundle_header(header,"B0",42,row["checkpoint_sha256"],legacy_provenance=audit)
            with self.assertRaisesRegex(ValueError,"checkpoint"):
                common.validate_bundle_header(dict(header,framework="modified"),"B0",42,row["checkpoint_sha256"],legacy_provenance=audit)

    def test_legacy_fresh_export_without_reuse_metadata_still_requires_unique_exact_cache(self):
        import final_statistics as statistics
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            output=Path(td).resolve();row,classes,cache,exported,record=legacy_fixture(output)
            exported["metadata"]={"seed":42};common.atomic_json(row["predictions"],exported)
            row["prediction_sha256"]=common.sha256(row["predictions"])
            statistics.validate_prediction_identity(row["predictions"],row,classes,output=output,groups={"x":"g"})
            self.assertEqual(row["prediction_identity"]["matching_images"],1)
            second=cache.parent/('b'*64);second.mkdir()
            common.atomic_json(second/"completed_run.json",dict(record,signature=second.name))
            with self.assertRaisesRegex(ValueError,"ambiguous"):
                statistics.validate_prediction_identity(row["predictions"],dict(row),classes,output=output,groups={"x":"g"})

    def test_export_signature_does_not_hide_ambiguous_completed_cache_bindings(self):
        import final_statistics as statistics
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            output=Path(td).resolve();row,classes,cache,exported,record=legacy_fixture(output)
            second=cache.parent/('b'*64);second.mkdir()
            common.atomic_json(second/"completed_run.json",dict(record,signature=second.name))
            with self.assertRaisesRegex(ValueError,"ambiguous"):
                statistics.validate_prediction_identity(row["predictions"],row,classes,output=output,groups={"x":"g"})

    def test_legacy_null_hash_rejects_changed_predictions_even_when_reported_ap_agrees(self):
        import final_statistics as statistics
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            output=Path(td).resolve();row,classes,cache,exported,record=legacy_fixture(output)
            exported["images"][0]["predictions"]=[dict(score=.9,class_name="crack")]
            common.atomic_json(row["predictions"],exported)
            with self.assertRaisesRegex(ValueError,"payload"):
                statistics.validate_prediction_identity(row["predictions"],row,classes,output=output,groups={"x":"g"})

    def test_legacy_null_hash_rejects_wrong_cache_binding_and_metric(self):
        import final_statistics as statistics
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            output=Path(td).resolve();row,classes,cache,exported,record=legacy_fixture(output)
            for field,value in [("best_checkpoint",str(cache/"another.pth")),("status","running")]:
                common.atomic_json(cache/"completed_run.json",dict(record,**{field:value}))
                with self.assertRaises(ValueError):statistics.validate_prediction_identity(row["predictions"],dict(row),classes,output=output,groups={"x":"g"})
            bad=dict(record,evaluation=dict(record["evaluation"],mask_ap=.5))
            common.atomic_json(cache/"completed_run.json",bad)
            with self.assertRaisesRegex(ValueError,"metric"):
                statistics.validate_prediction_identity(row["predictions"],dict(row),classes,output=output,groups={"x":"g"})

    def test_legacy_provenance_never_excuses_real_checkpoint_hash_conflict(self):
        import final_statistics as statistics
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            output=Path(td).resolve();row,classes,cache,exported,record=legacy_fixture(output)
            exported["checkpoint_sha256"]='f'*64;common.atomic_json(row["predictions"],exported)
            with self.assertRaisesRegex(ValueError,"checkpoint"):
                statistics.validate_prediction_identity(row["predictions"],row,classes,output=output,groups={"x":"g"})

    def test_legacy_provenance_survives_statistics_stage_without_mutating_originals(self):
        import os,sys
        project=Path(os.environ.get("SUPPLEMENT_TEST_PROJECT","/root/autodl-tmp/railway_tunnel_experiments"))
        if not (project/"src").is_dir():self.skipTest("real project unavailable")
        sys.path.insert(0,str(project/"src"))
        import final_statistics as statistics
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
            output=Path(td).resolve();row,classes,cache,exported,record=legacy_fixture(output)
            for metric in common.STRUCTURE:row[metric]=None;record["evaluation"][metric]=None
            common.atomic_json(cache/"completed_run.json",record)
            before={str(p):common.sha256(p) for p in output.rglob("*") if p.is_file()}
            statistics.validate_prediction_identity(row["predictions"],row,classes,output=output,groups={"x":"g"})
            protocol=dict(d3_groups={"x":"g"},class_names=classes)
            out=output/"supplements/statistics";common.atomic_json(out/"protocol.json",protocol)
            results,checks=statistics.cached_structure(protocol,row,out)
            self.assertEqual(len(results),6)
            self.assertTrue(all(r["count"]==0 for r in results))
            self.assertTrue(all(r["recomputed"] is None for r in checks))
            statistics.cached_structure(protocol,row,out)  # verified cache resume
            self.assertEqual({name:common.sha256(name) for name in before},before)

    def test_compact_rle_matching_agrees_with_real_full_mask_metrics(self):
        import os,sys
        project=Path(os.environ.get("SUPPLEMENT_TEST_PROJECT","/root/autodl-tmp/railway_tunnel_experiments"))
        if not (project/"src").is_dir():self.skipTest("real project unavailable")
        sys.path.insert(0,str(project/"src"))
        from morphoroute.data.annotations import encode_uncompressed_rle
        from morphoroute.metrics import structure as st
        from final_statistics import image_statistics,_rle,_rle_iou
        rng=np.random.default_rng(64)
        funcs=(st.cldice,st.crack_break_rate,st.component_count_error,st.boundary_f1,st.hd95,st.area_relative_error)
        for name in ("crack","spalling"):
            for index in range(12):
                a=np.zeros((24,31),bool);a[:16,:9]=1
                b=a.copy();b[:2,:2]=0;b[4:7,9:11]=1
                if index%2:a=rng.random((24,31))>.8;b=a.copy();b[3:6,8:10]^=True
                def inst(mask,score):return dict(class_name=name,score=score,mask_rle=encode_uncompressed_rle(mask))
                pa,gb=inst(a,.9),inst(b,1.)
                ra,rb=_rle(pa,24,31),_rle(gb,24,31)
                self.assertAlmostEqual(_rle_iou(ra,rb),np.logical_and(a,b).sum()/np.logical_or(a,b).sum())
                result=image_statistics(dict(height=24,width=31,predictions=[pa,pa],targets=[gb]),["crack","spalling"])
                self.assertEqual(result["false_positives"],1)
                for metric,fn in zip(common.STRUCTURE,funcs):
                    if (name=="crack")==metric.startswith("crack_"):
                        self.assertEqual(result["counts"][metric],1)
                        self.assertAlmostEqual(result["totals"][metric],fn(a,b),places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
