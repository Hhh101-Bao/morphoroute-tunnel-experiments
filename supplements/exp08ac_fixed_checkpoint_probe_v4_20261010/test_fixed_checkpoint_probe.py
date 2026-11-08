"""Regression tests for the read-only, post-hoc supplement protocol."""
import importlib
from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    probe = importlib.import_module("fixed_checkpoint_probe")
except ModuleNotFoundError:
    probe = None


class ProbeContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (Path(__file__).parent / "tmp").mkdir(exist_ok=True)

    def setUp(self):
        self.assertIsNotNone(probe, "the agreed supplement implementation is missing")

    def background_fixture(self, root):
        from PIL import Image
        Image.fromarray(np.zeros((8, 10), dtype=np.uint8)).save(root / "mask.png")
        Image.new("RGB", (10, 8)).save(root / "image.png")
        row = {"record_id": "TACK_negative", "group_id": "real_tunnel",
               "logical_split": "train", "source": "TACK", "classes_present": "",
               "class_instances_json": "{}", "destination_image_relative": "image.png",
               "destination_annotation_relative": "mask.png",
               "annotation_sha256": probe.sha256(root / "mask.png")}
        image = {"id": 7, "record_id": row["record_id"], "file_name": str(root / "image.png"),
                 "width": 10, "height": 8, "dataset_id": "D1"}
        return row, {"images": [image], "annotations": []}

    def cohort_auditor(self):
        auditor = getattr(probe, "audit_coco_cohort", None)
        self.assertIsNotNone(auditor, "empty COCO must be audited against original annotations, not rejected as missing")
        return auditor

    def test_true_background_is_audited_without_replacement_or_gradient(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent / "tmp") as folder:
            root = Path(folder)
            row, coco = self.background_fixture(root)
            original = dict(row)
            audit = self.cohort_auditor()([row], coco, root)
            self.assertEqual(row, original)
            self.assertEqual(len(audit), 1)
            self.assertEqual(audit[0]["record_id"], "TACK_negative")
            self.assertEqual(audit[0]["preflight_status"], "excluded_background")
            self.assertEqual(audit[0]["raw_mask_values"], [0])
            self.assertEqual(audit[0]["coco_image_id"], 7)

    def test_missing_image_is_not_treated_as_background(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent / "tmp") as folder:
            root = Path(folder)
            row, coco = self.background_fixture(root)
            coco["images"] = []
            with self.assertRaisesRegex(ValueError, "missing COCO image"):
                self.cohort_auditor()([row], coco, root)

    def test_matching_path_does_not_authorize_conflicting_explicit_coco_record_id(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent / "tmp") as folder:
            root = Path(folder)
            row, coco = self.background_fixture(root)
            coco["images"][0]["record_id"] = "other_record"
            with self.assertRaisesRegex(ValueError, "identity|record_id"):
                self.cohort_auditor()([row], coco, root)
            del coco["images"][0]["record_id"]
            # Idless COCO image metadata still maps by the exact verified path.
            audit = self.cohort_auditor()([row], coco, root)
            self.assertEqual(audit[0]["record_id"], "TACK_negative")
            self.assertEqual(audit[0]["preflight_status"], "excluded_background")

    def test_empty_coco_with_positive_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent / "tmp") as folder:
            root = Path(folder)
            row, coco = self.background_fixture(root)
            row.update(classes_present="crack", class_instances_json='{"crack": 1}')
            with self.assertRaisesRegex(ValueError, "foreground|contradict"):
                self.cohort_auditor()([row], coco, root)

    def test_empty_coco_with_nonzero_or_changed_original_mask_is_rejected(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent / "tmp") as folder:
            root = Path(folder)
            row, coco = self.background_fixture(root)
            Image.fromarray(np.full((8, 10), 40, dtype=np.uint8)).save(root / "mask.png")
            with self.assertRaisesRegex(ValueError, "hash"):
                self.cohort_auditor()([row], coco, root)
            row["annotation_sha256"] = probe.sha256(root / "mask.png")
            with self.assertRaisesRegex(ValueError, "nonzero"):
                self.cohort_auditor()([row], coco, root)

    def test_crowd_only_wrong_split_and_wrong_path_are_not_negative_controls(self):
        import copy
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent / "tmp") as folder:
            root = Path(folder)
            row, coco = self.background_fixture(root)
            crowd = copy.deepcopy(coco)
            crowd["annotations"] = [{"image_id": 7, "iscrowd": 1}]
            wrong_split = copy.deepcopy(coco)
            wrong_split["images"][0]["dataset_id"] = "D4"
            wrong_path = copy.deepcopy(coco)
            wrong_path["images"][0]["file_name"] = str(root / "other.png")
            for bad in (crowd, wrong_split, wrong_path):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    self.cohort_auditor()([row], bad, root)

    def test_positive_mapping_and_original_cohort_order_are_preserved(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent / "tmp") as folder:
            root = Path(folder)
            row, coco = self.background_fixture(root)
            positive = {**row, "record_id": "TACK_positive", "classes_present": "crack",
                        "class_instances_json": '{"crack":1}', "destination_image_relative": "positive.png"}
            (root / "positive.png").write_bytes((root / "image.png").read_bytes())
            coco["images"].append({**coco["images"][0], "id": 9, "record_id": "TACK_positive",
                                   "file_name": str(root / "positive.png")})
            coco["annotations"].append({"image_id": 9, "iscrowd": 0})
            audits = self.cohort_auditor()([row, positive], coco, root)
            self.assertEqual([a["record_id"] for a in audits], ["TACK_negative", "TACK_positive"])
            self.assertEqual([a["preflight_status"] for a in audits], ["excluded_background", "eligible_foreground"])

    def test_pilot_uses_first_five_positive_members_without_resampling(self):
        chooser = getattr(probe, "pilot_cohort", None)
        self.assertIsNotNone(chooser, "pilot must use the original positive subset rather than arbitrary first five")
        cohort = [{"record_id": "negative", "preflight_status": "excluded_background"}] + [
            {"record_id": str(i), "preflight_status": "eligible_foreground"} for i in range(7)]
        self.assertEqual([r["record_id"] for r in chooser({"cohort": cohort})], ["0", "1", "2", "3", "4"])
        self.assertEqual(len(cohort), 8)

    def test_negative_record_cannot_masquerade_as_gradient_observation(self):
        protocol = {"cohort": [{"record_id": "n", "group_id": "g", "coco_image_id": 7,
                                 "preflight_status": "excluded_background"}],
                    "checkpoints": {"42": {"sha256": "c" * 64, "iteration": 240000}}}
        row = {"seed": 42, "record_id": "n", "group_id": "g", "coco_image_id": 7,
               "checkpoint_hash": "c" * 64, "checkpoint_stage": "240000", "status": "valid",
               "gradient_cosine": 0.5, "model_state_sha256": "a" * 64,
               "model_state_verified_before_append": True, "training_update": 60000}
        with self.assertRaisesRegex(ValueError, "background|excluded"):
            probe.validate_observations(protocol, [row])

    def test_prepare_preserves_original_sample_and_writes_separate_negative_audit(self):
        """Only the unavailable MMEngine parser is doubled; real files/COCO/audits are used."""
        from PIL import Image
        from types import ModuleType
        from unittest.mock import patch
        import csv
        class AttrDict(dict):
            def __getattr__(self, name):
                return self[name]
        def attributes(value):
            if isinstance(value, dict):
                return AttrDict({k: attributes(v) for k, v in value.items()})
            if isinstance(value, list):
                return [attributes(v) for v in value]
            return value
        class FakeConfig:
            @classmethod
            def fromfile(cls, path):
                instance = cls()
                instance.payload = json.loads(Path(path).read_text())
                return instance
            def to_dict(self):
                return self.payload
            def __getattr__(self, key):
                return attributes(self.payload[key])
        module = ModuleType("mmengine.config")
        module.Config = FakeConfig
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent / "tmp") as folder:
            root = Path(folder)
            data, project, out = root / "data", root / "project", root / "output"
            (data / "metadata").mkdir(parents=True)
            project.mkdir()
            out.mkdir()
            Image.new("RGB", (10, 8)).save(data / "image.png")
            Image.fromarray(np.zeros((8, 10), dtype=np.uint8)).save(data / "negative.png")
            Image.fromarray(np.full((8, 10), 40, dtype=np.uint8)).save(data / "positive.png")
            rows, images, anns = [], [], []
            for i in range(21):
                negative = i == 0
                image_name, mask_name = f"image{i}.png", "negative.png" if negative else "positive.png"
                (data / image_name).write_bytes((data / "image.png").read_bytes())
                rows.append({"record_id": f"record{i}", "assignment_group_id": f"group{i}",
                             "logical_split": "train", "source": "TACK",
                             "classes_present": "" if negative else "crack",
                             "class_instances_json": "{}" if negative else '{"crack":1}',
                             "destination_image_relative": image_name, "destination_annotation_relative": mask_name,
                             "width": "10", "height": "8", "image_sha256": probe.sha256(data / image_name),
                             "annotation_sha256": probe.sha256(data / mask_name)})
                images.append({"id": i, "record_id": f"record{i}", "file_name": str(data / image_name),
                               "width": 10, "height": 8, "dataset_id": "D1"})
                if not negative:
                    anns.append({"image_id": i, "iscrowd": 0})
            manifest = data / "metadata/split_manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            annotation, prior = project / "D1.json", project / "prior.json"
            annotation.write_text(json.dumps({"images": images, "annotations": anns}), encoding="utf-8")
            prior.write_text("{}", encoding="utf-8")
            for seed in probe.SEEDS:
                directory = project / f"outputs/experiments/EXP-09/local_6gb/seed_{seed}/runs/B9/seed_{seed}"
                directory.mkdir(parents=True)
                (directory / "best_coco_segm_mAP_iter_240000.pth").write_bytes(b"checkpoint")
                cfg = {"model": {"type": "MorphoRouteMask2Former", "panoptic_head": {
                       "type": "MorphoRouteMask2FormerHead", "variant_id": "B9", "morphology_prior_path": str(prior)}},
                       "randomness": {"seed": seed}, "work_dir": str(directory),
                       "train_dataloader": {"dataset": {"type": "CocoDataset", "ann_file": str(annotation),
                       "metainfo": {"classes": ["crack", "water_leakage", "spalling", "efflorescence", "exposed_rebar"]}}}}
                (directory / "resolved_mmdet_config.py").write_text(json.dumps(cfg), encoding="utf-8")
            original_hash = probe.sha256(manifest)
            with patch.dict(sys.modules, {"mmengine": ModuleType("mmengine"), "mmengine.config": module}), redirect_stdout(io.StringIO()):
                protocol = probe.prepare(project, data, out, 21, 50)
            self.assertEqual(probe.sha256(manifest), original_hash)
            self.assertEqual(len(protocol["cohort"]), 21)
            self.assertEqual(len(probe.collection_cohort(protocol)), 20)
            self.assertEqual(protocol["preflight_exclusions"][0]["record_id"], "record0")
            self.assertTrue((out / "sample_exclusions_D1.csv").is_file())
            self.assertTrue((out / "foreground_eligibility_audit.json").is_file())
            with (out / "sample_manifest_D1.csv").open(encoding="utf-8-sig") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 21)

    def test_known_exif_copy_preserves_both_original_and_probe_fingerprints(self):
        validator = getattr(probe, "validate_image_fingerprint", None)
        self.assertIsNotNone(validator, "image provenance must distinguish the audited orientation copy")
        row = {"record_id": "DACL_train_dacl10k_v2_train_3520", "logical_split": "train",
               "source": "DACL", "width": "4032", "height": "3024",
               "destination_image_relative": "D1_D3_internal/train/images/DACL__dacl10k_v2_train_3520.jpg",
               "image_sha256": "2a85c12efdce62f89afc93e93984e1cf558a2cb594985fa40d675973d3a31d8f",
               "annotation_sha256": "6f7c51692efae19289d7c837b1b3fde21c3d2bce168e98622ae6b1b6a080841f"}
        original = dict(row)
        audit = validator(row, "739ae79710e05687ea0e94e9930918102409e9a018626a42849327c0f7a6ad0f",
                          (3024, 4032), row["annotation_sha256"], (3024, 4032), None)
        self.assertEqual(audit["status"], "audited_orientation_copy")
        self.assertEqual(audit["manifest_image_sha256"], original["image_sha256"])
        self.assertEqual(audit["probe_image_sha256"], "739ae79710e05687ea0e94e9930918102409e9a018626a42849327c0f7a6ad0f")
        self.assertEqual(row, original, "validation must not rewrite original manifest fields")
        for change in ({"record_id": "other"}, {"source": "TACK"}, {"logical_split": "D4"},
                       {"image_sha256": "f" * 64}, {"width": "3024"},
                       {"destination_image_relative": "another_image.jpg"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validator({**row, **change}, audit["probe_image_sha256"], (3024, 4032),
                          row["annotation_sha256"], (3024, 4032), None)

    def test_unknown_image_or_annotation_change_is_rejected(self):
        validator = getattr(probe, "validate_image_fingerprint", None)
        self.assertIsNotNone(validator, "integrity guard must still reject unknown changes")
        row = {"record_id": "regular", "logical_split": "D1", "source": "DACL",
               "image_sha256": "a" * 64, "annotation_sha256": "b" * 64,
               "width": "100", "height": "80"}
        audit = validator(row, "a" * 64, (100, 80), "b" * 64, (100, 80), None)
        self.assertEqual(audit["status"], "manifest_match")
        for actual, ann, size, orientation in [("c" * 64, "b" * 64, (100, 80), None),
                                              ("a" * 64, "c" * 64, (100, 80), None),
                                              ("a" * 64, "b" * 64, (80, 100), None),
                                              ("a" * 64, "b" * 64, (100, 80), 6)]:
            with self.subTest(actual=actual, ann=ann, size=size, orientation=orientation), self.assertRaises(ValueError):
                validator(row, actual, (100, 80), ann, size, orientation)

    def test_probe_input_drift_is_rejected_before_observation(self):
        verifier = getattr(probe, "verify_probe_input", None)
        self.assertIsNotNone(verifier, "collector must enforce the current frozen input fingerprint")
        import hashlib
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent / "tmp") as folder:
            root = Path(folder)
            (root / "image.jpg").write_bytes(b"image")
            (root / "annotation.json").write_bytes(b"annotation")
            row = {"record_id": "a", "destination_image_relative": "image.jpg",
                   "destination_annotation_relative": "annotation.json",
                   "probe_image_sha256": hashlib.sha256(b"image").hexdigest(),
                   "probe_annotation_sha256": hashlib.sha256(b"annotation").hexdigest()}
            verifier(row, root)
            (root / "image.jpg").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                verifier(row, root)

    def test_cohort_uses_real_groups_and_never_selects_d4(self):
        rows = [
            {"record_id": "a", "logical_split": "D1", "assignment_group_id": "tunnel1"},
            {"record_id": "b", "logical_split": "D1", "assignment_group_id": "tunnel1"},
            {"record_id": "c", "logical_split": "D1", "assignment_group_id": "tunnel2"},
            {"record_id": "external", "logical_split": "D4", "assignment_group_id": "tunnel3"},
        ]
        selected = probe.select_cohort(rows, limit=2, sampling_seed=20261010)
        self.assertEqual({r["group_id"] for r in selected}, {"tunnel1", "tunnel2"})
        self.assertNotIn("external", {r["record_id"] for r in selected})
        self.assertEqual(selected, probe.select_cohort(list(reversed(rows)), 2, 20261010))

    def test_duplicate_record_or_missing_group_is_not_silently_accepted(self):
        row = {"record_id": "a", "logical_split": "D1", "group_key": "real_group"}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            probe.select_cohort([row, row], 2, 1)
        with self.assertRaisesRegex(ValueError, "group"):
            probe.select_cohort([{**row, "group_key": ""}], 2, 1)

    def test_project_head_only_variant_schema_is_valid_b9(self):
        directory = Path(__file__).parent / "tmp" / "seed_42" / "runs" / "B9" / "seed_42"
        cfg = {"model": {"type": "MorphoRouteMask2Former",
                         "panoptic_head": {"type": "MorphoRouteMask2FormerHead", "variant_id": "B9"}},
               "randomness": {"seed": 42}, "work_dir": str(directory)}
        validator = getattr(probe, "validate_model_identity", None)
        self.assertIsNotNone(validator, "validator must support the actual head-only project schema")
        audit = validator(cfg, 42, directory)
        self.assertEqual(audit["head_variant"], "B9")
        self.assertIsNone(audit["top_variant"])

    def test_identity_conflicts_are_rejected_not_bypassed(self):
        directory = Path(__file__).parent / "tmp" / "seed_42" / "runs" / "B9" / "seed_42"
        cfg = {"model": {"type": "MorphoRouteMask2Former",
                         "panoptic_head": {"type": "MorphoRouteMask2FormerHead", "variant_id": "B9"}},
               "randomness": {"seed": 42}, "work_dir": str(directory)}
        validator = getattr(probe, "validate_model_identity", None)
        self.assertIsNotNone(validator, "identity guard must be implemented")
        import copy
        changes = [({"variant_id": "B11"}, None, None, None),
                   ({"type": "Mask2Former"}, None, None, None),
                   (None, {"variant_id": "B0"}, None, None),
                   (None, {"type": "Mask2FormerHead"}, None, None),
                   (None, None, 43, None), (None, None, None, directory.parent)]
        for model_change, head_change, seed_change, directory_change in changes:
            bad = copy.deepcopy(cfg)
            if model_change:
                bad["model"].update(model_change)
            if head_change:
                bad["model"]["panoptic_head"].update(head_change)
            if seed_change:
                bad["randomness"]["seed"] = seed_change
            if directory_change:
                bad["work_dir"] = str(directory_change)
            with self.assertRaisesRegex(ValueError, "identity"):
                validator(bad, 42, directory)

    def test_checkpoint_selection_never_chooses_latest_by_mtime(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent / "tmp") as folder:
            directory = Path(folder)
            (directory / "best_coco_segm_mAP_iter_240000.pth").write_bytes(b"best")
            (directory / "iter_400000.pth").write_bytes(b"latest")
            chosen, iteration = probe.select_checkpoint(directory)
            self.assertEqual(chosen.name, "best_coco_segm_mAP_iter_240000.pth")
            self.assertEqual(iteration, 240000)
            (directory / "best_coco_segm_mAP_iter_320000.pth").write_bytes(b"other")
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                probe.select_checkpoint(directory)

    def test_gradient_pair_zero_fills_unused_parameters_before_alignment(self):
        result = probe.gradient_pair([np.array([1., 0.]), None],
                                     [None, np.array([1.])], [(2,), (1,)])
        self.assertEqual(result["status"], "valid")
        self.assertAlmostEqual(result["gradient_cosine"], 0.)
        self.assertAlmostEqual(result["norm_ratio"], 1.)
        opposite = probe.gradient_pair([np.array([1., 2.])],
                                       [np.array([-1., -2.])], [(2,)])
        self.assertAlmostEqual(opposite["gradient_cosine"], -1.)

    def test_zero_and_nonfinite_gradients_are_not_fabricated_observations(self):
        result = probe.gradient_pair([np.zeros(2)], [np.ones(2)], [(2,)])
        self.assertEqual(result["status"], "zero_norm")
        self.assertIsNone(result["gradient_cosine"])
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            probe.gradient_pair([np.array([float("nan")])], [np.ones(1)], [(1,)])

    def test_resume_rejects_changed_protocol(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent / "tmp") as folder:
            path = Path(folder) / "protocol.json"
            probe.ensure_protocol(path, {"checkpoint": "sha1", "cohort": ["a", "b"]})
            probe.ensure_protocol(path, {"checkpoint": "sha1", "cohort": ["a", "b"]})
            with self.assertRaisesRegex(ValueError, "changed"):
                probe.ensure_protocol(path, {"checkpoint": "sha2", "cohort": ["a", "b"]})
            self.assertEqual(json.loads(path.read_text())["checkpoint"], "sha1")

    def test_partial_checkpoint_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent / "tmp") as folder:
            path = Path(folder) / "rows.jsonl"
            path.write_text('{"seed":42,"record_id":"a"}\n' * 2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                probe.load_records(path)

    def test_resume_requires_same_group_checkpoint_and_verified_model_state(self):
        protocol = {"cohort": [{"record_id": "a", "group_id": "real_group", "coco_image_id": 7}],
                    "checkpoints": {str(s): {"sha256": "c" * 64, "iteration": 240000}
                                    for s in probe.SEEDS}}
        row = {"seed": 42, "record_id": "a", "group_id": "real_group", "coco_image_id": 7,
               "checkpoint_hash": "c" * 64, "checkpoint_stage": "240000",
               "model_state_sha256": "a" * 64, "model_state_verified_before_append": True,
               "training_update": 60000}
        probe.validate_observations(protocol, [row])
        for change in ({"group_id": "wrong"}, {"checkpoint_hash": "wrong"},
                       {"model_state_verified_before_append": False}, {"training_update": 0}):
            with self.assertRaises(ValueError):
                probe.validate_observations(protocol, [{**row, **change}])
        with self.assertRaisesRegex(ValueError, "model state"):
            probe.validate_observations(protocol, [row, {**row, "model_state_sha256": "b" * 64}])

    def test_unloaded_or_invalid_training_counter_is_not_a_valid_checkpoint_stage(self):
        self.assertEqual(probe.validate_training_update(60000, 240000), 60000)
        for update in (0, -1, 240001):
            with self.assertRaisesRegex(ValueError, "training-update"):
                probe.validate_training_update(update, 240000)

    def test_group_summary_gives_each_image_equal_weight_across_seeds(self):
        rows = [
            {"seed": 42, "record_id": "a", "group_id": "g", "status": "valid", "gradient_cosine": -1., "norm_ratio": 1.},
            {"seed": 43, "record_id": "a", "group_id": "g", "status": "valid", "gradient_cosine": 1., "norm_ratio": 3.},
            {"seed": 42, "record_id": "b", "group_id": "g", "status": "valid", "gradient_cosine": 1., "norm_ratio": 5.},
            {"seed": 43, "record_id": "b", "group_id": "g", "status": "valid", "gradient_cosine": 1., "norm_ratio": 5.},
        ]
        summary = probe.summarize_groups(rows, expected_seeds=[42, 43])
        self.assertEqual(summary[0]["image_count"], 2)
        self.assertAlmostEqual(summary[0]["mean_cosine"], .5)
        self.assertAlmostEqual(summary[0]["negative_observation_rate"], .25)
        with self.assertRaisesRegex(ValueError, "paired"):
            probe.summarize_groups(rows[:-1], expected_seeds=[42, 43])

    def test_rank_deficiency_does_not_produce_significance(self):
        with self.assertRaisesRegex(ValueError, "rank"):
            probe.cluster_ols(np.ones((40, 2)), np.arange(40.),
                              [str(i) for i in range(40)], bootstrap=50)

    def test_cluster_ols_recovers_known_linear_relationship(self):
        x = np.column_stack([np.ones(40), np.linspace(-1., 1., 40)])
        y = 2. + 3. * x[:, 1]
        result = probe.cluster_ols(x, y, [str(i) for i in range(40)], bootstrap=50)
        self.assertAlmostEqual(result["coefficients"][0], 2.)
        self.assertAlmostEqual(result["coefficients"][1], 3.)
        self.assertEqual(result["group_count"], 40)

    def test_analysis_writes_only_posthoc_scope_not_final_completion(self):
        rng = np.random.default_rng(8)
        rows = []
        for i in range(40):
            features = rng.uniform(.1, 1., 5)
            fractions = rng.dirichlet(np.ones(5))
            for seed in (42, 43, 44):
                rows.append({"record_id": str(i), "group_id": str(i), "seed": seed,
                             "status": "valid", "gradient_cosine": float(features[0] / 2),
                             "norm_ratio": 2., "instance_area": float(rng.uniform(100, 1000)),
                             "class_fractions": fractions.tolist(),
                             **dict(zip(probe.FEATURES, features.tolist()))})
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent / "tmp") as folder:
            out = Path(folder)
            with redirect_stdout(io.StringIO()):
                probe.analysis({"bootstrap": 50, "notes": [], "cohort": [None] * 42,
                                "preflight_exclusions": [{"record_id": "negative1"}, {"record_id": "negative2"}]}, rows, out)
            manifest = json.loads((out / "run_manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed_posthoc_probe")
            self.assertFalse(manifest["EXP15_completed_by_this_script"])
            self.assertFalse(manifest["D4_access"])
            self.assertTrue((out / "M3_association_coefficients.csv").is_file())
            audit = json.loads((out / "M3_analysis_audit.json").read_text())
            self.assertEqual(audit["sampled_image_count"], 42)
            self.assertEqual(audit["complete_image_count"], 40)
            self.assertEqual(audit["excluded_image_count"], 2)
            self.assertEqual(audit["preflight_background_excluded_image_count"], 2)
            self.assertEqual(audit["actual_gradient_record_count"], 120)
            self.assertEqual(audit["preflight_excluded_record_ids"], ["negative1", "negative2"])

    def test_read_only_loss_gradients_preserve_torch_model_state(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Torch is not installed locally; this test must run on AutoDL")
        model = torch.nn.Linear(2, 1, bias=False)
        before = probe.model_hash(model)
        result = model(torch.tensor([[1., 2.]]))
        left = torch.autograd.grad(result.sum(), list(model.parameters()), retain_graph=True)
        right = torch.autograd.grad(-result.sum(), list(model.parameters()))
        measured = probe.gradient_pair([x.detach().numpy() for x in left],
                                        [x.detach().numpy() for x in right], [(1, 2)])
        self.assertAlmostEqual(measured["gradient_cosine"], -1.)
        self.assertEqual(probe.model_hash(model), before)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


class LauncherLogging(unittest.TestCase):
    def test_failed_preflight_is_logged_before_launch_refusal(self):
        import launch_fixed_checkpoint_probe as launcher
        runner = getattr(launcher, "run_logged", None)
        self.assertIsNotNone(runner, "preflight output must be persisted even before the worker starts")
        (Path(__file__).parent / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent / "tmp") as folder:
            path = Path(folder) / "startup.log"
            with path.open("w", encoding="utf-8") as log:
                with redirect_stdout(io.StringIO()), self.assertRaises(subprocess.CalledProcessError) as raised:
                    runner([sys.executable, "-c", "import sys; print('PREFLIGHT_FAILED_OUTPUT', flush=True); sys.exit(2)"], log)
            self.assertEqual(raised.exception.returncode, 2)
            self.assertIn("PREFLIGHT_FAILED_OUTPUT", path.read_text(encoding="utf-8"))

    def test_successful_preflight_stdout_and_stderr_are_both_logged(self):
        import launch_fixed_checkpoint_probe as launcher
        runner = getattr(launcher, "run_logged", None)
        self.assertIsNotNone(runner, "real preflight output must be persisted")
        (Path(__file__).parent / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent / "tmp") as folder:
            path = Path(folder) / "startup.log"
            with path.open("w", encoding="utf-8") as log:
                with redirect_stdout(io.StringIO()):
                    runner([sys.executable, "-c", "import sys; print('STDOUT_MARKER'); print('STDERR_MARKER', file=sys.stderr)"], log)
            result = path.read_text(encoding="utf-8")
            self.assertIn("STDOUT_MARKER", result)
            self.assertIn("STDERR_MARKER", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
