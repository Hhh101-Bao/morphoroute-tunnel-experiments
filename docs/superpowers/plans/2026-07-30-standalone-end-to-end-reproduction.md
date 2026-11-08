# Standalone End-to-End Experiment Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every EXP-00 through EXP-15 Python entry a no-argument, standalone workflow that automatically reads the registered dataset, resolves prerequisites, performs real training/evaluation, and produces every SOP-mandated artifact.

**Architecture:** A standard-library-only entry shim launches a shared orchestrator. The orchestrator validates data and dependencies, relays GPU jobs to pinned WSL environments, materializes YOLO/COCO data, invokes official Ultralytics or MMDetection backends, and runs custom MorphoRoute modules registered with MMDetection. Every stage emits typed prediction bundles, metrics, figures, hashes, and resumable manifests.

**Tech Stack:** Python 3.10; PyTorch 2.1.2; torchvision 0.16.2; CUDA 11.8; Ultralytics 8.x; MMEngine 0.10.x; MMCV 2.1.x; MMDetection 3.3.x; NumPy; pandas; SciPy; scikit-image; scikit-learn; pycocotools; OpenCV; matplotlib; openpyxl; pytest.

## Global Constraints

- Modify only `C:\Users\Baoju\Desktop\实验代码\铁路隧道多病害实例分割实验代码\railway_tunnel_experiments`.
- Do not modify the original ZIP or SOP DOCX.
- Preserve every existing Chinese filename, folder name, and experiment title.
- Dataset root is `C:\Users\Baoju\Desktop\铁路隧道多病害实例分割_原始影像严格划分_v1`.
- `split_manifest.csv` is the sole sample authority.
- D1 trains; D2 selects/checks/calibrates; D3 evaluates internally; D4 is one-shot after freeze.
- `local_6gb` keeps 640, physical batch 1, effective batch 8, AMP, accumulation, and checkpointing.
- `sop_full` keeps 1024 and 50,000 optimizer updates.
- No silent model degradation after OOM.
- No smoke-test artifact may enter final-paper aggregation.
- The project has no Git repository. The untouched original ZIP is the rollback point; each task records a verification checkpoint rather than a commit.

---

### Task 1: Standalone no-argument entry contract

**Files:**
- Create: `src/morphoroute/core/standalone.py`
- Modify: `experiments/_entry.py`
- Modify: all 19 `experiments/exp*.py`
- Modify: `scripts/run_experiment.py`
- Test: `tests/test_standalone_entries.py`

**Interfaces:**
- Produces: `EntrySettings`, `load_entry_settings()`, `run_standalone_experiment()`.
- Consumes: existing `execute_config()` and profile JSON files.

- [ ] **Step 1: Write failing entry-contract tests**

```python
class StandaloneEntryTests(unittest.TestCase):
    def test_every_entry_has_no_argument_defaults(self):
        for path in ENTRY_PATHS:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            names = {node.targets[0].id for node in tree.body
                     if isinstance(node, ast.Assign)
                     and isinstance(node.targets[0], ast.Name)}
            self.assertTrue(
                {"PROFILE", "SMOKE_TEST", "AUTO_RUN_PREREQUISITES", "RESUME"}
                <= names
            )

    def test_default_settings_point_to_authoritative_dataset(self):
        settings = EntrySettings.defaults()
        self.assertEqual(
            settings.dataset_root,
            Path(r"C:\Users\Baoju\Desktop\铁路隧道多病害实例分割_原始影像严格划分_v1"),
        )
        self.assertEqual(settings.profile, "local_6gb")
        self.assertTrue(settings.auto_run_prerequisites)
```

- [ ] **Step 2: Run RED**

Run:

```powershell
python -m pytest -q tests/test_standalone_entries.py
```

Expected: failure because `EntrySettings` and the constants do not exist.

- [ ] **Step 3: Implement the entry settings and launcher**

```python
@dataclass(frozen=True)
class EntrySettings:
    profile: str
    dataset_root: Path
    output_root: Path
    smoke_test: bool
    auto_run_prerequisites: bool
    resume: bool

    @classmethod
    def defaults(cls) -> "EntrySettings":
        project_root = Path(__file__).resolve().parents[3]
        return cls(
            profile="local_6gb",
            dataset_root=Path(
                r"C:\Users\Baoju\Desktop\铁路隧道多病害实例分割_原始影像严格划分_v1"
            ),
            output_root=project_root / "outputs",
            smoke_test=False,
            auto_run_prerequisites=True,
            resume=True,
        )
```

Expose the callable interface
`run_standalone_experiment(experiment_id: str, settings: EntrySettings, argv: Sequence[str] | None = None) -> int`.
Each entry passes its visible constants into this function. CLI values override constants only when explicitly supplied.

- [ ] **Step 4: Run GREEN and regression tests**

```powershell
python -m pytest -q tests/test_standalone_entries.py tests/test_cli.py
```

Expected: all pass.

- [ ] **Step 5: Record checkpoint**

Save `verification/task01-entry-contract.txt` containing the test command, exit code, and timestamp.

---

### Task 2: Windows-to-WSL environment relay and bootstrap

**Files:**
- Create: `src/morphoroute/runtime/environments.py`
- Create: `src/morphoroute/runtime/wsl_relay.py`
- Create: `scripts/bootstrap_environments.py`
- Modify: `scripts/setup_wsl.sh`
- Modify: `environment/requirements-main.txt`
- Modify: `environment/requirements-yolo.txt`
- Modify: `environment/requirements-stats.txt`
- Test: `tests/test_environment_relay.py`

**Interfaces:**
- Produces: `EnvironmentKind`, `EnvironmentResolver.for_experiment()`, `windows_to_wsl_path()`, `build_wsl_command()`, `ensure_environment()`.
- Consumes: experiment ID, project path, data path, requested profile.

- [ ] **Step 1: Write failing relay tests**

```python
def test_windows_path_is_converted_without_losing_chinese_text():
    assert windows_to_wsl_path(
        Path(r"C:\Users\Baoju\Desktop\实验代码")
    ) == PurePosixPath("/mnt/c/Users/Baoju/Desktop/实验代码")

def test_exp05_resolves_to_main_environment():
    assert EnvironmentResolver.for_experiment("EXP-05") is EnvironmentKind.MAIN

def test_wsl_command_passes_arguments_as_individual_tokens():
    command = build_wsl_command(project, entry, ["--profile", "local_6gb"])
    assert command[:4] == ["wsl.exe", "-d", "Ubuntu-22.04", "--"]
    assert "local_6gb" in command
```

- [ ] **Step 2: Run RED**

```powershell
python -m pytest -q tests/test_environment_relay.py
```

- [ ] **Step 3: Implement deterministic environment resolution**

Environment mapping:

```python
YOLO_EXPERIMENTS = {"EXP-02", "EXP-03"}
MAIN_EXPERIMENTS = {
    "EXP-04", "EXP-05", "EXP-06", "EXP-07",
    "EXP-08A", "EXP-08B", "EXP-08D", "EXP-09", "EXP-10", "EXP-11", "EXP-14",
}
STATS_EXPERIMENTS = {"EXP-00", "EXP-01", "EXP-08C", "EXP-12", "EXP-13", "EXP-15"}
```

The Windows shim re-executes the same entry in WSL. A relay guard environment variable prevents recursion. Environment manifests record exact package and CUDA versions.

- [ ] **Step 4: Implement idempotent bootstrap**

`bootstrap_environments.py` verifies Python, WSL distribution, NVIDIA passthrough, virtual environments, pinned packages, and imports. It installs only a missing environment and writes `environment/environment-lock-<kind>.json`.

- [ ] **Step 5: Run GREEN**

```powershell
python -m pytest -q tests/test_environment_relay.py
```

- [ ] **Step 6: Record checkpoint**

Save `verification/task02-environment-relay.txt`.

---

### Task 3: Acyclic prerequisite orchestration and staged EXP-12

**Files:**
- Create: `src/morphoroute/core/dependencies.py`
- Modify: `src/morphoroute/core/registry.py`
- Modify: `src/morphoroute/core/runner.py`
- Modify: `src/morphoroute/experiments/exp12.py`
- Test: `tests/test_dependency_graph.py`

**Interfaces:**
- Produces: `ArtifactRequirement`, `ExecutionNode`, `ExecutionGraph.plan()`, `ensure_prerequisites()`.
- EXP-12 produces `prefreeze` and `external` stage manifests.

- [ ] **Step 1: Write failing dependency tests**

```python
def test_every_execution_target_has_an_acyclic_plan():
    graph = ExecutionGraph.from_registry()
    for experiment_id in EXPERIMENTS:
        plan = graph.plan(experiment_id)
        assert len(plan) == len(set(plan))

def test_exp11_requires_freeze_not_exp09_directly():
    graph = ExecutionGraph.from_registry()
    assert graph.requirements("EXP-11") == (
        ArtifactRequirement("EXP-10", "complete"),
    )

def test_exp12_stages_break_the_old_cycle():
    graph = ExecutionGraph.from_registry()
    assert graph.plan("EXP-12:prefreeze")[-1] == "EXP-12:prefreeze"
    assert graph.plan("EXP-12:external")[-2:] == ["EXP-11", "EXP-12:external"]
```

- [ ] **Step 2: Run RED**

```powershell
python -m pytest -q tests/test_dependency_graph.py
```

Expected: old EXP-10/EXP-12 cycle is detected.

- [ ] **Step 3: Implement graph and artifact-aware reuse**

The graph is topologically sorted. Reuse requires a completed manifest with matching data, configuration, code, and required-output hashes. Missing artifacts invoke the prerequisite handler inside the same process/WSL environment transition.

- [ ] **Step 4: Split EXP-12 internally**

`prefreeze` fits D2 temperature scaling and validates on D3. `external` applies the frozen calibrator to sealed D4 predictions. The public filename and Chinese title remain unchanged.

- [ ] **Step 5: Run GREEN**

```powershell
python -m pytest -q tests/test_dependency_graph.py tests/test_registry.py tests/test_d4_guard.py
```

- [ ] **Step 6: Record checkpoint**

Save `verification/task03-dependency-graph.txt`.

---

### Task 4: Hash-addressed YOLO and COCO dataset materialization

**Files:**
- Create: `src/morphoroute/data/materialize.py`
- Create: `src/morphoroute/data/coco_dataset.py`
- Create: `src/morphoroute/data/yolo_dataset.py`
- Modify: `src/morphoroute/data/export.py`
- Modify: `src/morphoroute/data/annotations.py`
- Test: `tests/test_materialization.py`

**Interfaces:**
- Produces: `DatasetFingerprint`, `MaterializedDataset`, `ensure_yolo_dataset()`, `ensure_coco_dataset()`.
- Consumes: validated manifest records and allowed split tuple.

- [ ] **Step 1: Write failing synthetic materialization tests**

```python
def test_coco_materialization_preserves_all_instances_and_provenance(tmp_path):
    dataset = synthetic_five_class_dataset(tmp_path)
    result = ensure_coco_dataset(dataset.manifest, tmp_path / "derived", ("D1", "D2", "D3"))
    coco = json.loads(result.annotation_file.read_text(encoding="utf-8"))
    assert {category["name"] for category in coco["categories"]} == set(CLASS_NAMES)
    assert all("group_id" in image for image in coco["images"])
    assert len(coco["annotations"]) == dataset.instance_count

def test_repeated_materialization_reuses_matching_hash(tmp_path):
    first = ensure_yolo_dataset(manifest, output, ("D1", "D2", "D3"))
    second = ensure_yolo_dataset(manifest, output, ("D1", "D2", "D3"))
    assert first.fingerprint == second.fingerprint
    assert second.reused
```

- [ ] **Step 2: Run RED**

```powershell
python -m pytest -q tests/test_materialization.py
```

- [ ] **Step 3: Implement YOLO materialization**

Create `images/{train,val,test}`, `labels/{train,val,test}`, and `data.yaml`. Use hard links when possible and copy otherwise. Polygon labels use normalized coordinates and fixed class order.

- [ ] **Step 4: Implement COCO materialization**

Create split-specific COCO JSON plus a combined provenance index. Preserve record ID, source, tunnel, sequence, group ID, original paths, dimensions, and annotation hashes.

- [ ] **Step 5: Run GREEN**

```powershell
python -m pytest -q tests/test_materialization.py tests/test_annotations.py tests/test_manifest.py
```

- [ ] **Step 6: Record checkpoint**

Save `verification/task04-data-materialization.txt`.

---

### Task 5: Canonical prediction bundle, unified evaluator, tables, and figures

**Files:**
- Create: `src/morphoroute/evaluation/schema.py`
- Create: `src/morphoroute/evaluation/coco_eval.py`
- Create: `src/morphoroute/evaluation/structure_eval.py`
- Create: `src/morphoroute/evaluation/resource_eval.py`
- Create: `src/morphoroute/evaluation/pipeline.py`
- Modify: `src/morphoroute/visualization/plots.py`
- Test: `tests/test_evaluation_pipeline.py`

**Interfaces:**
- Produces: `PredictionInstance`, `PredictionImage`, `PredictionBundle`, `evaluate_bundle()`, `write_evaluation_outputs()`.
- Consumes: framework-specific prediction adapters.

- [ ] **Step 1: Write failing evaluator tests**

```python
def test_perfect_prediction_produces_perfect_mask_and_structure_scores(tmp_path):
    bundle = perfect_synthetic_bundle()
    result = evaluate_bundle(bundle, split="D3")
    assert result.mask_ap50 == pytest.approx(1.0)
    assert result.crack_cldice == pytest.approx(1.0)
    assert result.area_boundary_f1 == pytest.approx(1.0)

def test_output_writer_creates_registered_formats(tmp_path):
    paths = write_evaluation_outputs(result, tmp_path)
    assert paths.metrics_csv.is_file()
    assert paths.metrics_xlsx.is_file()
    assert paths.summary_json.is_file()
    assert paths.overlay_png.is_file()
```

- [ ] **Step 2: Run RED**

```powershell
python -m pytest -q tests/test_evaluation_pipeline.py
```

- [ ] **Step 3: Implement canonical schema and adapters**

Every prediction stores image ID, group, domain, class, score, box, RLE/polygon mask, logits, query ID, route probabilities, active experts, and latency when available.

- [ ] **Step 4: Implement common, structural, and resource metrics**

Use pycocotools for AP. Aggregate clDice only for crack instances and boundary metrics for area-like diseases. Preserve N/A when a domain lacks a class.

- [ ] **Step 5: Implement output validation**

Required output schemas are checked before a run becomes complete.

- [ ] **Step 6: Run GREEN**

```powershell
python -m pytest -q tests/test_evaluation_pipeline.py tests/test_metrics.py tests/test_outputs.py
```

- [ ] **Step 7: Record checkpoint**

Save `verification/task05-evaluation.txt`.

---

### Task 6: Real Ultralytics training, checkpoint selection, and evaluation

**Files:**
- Modify: `src/morphoroute/runtime/ultralytics_adapter.py`
- Modify: `src/morphoroute/experiments/exp02.py`
- Modify: `src/morphoroute/experiments/exp03.py`
- Create: `src/morphoroute/runtime/ultralytics_predictions.py`
- Test: `tests/test_yolo_end_to_end.py`

**Interfaces:**
- Produces: `UltralyticsRunResult`, `train_validate_predict_yolo()`.
- Consumes: `MaterializedDataset` and `EntrySettings`.

- [ ] **Step 1: Write failing end-to-end adapter tests**

Use an injected backend protocol that records real plan values without mocking project logic:

```python
def test_yolo_pipeline_trains_on_d1_selects_on_d2_and_predicts_d3(tmp_path):
    backend = RecordingUltralyticsBackend()
    result = train_validate_predict_yolo(config, materialized, backend=backend)
    assert backend.train_split == "D1"
    assert backend.validation_split == "D2"
    assert backend.prediction_split == "D3"
    assert result.best_checkpoint.is_file()
    assert result.predictions.split == "D3"
```

- [ ] **Step 2: Run RED**

```powershell
python -m pytest -q tests/test_yolo_end_to_end.py
```

- [ ] **Step 3: Implement real training**

Map optimizer updates to epochs using D1 size, physical batch, accumulation, and world size. Save exact Ultralytics arguments and version. Resume only compatible runs.

- [ ] **Step 4: Implement D2 best-checkpoint selection and frozen D3 prediction**

Convert Ultralytics results to `PredictionBundle`, then invoke the unified evaluator.

- [ ] **Step 5: Remove planning-only default**

EXP-02/03 execute training by default. `--dry-run` remains explicit. EXP-02 retains evidence-level labelling.

- [ ] **Step 6: Run GREEN**

```powershell
python -m pytest -q tests/test_yolo_end_to_end.py tests/test_experiments_00_07.py
```

- [ ] **Step 7: Record checkpoint**

Save `verification/task06-yolo.txt`.

---

### Task 7: Complete MMDetection baseline configuration and execution

**Files:**
- Create: `src/morphoroute/mmdet/config_builder.py`
- Create: `src/morphoroute/mmdet/datasets.py`
- Create: `src/morphoroute/mmdet/hooks.py`
- Create: `src/morphoroute/mmdet/predictions.py`
- Modify: `src/morphoroute/runtime/mmdet_adapter.py`
- Modify: `src/morphoroute/experiments/exp04.py`
- Modify: `src/morphoroute/experiments/exp05.py`
- Test: `tests/test_mmdet_end_to_end.py`

**Interfaces:**
- Produces: `MMDetResolvedConfig`, `MMDetRunResult`, `build_mask_rcnn_config()`, `build_mask2former_config()`, `train_validate_predict_mmdet()`.
- Consumes: COCO materialization and experiment profile.

- [ ] **Step 1: Write failing complete-config tests**

```python
REQUIRED_CFG_KEYS = {
    "model", "train_dataloader", "val_dataloader", "test_dataloader",
    "train_cfg", "val_cfg", "test_cfg", "optim_wrapper",
    "param_scheduler", "val_evaluator", "test_evaluator",
    "default_hooks", "visualizer", "work_dir",
}

def test_mask2former_config_is_runner_complete():
    cfg = build_mask2former_config(config, dataset)
    assert REQUIRED_CFG_KEYS <= set(cfg)
    assert cfg["model"]["panoptic_head"]["num_things_classes"] == 5
    assert cfg["train_dataloader"]["dataset"]["ann_file"].endswith("D1.json")
    assert cfg["val_dataloader"]["dataset"]["ann_file"].endswith("D2.json")
```

- [ ] **Step 2: Run RED**

```powershell
python -m pytest -q tests/test_mmdet_end_to_end.py
```

- [ ] **Step 3: Build complete official-derived configurations**

Use MMDetection 3.3.x base recipes and override all five-class dataset, loop, seed, input, optimizer, accumulation, checkpoint, evaluator, and output settings. Save the fully resolved Python config.

- [ ] **Step 4: Implement runner execution and prediction conversion**

Train on D1, validate on D2, select best D2 checkpoint, load it once, predict D3, convert to `PredictionBundle`, and evaluate.

- [ ] **Step 5: Make EXP-04/05 real by default**

Remove `status="planned"` from normal execution. A completed status requires checkpoints, D3 predictions, metrics, and resources.

- [ ] **Step 6: Run GREEN**

```powershell
python -m pytest -q tests/test_mmdet_end_to_end.py tests/test_adapters.py
```

- [ ] **Step 7: Record checkpoint**

Save `verification/task07-mmdet-baselines.txt`.

---

### Task 8: Registered MorphoRoute Mask2Former head and B0-B11 factory

**Files:**
- Create: `src/morphoroute/mmdet/models/__init__.py`
- Create: `src/morphoroute/mmdet/models/morphology_targets.py`
- Create: `src/morphoroute/mmdet/models/query_router.py`
- Create: `src/morphoroute/mmdet/models/line_expert.py`
- Create: `src/morphoroute/mmdet/models/area_expert.py`
- Create: `src/morphoroute/mmdet/models/morphoroute_head.py`
- Create: `src/morphoroute/mmdet/models/variant_factory.py`
- Modify: `src/morphoroute/models/variants.py`
- Test: `tests/test_morphoroute_head.py`
- Test: `tests/test_variant_factory_runtime.py`

**Interfaces:**
- Produces: registered `MorphoRouteMask2FormerHead`, `MorphologyTargetEncoder`, `QueryRouter`, `LineExpert`, `AreaExpert`, `build_variant_model_cfg()`.
- Consumes: MMDetection Mask2Former query features and Hungarian matching results.

- [ ] **Step 1: Write failing morphology-target tests**

```python
def test_only_matched_queries_receive_route_targets():
    encoder = MorphologyTargetEncoder(gmm=fitted_test_gmm())
    targets, weights = encoder(masks, matched_query_indices=[2, 7], query_count=10)
    assert weights.nonzero().flatten().tolist() == [2, 7]
    assert torch.allclose(targets.sum(dim=-1)[weights.bool()], torch.ones(2))
```

- [ ] **Step 2: Write failing head behavior tests**

```python
def test_b9_training_returns_segmentation_structure_and_route_losses():
    head = build_test_head("B9")
    losses = head.loss(features, samples)
    assert {"loss_cls", "loss_mask", "loss_dice", "loss_route",
            "loss_cldice", "loss_boundary"} <= set(losses)

def test_b9_inference_uses_rgb_derived_queries_only():
    head = build_test_head("B9").eval()
    output = head.predict(features, image_meta)
    assert output.route_probabilities.shape[-1] == 2
    assert output.ground_truth_inputs_used is False
```

- [ ] **Step 3: Run RED**

```powershell
python -m pytest -q tests/test_morphoroute_head.py tests/test_variant_factory_runtime.py
```

- [ ] **Step 4: Implement target encoder and router**

The target encoder receives only matched D1 masks during training. The router maps query features to two probabilities and computes supervised route loss plus load-balance regularization.

- [ ] **Step 5: Implement heterogeneous experts and sparse fusion**

`LineExpert` uses directional/deformable sampling for continuity. `AreaExpert` uses multi-scale context and boundary refinement. B9 applies Top-1 at high confidence and both experts below frozen threshold `tau`.

- [ ] **Step 6: Implement real B0-B11 configurations**

Every variant changes only the registered fields defined by the SOP. Runtime tests instantiate every model and execute a forward/backward pass.

- [ ] **Step 7: Run GREEN**

```powershell
python -m pytest -q tests/test_morphoroute_head.py tests/test_variant_factory_runtime.py tests/test_variants.py
```

- [ ] **Step 8: Record checkpoint**

Save `verification/task08-morphoroute-models.txt`.

---

### Task 9: Three-stage training, PCGrad, checkpoint resume, and diagnostics

**Files:**
- Create: `src/morphoroute/training/stages.py`
- Create: `src/morphoroute/training/engine.py`
- Create: `src/morphoroute/training/checkpoints.py`
- Create: `src/morphoroute/training/pcgrad_wrapper.py`
- Create: `src/morphoroute/training/diagnostics.py`
- Modify: `src/morphoroute/runtime/checkpoint.py`
- Test: `tests/test_training_engine.py`

**Interfaces:**
- Produces: `TrainingStage`, `StageSchedule`, `TrainingEngine.run()`, `ResumeState`, `QueryDiagnosticWriter`.
- Consumes: MMEngine runner, variant config, run manifest.

- [ ] **Step 1: Write failing stage and resume tests**

```python
def test_schedule_transitions_a_to_b_to_c_at_registered_updates():
    schedule = StageSchedule(warmup_end=1000, sparse_start=40000, max_updates=50000)
    assert schedule.stage_at(0) is TrainingStage.A
    assert schedule.stage_at(1000) is TrainingStage.B
    assert schedule.stage_at(40000) is TrainingStage.C

def test_resume_restores_rng_optimizer_scaler_and_stage(tmp_path):
    resumed = roundtrip_checkpoint(make_training_state(update=1234), tmp_path)
    assert resumed.update == 1234
    assert resumed.stage is TrainingStage.B
    assert resumed.rng_hash == original.rng_hash
```

- [ ] **Step 2: Run RED**

```powershell
python -m pytest -q tests/test_training_engine.py
```

- [ ] **Step 3: Implement stage controller**

Stage transitions toggle route supervision, expert activation, and frozen `tau` without rebuilding unrelated weights.

- [ ] **Step 4: Implement PCGrad wrapper**

B11 projects only negative task-gradient dot products after AMP unscale and before clipping.

- [ ] **Step 5: Implement diagnostics**

At registered checkpoints, export query logits, matched targets, morphology, route probabilities, expert outputs, load, entropy, losses, and shared-gradient summaries.

- [ ] **Step 6: Run GREEN**

```powershell
python -m pytest -q tests/test_training_engine.py tests/test_checkpoint_resume.py tests/test_torch_components.py
```

- [ ] **Step 7: Record checkpoint**

Save `verification/task09-training-engine.txt`.

---

### Task 10: Real execution for EXP-06, EXP-07, and EXP-09

**Files:**
- Modify: `src/morphoroute/experiments/exp06.py`
- Modify: `src/morphoroute/experiments/exp07.py`
- Modify: `src/morphoroute/experiments/exp09.py`
- Create: `src/morphoroute/training/resource_matching.py`
- Test: `tests/test_resource_and_ablation_execution.py`

**Interfaces:**
- Produces: `match_parameter_capacity()`, `match_flops()`, `run_variant_matrix()`.
- Consumes: B9 resource profile and B0-B11 factory.

- [ ] **Step 1: Write failing execution tests**

```python
def test_exp09_invokes_training_for_all_required_variants():
    backend = RecordingVariantBackend()
    run_exp09(config, backend=backend, smoke_test=True)
    assert backend.started_variants == list(VARIANT_IDS)
    assert all(backend.has_d3_predictions(v) for v in VARIANT_IDS)

def test_capacity_and_flops_match_within_registered_tolerance():
    matches = build_resource_matches(b0, b9, tolerance=0.05)
    assert abs(matches.b2.parameters / b9.parameters - 1) <= 0.05
    assert abs(matches.b3.flops / b9.flops - 1) <= 0.05
```

- [ ] **Step 2: Run RED**

```powershell
python -m pytest -q tests/test_resource_and_ablation_execution.py
```

- [ ] **Step 3: Implement B2/B3 matching search**

Search only registered decoder widths/depths using static D2-independent resource calculations. Save all candidates and the selected match.

- [ ] **Step 4: Implement EXP-07 runs**

Fit GMM on D1, then train and evaluate B4, B5, and B6 with common resource and output protocols.

- [ ] **Step 5: Implement EXP-09 serial matrix**

Run variants and seeds deterministically, resume partial runs, evaluate D3, and produce complete ablation and pairwise tables.

- [ ] **Step 6: Run GREEN**

```powershell
python -m pytest -q tests/test_resource_and_ablation_execution.py tests/test_experiments_00_07.py tests/test_experiments_08_11.py
```

- [ ] **Step 7: Record checkpoint**

Save `verification/task10-ablation.txt`.

---

### Task 11: Mechanism experiments EXP-08A through EXP-08D

**Files:**
- Modify: `src/morphoroute/experiments/exp08a.py`
- Modify: `src/morphoroute/experiments/exp08b.py`
- Modify: `src/morphoroute/experiments/exp08c.py`
- Modify: `src/morphoroute/experiments/exp08d.py`
- Create: `src/morphoroute/analysis/gradient_conflict.py`
- Create: `src/morphoroute/analysis/morphology_association.py`
- Create: `src/morphoroute/analysis/expert_interventions.py`
- Test: `tests/test_mechanism_execution.py`

**Interfaces:**
- Produces: `collect_gradient_conflict()`, `run_m2_models()`, `fit_m3_association()`, `run_expert_interventions()`.
- Consumes: frozen checkpoints and query diagnostics.

- [ ] **Step 1: Write failing mechanism tests**

```python
def test_m1_uses_same_checkpoint_batch_and_parameter_subset():
    result = collect_gradient_conflict(model, batch, parameter_names)
    assert result.line.checkpoint_hash == result.area.checkpoint_hash
    assert result.line.batch_hash == result.area.batch_hash
    assert result.parameter_names == parameter_names

def test_m4_forced_routes_preserve_nonexpert_computation():
    outputs = run_expert_interventions(model, batch)
    assert outputs.normal.shared_path_hash == outputs.force_area.shared_path_hash
    assert outputs.normal.route_mode != outputs.force_area.route_mode
```

- [ ] **Step 2: Run RED**

```powershell
python -m pytest -q tests/test_mechanism_execution.py
```

- [ ] **Step 3: Implement M1 and M2**

Collect unscaled pre-clipping gradients at registered early/middle/late updates. Train shared, independent, hard-split, and PCGrad diagnostic models with disclosed resources.

- [ ] **Step 4: Implement M3**

Build sequence-level tables, control class and log area, use cluster-robust/paired bootstrap inference, and generate partial-dependence/bin-trend figures.

- [ ] **Step 5: Implement M4**

Run normal, close-line, close-area, swap, force-line, and force-area paired predictions on preselected D3 queries.

- [ ] **Step 6: Run GREEN**

```powershell
python -m pytest -q tests/test_mechanism_execution.py tests/test_experiments_08_11.py
```

- [ ] **Step 7: Record checkpoint**

Save `verification/task11-mechanisms.txt`.

---

### Task 12: Freeze, D4 one-shot inference, and staged reliability

**Files:**
- Modify: `src/morphoroute/experiments/exp10.py`
- Modify: `src/morphoroute/experiments/exp11.py`
- Modify: `src/morphoroute/experiments/exp12.py`
- Modify: `src/morphoroute/core/d4_guard.py`
- Create: `src/morphoroute/evaluation/external.py`
- Test: `tests/test_freeze_external_reliability.py`

**Interfaces:**
- Produces: `build_freeze_manifest()`, `run_d4_once()`, `run_reliability_prefreeze()`, `run_reliability_external()`.
- Consumes: frozen checkpoints, predictions, thresholds, calibrator, and evaluator hash.

- [ ] **Step 1: Write failing freeze-chain tests**

```python
def test_d4_run_requires_all_frozen_artifacts_and_prefreeze_calibrator():
    with pytest.raises(FreezeValidationError):
        run_d4_once(incomplete_freeze_manifest)

def test_external_reliability_never_refits_temperature():
    calibrator = FrozenCalibrator(temperature=1.7)
    result = run_reliability_external(d4_predictions, calibrator)
    assert result.temperature == 1.7
    assert result.fit_calls == 0
```

- [ ] **Step 2: Run RED**

```powershell
python -m pytest -q tests/test_freeze_external_reliability.py
```

- [ ] **Step 3: Implement complete EXP-10 freeze**

Freeze candidate and baselines, checkpoints, code/config/environment hashes, GMM, `tau`, temperature, confidence/NMS/mask thresholds, overlap/stitching, post-processing, and evaluator.

- [ ] **Step 4: Implement one-shot multi-model D4 inference**

Verify the exact phrase, write the started marker before prediction, run all preregistered models through the same frozen inference/evaluation path, and preserve combined and domain results.

- [ ] **Step 5: Implement staged EXP-12**

Prefreeze uses D2/D3. External uses the frozen calibrator on EXP-11 predictions. Both write explicit stage manifests.

- [ ] **Step 6: Run GREEN**

```powershell
python -m pytest -q tests/test_freeze_external_reliability.py tests/test_d4_guard.py
```

- [ ] **Step 7: Record checkpoint**

Save `verification/task12-freeze-external.txt`.

---

### Task 13: Quantification, sequence efficiency, and final statistics

**Files:**
- Modify: `src/morphoroute/experiments/exp13.py`
- Modify: `src/morphoroute/experiments/exp14.py`
- Modify: `src/morphoroute/experiments/exp15.py`
- Create: `src/morphoroute/analysis/quantification_pipeline.py`
- Create: `src/morphoroute/analysis/sequence_benchmark.py`
- Create: `src/morphoroute/analysis/final_statistics.py`
- Test: `tests/test_late_experiment_execution.py`

**Interfaces:**
- Produces: `run_quantification()`, `benchmark_sequences()`, `aggregate_final_results()`.
- Consumes: frozen prediction bundles and reference metadata.

- [ ] **Step 1: Write failing late-stage tests**

```python
def test_quantification_without_scale_never_emits_physical_units(tmp_path):
    result = run_quantification(predictions, metadata_without_scale, tmp_path)
    assert result.physical_complete is False
    assert all(row["unit"] == "pixel" for row in result.measurements)
    assert result.template_path.is_file()

def test_final_aggregation_rejects_smoke_runs():
    with pytest.raises(ValueError, match="smoke"):
        aggregate_final_results([formal_run, smoke_run])
```

- [ ] **Step 2: Run RED**

```powershell
python -m pytest -q tests/test_late_experiment_execution.py
```

- [ ] **Step 3: Implement EXP-13**

Compute skeleton arc length, local normal width, and area. Use physical units only with valid scale/Jacobian and reference fields. Generate a fillable template when they are absent.

- [ ] **Step 4: Implement EXP-14**

Use fixed sequences, warm-up, CUDA synchronization, at least three batch-1 rounds, and stage-level timers. Report peak VRAM, throughput, active experts, dual activation, and continuous missed length.

- [ ] **Step 5: Implement EXP-15**

Aggregate every seed and sequence, preserve failures, run paired group bootstrap, effect sizes, registered Holm comparisons, M/E status, final tables/figures, and reproducibility hashes.

- [ ] **Step 6: Run GREEN**

```powershell
python -m pytest -q tests/test_late_experiment_execution.py tests/test_experiments_12_15.py tests/test_statistics.py
```

- [ ] **Step 7: Record checkpoint**

Save `verification/task13-late-experiments.txt`.

---

### Task 14: Full synthetic, real-data, WSL, and RTX 4050 verification

**Files:**
- Create: `scripts/run_all_smoke.py`
- Create: `scripts/verify_installation.py`
- Modify: `README.md`
- Modify: `docs/实验运行指南.md`
- Modify: `docs/实验与输出对照表.md`
- Test: `tests/test_smoke_orchestration.py`

**Interfaces:**
- Produces: a machine-readable final verification report.
- Consumes: all preceding components.

- [ ] **Step 1: Write failing smoke orchestration tests**

```python
def test_all_nineteen_entries_are_in_smoke_matrix():
    matrix = build_smoke_matrix()
    assert [item.experiment_id for item in matrix] == list(EXPERIMENTS)

def test_smoke_artifacts_are_rejected_by_formal_freeze():
    with pytest.raises(ValueError, match="formal"):
        build_freeze_manifest(smoke_artifacts)
```

- [ ] **Step 2: Run RED**

```powershell
python -m pytest -q tests/test_smoke_orchestration.py
```

- [ ] **Step 3: Implement all-entry synthetic smoke runner**

Generate a deterministic five-class D1-D4 synthetic dataset, execute all 19 workflows with tiny budgets, and verify every registered artifact schema.

- [ ] **Step 4: Run GREEN and the full unit/integration suite**

```powershell
python -m pytest -q
python -m compileall -q src scripts experiments tests
```

Expected: zero failures and zero unexpected skips.

- [ ] **Step 5: Run real-data preparation and CPU/GPU forward verification**

```powershell
python experiments/exp01_data_audit.py --smoke-test
python scripts/verify_installation.py --profile local_6gb --real-data
```

Expected: 9,370 records; no missing or hash mismatches; successful batch loading and model forward passes.

- [ ] **Step 6: Run RTX 4050 training-path smoke tests**

In WSL ENV-YOLO/ENV-MAIN, run train-validation-inference-evaluation smoke workflows for:

- YOLOv8n-seg;
- Mask R-CNN;
- Mask2Former B0;
- MorphoRoute B9.

Each must perform at least two optimizer updates, save/resume a checkpoint, predict D3 smoke samples, compute metrics, and generate a figure.

- [ ] **Step 7: Verify no-argument IDE behavior**

Execute representative entry files with no arguments from the project root and from an unrelated working directory. Confirm project/data discovery and automatic prerequisite resolution.

- [ ] **Step 8: Update Chinese user documentation**

Document the visible constants, click-Run workflow, first-run WSL bootstrap, expected runtime, resume behavior, outputs, D4 confirmation, EXP-13 metadata boundary, and formal-versus-smoke distinction.

- [ ] **Step 9: Final integrity audit**

```powershell
rg -n -i "TBD|TODO|implement later|fill in details|待实现|待补充" src scripts experiments configs docs README.md
python scripts/verify_installation.py --profile local_6gb --report verification/final-report.json
```

Expected: no implementation placeholders and a passing final report.

- [ ] **Step 10: Record final checkpoint**

Save:

- `verification/final-report.json`;
- `verification/test-output.txt`;
- `verification/gpu-smoke-report.json`;
- exact environment lock files and hashes.

The original ZIP remains unchanged. The extracted desktop project is the corrected deliverable.

---

## Plan Self-Review

### Spec coverage

- Standalone no-argument entries: Tasks 1-3.
- Windows/WSL automation: Task 2.
- Automatic data reading and materialization: Task 4.
- Real YOLO and official baselines: Tasks 6-7.
- MorphoRoute and B0-B11: Tasks 8-10.
- M1-M4: Task 11.
- Freeze, D4, and reliability: Task 12.
- Quantification, efficiency, and statistics: Task 13.
- Real-data and RTX 4050 acceptance: Task 14.

### Type consistency

- `EntrySettings` flows from Task 1 into environment, data, and runtime layers.
- `MaterializedDataset` from Task 4 is consumed by Tasks 6-10.
- `PredictionBundle` from Task 5 is produced by all model backends and consumed by Tasks 11-13.
- `ArtifactRequirement` and stage manifests from Task 3 govern all prerequisite reuse.

### Placeholder check

The plan contains no deferred implementation marker. Every task identifies its concrete files, public interfaces, failing behavior, implementation contract, and verification command.
