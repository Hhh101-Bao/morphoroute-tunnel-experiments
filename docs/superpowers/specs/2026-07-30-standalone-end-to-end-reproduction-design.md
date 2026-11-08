# Standalone End-to-End Experiment Reproduction Design

## 1. Goal

Convert the existing railway-tunnel multi-disease instance-segmentation project from a planning-oriented experiment scaffold into a genuinely executable reproduction package.

Each of the 19 experiment entry files must be independently launchable from PyCharm, VS Code, or a terminal. Running one entry must automatically load the registered dataset, resolve and execute required prerequisites, train or resume the required models, select checkpoints on D2, evaluate on the experiment's allowed splits, and generate the SOP-mandated tables and figures. The user must never need to combine several Python files manually.

The existing Chinese folder names, archive name, SOP document name, and Chinese experiment titles must remain unchanged.

## 2. Target Project

The implementation target is the extracted project:

`C:\Users\Baoju\Desktop\实验代码\铁路隧道多病害实例分割实验代码\railway_tunnel_experiments`

The original ZIP and the SOP DOCX remain untouched as rollback and requirements references.

The authoritative dataset root is:

`C:\Users\Baoju\Desktop\铁路隧道多病害实例分割_原始影像严格划分_v1`

`split_manifest.csv` remains the only authoritative sample index.

## 3. Standalone Entry Contract

Every `experiments/exp*.py` file contains a short editable settings block and calls the same shared launcher:

```python
PROFILE = "local_6gb"
SMOKE_TEST = False
AUTO_RUN_PREREQUISITES = True
RESUME = True

if __name__ == "__main__":
    raise SystemExit(run_standalone_experiment("EXP-05", ...))
```

The default no-argument execution contract is:

1. Locate the project and dataset roots.
2. Validate the manifest, data version, profile, and split permissions.
3. Detect Windows or Linux execution.
4. Select ENV-YOLO, ENV-MAIN, or ENV-STATS.
5. On Windows, relay GPU experiments to WSL2 with converted paths.
6. Verify or bootstrap the required environment.
7. Resolve the experiment dependency graph.
8. Reuse only completed upstream artifacts whose configuration and data hashes still match.
9. Automatically run missing or invalid prerequisites inside the same invocation.
10. Prepare the required derived dataset format.
11. Train or resume the required model and seeds.
12. Select the best checkpoint using D2 only.
13. Freeze thresholds and run the permitted D3 or D4 inference.
14. Compute metrics, resources, tables, figures, and hashes.
15. Mark the experiment complete only after all required outputs pass validation.

Command-line options remain available, but are not required for normal use.

## 4. Profiles

### 4.1 `local_6gb`

- Intended for the RTX 4050 Laptop GPU with 6 GB VRAM.
- Input size: 640 x 640.
- Physical batch: 1.
- Effective batch: 8 through gradient accumulation.
- AMP enabled.
- Activation checkpointing enabled where supported.
- Two data-loader workers by default.
- Models and seeds run serially.
- No silent architecture degradation is allowed after an out-of-memory error.

### 4.2 `sop_full`

- Input size: 1024 x 1024.
- Training budget: 50,000 optimizer updates.
- Effective batch: 8.
- Preserves the SOP formal protocol.
- May require a GPU with at least 24 GB VRAM or substantially longer CPU/offloaded execution.

### 4.3 `local_6gb_fallback`

- Intended only for debugging and pre-experiment validation.
- Its results cannot be used as formal paper results.

### 4.4 `--smoke-test`

- Uses a deterministic small subset and a small update budget.
- Exercises preparation, forward/backward, checkpointing, validation, inference, metrics, tables, and figures.
- Smoke-test outputs are permanently labelled and cannot enter final statistics.

## 5. Data Pipeline

The shared data layer performs:

1. Full manifest schema and path validation.
2. Image and annotation SHA-256 verification.
3. TACK single-channel PNG parsing and DACL polygon JSON parsing.
4. Five-class category normalization.
5. Instance validation, including empty masks, degenerate polygons, bounds, and disconnected components.
6. Leakage checks using `source`, `tunnel`, `sequence`, and `group_id`.
7. Hash-addressed generation of:
   - Ultralytics YOLO segmentation datasets;
   - COCO instance-segmentation datasets;
   - query/morphology analysis tables.
8. Derived-data caching with provenance back to every manifest row.

An experiment may access only the splits permitted by its SOP card. D4 metadata, labels, or predictions cannot be exposed before EXP-10 completes.

## 6. Framework Strategy

### 6.1 YOLO

EXP-02 and EXP-03 use the pinned Ultralytics YOLOv8n-seg implementation.

EXP-02 uses the archived original configuration when available. If the original version, weights, or logs are unavailable, it is labelled `public_default_reconstruction`, matching the SOP evidence boundary.

EXP-03 changes only the strict split and explicitly registered corrections. Both experiments perform real training, D2 checkpoint selection where applicable, frozen D3 inference, unified evaluation, resource measurement, and prediction export.

YOLO results remain restricted to RQ1 and are excluded from MorphoRoute candidate selection.

### 6.2 Mask R-CNN and Mask2Former

EXP-04 and EXP-05 use pinned MMDetection 3.3.x configurations derived from official Mask R-CNN and Mask2Former recipes.

The project supplies complete model, dataset, pipeline, optimizer, scheduler, loop, evaluator, hook, and checkpoint configurations. The current partial `Runner.from_cfg` mapping is replaced by a valid resolved MMDetection configuration.

### 6.3 MorphoRoute-Tunnel

MorphoRoute is implemented as registered MMDetection modules that reuse the same Mask2Former backbone, pixel decoder, and shared query decoder used by B0.

The extension supplies:

- five-dimensional morphology targets computed from matched D1 masks;
- a D1-only standardized two-component GMM;
- query-level soft morphology routing;
- a line expert;
- an area expert;
- low-confidence dual activation;
- Top-1 sparse inference;
- matched-query-only routing supervision;
- clDice and boundary-aware structural losses;
- optional PCGrad;
- intervention hooks and query-level diagnostic export.

Inference consumes RGB only. Ground-truth class, masks, morphology targets, and GMM component labels cannot enter the inference path.

## 7. Training Stages

MorphoRoute training follows the SOP:

1. Stage A: shared Mask2Former warm-up.
2. Stage B: morphology soft routing and dual-expert joint training.
3. Stage C: freeze the D2-selected routing threshold and train or fine-tune with Top-1 sparse activation plus registered low-confidence dual activation.

Checkpoints include model, optimizer, scheduler, AMP scaler, update number, training stage, RNG states, configuration hash, dataset hash, and initial-weight hash.

Re-running the same experiment with `RESUME = True` resumes the newest compatible checkpoint. A mismatched profile, data hash, or configuration is rejected rather than silently resumed.

## 8. B0-B11 Model Factory

The model factory constructs real executable variants:

- B0: shared Mask2Former;
- B1: B0 plus structural losses;
- B2: parameter-capacity-matched shared decoder;
- B3: FLOPs-matched shared decoder;
- B4: generic two-expert MoE;
- B5: class hard-routing diagnostic control;
- B6: morphology routing with homogeneous experts;
- B7: B6 plus the line expert;
- B8: B6 plus the area expert;
- B9: full MorphoRoute-Tunnel;
- B10: B9 without low-confidence dual activation;
- B11: B9 plus PCGrad.

Each variant records an audited diff from B0. EXP-09 trains the required variants and seeds serially, performs D2 selection and D3 inference, and writes prediction-, metric-, structure-, routing-, and resource-level outputs.

## 9. Experiment Dependency Orchestration

Users never run prerequisites manually. The shared orchestrator resolves them.

Examples:

- Running EXP-05 automatically validates or runs EXP-00 and EXP-01, builds COCO data, then trains and evaluates B0.
- Running EXP-09 automatically prepares all required B0 and routing artifacts, then runs B0-B11.
- Running EXP-15 automatically validates that all frozen upstream outputs exist and are compatible before producing final statistics.

Dependencies are represented as an acyclic graph. The EXP-10/EXP-12 cycle is removed by giving EXP-12 two substages under the same unchanged entry:

1. `prefreeze`: after EXP-09, fit and validate calibration using D2/D3.
2. `external`: after EXP-11, apply the already frozen calibrator to D4 predictions without refitting.

EXP-10 requires the EXP-12 `prefreeze` artifact. EXP-11 requires EXP-10. EXP-12 `external` requires EXP-11.

## 10. Evaluation and Outputs

The shared evaluator generates:

- Box AP, Mask AP, AP50, AP75, macro AP, per-class AP, and worst-class AP;
- crack clDice, break rate, and connected-component error;
- area-disease Boundary-F1, HD95, and area error;
- routing entropy, expert load, dual-activation rate, and output similarity;
- parameters, FLOPs, peak VRAM, batch-1 latency, throughput, and mean active experts;
- ECE, AUROC, Brier score, risk-coverage, and review-budget capture rate;
- measurement MAE, relative error, bias, agreement interval, and unmeasurable rate;
- sequence-stage latency and continuous missed-detection length.

Every experiment writes the SOP-mandated JSON/CSV/XLSX files and PNG/PDF figures. Predictions are stored per image and per instance, not only as aggregate metrics.

Every output is associated with a run manifest containing status, seed, environment, hardware, hashes, timestamps, and failure information.

## 11. D4 One-Shot Guard

EXP-11 remains standalone but requires the exact interactive confirmation:

`I UNDERSTAND D4 IS ONE-SHOT`

Before inference, it verifies the complete freeze manifest and writes a started marker. A second run is rejected. Infrastructure-only recovery is allowed only when no D4 prediction or performance feedback was exposed, and the recovery event is audited.

## 12. Quantification Boundary

EXP-13 automatically reads scale, calibration, and reference-measurement metadata when available.

If those inputs are absent, it:

- generates a fillable quantification template;
- computes pixel-domain quantities;
- marks physical quantification incomplete;
- never invents millimetres or physical area.

This is a scientific input requirement, not a code placeholder.

## 13. Error Handling

- Missing data: fail before training and identify exact manifest rows and paths.
- Dependency failure: record the failing prerequisite and preserve completed artifacts.
- OOM: save a recoverable checkpoint and measured peak memory; do not silently alter the model.
- Interrupted training: resume from the newest compatible checkpoint.
- Corrupt upstream output: detect through hashes and rerun the affected prerequisite.
- Failed seed: preserve and report the failure; do not omit it from final accounting.
- Missing environment: identify and bootstrap the exact pinned environment when network access is available.
- Incomplete required output: the experiment cannot receive `completed` status.

## 14. Verification and Acceptance

The implementation is accepted only when:

1. All 19 entry files run with no required command-line arguments.
2. No user manually combines or sequences experiment Python files.
3. All entries automatically read the authoritative manifest and permitted splits.
4. All 19 complete smoke-test workflows pass on a synthetic five-class dataset.
5. Real-data conversion, loading, and small-batch forward passes pass.
6. RTX 4050 smoke tests pass for YOLO, Mask R-CNN, Mask2Former B0, and MorphoRoute B9.
7. EXP-09 constructs and starts real B0-B11 training runs.
8. EXP-10/EXP-12 have no circular dependency.
9. Checkpoint resume is numerically and structurally validated.
10. Expected tables and figures exist and pass schema checks.
11. D4 access controls fail closed.
12. Unit, integration, syntax, configuration, and output integrity tests pass.
13. No smoke-test result is accepted by final-paper aggregation.

Full 50,000-update, five-seed experiments are not required to finish during implementation verification, but their complete execution path must be the same path exercised by smoke tests.

## 15. Non-Goals and Scientific Limits

- The implementation does not fabricate original YOLO logs or unavailable weights.
- It does not fabricate physical scale or human reference measurements.
- It does not use D4 to tune any model, threshold, calibration, or post-processing.
- It does not claim formal performance from smoke-test outputs.
- It does not promise that a 1024-pixel full SOP run fits in 6 GB VRAM.
- It does not change existing Chinese names or the SOP document.
