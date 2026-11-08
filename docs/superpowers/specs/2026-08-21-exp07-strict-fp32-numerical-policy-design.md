# EXP-07 Strict FP32 Numerical Policy Design

## Context

EXP-07 trains the B4, B5, and B6 routing-control variants with the same
Mask2Former backbone, dataset splits, optimizer budget, seed, and evaluation
pipeline. Two formal mixed-precision runs failed inside SciPy Hungarian
assignment with `matrix contains invalid numeric entries`, first near raw
iteration 55,600 and again near raw iteration 49,200. A partial precision
island that moved the matching head and matching loss to FP32 did not prevent
the second failure. The 40,000-iteration checkpoint from the second run has
655 finite model tensors, which shows that the saved model was not corrupted,
but the mixed-precision execution path remains unsuitable for a formal run.

## Decision

EXP-07 B4, B5, and B6 will use one strict IEEE-FP32 training policy. AMP,
GradScaler, BF16, and TF32 will be disabled. The model architecture, dataset,
split manifest, seed, optimizer type and hyperparameters, learning-rate
schedule, gradient accumulation, input size, augmentation, training budget,
budget gate, and evaluation definitions will not change.

The failed AMP runs and their checkpoints are diagnostic evidence only. They
must not contribute predictions, metrics, checkpoint selection, or reported
paper results.

## Alternatives Considered

### BF16 AMP

BF16 offers FP32-like exponent range and is faster than FP32 on the rented
GPU, but the exact MMDetection 3.3/MMCV 2.1 execution path has not been
validated for BF16. A compatibility failure after another long run would
waste more compute.

### A larger partial FP32 island

Moving the pixel decoder and transformer decoder to FP32 while leaving other
operations under AMP would be faster. However, two previous partial-boundary
fixes have demonstrated that proving the boundary complete is difficult. This
option is rejected for the formal experiment.

### Strict FP32

Strict FP32 is slower, but the RTX 4080 SUPER has sufficient memory and it
removes FP16 overflow and GradScaler step-skipping from the entire EXP-07
comparison. It is the selected policy.

## Runtime Architecture

The deliverable will contain a project-owned runner named
`run_exp07_strict_fp32.py`. It will execute the existing efficient EXP-07
runner as an inner runner so the verified D2 AP-only aggregation, full D3
evaluation, resource-audit compaction, and B4-to-B6 sequencing remain intact.

Before handing control to the inner runner, it will install the following
runtime policy at the final MMDetection configuration boundary:

1. Replace `AmpOptimWrapper` with a registered `StrictFiniteOptimWrapper`.
2. Remove `loss_scale` and AMP dtype configuration.
3. Set `torch.backends.cuda.matmul.allow_tf32 = False`.
4. Set `torch.backends.cudnn.allow_tf32 = False`.
5. Set `torch.set_float32_matmul_precision("highest")`.
6. Assert that the resolved configuration contains no AMP optimizer wrapper.
7. Add a `numerical_stability_audit` object to every B4/B5/B6 resolved
   configuration.

The strict optimizer wrapper will:

- reject a non-finite scalar loss before backward propagation;
- reject any non-finite gradient before clipping or optimizer update;
- reject a non-finite gradient norm;
- never clamp, replace, or silently skip a non-finite update;
- raise an error before model parameters can be modified by invalid values.

The runner will instrument the Hungarian boundary. If a cost matrix contains
NaN or Inf, it will save the matrix and a JSON numerical report before raising
the original error. It will not replace invalid entries or alter assignment.

## Audit and Provenance

Each resolved B4/B5/B6 configuration will record:

- policy revision `morphoroute_strict_fp32_v1`;
- AMP disabled;
- TF32 matmul disabled;
- TF32 cuDNN disabled;
- optimizer-wrapper class;
- dataset-manifest hash and existing configuration hash;
- seed, variant, and profile;
- whether the run is a diagnostic probe or a formal restart.

The runner will emit a single startup marker for each variant and will refuse
to start if the effective optimizer wrapper is AMP-based.

## Validation Sequence

### Unit and integration tests

Tests must be written and observed failing before implementation. They will
verify:

1. B4, B5, and B6 all resolve to the same strict FP32 policy.
2. `loss_scale` and AMP wrappers are absent.
3. TF32 is disabled before runner construction.
4. A finite loss and finite gradients allow one real optimizer update.
5. A NaN loss fails before backward.
6. A NaN or Inf gradient fails before clipping and parameter update.
7. An invalid Hungarian matrix is saved and rejected, never repaired.
8. The existing compact evaluation wrapper is still invoked.

Existing numerical-policy, Mask2Former, variant, registry, and end-to-end
tests must continue to pass.

### 40,000-to-60,000 diagnostic probe

The verified finite `iter_40000.pth` may be used only for a diagnostic probe.
The probe will switch to strict FP32, resume optimizer and scheduler state,
cross both prior failure regions, and stop at 60,000. It will be successful
only if:

- no loss, gradient, parameter, or Hungarian cost is non-finite;
- the 60,000-iteration checkpoint contains only finite tensors;
- the effective optimizer wrapper is strict FP32;
- the process exits with the dedicated probe-success marker.

The resulting 60,000 checkpoint remains diagnostic and cannot be reused by
the formal run.

### Formal run

After the probe passes, the complete failed EXP-07 directory will be moved to
a timestamped backup. Only immutable D1 morphology artifacts
`morphology_features_D1.csv` and `gmm_D1.json` may be copied into the new
directory. No checkpoint, optimizer state, prediction, metric, run manifest,
or resolved training configuration may be reused.

The formal command will use `--no-resume`. B4, B5, and B6 will start from raw
iteration zero and run sequentially under the identical policy. Completion
requires:

- B4, B5, and B6 outputs present;
- no numerical-failure marker in the formal log;
- all final checkpoints finite;
- `run_manifest.json` status `completed`;
- `metrics_resources.csv`, `router_diagnostics.csv`, and the run index present.

## Scientific Rigor

The selected change is an execution-precision policy, not a model or data
change. Applying it uniformly to all three EXP-07 routing controls preserves
the intended B4/B5/B6 comparison. The paper and reproducibility materials
must disclose strict FP32 execution for EXP-07 and must exclude every failed
AMP attempt from statistical analysis.

No recovery mechanism may silently alter labels, masks, predictions, costs,
losses, gradients, optimizer updates, or evaluation thresholds.

## Expected Cost

Strict FP32 may be approximately 25% to 60% slower than the current FP16 AMP
run. The local_6gb physical batch of one and the 32 GB GPU provide ample
memory headroom. The configuration will retain the original physical batch
and gradient accumulation to avoid introducing a batch-size confound.

