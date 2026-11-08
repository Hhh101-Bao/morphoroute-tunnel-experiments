# EXP-09 B11 true-FP32 recovery

This patch changes only the numerical execution wrapper of EXP-09 B11.
B0--B10 caches and all scientific settings remain unchanged. B11 keeps its
model, PCGrad projection, AdamW state, learning-rate scheduler, accumulated
batch size, losses, data order, seed, and 400000-iteration budget.

The old `PCGradAmpOptimWrapper` is replaced at runtime by
`PCGradFP32OptimWrapper`, which inherits MMEngine `OptimWrapper`. It therefore
uses neither autocast nor GradScaler. On resume, only the obsolete
`loss_scaler` field is removed; model, AdamW, scheduler, iteration, and message
hub state are restored normally. Finite checks reject invalid losses or
gradients before a parameter update and save diagnostic evidence.

The recovery also replaces only B11's unsafe `BatchFixedSizePad` with
`MorphoRouteSafeBatchFixedSizePad`. The original random resize, image pixels,
masks, data order, and configured minimum size are retained. If one resized
dimension already exceeds 640, the safe pad preserves it, pads only the short
dimension, and records that actual final shape consistently for the input and
instance masks. This prevents the observed `2 * 640 * 704` mask array from
being incorrectly reshaped as `640 * 640`; it performs no crop or resampling.

Post-augmentation masks that contain no foreground pixels are treated as
unobservable for morphology only. No pixel or synthetic morphology target is
created. The affected matched query is excluded from the route KL loss, while
its ordinary Mask2Former losses, structural losses, sample order, and training
budget remain unchanged. Each occurrence is logged with a cumulative count.
The original D1 morphology-table builder remains strict and continues to reject
empty source annotations; this policy applies only to masks emptied by online
augmentation.

The default B11 configuration builder now selects the same true-FP32 wrapper,
so the correction is not limited to the recovery command. Other variants do
not enable PCGrad and retain their existing configuration.

Version 4 additionally replaces only B11's instance postprocessor with
`MorphoRouteMemoryEfficientMaskFormerFusionHead`. MMDetection's reference
implementation duplicates all top-100, original-resolution mask logits and
creates several full FP32 temporaries. The replacement preserves the same
softmax ranking, top-100 candidates, bilinear-rescaled logits, zero threshold,
mask-quality score, bounding boxes, mask order, original resolution, and COCO
evaluator, but computes one unique query at a time. It retains only the final
boolean masks, preventing the observed 7.44-GiB transient allocation without
changing the scientific evaluation contract.

The recovery launcher verifies the known-good B11 seed42 `iter_80000.pth`,
runs focused tests, and then executes a complete 1158-image D2 evaluation in a
model-only process. Formal recovery is launched only if that exact evaluation
finishes with `EXP09_B11_MEMORY_EFFICIENT_D2=PASSED`. It then resumes EXP-09,
updates the standard PID/log pointers, and restarts the downstream autostart
chain if present.
