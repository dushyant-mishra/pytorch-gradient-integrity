# Experiment contract — exploratory, not a bug claim

## Question

Can a generic masked multi-view student/predictor loop using FP16 autocast produce finite, entirely zero gradients for selected active student parameter tensors while the training objective still falls? Does backward autocast state, or the lifetime of the autocast scope across multiple views, change the result?

## Controls and invariants

1. Synthetic inputs only. Same model and input-generator seed per arm; compare initial parameter SHA-256.
2. `inside_per_view` and `outside_per_view` have identical one-scope-per-view forward lifetimes; only the backward placement changes.
3. `inside_shared` and `nested_disabled` keep one outer scope across views; only the backward autocast enabled state changes. `nested_disabled` is **not** equivalent to leaving the outer scope.
4. The frozen teacher is not expected to receive gradients and is excluded from the reported student gradient count.
5. Gradient classifications use exact element tests, not a squared norm that might underflow. Gradients are examined after accumulation and unscale, before the optimizer step.
6. A missing/nonfinite-only monitor is reported as a **deliberately limited** check. All-zero gradients are observational, not automatically erroneous.
7. Loss changes are observational. Different arms can diverge after training updates; individual gradients and Adam moments are the primary evidence.
8. CPU/BF16 runs are wiring tests, never evidence for CUDA/FP16.

## Decision rules

- **No reproduction:** all FP16 modes remain live, or results do not persist across repeats. Report a negative or inconclusive result, not a PyTorch defect.
- **Possible scoped failure:** a specific FP16 mode repeatedly yields zero gradients in a parameter expected to be active while matched controls do not, with common initial weights and a valid gradient path. Narrow the reproducer further before considering an upstream report.
- **Not attributable to backward placement:** zeros occur in both inside and outside arms; investigate architecture, masking, checkpointing, scale, precision, and data before attributing cause.
- **Not interpretable:** nonfinite gradients, missing/disabled optimizer steps, OOM, or mismatched initialization. Fix the experimental control first.

## What is not established

No claims about scientific applications, GPU hardware generally, PyTorch regression versions, or causal mechanisms. The test is synthetic and runs a deliberately compact model; a reproduced issue would still require independent reduction and comparison with known PyTorch behavior.
