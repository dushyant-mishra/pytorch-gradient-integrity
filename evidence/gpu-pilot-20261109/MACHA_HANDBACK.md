# Macha hand-back: CUDA/FP16 gradient-integrity probe

**Verdict: NOT_REPRODUCED**

Every FP16 arm in every run kept all student gradient tensors live. The four arms are `inside_shared`, `nested_disabled`, `inside_per_view` and `outside_per_view`. The runs are the smoke run, the pilot with seeds 17, 29 and 43, and the BF16 control. There were no exactly-zero, missing or nonfinite student gradients and no zero Adam moments. Under the contract's decision rules this is **No reproduction**, so the result does not show a PyTorch defect. Nothing was escalated beyond the pilot size, as the handoff requires.

## Provenance

| Item | Value |
|---|---|
| Commit | `bbbd063383483fad31782e1fb3438bc07ec1fb24` (clean clone of `main`) |
| Date | 2026-11-09, runs 12:48–12:50 ET |
| OS | Windows 11 Pro 10.0.26200 (`Windows-10-10.0.26200-SP0` per Python) |
| GPU | NVIDIA GeForce RTX 3080 Laptop GPU, 16 GB, compute capability 8.6, WDDM |
| Driver | 610.74 (nvidia-smi reports CUDA UMD 13.3) |
| Python | 3.11.15 (conda-forge) |
| PyTorch | 2.7.0+cu128, CUDA runtime 12.8, cuDNN 90701, git `134179474539648ba7dee1317959529fbd0e7f89` |
| BF16 supported | True |

**Interpreter:** the `python.exe` of an existing local conda environment that belongs to an unrelated project. Its name and path are redacted; see `PROVENANCE.md`. This existing CUDA environment was used read-only, called by its full path. Nothing was installed or changed in it, and no research repository was entered or imported. `torch` loaded from that environment's own site-packages, and no `sitecustomize` was present. The base Anaconda Python has CPU-only torch 2.1.2, so it was not used.

## Commands (in order, all exit 0)

```
python -m unittest discover -s tests -v                                   # 3/3 OK
python autocast_backward_probe.py --device cuda --dtype fp16 --json artifacts/fp16_smoke.json
python autocast_backward_probe.py --device cuda --dtype fp16 --tokens 1024 --width 64 --heads 4 --depth 3 --views 4 --batch 4 --microbatch 2 --blocks 4 --block-size 8 --updates 5 --json artifacts/fp16_pilot_seed17.json
python autocast_backward_probe.py --device cuda --dtype bf16 <same pilot args> --json artifacts/bf16_pilot_seed17.json
python autocast_backward_probe.py --device cuda --dtype fp16 <same pilot args> --seed 29 --json artifacts/fp16_pilot_seed29.json
python autocast_backward_probe.py --device cuda --dtype fp16 <same pilot args> --seed 43 --json artifacts/fp16_pilot_seed43.json
python artifacts/verify_json.py artifacts/fp16_smoke.json artifacts/fp16_pilot_seed17.json artifacts/bf16_pilot_seed17.json artifacts/fp16_pilot_seed29.json artifacts/fp16_pilot_seed43.json   # handoff checklist, read-only
```

The full console output of each command is in `00_preflight.log` through `06_json_verification.log`. It is unedited except for three redacted lines in `00_preflight.log`, which are listed in `PROVENANCE.md`.

## Results

| Run | Arms OK | Init SHA matched | Zero / total student tensors (each arm) | Missing | Nonfinite | Zero Adam moments | Loss fell |
|---|---|---|---|---|---|---|---|
| FP16 smoke, seed 17 | 4/4 | yes | 0/39 | 0 | 0 | 0 | yes |
| FP16 pilot, seed 17 | 4/4 | yes | 0/55 | 0 | 0 | 0 | yes |
| BF16 pilot, seed 17 | 4/4 | yes | 0/55 | 0 | 0 | 0 | yes |
| FP16 pilot, seed 29 | 4/4 | yes | 0/55 | 0 | 0 | 0 | yes |
| FP16 pilot, seed 43 | 4/4 | yes | 0/55 | 0 | 0 | 0 | yes |

Results of the handoff checklist, verified from the JSON files:

- `initial_weights_matched == true` in all 5 files, and all 20 arms succeeded. There were no OOM errors.
- `backward_autocast_enabled` was `[true]` for both inside arms and `[false]` for `nested_disabled` and `outside_per_view` in every file.
- No student tensor was zero in any arm. Every role in `by_role` had `zero=0`, and the per-tensor `gradients` records agree with the summary counts.
- No student parameter lacked Adam state, and no parameter had both Adam moments exactly zero.
- The pattern was the same across seeds 17, 29 and 43. BF16 matches FP16, with all tensors live.

## Observations for the reviewer (not findings)

1. **Loss trajectories group by backward autocast state, not scope lifetime.** In every run, `inside_shared` and `inside_per_view` produce identical loss lists. `nested_disabled` and `outside_per_view` also match each other exactly. The two groups differ by at most 1.2e-6 to 5.4e-6 in loss for FP16, and 1.7e-5 for BF16. In this setup, keeping one autocast scope across views changed nothing numerically. The nested-disabled control behaved exactly like genuinely outside autocast, even though the outer scope was still open. Why backward with autocast enabled shifts results slightly was not investigated.
2. **Closest-to-zero tensor:** in every run, the student tensor with the smallest maximum gradient is an `attention.key.bias`. Its value is about 6e-6 to 8e-6 for the pilot and about 4.8e-5 for the smoke run. After scaling by the GradScaler's default 65536, that is about 0.4 in FP16, far from the subnormal range. Underflow is not a concern at this scale. This bias tensor is the one to watch if a larger configuration is ever approved.

## Limits and alternate explanations

- **Scale and setting.** Only the compact synthetic pilot was run. A negative result here does not rule out the behaviour originally observed in a larger or different model. The handoff forbids escalating without a separate analysis, so none was attempted.
- **Gradients are summarized only at the final update.** If an earlier FP16 step was skipped by the GradScaler because of inf, that would not show up. The JSON records neither the scaler scale nor the skipped steps nor Adam's `step` count. Nonzero Adam moments prove that at least one step was applied, not all five. Recording `scaler.get_scale()` and `optimizer.state[p]["step"]` per update would close this gap.
- **Detection is per tensor.** A tensor is classified as zero only if every element is zero. Partial loss, such as dead rows in `identity.weight`, would not be detected. This matches the contract's target but is a sensitivity limit.
- **Checkpointing.** All runs used the default `checkpointing=True` (non-reentrant). A `--no-checkpoint` variant was not run because the handoff did not include it.
- **Single platform.** The results come from one GPU, one driver and one torch build (2.7.0+cu128 on Windows with WDDM). Nothing can be said about other versions.

## Files in this bundle (`evidence/gpu-pilot-20261109/`)

`fp16_smoke.json`, `fp16_pilot_seed17.json`, `bf16_pilot_seed17.json`, `fp16_pilot_seed29.json`, `fp16_pilot_seed43.json`, `00_preflight.log` through `06_json_verification.log`, `verify_json.py`, `PROVENANCE.md`, and this file.

These files were copied from the git-ignored `artifacts/` folder where the runs wrote them. `PROVENANCE.md` gives SHA-256 hashes and lists exactly what differs from the originals. No issue or article was drafted. Independent review comes next, as the handoff specifies.
