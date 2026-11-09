# PyTorch Gradient Integrity Lab

An **independent, synthetic** experiment asking whether multi-view, mixed-precision training can produce exactly-zero gradients for some active parameter tensors while the overall loss changes and missing/nonfinite-only checks pass.

**Status: investigation only.** No PyTorch defect or general failure rate has been established. The initial CPU sanity checks do **not** test CUDA/FP16.

## Why four arms?

The original exploratory script labeled a nested `autocast(enabled=False)` block as “outside autocast,” although the surrounding autocast scope remained open. This package separates that diagnostic from a genuinely outside-autocast arm.

| Mode | Forward context | Backward context | Purpose |
|---|---|---|---|
| `inside_shared` | One autocast scope across views | Autocast **enabled** in same shared scope | Suspected multi-view trigger |
| `nested_disabled` | One autocast scope across views | Autocast **disabled** in nested scope, outer scope remains open | Control for enabled state without changing shared scope lifetime |
| `inside_per_view` | One autocast scope per view | Autocast **enabled** in that view scope | Paired control for outside-per-view |
| `outside_per_view` | One autocast scope per view | **Outside** all autocast scopes | Standard AMP placement |

Every arm independently resets the same model seed and synthetic input generator. The script records a SHA-256 of the initial student model and refuses to compare arms whose initial weights differ. The experiment uses a generic token encoder, a frozen teacher, a predictor, masked views, linear attention, AdamW and a GradScaler. All inputs are random; it contains no external research data.

## Run on a CUDA GPU

Use a CUDA-enabled PyTorch environment. Check it before starting:

```bash
python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available(), 'runtime', torch.version.cuda)"
python -m unittest discover -s tests -v
python autocast_backward_probe.py --device cuda --dtype fp16 --json artifacts/fp16_smoke.json
```

For a more demanding but still bounded pilot **after the smoke run works**:

```bash
python autocast_backward_probe.py --device cuda --dtype fp16 \
  --tokens 1024 --width 64 --heads 4 --depth 3 \
  --views 4 --batch 4 --microbatch 2 --blocks 4 --block-size 8 \
  --updates 5 --json artifacts/fp16_pilot.json
```

On GPUs with bfloat16 support, rerun the *same* pilot with `--dtype bf16 --json artifacts/bf16_pilot.json`. Then repeat the FP16 pilot with `--seed 29` and `--seed 43`. No automatic wide grid or high-memory runs are enabled.

For a CPU **wiring-only** check:

```bash
python autocast_backward_probe.py --device cpu --dtype bf16 --tokens 64 \
  --width 16 --heads 4 --depth 1 --views 2 --batch 2 \
  --microbatch 1 --blocks 2 --block-size 4 --updates 2 \
  --json artifacts/cpu_wiring.json
```

## What to inspect

The JSON report records environment and CUDA details, backward autocast state, initial model hash, per-update loss, exact-zero / missing / nonfinite gradients, per-tensor maximum magnitude, and whether both Adam moment tensors remain exactly zero. It inspects gradients **after one unscale, before the optimizer step**, not after `zero_grad`.

A falling loss or exactly-zero tensor alone is **not proof of a PyTorch defect**. A meaningful finding needs consistent same-seed comparisons, a healthy positive control, repeat runs, and examination of whether each zero-gradient parameter was expected to participate.

**GPU execution handoff:** [MACHA_GPU_HANDOFF.md](MACHA_GPU_HANDOFF.md).  **Testing limits:** [EXPERIMENT_CONTRACT.md](EXPERIMENT_CONTRACT.md).

Reference: [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html) recommend that backward passes run outside autocast. This project tests a specific concern; it does not challenge that documented recommendation.
