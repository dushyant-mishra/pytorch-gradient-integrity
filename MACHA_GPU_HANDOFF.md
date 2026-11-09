# Macha — GPU execution handoff

## Objective

Run the new public, synthetic PyTorch probe on the GPU laptop. Determine whether a **real CUDA/FP16** test shows all-zero student gradients under `inside_shared`, whether they differ from `nested_disabled` and `outside_per_view`, and whether an apparent effect repeats. **Do not infer a PyTorch bug merely from a falling loss or a zero gradient.**

Repository: `https://github.com/dushyant-mishra/pytorch-gradient-integrity.git`.

## Host and environment preflight

Use a separate work directory and CUDA-enabled PyTorch environment. Do not run in, import from, or change any unrelated research repository. Do not add tokens or credentials to the public repo.

```bash
git clone https://github.com/dushyant-mishra/pytorch-gradient-integrity.git
cd pytorch-gradient-integrity
git rev-parse HEAD
nvidia-smi
python -c "import sys,torch; print(sys.version); print(torch.__version__,torch.version.cuda,torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
python -m unittest discover -s tests -v
```

If CUDA is unavailable, **stop the GPU phase**; a CPU/BF16 smoke run is not a substitute. Use an existing isolated, working GPU environment or install official CUDA PyTorch in a new virtual environment; never overwrite another project's pinned packages merely to run this probe.

## Execution sequence

Run these in order; each command writes local JSON under git-ignored `artifacts/`.

**1. Small CUDA/FP16 check** (default 128 tokens, four modes, three updates):

```bash
python autocast_backward_probe.py --device cuda --dtype fp16 --json artifacts/fp16_smoke.json
```

**2. Paired CUDA/FP16 pilot** (only if the first completed without error/OOM):

```bash
python autocast_backward_probe.py --device cuda --dtype fp16 \
  --tokens 1024 --width 64 --heads 4 --depth 3 \
  --views 4 --batch 4 --microbatch 2 --blocks 4 --block-size 8 \
  --updates 5 --json artifacts/fp16_pilot_seed17.json
```

**3. BF16 precision control** (only if `torch.cuda.is_bf16_supported()`):

```bash
python autocast_backward_probe.py --device cuda --dtype bf16 \
  --tokens 1024 --width 64 --heads 4 --depth 3 \
  --views 4 --batch 4 --microbatch 2 --blocks 4 --block-size 8 \
  --updates 5 --json artifacts/bf16_pilot_seed17.json
```

**4. Replicate the FP16 pilot:** repeat step 2 with `--seed 29` and then `--seed 43`, writing new JSON files. If the suspected effect is absent in the pilot, report that; do not escalate model size without a separate analysis.

## Evidence to return

Provide the exact Git commit, OS/driver/GPU, Python/PyTorch/CUDA versions, command lines, **full unedited console output**, and four JSON files (more if repeating seeds). Keep errors and negative results. Do not publish partial results as established findings.

Before drawing a conclusion, confirm from JSON:

- `initial_weights_matched == true` and all arms succeeded.
- `backward_autocast_enabled == [true]` for inside arms, `[false]` for the nested-disabled and genuinely outside arms.
- Which named **student** gradients, if any, are finite but exactly zero; check `by_role` and `gradients`.
- Whether Adam moments are zero, with no missing state or nonfinite gradients.
- Whether the same tensor-level pattern holds across seeds, and how BF16 compares.

If a mode fails, record the failure. Do not silently discard OOM, suppress failing tests, or change the target classification after looking at the outcome. Do not broaden to thousands of experiments until the minimal result is independently reviewed.

## Hand back for independent review

Report one of: **REPRODUCED_WITH_CONTROLS**, **NOT_REPRODUCED**, or **INCONCLUSIVE**. Include evidence and any alternate explanations. Do not open a PyTorch issue or write an article before independent review and prior-art check.
