# Provenance of this evidence bundle

All runs used commit `bbbd063383483fad31782e1fb3438bc07ec1fb24` on 2026-11-09, between 12:48 and 12:50 ET. They wrote their output to the git-ignored `artifacts/` folder, and the files were then copied here. The verdict and full details are in `MACHA_HANDBACK.md`.

## Byte-identical to the run outputs

| File | SHA-256 |
|---|---|
| `fp16_smoke.json` | `4a523988b1fae8da41143fbc9bbf20c40eb72221e2a3d4600d9bfa872e704bb7` |
| `fp16_pilot_seed17.json` | `870449bf7d15cea7ee8c19ac5d4c7569d208d9b5aab2438f769252c68a86759d` |
| `bf16_pilot_seed17.json` | `ca2cd5d858d3ee3f74bd977fe0656f2e95815beeee7da09496bb576f534e7c2b` |
| `fp16_pilot_seed29.json` | `1899371f4424fa099ca57b2286eecd2f49332cca9377ef421208860e89a5548b` |
| `fp16_pilot_seed43.json` | `caecdd7840d4d68e520f971239b910813fd8f3ec2c5627116f26c4582de466e7` |
| `01_fp16_smoke.log` | `d11ab1b3c2cdbdee95d7e07868871ea45ef2c61312a72ab82bda334ff6e65428` |
| `02_fp16_pilot_seed17.log` | `53ae132eecf9ca0cb8755177958103e59f899f5dd6f68f8a1b786527bcdd2d61` |
| `03_bf16_pilot_seed17.log` | `27a79a0d921230fdc1e85e0fa5d70109cdda4d65cec87ce6d6113cd2eeaa4f66` |
| `04_fp16_pilot_seed29.log` | `8ecbf8a7ff7b01853cc1f4654565bb9f1d9d5cfed04a90e0b75c355a633e067f` |
| `05_fp16_pilot_seed43.log` | `ea7054d00e0330d289b31ac3ddaa62245a300a976b0102c85e36abdca6132353` |
| `06_json_verification.log` | `cfb4c4e7646214098baaa799fa596aa142ddfb3a644018204fd224b365d049ac` |
| `verify_json.py` | `9e3323ccc34c7652f625c5fd57545e622f3e0740f73a40f94d1d1f8f40226bb9` |

## Changed before committing

The repository is public, so local machine details were redacted. No credentials were found. The JSON results contain no paths or user information and were not changed.

**`00_preflight.log`:** 3 of 47 lines were changed by exact byte replacement, and line endings were preserved. Original SHA-256: `c64b4287ebac0a092e443c624889bf458b4fbf355a03a1b4c02711d5bf24d255`. Committed SHA-256: `e255977dd4562e2a7686f2a39ccd7ab1b8c003c3e7b4b9617d335a1e9c5c4084`.

| Line | Content | Reason |
|---|---|---|
| 2 | Interpreter path → `[REDACTED: local conda env path]/python.exe` | The path contains the local username and the name of an environment belonging to an unrelated project. The torch build is recorded in full on lines 34–38 and in every JSON. |
| 24 | `nvidia-smi` process row: PID and app name → `[PID]`, `[REDACTED: unrelated desktop app]` | This was an unrelated desktop app. The row's type (`C+G`) and GPU memory (`N/A`) are kept. |
| 33 | `cmd.exe` prompt echo → `[REDACTED: local path]>` | The prompt contains the local username. |

**`MACHA_HANDBACK.md`:** Original SHA-256: `67bb143c6ca2f0b6cbdf287cfedac6d106765a0cc66f342a898efb50b298a8bf`. Committed SHA-256: `a301bce3896ba898e66a031a70b886fe30c2d409b59e80356909743790118541`. Three edits were made:

- The interpreter path was redacted, for the same reason as line 2 above.
- The `verify_json.py` command line was corrected to the exact invocation used. The original wrongly showed `artifacts/*.json`, but the files were passed in the order shown.
- The bundle section was updated to point at this folder.

None of the results, checks, observations or limits were changed.

The unredacted originals are kept locally and can be shown to the reviewer on request.

## Repository handling

- `.gitattributes` (`* -text`) turns off end-of-line conversion here. The files were written on Windows and contain CRLF line endings, mixed with LF in the logs. Without this setting, git would normalize them and the hashes above would not match a clone.
- The repository's `.gitignore` ignores `*.log`, so the logs were added with `git add -f`.

## Re-running the checklist

```bash
python evidence/gpu-pilot-20261109/verify_json.py \
  evidence/gpu-pilot-20261109/fp16_smoke.json evidence/gpu-pilot-20261109/fp16_pilot_seed17.json \
  evidence/gpu-pilot-20261109/bf16_pilot_seed17.json evidence/gpu-pilot-20261109/fp16_pilot_seed29.json \
  evidence/gpu-pilot-20261109/fp16_pilot_seed43.json
```

This command was run on these committed copies before pushing. Its output was identical to `06_json_verification.log`, ignoring line endings. The order of the files matters: a shell glob would print the same results in a different order.

## What `verify_json.py` does and does not establish

These limits are recorded so the reviewer does not have to rediscover them.

**What it asserts** (and fails on):

- `initial_weights_matched` is true, and there is one distinct initial SHA across all successful arms.
- No arm has an error.
- Exactly the four expected modes are present.
- `backward_autocast_enabled` is `[true]` for the inside arms and `[false]` for the other two.
- The per-tensor zero and live counts and the tensor total agree with the summary fields.
- No missing or nonfinite student gradients.
- `zero_adam_moments == 0` and `no_adam_state == 0`.

**What it does not do:**

- **It does not recompute evidence independently.** The Adam-moment and autocast-state checks read fields that the probe itself computed. A bug in the probe's own bookkeeping would pass straight through.
- **It does not detect skipped optimizer steps.** The probe does not record the GradScaler scale, skipped steps or Adam's `step` count. Nonzero Adam moments show only that at least one step was applied, not that all of them were.
- **It only covers the final update.** Gradients are summarized at the last update only, so a zero or nonfinite gradient at an earlier update would not appear.
- **It does not assert the run configuration.** It prints the configuration and environment but does not fail on a wrong configuration, dtype or device. For example, it would not fail on a CPU run or a dtype other than the one intended. These have to be checked against the printed headers. In this bundle they match the handoff commands, and every run is CUDA on the RTX 3080 Laptop GPU.
- **It does not require the loss to fall.** `loss_decreased` is printed but not asserted, because the contract treats loss as observational.
- **It only detects all-zero tensors.** A tensor counts as zero only if every element is zero, so partially dead tensors are not detected.
