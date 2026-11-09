"""Check the MACHA_GPU_HANDOFF.md 'Evidence to return' checklist against the run JSONs.

Read-only: parses artifacts/*.json and prints a verdict per check. Does not rerun anything.
Usage: python artifacts/verify_json.py artifacts/fp16_smoke.json artifacts/fp16_pilot_seed17.json ...
"""
import json
import sys
from pathlib import Path

INSIDE = {"inside_shared", "inside_per_view"}
MODES = ["inside_shared", "nested_disabled", "inside_per_view", "outside_per_view"]


def check(path: Path) -> bool:
    rep = json.loads(path.read_text(encoding="utf-8"))
    env, cfg, results = rep["environment"], rep["configuration"], rep["results"]
    ok = True

    def fail(msg):
        nonlocal ok
        ok = False
        print(f"  FAIL: {msg}")

    print(f"== {path.name}  dtype={results[0].get('dtype')} seed={cfg['seed']} tokens={cfg['tokens']} "
          f"width={cfg['width']} depth={cfg['depth']} views={cfg['views']} updates={cfg['updates']} "
          f"checkpointing={cfg['checkpointing']}")
    print(f"   torch={env['torch']} cuda={env['cuda_runtime']} gpu={env.get('gpu')} driver={env.get('driver_version')}")

    # 1. initial weights matched, all arms succeeded
    if rep["initial_weights_matched"] is not True:
        fail("initial_weights_matched is not true")
    errors = [r for r in results if "error" in r]
    if errors:
        fail(f"arms with errors: {[(r['mode'], r['error']) for r in errors]}")
    if sorted(r["mode"] for r in results) != sorted(MODES):
        fail(f"modes present: {[r['mode'] for r in results]}")
    shas = {r["initial_student_sha256"] for r in results if "error" not in r}
    print(f"   initial_weights_matched={rep['initial_weights_matched']} distinct_sha={len(shas)} "
          f"sha={next(iter(shas))[:16]}... errors={len(errors)}")

    for r in results:
        if "error" in r:
            continue
        m = r["mode"]
        # 2. backward autocast state
        want = [m in INSIDE]
        if r["backward_autocast_enabled"] != want:
            fail(f"{m}: backward_autocast_enabled={r['backward_autocast_enabled']} expected {want}")
        # 3. zero / missing / nonfinite student tensors, cross-checked against per-tensor records
        recs = r["gradients"]
        zero_names = [g["name"] for g in recs if g["status"] == "zero"]
        counts = {s: sum(g["status"] == s for g in recs) for s in ("zero", "live", "missing", "nonfinite")}
        if counts["zero"] != r["zero_tensors"] or counts["live"] != r["live_tensors"] or len(recs) != r["student_tensors"]:
            fail(f"{m}: record counts disagree with summary fields")
        role_zero = {k: v["zero"] for k, v in r["by_role"].items() if v["zero"]}
        if role_zero or zero_names:
            print(f"   {m}: ZERO tensors: {zero_names} by_role={role_zero}")
        if counts["missing"] or counts["nonfinite"]:
            fail(f"{m}: missing={counts['missing']} nonfinite={counts['nonfinite']}")
        # 4. Adam moments / optimizer state
        if r["zero_adam_moments"] or r["no_adam_state"]:
            fail(f"{m}: zero_adam_moments={r['zero_adam_moments']} no_adam_state={r['no_adam_state']}")
        mags = [(g["max_abs_grad"], g["name"]) for g in recs if g["max_abs_grad"] is not None]
        smallest = min(mags)
        print(f"   {m:17s} bwd_autocast={r['backward_autocast_enabled']} zero={counts['zero']}/{len(recs)} "
              f"live={counts['live']} missing={counts['missing']} nonfinite={counts['nonfinite']} "
              f"zero_adam={r['zero_adam_moments']} no_state={r['no_adam_state']} "
              f"loss_decreased={r['loss_decreased']} min_max|g|={smallest[0]:.3e} ({smallest[1]}) t={r['seconds']}s")

    # 5. loss-trajectory grouping across arms (observational only)
    by_mode = {r["mode"]: r["losses"] for r in results if "error" not in r}
    if len(by_mode) == 4:
        def maxdiff(a, b):
            return max(abs(x - y) for x, y in zip(by_mode[a], by_mode[b]))
        print(f"   max|dloss| inside_shared vs inside_per_view   = {maxdiff('inside_shared', 'inside_per_view'):.3e}")
        print(f"   max|dloss| nested_disabled vs outside_per_view = {maxdiff('nested_disabled', 'outside_per_view'):.3e}")
        print(f"   max|dloss| inside_shared vs nested_disabled    = {maxdiff('inside_shared', 'nested_disabled'):.3e}")
        print(f"   max|dloss| inside_per_view vs outside_per_view = {maxdiff('inside_per_view', 'outside_per_view'):.3e}")
    print(f"   checklist: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]]
    results = [check(p) for p in paths]
    print(f"\nALL CHECKLISTS PASS: {all(results)} ({sum(results)}/{len(results)} files)")
    sys.exit(0 if all(results) else 1)
