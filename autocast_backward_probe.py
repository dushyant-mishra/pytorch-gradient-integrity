"""Independent synthetic PyTorch AMP gradient-integrity experiment.

This is NOT a reproduction of a confirmed PyTorch defect. No external data,
project-specific architectures, checkpoints or credentials are used.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

MODES = ("inside_shared", "nested_disabled", "inside_per_view", "outside_per_view")


class Attention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        b, n, w = x.shape
        shape = (b, n, self.heads, self.head_dim)
        q0 = self.query(x).reshape(shape)
        k0 = self.key(x).reshape(shape)
        v0 = self.value(x).reshape(shape)
        with torch.autocast(device_type=x.device.type, enabled=False):
            q = (F.elu(q0.float()) + 1).transpose(1, 2)
            k = (F.elu(k0.float()) + 1).transpose(1, 2)
            v = v0.float().transpose(1, 2)
            m = valid[:, None, :, None]
            k, v = k * m, v * m
            kv = torch.einsum("bhnd,bhne->bhde", k, v)
            den = torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2)).clamp_min(1e-6)
            out = (torch.einsum("bhnd,bhde->bhne", q, kv) / den[..., None]).transpose(1, 2).reshape(b, n, w)
        # Keep low-precision cast in the common forward path, regardless of mode.
        return self.output(out.to(q0.dtype))


class Block(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = Attention(width, heads)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), valid)
        return x + self.ff(self.norm2(x))


class Encoder(nn.Module):
    def __init__(self, tokens: int, width: int, heads: int, depth: int, checkpointing: bool):
        super().__init__()
        self.identity = nn.Embedding(tokens, width)
        self.features = nn.Linear(1, width)
        self.norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(depth)])
        self.out = nn.LayerNorm(width)
        self.checkpointing = checkpointing

    def forward(self, ids: torch.Tensor, values: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.identity(ids) + self.features(values.masked_fill(~visible, 0).unsqueeze(-1)))
        for block in self.blocks:
            x = checkpoint(block, x, visible, use_reentrant=False) if self.checkpointing and self.training else block(x, visible)
        return self.out(x)


class Predictor(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))

    def forward(self, visible_states: torch.Tensor, visible: torch.Tensor, block_ids: torch.Tensor) -> torch.Tensor:
        weights = visible.to(visible_states.dtype)
        summary = (visible_states * weights[..., None]).sum(dim=1) / weights.sum(dim=1, keepdim=True).clamp_min(1)
        return self.net(summary)[:, None, :].expand(-1, block_ids.shape[1], -1)


@dataclass(frozen=True)
class Config:
    tokens: int = 128
    width: int = 32
    heads: int = 4
    depth: int = 2
    views: int = 2
    batch: int = 2
    microbatch: int = 1
    blocks: int = 2
    block_size: int = 4
    updates: int = 3
    learning_rate: float = 0.0003
    seed: int = 17
    checkpointing: bool = True

    def validate(self) -> None:
        if min(self.tokens, self.width, self.heads, self.depth, self.views, self.batch,
               self.microbatch, self.blocks, self.block_size, self.updates) <= 0:
            raise ValueError("All integer dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if self.batch % self.microbatch:
            raise ValueError("batch must be divisible by microbatch")
        if self.blocks * self.block_size >= self.tokens:
            raise ValueError("must leave visible tokens after hiding blocks")
        if not 0 < self.learning_rate <= 0.1:
            raise ValueError("invalid learning rate")


def role(name: str) -> str:
    for part in ("attention.query", "attention.key", "attention.value", "attention.output", "norm1", "norm2"):
        if part in name:
            return part
    return "other"


def model_sha(module: nn.Module) -> str:
    h = hashlib.sha256()
    for name, v in sorted(module.state_dict().items()):
        raw = v.detach().cpu().contiguous().numpy().tobytes()
        h.update(name.encode() + b"\x00" + str(v.dtype).encode() + b"\x00")
        h.update(str(tuple(v.shape)).encode() + b"\x00" + raw)
    return h.hexdigest()


def sample_views(gen: torch.Generator, cfg: Config) -> list[tuple[torch.Tensor, torch.Tensor]]:
    out = []
    for _ in range(cfg.views):
        idx = torch.stack([torch.randperm(cfg.tokens, generator=gen)[:cfg.blocks * cfg.block_size]
                           for _ in range(cfg.batch)]).reshape(cfg.batch, cfg.blocks, cfg.block_size)
        hidden = torch.zeros(cfg.batch, cfg.tokens, dtype=torch.bool)
        hidden.scatter_(1, idx.flatten(1), True)
        out.append((idx, hidden))
    return out


def summarize_gradients(model: Encoder) -> dict:
    records = []
    for name, p in model.named_parameters():
        grad = p.grad
        if grad is None:
            status, mx = "missing", None
        elif not torch.isfinite(grad).all().item():
            status, mx = "nonfinite", None
        else:
            status = "zero" if torch.count_nonzero(grad).item() == 0 else "live"
            mx = float(grad.abs().max().item())
        records.append({"name": name, "role": role(name), "status": status, "max_abs_grad": mx})
    by_role = {}
    for rec in records:
        group = by_role.setdefault(rec["role"], {"total": 0, "zero": 0, "live": 0, "missing": 0, "nonfinite": 0})
        group["total"] += 1
        group[rec["status"]] += 1
    return {"records": records, "by_role": by_role,
            "zero": sum(r["status"] == "zero" for r in records),
            "missing": sum(r["status"] == "missing" for r in records),
            "nonfinite": sum(r["status"] == "nonfinite" for r in records),
            "live": sum(r["status"] == "live" for r in records)}


def train_arm(cfg: Config, device: torch.device, dtype: torch.dtype, mode: str) -> dict:
    """Each arm is independently seeded and trains with the same synthetic batches."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode}")
    cfg.validate()
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    student = Encoder(cfg.tokens, cfg.width, cfg.heads, cfg.depth, cfg.checkpointing).to(device).train()
    teacher = copy.deepcopy(student).requires_grad_(False).eval()
    predictor = Predictor(cfg.width).to(device).train()
    initial_sha = model_sha(student)
    optimizer = torch.optim.AdamW(list(student.parameters()) + list(predictor.parameters()),
                                   lr=cfg.learning_rate, weight_decay=0.05)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda" and dtype == torch.float16))
    micro_steps = cfg.batch // cfg.microbatch
    expected_grad_context = mode in ("inside_shared", "inside_per_view")
    seen_backward_autocast = []
    losses = []
    final_grad = None
    for update in range(cfg.updates):
        ids = torch.arange(cfg.tokens, device=device)[None, :].expand(cfg.batch, -1)
        values = torch.rand(cfg.batch, cfg.tokens, generator=gen).log1p().to(device)
        measured = (torch.rand(cfg.batch, cfg.tokens, generator=gen) > 0.4).to(device)
        views = sample_views(gen, cfg)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for begin in range(0, cfg.batch, cfg.microbatch):
            sl = slice(begin, begin + cfg.microbatch)
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
                teacher_states = teacher(ids[sl], values[sl], measured[sl])

            def compute(idx: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
                visible = measured[sl] & ~hidden
                student_states = student(ids[sl], values[sl], visible)
                prediction = predictor(student_states, visible, idx)
                rows = torch.arange(len(idx), device=device)[:, None, None]
                target = teacher_states[rows, idx].mean(dim=2)
                return F.mse_loss(prediction.float(), target.float()) / (cfg.views * micro_steps)

            def backward(loss: torch.Tensor) -> None:
                enabled = torch.is_autocast_enabled(device.type)
                seen_backward_autocast.append(bool(enabled))
                if bool(enabled) != expected_grad_context:
                    raise AssertionError(f"mode={mode} unexpected autocast enabled={enabled} during backward")
                scaler.scale(loss).backward()

            if mode in ("inside_shared", "nested_disabled"):
                # Shared region across multiple backward calls: the suspected trigger.
                with torch.autocast(device_type=device.type, dtype=dtype):
                    for idx_cpu, hidden_cpu in views:
                        idx = idx_cpu[sl].to(device)
                        hidden = hidden_cpu[sl].to(device)
                        loss = compute(idx, hidden)
                        total_loss += float(loss.detach())
                        if mode == "inside_shared":
                            backward(loss)
                        else:
                            # Diagnostic ONLY: the outer context remains live.
                            with torch.autocast(device_type=device.type, enabled=False):
                                backward(loss)
            else:
                # Same one-region-per-view lifetime in both arms; only backward placement differs.
                for idx_cpu, hidden_cpu in views:
                    idx = idx_cpu[sl].to(device)
                    hidden = hidden_cpu[sl].to(device)
                    with torch.autocast(device_type=device.type, dtype=dtype):
                        loss = compute(idx, hidden)
                        if mode == "inside_per_view":
                            backward(loss)
                    if mode == "outside_per_view":
                        backward(loss)  # genuinely outside every autocast region
                    total_loss += float(loss.detach())
        losses.append(total_loss)
        scaler.unscale_(optimizer)  # exactly once, after all accumulation for this step
        if update == cfg.updates - 1:
            final_grad = summarize_gradients(student)
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            for t, s in zip(teacher.parameters(), student.parameters()):
                t.mul_(0.996).add_(s.detach(), alpha=0.004)
    assert final_grad is not None
    moments_zero = 0
    no_state = 0
    for p in student.parameters():
        state = optimizer.state.get(p)
        if not state:
            no_state += 1
        elif torch.count_nonzero(state["exp_avg"]).item() == 0 and torch.count_nonzero(state["exp_avg_sq"]).item() == 0:
            moments_zero += 1
    return {
        "mode": mode, "dtype": str(dtype), "initial_student_sha256": initial_sha,
        "losses": losses, "loss_decreased": len(losses) >= 2 and losses[-1] < losses[0],
        "backward_autocast_enabled": sorted(set(seen_backward_autocast)),
        "student_tensors": len(final_grad["records"]),
        "zero_tensors": final_grad["zero"], "missing_tensors": final_grad["missing"],
        "nonfinite_tensors": final_grad["nonfinite"], "live_tensors": final_grad["live"],
        "weak_check_passes": final_grad["missing"] == 0 and final_grad["nonfinite"] == 0,
        "zero_adam_moments": moments_zero, "no_adam_state": no_state,
        "by_role": final_grad["by_role"], "gradients": final_grad["records"],
    }


def environment(device: torch.device) -> dict:
    env = {"python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform(),
           "device": device.type, "cuda_runtime": torch.version.cuda,
           "torch_git_version": getattr(torch.version, "git_version", None)}
    if device.type == "cuda":
        env.update({"gpu": torch.cuda.get_device_name(device),
                    "gpu_capability": torch.cuda.get_device_capability(device),
                    "cudnn_version": torch.backends.cudnn.version(),
                    "bf16_supported": torch.cuda.is_bf16_supported()})
        try:
            p = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=5)
            env["driver_version"] = p.stdout.splitlines()[0].strip() if p.returncode == 0 and p.stdout.strip() else None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            env["driver_version"] = None
    return env


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--json", type=Path, default=None, help="write full evidence JSON locally")
    return parser.parse_args()


def main() -> None:
    args = parse_cli()
    cfg = Config(**{name: getattr(args, name) for name in Config.__dataclass_fields__ if name in vars(args)},
                 checkpointing=not args.no_checkpoint)
    cfg.validate()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available. Use --device cpu --dtype bf16 for a wiring-only test.")
    if args.device == "cpu" and args.dtype != "bf16":
        raise SystemExit("CPU wiring test must use --dtype bf16; FP16 conclusion needs CUDA.")
    if args.device == "cuda" and args.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("This CUDA device does not support bf16.")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    print("Environment:", json.dumps(environment(device), sort_keys=True), flush=True)
    results = []
    for mode in args.modes:
        start = time.monotonic()
        try:
            result = train_arm(cfg, device, dtype, mode)
            result["seconds"] = round(time.monotonic() - start, 3)
            results.append(result)
            print(f"{mode}: zero={result['zero_tensors']}/{result['student_tensors']} "
                  f"missing={result['missing_tensors']} nonfinite={result['nonfinite_tensors']} "
                  f"zero_adam={result['zero_adam_moments']} weak_pass={result['weak_check_passes']} "
                  f"losses={[round(x, 6) for x in result['losses']]} ", flush=True)
        except torch.cuda.OutOfMemoryError as exc:
            results.append({"mode": mode, "error": "CUDA_OOM", "details": str(exc)[:500]})
            print(f"{mode}: CUDA_OOM; no result inferred", flush=True)
            torch.cuda.empty_cache()
    successful = [r for r in results if "error" not in r]
    comparable = len({r["initial_student_sha256"] for r in successful}) <= 1
    if not comparable:
        raise AssertionError("initial student weights differ between arms")
    report = {"environment": environment(device), "configuration": vars(cfg),
              "results": results, "initial_weights_matched": comparable,
              "interpretation": "Exploratory synthetic test; zero tensor alone is not proof of a software defect."}
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print("Saved", args.json, flush=True)
    affected = [r for r in successful if r["weak_check_passes"] and r["loss_decreased"] and r["zero_tensors"] > 0]
    print("Observed zero gradients with falling loss in:", [r["mode"] for r in affected], flush=True)
    print("A causal claim requires matched controls and repeated CUDA results.", flush=True)


if __name__ == "__main__":
    main()
