"""CPU-only contract tests. Passing these does not establish CUDA/FP16 behavior."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autocast_backward_probe import Config, Encoder, MODES, summarize_gradients, train_arm


class ProbeContractTests(unittest.TestCase):
    def test_invalid_experiment_geometry_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            Config(width=15, heads=4).validate()
        with self.assertRaisesRegex(ValueError, "visible"):
            Config(tokens=16, blocks=2, block_size=8).validate()
        with self.assertRaisesRegex(ValueError, "microbatch"):
            Config(batch=3, microbatch=2).validate()

    def test_zero_gradient_is_not_missing_or_nonfinite(self):
        module = Encoder(tokens=16, width=8, heads=2, depth=1, checkpointing=False)
        params = dict(module.named_parameters())
        # Only one parameter has a gradient. The rest are intentionally missing.
        params["blocks.0.attention.query.weight"].grad = torch.zeros_like(params["blocks.0.attention.query.weight"])
        report = summarize_gradients(module)
        row = next(x for x in report["records"] if x["name"] == "blocks.0.attention.query.weight")
        self.assertEqual(row["status"], "zero")
        self.assertGreater(report["missing"], 0)
        params["blocks.0.attention.query.weight"].grad.view(-1)[0] = 1e-23
        row = next(x for x in summarize_gradients(module)["records"] if x["name"] == "blocks.0.attention.query.weight")
        self.assertEqual(row["status"], "live")  # no squaring-induced underflow

    def test_cpu_autocast_scoping_and_shared_seed(self):
        cfg = Config(tokens=32, width=16, heads=4, depth=1, views=2,
                     batch=2, microbatch=1, blocks=2, block_size=4,
                     updates=1, checkpointing=False)
        results = [train_arm(cfg, torch.device("cpu"), torch.bfloat16, mode) for mode in MODES]
        self.assertEqual(len({x["initial_student_sha256"] for x in results}), 1)
        self.assertEqual({x["mode"] for x in results}, set(MODES))
        for result in results:
            self.assertEqual(result["backward_autocast_enabled"], [result["mode"] in ("inside_shared", "inside_per_view")])
            self.assertGreater(result["live_tensors"], 0)
            self.assertEqual(result["missing_tensors"], 0)
            self.assertEqual(result["nonfinite_tensors"], 0)
            self.assertEqual(len(result["losses"]), 1)


if __name__ == "__main__":
    unittest.main()
