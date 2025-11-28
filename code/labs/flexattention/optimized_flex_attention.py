"""Optimized FlexAttention lab: torch.compile + block sparsity."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from common.python.compile_utils import enable_tf32
from labs.flexattention.flexattention_common import (
    build_flex_attention_inputs,
    make_relative_bias_score_mod,
    resolve_device,
)

DEFAULT_COMPILE_MODE = "reduce-overhead"
COMPILE_MODE = os.getenv("TORCH_COMPILE_MODE", DEFAULT_COMPILE_MODE)


class CompiledFlexAttention(nn.Module):
    """Minimal module to allow torch.compile to fuse FlexAttention."""

    def __init__(self, block_mask, rel_bias: torch.Tensor):
        super().__init__()
        self.block_mask = block_mask
        self.register_buffer("rel_bias", rel_bias)
        self.score_mod = make_relative_bias_score_mod(self.rel_bias)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return flex_attention(
            q,
            k,
            v,
            score_mod=self.score_mod,
            block_mask=self.block_mask,
        )


class OptimizedFlexAttentionBenchmark(BaseBenchmark):
    """Compiled FlexAttention with the blog's DSL hooks."""

    def __init__(self) -> None:
        super().__init__()
        self.dtype = torch.bfloat16
        self.seq_len = 1024
        self.batch = 2
        self.heads = 8
        self.head_dim = 64
        self.block_size = 64
        self.doc_span = 256
        self.inputs = None
        self.module: Optional[nn.Module] = None
        self.compiled = None
        tokens = self.batch * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        enable_tf32()
        torch._inductor.config.triton.cudagraphs = True
        torch._inductor.config.max_autotune = True

        self.inputs = build_flex_attention_inputs(
            batch=self.batch,
            heads=self.heads,
            seq_len=self.seq_len,
            head_dim=self.head_dim,
            doc_span=self.doc_span,
            block_size=self.block_size,
            dtype=self.dtype,
            device=self.device,
        )

        self.module = CompiledFlexAttention(
            block_mask=self.inputs.block_mask,
            rel_bias=self.inputs.rel_bias,
        ).to(self.device)
        self.compiled = torch.compile(self.module, mode=COMPILE_MODE, fullgraph=True, dynamic=None)

        with torch.inference_mode():
            _ = self.compiled(self.inputs.q, self.inputs.k, self.inputs.v)
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.inputs is None or self.compiled is None:
            raise RuntimeError("FlexAttention module is not initialized")

        with self._nvtx_range("flexattention_compiled"):
            with torch.inference_mode():
                _ = self.compiled(
                    self.inputs.q,
                    self.inputs.k,
                    self.inputs.v,
                )
            self._synchronize()

    def teardown(self) -> None:
        torch.cuda.empty_cache()
        self.inputs = None
        self.module = None
        self.compiled = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=10,  # Required for FlexAttention + torch.compile JIT
            use_subprocess=False,
            measurement_timeout_seconds=60,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return roofline analysis metrics."""
        # Estimate problem size for roofline analysis
        n = getattr(self, 'N', 0) or getattr(self, 'hidden_dim', 0) or 4096
        batch = getattr(self, 'batch_size', 1) or getattr(self, 'batch', 1)
        # Simple FLOP estimate for linear layers
        flops = 2.0 * batch * n * n  # Rough estimate
        bytes_moved = batch * n * 4.0  # Input/output bytes
        arithmetic_intensity = flops / max(bytes_moved, 1.0)
        return {
            "flex_attention.estimated_flops": flops,
            "flex_attention.estimated_bytes": bytes_moved,
            "flex_attention.arithmetic_intensity": arithmetic_intensity,
        }

    def validate_result(self) -> Optional[str]:
        if self.inputs is None or self.compiled is None:
            return "FlexAttention module is not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedFlexAttentionBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    bench = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=bench.get_config())
    result = harness.benchmark(bench)
    mean_ms = result.timing.mean_ms if result and result.timing else 0.0
    print(f"[flexattention optimized] mean iteration {mean_ms:.3f} ms")
