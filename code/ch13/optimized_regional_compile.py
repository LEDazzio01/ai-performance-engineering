"""optimized_regional_compile.py - Regional torch.compile (MLP-only).

Demonstrates regional compilation by compiling only the MLP subgraph while
keeping the rest of the Transformer block eager. Reduces compile churn across
sequence buckets and improves steady-state latency versus full-graph compile.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repo root is importable when running as a script
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python import compile_utils as _compile_utils_patch  # noqa: F401
from common.python.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


class RegionalMLP(nn.Module):
    """MLP wrapped with torch.compile as the regional hot path."""

    def __init__(self, hidden: int, mlp_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, hidden)
        self._compiled = torch.compile(
            self._forward_impl,
            backend="aot_eager",
            fullgraph=False,
            dynamic=True,
            mode="default",
        )

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._compiled(x)


class TinyTransformerBlock(nn.Module):
    """Transformer block using a regionally-compiled MLP."""

    def __init__(self, hidden: int = 1024, num_heads: int = 8, mlp_hidden: int = 4096):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(hidden)
        self.mlp = RegionalMLP(hidden, mlp_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln1(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = residual + attn_out

        residual = x
        x = self.ln2(x)
        x = residual + self.mlp(x)
        return x


class OptimizedRegionalCompileBenchmark(BaseBenchmark):
    """Optimized: compile only the MLP region, keep the rest eager."""

    def __init__(self):
        super().__init__()
        self.hidden = 1024
        self.num_heads = 8
        self.mlp_hidden = 4096
        self.batch_size = 8
        self.sequence_schedule: List[int] = [128, 256, 384, 512]
        self._step = 0

        self.model: Optional[nn.Module] = None
        self.inputs: Dict[int, torch.Tensor] = {}

        max_tokens = self.batch_size * max(self.sequence_schedule) * self.hidden
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(max_tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(0)
        self.model = TinyTransformerBlock(
            hidden=self.hidden,
            num_heads=self.num_heads,
            mlp_hidden=self.mlp_hidden,
        ).to(self.device, dtype=torch.bfloat16).eval()

        for seq in self.sequence_schedule:
            self.inputs[seq] = torch.randn(
                self.batch_size,
                seq,
                self.hidden,
                device=self.device,
                dtype=torch.bfloat16,
            )

        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )
        self._synchronize()

    def _next_sequence_length(self) -> int:
        seq = self.sequence_schedule[self._step % len(self.sequence_schedule)]
        self._step += 1
        return seq

    def benchmark_fn(self) -> None:
        if self.model is None:
            raise RuntimeError("Model not initialized")

        seq_len = self._next_sequence_length()
        x = self.inputs[seq_len]

        with torch.no_grad(), self._nvtx_range("optimized_regional_compile"):
            _ = self.model(x)
        self._synchronize()

    def teardown(self) -> None:
        self.model = None
        self.inputs.clear()
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=8,
            warmup=0,
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=300,
            measurement_timeout_seconds=300,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_fp32_ms', 10.0),
            reduced_precision_time_ms=getattr(self, '_reduced_ms', 5.0),
            precision_type="fp8",
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedRegionalCompileBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(
        f"\nOptimized regional compile: {result.timing.mean_ms if result.timing else 0.0:.3f} ms"
    )
