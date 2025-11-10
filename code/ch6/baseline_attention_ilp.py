"""baseline_attention_ilp.py - Baseline attention with low ILP.

Demonstrates attention operations that limit instruction-level parallelism.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
from typing import Optional
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)
from ch6.workload_config import WORKLOAD, is_smoke_test


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch6")
    return torch.device("cuda")


class BaselineAttentionILPBenchmark(Benchmark):
    """Baseline: Attention with low ILP (sequential operations)."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None
        self.workload = WORKLOAD
        self.smoke_test = is_smoke_test()
        self.batch = self.workload.attention_batch
        self.embed_dim = self.workload.attention_embed_dim
        self.tokens = self.workload.attention_tokens_for_mode(self.smoke_test)
        self.query_chunk = self.workload.attention_chunk_for_mode(self.smoke_test)

    def setup(self) -> None:
        """Setup: Initialize attention model."""
        torch.manual_seed(42)
        # Baseline: Attention with low ILP
        # Sequential attention operations limit instruction-level parallelism
        # This baseline does not optimize for ILP
        self.model = nn.MultiheadAttention(
            self.embed_dim,
            self.workload.attention_heads,
            batch_first=True,
        ).to(self.device).eval()
        self.model = self.model.half()
        self.input = torch.randn(
            self.batch,
            self.tokens,
            self.embed_dim,
            device=self.device,
            dtype=torch.float16,
        )
        self._last_sum = torch.tensor(0.0, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Attention with low ILP."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_attention_ilp", enable=enable_nvtx):
            with torch.no_grad():
                accum = torch.zeros(1, device=self.device, dtype=self.input.dtype)
                for query in self.input.split(self.query_chunk, dim=1):
                    out = self.model(query, self.input, self.input)[0]
                    accum += out.sum()
                    torch.cuda.synchronize()
                self._last_sum = accum

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=self.workload.ilp_iterations,
            warmup=self.workload.ilp_warmup,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineAttentionILPBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Attention ILP: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print("  Note: Sequential attention operations limit instruction-level parallelism")
