"""baseline_flash_attention.py - Baseline attention without FlashAttention in GEMM context."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from typing import Optional

from common.python.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


class BaselineFlashAttentionBenchmark(BaseBenchmark):
    """Baseline: Standard attention without FlashAttention.
    
    Flash attention: This baseline does not use FlashAttention.
    Uses standard attention with quadratic memory complexity.
    """
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.input = None
        self.batch_size = 4
        self.seq_len = 512
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize attention model without FlashAttention."""
        torch.manual_seed(42)
        # Baseline: Standard attention - no FlashAttention optimization
        # FlashAttention reduces memory complexity from O(seq_len^2) to O(seq_len)
        # This baseline does not use FlashAttention
        
        hidden_dim = 256
        num_heads = 8
        
        self.model = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        ).to(self.device).eval()
        
        # Input sequence
        self.input = torch.randn(self.batch_size, self.seq_len, hidden_dim, device=self.device)
        torch.cuda.synchronize(self.device)
    
    def benchmark_fn(self) -> None:
        """Benchmark: Standard attention without FlashAttention."""
        with self._nvtx_range("baseline_flash_attention"):
            with torch.no_grad():
                _output, _ = self.model(self.input, self.input, self.input)
            self._synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_pipeline_metrics
        return compute_pipeline_metrics(
            num_stages=getattr(self, 'num_stages', 4),
            stage_times_ms=getattr(self, '_stage_times_ms', [1.0]),
        )
}

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.input is None:
            return "Input not initialized"
        return None

def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineFlashAttentionBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineFlashAttentionBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: Flash Attention")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
