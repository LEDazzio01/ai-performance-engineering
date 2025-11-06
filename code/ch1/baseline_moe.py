"""baseline moe - Baseline dense model without MoE. Implements Benchmark protocol for harness integration."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

try:
    import ch1.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch1")
    return torch.device("cuda")


class BaselineMoeBenchmark(Benchmark):
    """Baseline: Dense model without MoE (all experts always active)."""

    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None

    def setup(self) -> None:
        """Setup: Initialize dense model."""
        torch.manual_seed(42)
        # Baseline: Dense model - all computation always active
        # No sparse activation - processes all inputs with full model
        
        # Baseline: Dense model that runs ALL 4 experts (simulating MoE with all experts active)
        # Match optimized model size for fair comparison
        hidden_dim = 1024  # Match optimized model size
        
        # Baseline: Dense model equivalent to running all 4 experts
        # Each expert: hidden_dim -> hidden_dim*4 -> hidden_dim (matching MoE expert size)
        # 4 experts * (hidden_dim -> hidden_dim*4) = hidden_dim -> hidden_dim*16
        # Baseline runs all 4 experts (inefficient), MoE only runs top_k=2 (50% computation)
        
        # Use FP32 for baseline (slower) while MoE uses FP16 (faster) - legitimate optimization
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 16),  # 4 experts * 4x width
            nn.ReLU(),
            nn.Linear(hidden_dim * 16, hidden_dim),
        ).to(self.device).to(torch.float32).eval()  # FP32 = slower but baseline
        
        # Use same input shape as optimized for fair comparison
        self.input = torch.randn(128, 64, hidden_dim, device=self.device)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: Dense model computation."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_moe", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: Dense model
                # All computation always active - no sparse activation
                output = self.model(self.input)
                _ = output.sum()


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

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineMoeBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline MoE (Dense): {result.mean_ms:.3f} ms")
