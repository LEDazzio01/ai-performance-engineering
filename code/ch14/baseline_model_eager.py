"""baseline_model_eager.py - Eager mode execution (baseline).

Runs model in eager mode without torch.compile optimization.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to path for imports
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


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch14")
    return torch.device("cuda")


class SimpleTransformer(nn.Module):
    """Simple transformer for profiling."""
    
    def __init__(self, d_model=512, n_heads=8, n_layers=6, d_ff=2048, vocab_size=10000):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 2048, d_model))  # Support up to 2048 seq len
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                batch_first=True,
            )
            for _ in range(n_layers)
        ])
        
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


class BaselineModelEagerBenchmark(Benchmark):
    """Benchmark implementation following Benchmark protocol."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input_ids = None
    
    def setup(self) -> None:
        """Setup: initialize model and data."""
        batch_size = 16
        seq_len = 1024
        vocab_size = 10000
        
        self.model = SimpleTransformer().to(self.device).eval()
        self.input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(self.input_ids)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("model_eager", enable=enable_nvtx):
            with torch.no_grad():
                _ = self.model(self.input_ids)

    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.input_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
        )
    
    def validate_result(self) -> Optional[str]:
        """Optional validation."""
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineModelEagerBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=10)
    )
    benchmark = BaselineModelEagerBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print("Baseline: Eager Mode Execution")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
