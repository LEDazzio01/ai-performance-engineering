"""baseline_model_eager.py - Eager mode execution (baseline)."""

from __future__ import annotations

from common.python import compile_utils as _compile_utils_patch  # noqa: F401
import sys
from pathlib import Path

# Add repo root to path for imports
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


class BaselineModelEagerBenchmark(BaseBenchmark):
    """Benchmark implementation following BaseBenchmark."""
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.input_ids = None
        # Increase work so compile path has enough compute to amortize overhead.
        self.batch_size = 24
        self.seq_len = 1536
        self.vocab_size = 10000
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: initialize model and data."""
        self.model = SimpleTransformer().to(self.device).eval()
        self.input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len), device=self.device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(self.input_ids)
        torch.cuda.synchronize(self.device)
    
    def benchmark_fn(self) -> None:
        """Function to benchmark."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("model_eager", enable=enable_nvtx):
            with torch.no_grad():
                _ = self.model(self.input_ids)
        self._synchronize()

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
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_triton_metrics
        return compute_triton_metrics(
            num_elements=getattr(self, 'N', getattr(self, 'num_elements', 1024)),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            block_size=getattr(self, 'BLOCK_SIZE', 1024),
            num_warps=getattr(self, 'num_warps', 4),
        )
}

    def validate_result(self) -> Optional[str]:
        """Optional validation."""
        return None


def get_benchmark() -> BaseBenchmark:
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
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
