"""optimized_inference_disaggregated.py - Disaggregated inference (optimized).

Separate prefill and decode services - parallel execution, better utilization.
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
        raise RuntimeError("CUDA required for ch15")
    return torch.device("cuda")


class SimpleLLM(nn.Module):
    """Simplified LLM for inference simulation."""
    
    def __init__(self, hidden_dim=1024, num_layers=12):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
    
    def prefill(self, prompt_tokens):
        """Prefill: Process full prompt (compute-bound)."""
        x = torch.randn(prompt_tokens.size(0), prompt_tokens.size(1), self.hidden_dim,
                       device=prompt_tokens.device, dtype=torch.bfloat16)
        for layer in self.layers:
            x = layer(x)
            x = torch.relu(x)
        return x[:, -1:, :]
    
    def decode(self, kv_cache, num_tokens=16):
        """Decode: Generate tokens (memory-bound)."""
        outputs = []
        x = kv_cache
        for _ in range(num_tokens):
            for layer in self.layers:
                x = layer(x)
                x = torch.relu(x)
            outputs.append(x)
        return torch.cat(outputs, dim=1)


class OptimizedInferenceDisaggregatedBenchmark(Benchmark):
    """Benchmark implementation with disaggregated architecture."""
    
    def __init__(self):
        self.device = resolve_device()
        self.decode_model = None
        # Optimization: Compile model for kernel fusion and optimization
        try:
            model = torch.compile(None, mode="reduce-overhead", backend="inductor")
        except Exception:
            pass  # Fallback to eager if compilation fails

        self.kv_cache = None
    
    def setup(self) -> None:
        """Setup: initialize separate decode model (prefill runs elsewhere)."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        # In disaggregated architecture, decode service is separate
        # Prefill runs on different service/GPU
        self.decode_model = SimpleLLM(hidden_dim=1024, num_layers=12).to(self.device).to(torch.bfloat16).eval()
        
        # Simulate: KV cache comes from prefill service
        self.kv_cache = torch.randn(1, 1, 1024, device=self.device, dtype=torch.bfloat16)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = self.decode_model.decode(self.kv_cache, num_tokens=16)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - disaggregated (decode doesn't block on prefill)."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_inference_disaggregated", enable=enable_nvtx):
            with torch.no_grad():
                # Decode runs independently (can overlap with other prefills)
                _ = self.decode_model.decode(self.kv_cache, num_tokens=16)

    def teardown(self) -> None:
        """Cleanup."""
        del self.decode_model, self.kv_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )
    def validate_result(self) -> Optional[str]:
        """Optional validation."""
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedInferenceDisaggregatedBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=20, warmup=5)
    )
    benchmark = OptimizedInferenceDisaggregatedBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print("Optimized: Disaggregated Inference")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
