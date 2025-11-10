"""baseline_inference_monolithic.py - Monolithic inference (baseline).

Single service handles both prefill and decode - blocks each other.
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


class BaselineInferenceMonolithicBenchmark(Benchmark):
    """Benchmark implementation following Benchmark protocol."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.prompt = None
        self.kv_cache = None
    
    def setup(self) -> None:
        """Setup: initialize model and data."""
        self.model = SimpleLLM(hidden_dim=1024, num_layers=12).to(self.device).to(torch.bfloat16).eval()
        self.prompt = torch.randint(0, 10000, (1, 256), device=self.device)
        
        # Pre-compute KV cache
        with torch.no_grad():
            self.kv_cache = self.model.prefill(self.prompt)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> Optional[dict]:
        """Function to benchmark - monolithic (prefill blocks decode).
        
        Returns:
            Optional dict with 'ttft_times_ms' and 'tpot_times_ms' keys for inference timing,
            or None if not measuring inference timing.
        """
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled
        import time

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("inference_monolithic", enable=enable_nvtx):
            with torch.no_grad():
                # Measure TTFT: Time from request start to first token generation
                request_start = time.perf_counter()
                
                # Prefill phase (simulates processing prompt to get first token)
                # In real inference, this would be the prefill computation
                torch.cuda.synchronize()
                prefill_start = time.perf_counter()
                kv_cache = self.model.prefill(self.prompt)
                torch.cuda.synchronize()
                prefill_end = time.perf_counter()
                
                # TTFT is the time from request start to first token ready
                ttft_ms = (prefill_end - request_start) * 1000
                
                # Measure TPOT: Time per token during decode phase
                num_tokens = 16
                tpot_times_ms = []
                
                decode_start = time.perf_counter()
                for i in range(num_tokens):
                    token_start = time.perf_counter()
                    # Simulate generating one token
                    if i == 0:
                        # First token uses kv_cache from prefill
                        token_output = self.model.decode(kv_cache, num_tokens=1)
                    else:
                        # Subsequent tokens use previous output
                        token_output = self.model.decode(token_output[:, -1:, :], num_tokens=1)
                    torch.cuda.synchronize()
                    token_end = time.perf_counter()
                    
                    # TPOT is the time per token during decode
                    tpot_ms = (token_end - token_start) * 1000
                    tpot_times_ms.append(tpot_ms)
                
                # Return inference timing data
                return {
                    "ttft_times_ms": [ttft_ms],
                    "tpot_times_ms": tpot_times_ms,
                }

    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.prompt, self.kv_cache
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
    return BaselineInferenceMonolithicBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=20, warmup=5)
    )
    benchmark = BaselineInferenceMonolithicBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print("Baseline: Monolithic Inference")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
