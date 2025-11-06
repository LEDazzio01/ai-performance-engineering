"""baseline_inference_monolithic.py - Monolithic inference baseline (baseline).

Single-node inference where prefill and decode phases run sequentially on same GPU.
No separation of concerns, leading to resource underutilization.

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
    BenchmarkHarness,
    BenchmarkMode,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch20")
    return torch.device("cuda")


class SimpleTransformer(nn.Module):
    """Simple transformer for inference demonstration."""
    
    def __init__(self, hidden_dim: int = 1024, num_layers: int = 6):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                dtype=torch.float16
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class BaselineInferenceMonolithicBenchmark(Benchmark):
    """Monolithic inference baseline - prefill and decode on same GPU."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.prompt = None
        self.hidden_dim = 1024
        self.prompt_len = 512
        self.decode_steps = 10
    
    def setup(self) -> None:
        """Setup: Initialize model and prompt."""
        torch.manual_seed(42)
        
        self.model = SimpleTransformer(hidden_dim=self.hidden_dim, num_layers=6).to(self.device).half().eval()
        self.prompt = torch.randn(1, self.prompt_len, self.hidden_dim, device=self.device, dtype=torch.float16)
        
        # Warmup
        with torch.no_grad():
            _ = self.model(self.prompt)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - monolithic inference."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_inference_monolithic", enable=enable_nvtx):
            with torch.no_grad():
                # Monolithic: prefill and decode on same GPU sequentially
                # 1. Prefill phase (process entire prompt)
                prefill_output = self.model(self.prompt)
                
                # 2. Decode phase (generate tokens one by one)
                # Use last token from prefill as starting point
                current_token = prefill_output[:, -1:, :]
                for _ in range(self.decode_steps):
                    # Generate next token
                    next_token = self.model(current_token)
                    current_token = next_token[:, -1:, :]

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.prompt
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineInferenceMonolithicBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Monolithic Inference: {result.mean_ms:.3f} ms")

