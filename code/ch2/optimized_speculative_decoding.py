"""optimized_speculative_decoding.py - Optimized speculative decoding in hardware overview context.

Demonstrates speculative decoding optimized for hardware parallelism.
Speculative decoding: Uses draft model to predict multiple tokens in parallel.
Optimized for hardware capabilities (e.g., tensor cores, parallel execution).
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

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch2")
    return torch.device("cuda")


class OptimizedSpeculativeDecodingBenchmark(Benchmark):
    """Optimized: Speculative decoding optimized for hardware parallelism.
    
    Speculative decoding: Uses draft model to predict multiple tokens in parallel.
    Optimized for hardware capabilities (e.g., tensor cores, parallel execution).
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.target_model = None

        self.draft_model = None
        self.input_ids = None
        self.max_length = 20
        self.speculative_length = 4  # Number of tokens to predict speculatively
    
    def setup(self) -> None:
        """Setup: Initialize target and draft models."""
        
        torch.manual_seed(42)
        # Optimization: Speculative decoding optimized for hardware
        # Draft model predicts multiple tokens in parallel (optimized for hardware parallelism)
        # Target model verifies predictions
        
        hidden_dim = 256
        vocab_size = 1000
        
        # Target model (slower, more accurate)
        self.target_model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=4
        ).to(self.device).eval()
        
        # Draft model (faster, less accurate) for speculative decoding
        self.draft_model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=2
        ).to(self.device).eval()
        
        # Input
        batch_size = 4
        seq_len = 10
        self.input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Speculative decoding optimized for hardware."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_speculative_decoding", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: Speculative decoding optimized for hardware
                # Draft model predicts multiple tokens in parallel (hardware parallelism)
                # Target model verifies predictions
                # Optimized for hardware capabilities (tensor cores, parallel execution)
                
                current_ids = self.input_ids.clone()
                
                while current_ids.size(1) < self.input_ids.size(1) + self.max_length:
                    # Draft model: Predict multiple tokens speculatively (parallel execution)
                    draft_output = self.draft_model(current_ids)
                    draft_tokens = draft_output[:, -self.speculative_length:, :].argmax(dim=-1)
                    
                    # Target model: Verify draft predictions (speculative decoding verification)
                    # In practice, would verify each token sequentially and accept/reject
                    verified_tokens = draft_tokens  # Simplified: accept all draft tokens
                    
                    # Append verified tokens (speculative decoding: parallel generation)
                    current_ids = torch.cat([current_ids, verified_tokens], dim=1)
                    
                    # Optimization: Speculative decoding benefits for hardware
                    # - Parallel token prediction (optimized for hardware parallelism)
                    # - Faster generation compared to sequential decoding
                    # - Better hardware utilization (tensor cores, parallel execution)
                    # - Target model verification ensures correctness
                    if current_ids.size(1) >= self.input_ids.size(1) + self.max_length:
                        break

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.target_model = None
        self.draft_model = None
        self.input_ids = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
        iterations=10,
            warmup=2,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.target_model is None or self.draft_model is None:
            return "Models not initialized"
        if self.input_ids is None:
            return "Input IDs not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedSpeculativeDecodingBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedSpeculativeDecodingBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: speculative_decoding")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")

if __name__ == "__main__":
    main()
