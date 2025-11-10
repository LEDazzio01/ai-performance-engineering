"""baseline_speculative_decoding.py - Baseline decoding without speculative execution in inference/profiling.

Demonstrates standard autoregressive decoding without speculative decoding optimization.
Speculative decoding: This baseline does not use speculative decoding.
Generates tokens one at a time, sequential and slow.
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
        raise RuntimeError("CUDA required for ch17")
    return torch.device("cuda")


class BaselineSpeculativeDecodingBenchmark(Benchmark):
    """Baseline: Standard autoregressive decoding (no speculative execution)."""
    
    def __init__(self):
        self.device = resolve_device()
        self.embedding = None
        self.decoder = None
        self.output_head = None
        self.input_ids = None
        self.hidden_state = None
        self.memory = None
        self.max_length = 64
        self.hidden_dim = 512
        self.vocab_size = 16000
        self.batch_size = 8
        self.seq_len = 64
        self.num_layers = 1
    
    def setup(self) -> None:
        """Setup: Initialize model and input."""
        torch.manual_seed(42)
        dtype = torch.float32
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim, device=self.device, dtype=dtype)
        self.decoder = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        ).to(self.device, dtype=dtype).eval()
        self.output_head = nn.Linear(self.hidden_dim, self.vocab_size, device=self.device, dtype=dtype)
        
        self.input_ids = torch.randint(
            0,
            self.vocab_size,
            (self.batch_size, self.seq_len),
            device=self.device,
            dtype=torch.long,
        )
        self.hidden_state = torch.zeros(
            self.num_layers,
            self.batch_size,
            self.hidden_dim,
            device=self.device,
            dtype=dtype,
        )
        self.memory = torch.zeros(
            self.batch_size,
            self.seq_len,
            self.hidden_dim,
            device=self.device,
            dtype=dtype,
        )
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Standard autoregressive decoding."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("baseline_speculative_decoding", enable=enable_nvtx):
            with torch.no_grad():
                tokens = self.input_ids.clone()
                for _ in range(self.max_length):
                    tgt_embeddings = self.embedding(tokens)
                    out, _ = self.decoder(tgt_embeddings, self.hidden_state)
                    logits = self.output_head(out[:, -1, :])
                    next_token = logits.argmax(dim=-1, keepdim=True)
                    tokens = torch.cat([tokens, next_token], dim=1)
        torch.cuda.synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.decoder = None
        self.embedding = None
        self.output_head = None
        self.input_ids = None
        self.memory = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.decoder is None or self.embedding is None or self.output_head is None:
            return "Model not initialized"
        if self.input_ids is None:
            return "Input IDs not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineSpeculativeDecodingBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineSpeculativeDecodingBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: speculative_decoding")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
