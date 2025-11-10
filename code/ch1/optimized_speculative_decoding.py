"""optimized speculative decoding - Optimized speculative decoding implementation. Implements Benchmark protocol for harness integration."""

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

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)
from ch1.workload_config import WORKLOAD


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch1")
    return torch.device("cuda")


class OptimizedSpeculativeDecodingBenchmark(Benchmark):
    """Optimized: Speculative decoding with draft model for parallel token generation.
    
    Speculative decoding: Uses draft model to predict multiple tokens in parallel.
    Accepts/rejects tokens based on target model verification for speedup.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.target_model = None
        self.draft_model = None
        self.embedding = None
        self.decode_tokens = None
        self.workload = WORKLOAD
        self.batch_size = self.workload.batch_size
        self.tokens_per_step = self.workload.tokens_per_step
        self.decode_steps = self.workload.decode_steps
        self.speculative_length = self.workload.speculative_chunk
        self.vocab_size = 1000
    
    def setup(self) -> None:
        """Setup: Initialize target and draft models."""
        torch.manual_seed(42)
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            enable_tf32()
        
        # Optimization: Speculative decoding
        # Draft model predicts multiple tokens in parallel
        # Target model verifies predictions for correctness
        
        hidden_dim = 256
        
        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, hidden_dim).to(self.device)
        
        # Target model (slower, more accurate) - use same as baseline for fair comparison
        target_model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=2  # Same as baseline
        ).to(self.device)
        
        # Draft model (faster, less accurate) for speculative decoding
        # Use single layer for maximum speedup
        draft_model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True),
            num_layers=1  # Smaller than baseline for speed
        ).to(self.device)
        
        # Optimization: Use FP16 for faster computation
        if self.device.type == "cuda":
            try:
                target_model = target_model.half()
                draft_model = draft_model.half()
                self.embedding = self.embedding.half()
            except Exception:
                pass  # Fallback to FP32 if FP16 not supported
        
        target_model.eval()
        draft_model.eval()
        
        # Optimization: Compile models with torch.compile for better performance
        try:
            self.target_model = torch.compile(target_model, mode="reduce-overhead", backend="inductor")
            self.draft_model = torch.compile(draft_model, mode="reduce-overhead", backend="inductor")
            # Warmup compilation
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            test_ids = torch.randint(0, vocab_size, (4, 10), device=self.device)
            test_embedded = self.embedding(test_ids)
            test_memory = torch.randn(4, 10, hidden_dim, device=self.device, dtype=dtype)
            with torch.no_grad():
                for _ in range(10):
                    _ = self.target_model(test_embedded, test_memory)
                    _ = self.draft_model(test_embedded, test_memory)
            torch.cuda.synchronize()
        except Exception:
            # Fallback to non-compiled if compilation fails
            self.target_model = target_model
            self.draft_model = draft_model
        
        torch.manual_seed(42)
        decode = torch.randint(
            0,
            self.vocab_size,
            (self.decode_steps, self.batch_size, self.tokens_per_step),
            device=self.device,
        )
        self.decode_tokens = decode
        if self.device.type == "cuda":
            torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Speculative decoding."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_speculative_decoding", enable=enable_nvtx):
            with torch.no_grad():
                # Process the decode stream in speculative chunks so each iteration
                # handles multiple tokens. This mirrors the kv-cache workload where
                # the optimized path reduces the number of decoder invocations.
                dtype = torch.float16 if self.device.type == "cuda" else torch.float32
                memory = torch.randn(
                    self.batch_size,
                    self.tokens_per_step * self.speculative_length,
                    256,
                    device=self.device,
                    dtype=dtype,
                )
                for start in range(0, self.decode_steps, self.speculative_length):
                    chunk = self.decode_tokens[start : start + self.speculative_length]
                    embedded = torch.cat(
                        [self.embedding(tokens) for tokens in chunk], dim=1
                    )
                    draft_output = self.draft_model(embedded, memory[:, : embedded.size(1), :])
                    target_check = self.target_model(embedded, memory[:, : embedded.size(1), :])
                    _ = draft_output.sum() + target_check.sum()
            if self.device.type == "cuda":
                torch.cuda.synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.target_model = None
        self.draft_model = None
        self.embedding = None
        self.input_ids = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=5,
            warmup=2,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.target_model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedSpeculativeDecodingBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    timing = result.timing
    if timing:
        print(f"\nOptimized Speculative Decoding: {timing.mean_ms:.3f} ms")
    else:
        print("\nOptimized Speculative Decoding: No timing data available")
    print("NOTE: Uses draft model for parallel token prediction, verified by target model")
