"""optimized_flash_attention.py - Optimized FlashAttention in disaggregated inference.

Demonstrates FlashAttention for memory-efficient attention computation.
Flash attention: Uses FlashAttention to reduce memory complexity.
Tiles attention computation to avoid storing full attention matrix.
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
from contextlib import contextmanager

from typing import Optional, Tuple

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch15")
    return torch.device("cuda")

class OptimizedFlashAttentionBenchmark(Benchmark):
    """Optimized: FlashAttention for memory-efficient attention.

    Flash attention: Uses FlashAttention to reduce memory complexity.
    Tiles attention computation to avoid storing full attention matrix.
    """

    def __init__(self):
        self.device = resolve_device()
        self.hidden_dim = 512
        self.num_heads = 8
        self.head_dim = self.hidden_dim // self.num_heads
        self.seq_len = 2048
        self.batch_size = 4
        self.qkv_proj: Optional[nn.Linear] = None
        self.inputs: Optional[torch.Tensor] = None

    def setup(self) -> None:
        """Setup: Initialize attention model with FlashAttention."""

        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)

        self.qkv_proj = nn.Linear(self.hidden_dim, self.hidden_dim * 3, bias=True).to(
            self.device, dtype=torch.bfloat16
        )
        self.inputs = torch.randn(
            self.batch_size,
            self.seq_len,
            self.hidden_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        torch.cuda.synchronize()

    def _project_qkv(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.qkv_proj is not None
        qkv = self.qkv_proj(x)
        return qkv.chunk(3, dim=-1)

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = tensor.shape
        return tensor.view(batch, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    @contextmanager
    def _force_flash_kernels(self):
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=False,
            ):
                yield
        else:
            yield

    def benchmark_fn(self) -> None:
        """Benchmark: FlashAttention computation."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        if self.inputs is None or self.qkv_proj is None:
            raise RuntimeError("Inputs not initialized")

        with nvtx_range("optimized_flash_attention", enable=enable_nvtx):
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                q, k, v = self._project_qkv(self.inputs)
                q = self._reshape_heads(q)
                k = self._reshape_heads(k)
                v = self._reshape_heads(v)
                with self._force_flash_kernels():
                    _ = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, is_causal=False
                    )
        torch.cuda.synchronize()

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.qkv_proj = None
        self.inputs = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=3,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.qkv_proj is None:
            return "Projection layer not initialized"
        if self.inputs is None:
            return "Input not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedFlashAttentionBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedFlashAttentionBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: flash_attention")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")

if __name__ == "__main__":
    main()
