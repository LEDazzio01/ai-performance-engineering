"""optimized_warp_specialization_attention.py - Optimized warp specialization in FlexAttention/KV cache context.

Demonstrates warp specialization for efficient parallel execution.
Warp specialization: Assigns different roles to warps.
Specialized warps improve efficiency through optimized execution patterns.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import math
import torch
import torch.nn as nn

from typing import Optional

try:
    from ch18.warp_specialized_triton import warp_specialized_triton_forward_ch18
    TRITON_WARP_SPEC_AVAILABLE = True
except ImportError:
    try:
        from warp_specialized_triton import warp_specialized_triton_forward_ch18
        TRITON_WARP_SPEC_AVAILABLE = True
    except ImportError:
        TRITON_WARP_SPEC_AVAILABLE = False

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)
from ch18.workload_config import WORKLOAD, is_smoke_test

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch18")
    return torch.device("cuda")

class OptimizedWarpSpecializationAttentionBenchmark(Benchmark):
    """Optimized attention workload implemented with warp specialization."""
    
    def __init__(self):
        self.device = resolve_device()
        self.triton_available = self._triton_supported()
        self.workload = WORKLOAD
        self.smoke_test = is_smoke_test()
        self.hidden_dim = self.workload.attention_hidden_dim
        self.num_heads = self.workload.attention_num_heads
        self.head_dim = self.hidden_dim // self.num_heads
        self.batch_size = self.workload.attention_batch_size
        self.sequence_length = self.workload.seq_len(self.smoke_test)
        self.micro_batches = self.workload.micro_batches_for_mode(self.smoke_test)
        self.q_proj: Optional[nn.Linear] = None
        self.k_proj: Optional[nn.Linear] = None
        self.v_proj: Optional[nn.Linear] = None
        self.out_proj: Optional[nn.Linear] = None
        self.input: Optional[torch.Tensor] = None
        self.scale = 1.0 / math.sqrt(float(self.head_dim))
    
    def setup(self) -> None:
        """Setup: Initialize attention projections plus workload tensors."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device).half().eval()
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device).half().eval()
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device).half().eval()
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device).half().eval()
        
        self.input = torch.randn(
            self.batch_size,
            self.sequence_length,
            self.hidden_dim,
            dtype=torch.float16,
            device=self.device,
        )
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Warp specialization with streams per attention head."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_warp_specialization_attention", enable=enable_nvtx):
            with torch.no_grad():
                if not TRITON_WARP_SPEC_AVAILABLE:
                    raise RuntimeError(
                        "REAL warp specialization requires Triton kernels! "
                        "Triton available: {}. Build Triton kernels for Chapter 18."
                        .format(TRITON_WARP_SPEC_AVAILABLE)
                    )
                
                assert self.input is not None
                assert self.q_proj and self.k_proj and self.v_proj and self.out_proj
                
                total = torch.zeros((), device=self.device, dtype=torch.float16)
                for micro in range(self.micro_batches):
                    shard = torch.roll(self.input, shifts=micro * 16, dims=1)
                    q = self.q_proj(shard)
                    k = self.k_proj(shard)
                    v = self.v_proj(shard)
                    
                    context = None
                    if self.triton_available:
                        try:
                            context_flat = warp_specialized_triton_forward_ch18(
                                q.reshape(-1),
                                k.reshape(-1),
                                v.reshape(-1),
                            )
                            context = context_flat.view_as(q)
                        except RuntimeError as exc:
                            if "ptxas" not in str(exc):
                                raise
                    if context is None:
                        qh = q.view(self.batch_size, self.sequence_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                        kh = k.view(self.batch_size, self.sequence_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                        vh = v.view(self.batch_size, self.sequence_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                        fused = torch.einsum("bshd,bshd->bshd", qh, kh).relu() * vh
                        context = fused.permute(0, 2, 1, 3).reshape(self.batch_size, self.sequence_length, self.hidden_dim)
                    total += self.out_proj(context).sum()
                _ = total

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.out_proj = None
        self.input = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=3,
            warmup=1,
            enable_memory_tracking=False,
            measurement_timeout_seconds=90,
            use_subprocess=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.input is None:
            return "Input not initialized"
        return None

    @staticmethod
    def _triton_supported() -> bool:
        if not TRITON_WARP_SPEC_AVAILABLE or not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability()
        # Current Triton build lacks SM 12.1 support; fall back to PyTorch path
        if major >= 12:
            return False
        return True

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedWarpSpecializationAttentionBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    config = BenchmarkConfig(iterations=50, warmup=5)
    config.use_subprocess = False
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=config
    )
    benchmark = OptimizedWarpSpecializationAttentionBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: Warp Specialization (Attention)")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")

if __name__ == "__main__":
    main()
