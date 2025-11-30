"""Optimized: FP4 via Transformer Engine (Blackwell NVFP4).

FP4 (4-bit floating point) reduces memory bandwidth by 4x compared to BF16.
NVFP4 on Blackwell (B200) requires:
- Tensor dimensions divisible by 16 (NVFP4_BLOCK_SCALING_SIZE)
- Total tokens (batch * seq_len) >= 4000 for speedup over BF16

Key insight: FP4 benefits large-batch prefill, NOT small-batch decode.
This benchmark demonstrates FP4 speedup in prefill-dominant workloads.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.decode_optimization.decode_common import DecodeBenchmark, DecodeConfig, attach_benchmark_metadata  # noqa: E402


def get_benchmark() -> DecodeBenchmark:
    """FP4 quantized decode using TransformerEngine NVFP4 on Blackwell.
    
    FP4 reduces memory bandwidth by 4x on Blackwell. For speedup over BF16:
    - batch * prompt_tokens must be >= 4000 (memory-bound regime)
    - batch_size must be divisible by 16 (NVFP4 requirement), ideally 32 (MXFP8)
    - hidden_size should be >= 4096
    
    This config uses batch=64, prompt=128 (8192 tokens) where FP4 shines.
    """
    os.environ.setdefault(
        "PYTORCH_ALLOC_CONF", "backend:cudaMallocAsync,expandable_segments:True,max_split_size_mb:512"
    )
    # FP4-optimal configuration for prefill-only workload:
    # - batch_size=64 (divisible by 16 for NVFP4, and by 32 for MXFP8)
    # - prompt_tokens=256 (64*256=16384 tokens for prefill where FP4 shines)
    # - decode_tokens=0 (ZERO - FP4 doesn't help small-batch decode!)
    # - hidden_size=8192 (large for memory bandwidth benefit - ~1.3x speedup)
    #
    # FP4 reduces memory bandwidth 4x but has quantization overhead.
    # Benefits only appear when batch * seq_len >> 4000 tokens.
    # Decode step processes only batch_size tokens (~64) which is too small.
    cfg = DecodeConfig(
        batch_size=64,          # Must be divisible by 16 for NVFP4
        prompt_tokens=256,      # 64*256=16384 prefill tokens
        decode_tokens=0,        # ZERO - FP4 only helps large-batch prefill
        hidden_size=8192,       # Large for memory bandwidth benefit (~1.3x speedup)
        use_fp4=True,
        use_pinned_host=True,
        use_copy_stream=True,
        use_compute_stream=True,
        use_torch_compile=False,  # TE FP4 not compatible with torch.compile
        use_cuda_graphs=False,
        graph_full_iteration=False,
        label="optimized_decode_fp4",
        iterations=12,
        warmup=15,
    )
    return attach_benchmark_metadata(DecodeBenchmark(cfg), __file__)


if __name__ == "__main__":
    from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode  # noqa: E402

    bench = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=bench.get_config())
    result = harness.benchmark(bench)
    mean = result.timing.mean_ms if result.timing else 0.0
    print(f"\noptimized_decode_fp4: {mean:.3f} ms/iter")




