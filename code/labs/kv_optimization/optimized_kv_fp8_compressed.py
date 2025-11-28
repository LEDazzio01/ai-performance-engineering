#!/usr/bin/env python3
"""Optimized: FP8 compressed KV cache for Blackwell.

Optimized KV cache with:
- FP8 E4M3 quantization (2× memory savings)
- Dynamic scaling per layer
- Optional NVFP4 for extreme compression (4× savings)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
import sys
from pathlib import Path

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkHarness,
    BenchmarkConfig,
    BenchmarkMode,
)
from common.python.logger import get_logger

logger = get_logger(__name__)


class OptimizedKVFP8Compressed(BaseBenchmark):
    """Optimized FP8 compressed KV cache."""

    def __init__(
        self,
        batch_size: int = 8,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        max_seq_length: int = 8192,
        use_fp8: bool = True,
        use_fp4: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_length = max_seq_length
        self.use_fp8 = use_fp8
        self.use_fp4 = use_fp4
        self._last_metrics: Dict[str, Any] = {}

        # Determine precision
        if use_fp4 and hasattr(torch, 'float4_e2m1fn'):
            self.cache_dtype = torch.float4_e2m1fn
            self.bytes_per_element = 0.5
            compression_ratio = 4
        elif use_fp8 and hasattr(torch, 'float8_e4m3fn'):
            self.cache_dtype = torch.float8_e4m3fn
            self.bytes_per_element = 1
            compression_ratio = 2
        else:
            self.cache_dtype = torch.bfloat16
            self.bytes_per_element = 2
            compression_ratio = 1

        self.precision_label = str(self.cache_dtype).split(".")[-1]
        memory_per_token = num_layers * 2 * num_heads * head_dim * self.bytes_per_element
        total_memory_gb = (batch_size * max_seq_length * memory_per_token) / (1024**3)

        logger.info(f"Optimized KV Cache ({self.cache_dtype})")
        logger.info(f"  Compression: {compression_ratio}×")
        logger.info(f"  Estimated memory: {total_memory_gb:.2f} GB")

    def _resolve_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup(self):
        """Initialize compressed KV cache."""
        # Pre-allocate KV cache in compressed format
        self.kv_cache = torch.zeros(
            self.batch_size,
            self.num_layers,
            2,
            self.num_heads,
            self.max_seq_length,
            self.head_dim,
            device=self.device,
            dtype=self.cache_dtype
        )

        # Scaling factors (per layer, updated dynamically)
        self.k_scales = torch.ones(self.num_layers, device=self.device)
        self.v_scales = torch.ones(self.num_layers, device=self.device)

        # Sequence lengths
        self.seq_lengths = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        # EMA for scale smoothing
        self.scale_ema = 0.95

        logger.info("Compressed KV cache allocated")
    
    def _compute_scale(self, x: torch.Tensor) -> float:
        """Compute dynamic scaling factor."""
        absmax = x.abs().max().item()
        
        if self.use_fp4:
            max_val = 6.0  # FP4 range
        else:  # FP8
            max_val = 448.0  # FP8 E4M3 range
        
        scale = max_val / (absmax + 1e-12)
        return scale
    
    def append_kv(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None
    ):
        """Append K/V with compression."""
        if batch_indices is None:
            batch_indices = torch.arange(self.batch_size)
        
        # Update scales with EMA
        k_scale = self._compute_scale(k)
        v_scale = self._compute_scale(v)
        
        self.k_scales[layer_idx] = (
            self.scale_ema * self.k_scales[layer_idx] +
            (1 - self.scale_ema) * k_scale
        )
        self.v_scales[layer_idx] = (
            self.scale_ema * self.v_scales[layer_idx] +
            (1 - self.scale_ema) * v_scale
        )
        
        # Quantize and store
        k_quantized = (k * self.k_scales[layer_idx]).to(self.cache_dtype)
        v_quantized = (v * self.v_scales[layer_idx]).to(self.cache_dtype)
        
        for i, batch_idx in enumerate(batch_indices):
            pos = self.seq_lengths[batch_idx].item()
            self.kv_cache[batch_idx, layer_idx, 0, :, pos] = k_quantized[i]
            self.kv_cache[batch_idx, layer_idx, 1, :, pos] = v_quantized[i]
            self.seq_lengths[batch_idx] += 1
    
    def get_kv(
        self,
        layer_idx: int,
        batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve and dequantize K/V."""
        seq_len = self.seq_lengths[batch_idx].item()
        
        k_quantized = self.kv_cache[batch_idx, layer_idx, 0, :, :seq_len]
        v_quantized = self.kv_cache[batch_idx, layer_idx, 1, :, :seq_len]
        
        # Dequantize
        k = k_quantized.to(torch.bfloat16) / self.k_scales[layer_idx]
        v = v_quantized.to(torch.bfloat16) / self.v_scales[layer_idx]
        
        return k, v
    
    def benchmark_fn(self) -> None:
        """Benchmark compressed KV cache."""
        import time

        num_decode_steps = 100

        self._synchronize()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()

        for _ in range(num_decode_steps):
            new_k = torch.randn(
                self.batch_size, self.num_heads, self.head_dim,
                device=self.device, dtype=torch.bfloat16
            )
            new_v = torch.randn_like(new_k)

            for layer_idx in range(min(4, self.num_layers)):
                self.append_kv(layer_idx, new_k, new_v)

        self._synchronize()
        elapsed = time.perf_counter() - start

        if torch.cuda.is_available():
            memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)
        else:
            memory_gb = 0.0
        tokens_per_sec = (self.batch_size * num_decode_steps) / elapsed

        logger.info(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
        logger.info(f"Memory: {memory_gb:.2f} GB (FP8 compressed)")

        self._last_metrics = {
            "latency_ms": elapsed * 1000,
            "tokens_per_sec": tokens_per_sec,
            "memory_gb": memory_gb,
            "compression_ratio": 2.0 / self.bytes_per_element,
        }

    def get_custom_metrics(self) -> Dict[str, Any]:
        return self._last_metrics

    def get_optimization_goal(self) -> str:
        """Memory optimization - lower memory usage is better."""
        return "memory"

    def teardown(self):
        """Clean up."""
        del self.kv_cache
        super().teardown()


def run_benchmark(
    batch_size: int = 8,
    num_layers: int = 32,
    max_seq_length: int = 8192,
    use_fp8: bool = True,
    use_fp4: bool = False,
    profile: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """Run optimized KV cache benchmark."""

    benchmark = OptimizedKVFP8Compressed(
        batch_size=batch_size,
        num_layers=num_layers,
        max_seq_length=max_seq_length,
        use_fp8=use_fp8,
        use_fp4=use_fp4,
    )

    config = BenchmarkConfig(
        iterations=1,
        warmup=5,
        profile_mode=profile,
        use_subprocess=False,  # keep metrics in-process
    )
    harness = BenchmarkHarness(mode=BenchmarkMode.INFERENCE, config=config)

    result = harness.benchmark(benchmark, name="optimized_kv_fp8_compressed")

    metrics = result.custom_metrics or {}
    return {
        "mean_time_ms": result.timing.mean_ms,
        "precision": benchmark.precision_label,
        **metrics,
    }


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedKVFP8Compressed()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized FP8 KV Cache")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--use-fp4", action="store_true",
                       help="Use FP4 for 4× compression (Blackwell only)")
    parser.add_argument("--profile", type=str, default="none")
    
    args = parser.parse_args()
    
    result = run_benchmark(
        batch_size=args.batch_size,
        num_layers=args.num_layers,
        max_seq_length=args.max_seq_length,
        use_fp4=args.use_fp4,
        profile=args.profile,
    )
    
    print(f"\n{'='*60}")
    print(f"Optimized FP8 KV Cache Results")
    print(f"{'='*60}")
    print(f"Precision: {result['precision']}")
    print(f"Compression: {result['compression_ratio']:.1f}×")
    print(f"Memory: {result['memory_gb']:.2f} GB")
    print(f"Throughput: {result['tokens_per_sec']:.2f} tokens/sec")
    print(f"{'='*60}\n")
    print(f"Expected: {result['compression_ratio']:.0f}× memory savings on Blackwell")

