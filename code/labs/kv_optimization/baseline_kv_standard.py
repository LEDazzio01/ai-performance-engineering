#!/usr/bin/env python3
"""Baseline: Standard KV cache without compression.

Standard KV cache using BF16 precision without optimization.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple
import sys
from pathlib import Path

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkHarness,
    BenchmarkConfig,
    BenchmarkMode,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.utils.logger import get_logger

logger = get_logger(__name__)


class BaselineKVStandard(VerificationPayloadMixin, BaseBenchmark):
    """Baseline KV cache (BF16, no compression).
    
    Goal: memory - This benchmark measures memory usage for KV cache.
    """

    def __init__(
        self,
        batch_size: int = 8,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        max_seq_length: int = 8192,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_length = max_seq_length
        self._last_metrics: Dict[str, Any] = {}
        self.precision_label = "bf16"
        self.output: Optional[torch.Tensor] = None
        self.register_workload_metadata(requests_per_iteration=1.0)

        hidden_size = num_heads * head_dim
        memory_per_token = num_layers * 2 * num_heads * head_dim * 2  # 2 for K/V, 2 bytes for BF16
        total_memory_gb = (batch_size * max_seq_length * memory_per_token) / (1024**3)

        logger.info(f"Baseline KV Cache (BF16)")
        logger.info(f"  Estimated memory: {total_memory_gb:.2f} GB")

    def _resolve_device(self):
        # Allow CPU fallback for environments without CUDA.
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup(self):
        """Initialize KV cache."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        # Pre-allocate KV cache
        # Shape: [batch, num_layers, 2, num_heads, max_seq, head_dim]
        self.kv_cache = torch.zeros(
            self.batch_size,
            self.num_layers,
            2,  # K and V
            self.num_heads,
            self.max_seq_length,
            self.head_dim,
            device=self.device,
            dtype=torch.bfloat16
        )

        # Current sequence lengths per batch
        self.seq_lengths = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)

        logger.info("KV cache allocated")

    def append_kv(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_indices: Optional[torch.Tensor] = None
    ):
        """Append K/V to cache."""
        if batch_indices is None:
            batch_indices = torch.arange(self.batch_size, device=self.device)

        for i, batch_idx in enumerate(batch_indices):
            pos = self.seq_lengths[batch_idx].item()
            self.kv_cache[batch_idx, layer_idx, 0, :, pos] = k[i]
            self.kv_cache[batch_idx, layer_idx, 1, :, pos] = v[i]
            self.seq_lengths[batch_idx] += 1

    def get_kv(
        self,
        layer_idx: int,
        batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve K/V from cache."""
        seq_len = self.seq_lengths[batch_idx].item()
        k = self.kv_cache[batch_idx, layer_idx, 0, :, :seq_len]
        v = self.kv_cache[batch_idx, layer_idx, 1, :, :seq_len]
        return k, v

    def benchmark_fn(self) -> None:
        """Benchmark KV cache operations."""
        import time

        # Simulate decoding
        num_decode_steps = 100

        self._synchronize()
        start = time.perf_counter()

        for _ in range(num_decode_steps):
            # Generate new K/V (simulating decode step)
            new_k = torch.randn(
                self.batch_size, self.num_heads, self.head_dim,
                device=self.device, dtype=torch.bfloat16
            )
            new_v = torch.randn_like(new_k)

            # Append to cache
            for layer_idx in range(min(4, self.num_layers)):  # Test with 4 layers
                self.append_kv(layer_idx, new_k, new_v)

        self._synchronize()
        elapsed = time.perf_counter() - start

        # Memory usage
        if torch.cuda.is_available():
            memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)
        else:
            memory_gb = 0.0

        tokens_per_sec = (self.batch_size * num_decode_steps) / elapsed

        logger.info(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
        logger.info(f"Memory: {memory_gb:.2f} GB")

        self._last_metrics = {
            "latency_ms": elapsed * 1000,
            "tokens_per_sec": tokens_per_sec,
            "memory_gb": memory_gb,
        }

        # Capture a slice of KV cache for verification (layer 0, first token/head window)
        view = self.kv_cache[:1, :1, :, :, : min(1, self.kv_cache.shape[4]), : min(8, self.kv_cache.shape[5])]
        self.output = view.detach().float().clone()

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={
                "batch_size": torch.tensor([self.batch_size], dtype=torch.int64, device="cpu"),
                "seq_lengths": self.seq_lengths.detach().clone(),
            },
            output=self.output,
            batch_size=self.batch_size,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": True, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(0.1, 1.0),
        )

    def get_custom_metrics(self) -> Dict[str, Any]:
        return self._last_metrics

    def get_optimization_goal(self) -> str:
        """Memory optimization - lower memory usage is better."""
        return "memory"

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=5,
            enable_memory_tracking=True,
        )

    def teardown(self):
        """Clean up."""
        del self.kv_cache
        self.output = None
        super().teardown()


def run_benchmark(
    batch_size: int = 8,
    num_layers: int = 32,
    num_heads: int = 32,
    head_dim: int = 128,
    max_seq_length: int = 8192,
    profile: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """Run baseline KV cache benchmark."""

    benchmark = BaselineKVStandard(
        batch_size=batch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        max_seq_length=max_seq_length,
    )

    config = BenchmarkConfig(
        iterations=1,
        warmup=5,
        profile_mode=profile,
        use_subprocess=False,  # keep metrics available to caller
    )
    harness = BenchmarkHarness(mode=BenchmarkMode.INFERENCE, config=config)

    result = harness.benchmark(benchmark, name="baseline_kv_standard")

    metrics = result.custom_metrics or {}
    return {
        "mean_time_ms": result.timing.mean_ms,
        "precision": benchmark.precision_label,
        **metrics,
    }


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineKVStandard()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
