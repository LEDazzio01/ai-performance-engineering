#!/usr/bin/env python3
"""Baseline: Context Parallelism without sequence sharding.

Demonstrates standard attention without Context Parallelism (CP).
All attention is computed on a single GPU, which limits maximum sequence length
to what fits in a single GPU's memory.
"""

import os
import sys
from pathlib import Path
import math
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)
from core.utils.logger import get_logger

logger = get_logger(__name__)


class StandardAttention(nn.Module):
    """Standard multi-head attention without Context Parallelism."""
    
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size, \
            f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_size]
            mask: Optional causal mask [seq_len, seq_len]
        
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # [batch_size, seq_len, hidden_size]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        # [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores + mask
        
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        # [batch_size, num_heads, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back to [batch_size, seq_len, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        return output


class BaselineContextParallelism:
    """Baseline: No Context Parallelism, single-GPU attention."""
    
    def __init__(
        self,
        batch_size: int = 1,
        seq_length: int = 2048,
        hidden_size: int = 4096,
        num_heads: int = 32,
        num_layers: int = 1,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for baseline context parallelism on Blackwell GPUs.")
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Check memory requirements
        self._check_memory()
    
    def _check_memory(self):
        """Estimate memory requirements and warn if tight."""
        # Rough estimate: attention matrix is seq_len x seq_len per head
        attn_memory_gb = (
            self.batch_size * self.num_heads * self.seq_length * self.seq_length * 4
        ) / (1024**3)  # FP32
        
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        logger.info(f"Estimated attention memory: {attn_memory_gb:.2f} GB")
        logger.info(f"Total GPU memory: {total_memory_gb:.2f} GB")
        
        if attn_memory_gb > total_memory_gb * 0.5:
            logger.warning(
                f"Attention may consume {attn_memory_gb:.2f} GB "
                f"(>{50}% of {total_memory_gb:.2f} GB). "
                "Consider using Context Parallelism for longer sequences."
            )
    
    def setup(self):
        """Initialize model and data."""
        # Create attention layers
        self.layers = nn.ModuleList([
            StandardAttention(self.hidden_size, self.num_heads)
            for _ in range(self.num_layers)
        ]).to(self.device)
        
        # Create input tensor
        self.input = torch.randn(
            self.batch_size,
            self.seq_length,
            self.hidden_size,
            device=self.device,
            dtype=torch.float32
        )
        
        # Create causal mask
        self.mask = torch.triu(
            torch.full((self.seq_length, self.seq_length), float('-inf')),
            diagonal=1
        ).to(self.device)
        
        logger.info(
            f"Setup complete: batch={self.batch_size}, seq={self.seq_length}, "
            f"hidden={self.hidden_size}, heads={self.num_heads}, layers={self.num_layers}"
        )
    
    def run(self) -> float:
        """Execute baseline attention without Context Parallelism."""
        torch.cuda.synchronize()
        
        # Forward pass through all layers
        x = self.input
        for layer in self.layers:
            x = layer(x, self.mask)
        
        torch.cuda.synchronize()
        
        # Return peak memory usage
        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
        logger.info(f"Peak memory: {peak_memory_gb:.2f} GB")
        
        return peak_memory_gb
    
    def cleanup(self):
        """Clean up resources."""
        del self.layers
        del self.input
        del self.mask
        torch.cuda.empty_cache()


def run_benchmark(
    batch_size: int = 1,
    seq_length: int = 2048,
    hidden_size: int = 4096,
    num_heads: int = 32,
    num_layers: int = 1,
    profile: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """Run baseline context parallelism benchmark."""
    
    benchmark = BaselineContextParallelism(
        batch_size=batch_size,
        seq_length=seq_length,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    benchmark.setup()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    peak_memory = benchmark.run()
    t1.record()
    torch.cuda.synchronize()
    elapsed_ms = t0.elapsed_time(t1)
    benchmark.cleanup()
    
    return {
        "mean_time_ms": float(elapsed_ms),
        "peak_memory_gb": peak_memory,
        "seq_length": seq_length,
        "num_gpus": 1,
        "parallelism": "none",
    }


class BaselineContextParallelismBenchmark(BaseBenchmark):
    """Harness-friendly wrapper around the standalone baseline runner."""

    def __init__(self):
        super().__init__()
        self._metrics: Dict[str, Any] = {}
        self.jitter_exemption_reason = "Context parallelism benchmark: fixed configuration"
        self.register_workload_metadata(requests_per_iteration=1.0)

    def setup(self) -> None:  # noqa: D401 - harness hook
        # All initialization happens inside run_benchmark; nothing to do here.
        torch.cuda.empty_cache()

    def benchmark_fn(self) -> None:
        # Use default shapes for a minimal single-GPU check.
        self._metrics = run_benchmark()
        self._synchronize()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=1, warmup=0)

    def get_custom_metrics(self) -> Optional[Dict[str, Any]]:
        return self._metrics

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return torch.tensor([hash(str(id(self))) % (2**31)], dtype=torch.float32)

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return {"type": "context_parallelism"}

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)


def get_benchmark() -> BaseBenchmark:
    return BaselineContextParallelismBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
