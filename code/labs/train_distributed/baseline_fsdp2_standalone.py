#!/usr/bin/env python3
"""Baseline: FSDP2 with BF16 mixed precision.

Demonstrates standard FSDP2 training without FP8 optimization.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from typing import Dict, Any
import sys
from pathlib import Path
import time
import os

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)
from common.python.logger import get_logger

logger = get_logger(__name__)


class BaselineFSDP2Standard(BaseBenchmark):
    """Baseline FSDP2 with BF16."""

    def __init__(
        self,
        batch_size: int = 4,
        seq_length: int = 2048,
        hidden_size: int = 4096,
        num_layers: int = 8,
        micro_batch_size: int = 1,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.micro_batch_size = micro_batch_size
        self._last_metrics: Dict[str, float] = {}

        self._init_distributed()

        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        logger.info(f"FSDP2 Rank {self.rank}/{self.world_size}: BF16 baseline")

    def _init_distributed(self):
        """Initialize distributed."""
        if not dist.is_initialized():
            if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                dist.init_process_group(backend='nccl')
            else:
                logger.warning("Simulation mode")
                self.rank = 0
                self.world_size = 1
                self.local_rank = 0
                return
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = self.rank % torch.cuda.device_count()

    def setup(self):
        """Initialize FSDP model and inputs."""
        self.register_workload_metadata(
            tokens_per_iteration=float(self.batch_size * self.seq_length),
        )

        # Create simple model
        class SimpleTransformerLayer(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.linear1 = nn.Linear(hidden_size, hidden_size * 4, bias=False)
                self.linear2 = nn.Linear(hidden_size * 4, hidden_size, bias=False)
                self.norm = nn.LayerNorm(hidden_size)

            def forward(self, x):
                return x + self.linear2(torch.relu(self.linear1(self.norm(x))))

        model = nn.ModuleList([
            SimpleTransformerLayer(self.hidden_size)
            for _ in range(self.num_layers)
        ]).to(self.device)

        # Wrap with FSDP2
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        self.model = FSDP(
            model,
            mixed_precision=mixed_precision,
            use_orig_params=True,
        )

        # Create input
        self.input = torch.randn(
            self.micro_batch_size,
            self.seq_length,
            self.hidden_size,
            device=self.device,
            dtype=torch.bfloat16
        )

        # Create optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        logger.info(f"FSDP2 model initialized (Rank {self.rank})")

    def benchmark_fn(self) -> None:
        """Execute FSDP2 training step and stash metrics for the harness."""
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()

        # Training step
        self.optimizer.zero_grad()

        # Forward
        output = self.input
        for layer in self.model:
            output = layer(output)

        # Loss
        loss = output.mean()

        # Backward
        loss.backward()

        # Optimizer step
        self.optimizer.step()

        self._synchronize()
        elapsed = time.perf_counter() - start

        # Memory metrics
        peak_memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)
        tokens_per_sec = (self.batch_size * self.seq_length) / elapsed if elapsed > 0 else 0.0

        logger.info(f"Rank {self.rank}: {elapsed*1000:.2f} ms, peak mem: {peak_memory_gb:.2f} GB")

        self._last_metrics = {
            "latency_ms": elapsed * 1000,
            "peak_memory_gb": peak_memory_gb,
            "loss": loss.item(),
            "tokens_per_sec": tokens_per_sec,
        }

    def get_custom_metrics(self) -> Dict[str, float]:
        """Expose last-run metrics to the harness."""
        return self._last_metrics

    def teardown(self):
        """Clean up resources."""
        del self.model, self.optimizer, self.input
        super().teardown()


def run_benchmark(
    batch_size: int = 4,
    seq_length: int = 2048,
    hidden_size: int = 4096,
    num_layers: int = 8,
    micro_batch_size: int = 1,
    profile: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """Run baseline FSDP2 benchmark."""

    benchmark = BaselineFSDP2Standard(
        batch_size=batch_size,
        seq_length=seq_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        micro_batch_size=micro_batch_size,
    )

    config = BenchmarkConfig(
        iterations=3,
        warmup=5,
        profile_mode=profile,
    )

    harness = BenchmarkHarness(mode=BenchmarkMode.TRAINING, config=config)

    result = harness.benchmark(benchmark, name="baseline_fsdp2_standard")

    metrics = result.custom_metrics or {}
    return {
        "mean_time_ms": result.timing.mean_ms,
        "peak_memory_gb": metrics.get("peak_memory_gb"),
        "loss": metrics.get("loss"),
        "tokens_per_sec": metrics.get("tokens_per_sec"),
        "precision": "bf16",
    }


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineFSDP2Standard()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline FSDP2")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--profile", type=str, default="none")
    
    args = parser.parse_args()
    
    result = run_benchmark(
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        micro_batch_size=args.micro_batch_size,
        profile=args.profile,
    )
    
    print(f"\n{'='*60}")
    print(f"Baseline FSDP2 Results")
    print(f"{'='*60}")
    print(f"Precision: {result['precision']}")
    print(f"Mean time: {result['mean_time_ms']:.2f} ms")
    print(f"Peak memory: {result['peak_memory_gb']:.2f} GB")
    print(f"Loss: {result['loss']:.6f}")
    print(f"{'='*60}\n")
    print(f"Launch with: torchrun --nproc_per_node=2 {__file__}")

