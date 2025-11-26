#!/usr/bin/env python3
"""Baseline: Pipeline Parallelism (GPipe style).

Demonstrates basic pipeline parallelism with sequential micro-batches.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path
import time
import os

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from common.python.logger import get_logger

logger = get_logger(__name__)


class BaselinePipelineParallel:
    """Baseline pipeline parallelism (GPipe style)."""
    
    def __init__(
        self,
        batch_size: int = 32,
        seq_length: int = 2048,
        hidden_size: int = 4096,
        num_layers: int = 8,
        num_micro_batches: int = 4,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_micro_batches = num_micro_batches
        
        # Initialize distributed
        self._init_distributed()
        
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)
        
        # Layers per stage
        self.layers_per_stage = num_layers // self.world_size
        self.stage_id = self.rank
        
        self.micro_batch_size = batch_size // num_micro_batches
        
        logger.info(
            f"PP Stage {self.stage_id}/{self.world_size}: "
            f"{self.layers_per_stage} layers, {num_micro_batches} micro-batches"
        )
    
    def _init_distributed(self):
        """Initialize distributed process group."""
        if not dist.is_initialized():
            if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                dist.init_process_group(backend='nccl')
            else:
                logger.warning("Running in simulation mode")
                self.rank = 0
                self.world_size = 1
                self.local_rank = 0
                return
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = self.rank % torch.cuda.device_count()
    
    def setup(self):
        """Initialize pipeline stage."""
        # Each stage gets a subset of layers
        self.stage_layers = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            for _ in range(self.layers_per_stage)
        ]).to(self.device).to(torch.bfloat16)
        
        # Create input (only on first stage)
        if self.stage_id == 0:
            self.input = torch.randn(
                self.batch_size,
                self.seq_length,
                self.hidden_size,
                device=self.device,
                dtype=torch.bfloat16
            )
        else:
            self.input = None
        
        logger.info(f"Stage {self.stage_id} setup complete")
    
    def _forward_micro_batch(self, micro_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass for one micro-batch."""
        x = micro_batch
        
        # Process through local layers
        for layer in self.stage_layers:
            x = torch.relu(layer(x))
        
        return x
    
    def run(self) -> float:
        """Execute baseline pipeline parallel (GPipe)."""
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # GPipe: Forward all micro-batches, then backward
        # Baseline: Sequential processing (large bubble)
        
        outputs = []
        
        for micro_idx in range(self.num_micro_batches):
            if self.stage_id == 0:
                # First stage: split input
                start_idx = micro_idx * self.micro_batch_size
                end_idx = start_idx + self.micro_batch_size
                micro_batch = self.input[start_idx:end_idx]
            else:
                # Receive from previous stage
                if self.world_size > 1 and dist.is_initialized():
                    micro_batch = torch.empty(
                        self.micro_batch_size,
                        self.seq_length,
                        self.hidden_size,
                        device=self.device,
                        dtype=torch.bfloat16
                    )
                    dist.recv(micro_batch, src=self.stage_id - 1)
                else:
                    # Single-GPU: use input directly
                    start_idx = micro_idx * self.micro_batch_size
                    end_idx = start_idx + self.micro_batch_size
                    micro_batch = self.input[start_idx:end_idx]
            
            # Forward through local layers
            output = self._forward_micro_batch(micro_batch)
            
            # Send to next stage or save output
            if self.world_size > 1 and dist.is_initialized():
                if self.stage_id < self.world_size - 1:
                    dist.send(output, dst=self.stage_id + 1)
                else:
                    outputs.append(output)
            else:
                # Single-GPU: save output directly
                outputs.append(output)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        # Calculate bubble time (idle time)
        # In GPipe, bubble = (num_stages - 1) / num_micro_batches
        bubble_pct = ((self.world_size - 1) / self.num_micro_batches) * 100
        
        logger.info(f"Stage {self.stage_id}: {elapsed*1000:.2f} ms")
        logger.info(f"Expected bubble: ~{bubble_pct:.1f}%")
        
        return elapsed * 1000
    
    def cleanup(self):
        """Clean up resources."""
        del self.stage_layers
        if self.input is not None:
            del self.input
        torch.cuda.empty_cache()


def run_benchmark(
    batch_size: int = 32,
    seq_length: int = 2048,
    hidden_size: int = 4096,
    num_layers: int = 8,
    num_micro_batches: int = 4,
    profile: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """Run baseline pipeline parallel benchmark."""
    
    benchmark = BaselinePipelineParallel(
        batch_size=batch_size,
        seq_length=seq_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_micro_batches=num_micro_batches,
    )
    benchmark.setup()
    
    config = BenchmarkConfig(
        iterations=3,
        warmup=1,
        profile_mode=profile,
    )
    
    harness = BenchmarkHarness(mode=BenchmarkMode.TRAINING, config=config)
    
    result = harness.benchmark(
        benchmark.run,
        name="baseline_pipeline_parallel"
    )
    
    benchmark.cleanup()
    
    bubble_pct = ((benchmark.world_size - 1) / num_micro_batches) * 100
    
    return {
        "mean_time_ms": result.timing.mean_ms,
        "num_stages": benchmark.world_size,
        "micro_batches": num_micro_batches,
        "expected_bubble_pct": bubble_pct,
        "parallelism": "pipeline_gpipe_baseline",
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline Pipeline Parallelism")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-micro-batches", type=int, default=4)
    parser.add_argument("--profile", type=str, default="none")
    
    args = parser.parse_args()
    
    result = run_benchmark(
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_micro_batches=args.num_micro_batches,
        profile=args.profile,
    )
    
    print(f"\n{'='*60}")
    print(f"Baseline Pipeline Parallelism Results")
    print(f"{'='*60}")
    print(f"Num stages: {result['num_stages']}")
    print(f"Micro-batches: {result['micro_batches']}")
    print(f"Expected bubble: {result['expected_bubble_pct']:.1f}%")
    print(f"Mean time: {result['mean_time_ms']:.2f} ms")
    print(f"{'='*60}\n")
    print(f"Launch with: torchrun --nproc_per_node=2 {__file__}")


#============================================================================
# Benchmark Harness Integration
#============================================================================

class PipelineParallelBenchmark(BaseBenchmark):
    """Benchmark harness wrapper for baseline pipeline parallelism."""

    def __init__(self):
        super().__init__()
        self.pp = None
        self.batch_size = 32
        self.seq_length = 2048
        self.hidden_size = 4096
        self._last = 0.0
        
        tokens = self.batch_size * self.seq_length
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        """Setup: Initialize baseline pipeline parallelism."""
        torch.manual_seed(42)
        self.pp = BaselinePipelineParallel(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            num_layers=8,
            num_micro_batches=4,
        )
        self.pp.setup()

    def benchmark_fn(self) -> None:
        """Benchmark: Baseline PP forward pass."""
        if self.pp is not None:
            self._last = self.pp.run()
        self._synchronize()

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        if self.pp is not None:
            self.pp.cleanup()
            self.pp = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=3)
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        if self.pp is None:
            return "Pipeline parallelism not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return PipelineParallelBenchmark()

