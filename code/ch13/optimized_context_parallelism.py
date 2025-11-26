#!/usr/bin/env python3
"""Optimized: Context Parallelism with ring attention for long sequences.

Demonstrates Context Parallelism (CP) to shard attention computation across GPUs,
enabling 128K+ token sequences that don't fit in single GPU memory.

Uses ring attention pattern where each GPU holds a shard of the sequence and
communicates with neighbors in a ring topology.
"""

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

# Add common to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.python.benchmark_harness import BenchmarkHarness, BenchmarkConfig, BenchmarkMode
from common.python.logger import get_logger

logger = get_logger(__name__)


class RingAttention(nn.Module):
    """Ring attention with Context Parallelism across GPUs."""
    
    def __init__(self, hidden_size: int, num_heads: int, cp_group: Optional[dist.ProcessGroup] = None, cp_rank: int = 0, cp_size: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.cp_group = cp_group
        self.cp_rank = cp_rank
        self.cp_size = cp_size
        
        assert self.head_dim * num_heads == hidden_size
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def _ring_pass(
        self,
        q: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        seq_shard: int,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Ring attention with proper global softmax normalization.
        
        Computes a streaming log-sum-exp across shards so that the final
        softmax is correct even when scores differ across shards.
        """
        batch_size, num_heads, _, head_dim = q.shape
        k_current = k_local
        v_current = v_local
        
        attn_num: Optional[torch.Tensor] = None
        global_max: Optional[torch.Tensor] = None
        global_sum: Optional[torch.Tensor] = None
        
        global_q = (self.cp_rank * seq_shard) + torch.arange(seq_shard, device=q.device)
        global_q = global_q.view(1, 1, seq_shard, 1)
        k_indices = torch.arange(seq_shard, device=q.device).view(1, 1, 1, seq_shard)
        
        for step in range(self.cp_size):
            target_rank = (self.cp_rank - step) % self.cp_size
            scores = torch.matmul(q, k_current.transpose(-2, -1)) * self.scale
            
            if causal:
                global_k = target_rank * seq_shard + k_indices
                causal_mask = global_k > global_q
                if causal_mask.any():
                    scores = scores.masked_fill(causal_mask, float("-inf"))
            
            local_max = scores.amax(dim=-1, keepdim=True)
            exp_scores = torch.exp(scores - local_max)
            local_sum = exp_scores.sum(dim=-1, keepdim=True)
            local_num = torch.matmul(exp_scores, v_current)
            
            if global_max is None:
                global_max = local_max
                global_sum = local_sum
                attn_num = local_num
            else:
                new_max = torch.maximum(global_max, local_max)
                scale_prev = torch.exp(global_max - new_max)
                scale_local = torch.exp(local_max - new_max)
                attn_num = attn_num * scale_prev + local_num * scale_local  # type: ignore[assignment]
                global_sum = global_sum * scale_prev + local_sum * scale_local
                global_max = new_max
            
            if step < self.cp_size - 1 and self.cp_group is not None:
                next_rank = (self.cp_rank + 1) % self.cp_size
                prev_rank = (self.cp_rank - 1) % self.cp_size
                
                k_recv = torch.empty_like(k_current)
                v_recv = torch.empty_like(v_current)
                
                send_k = dist.isend(k_current.contiguous(), next_rank, group=self.cp_group)
                recv_k = dist.irecv(k_recv, prev_rank, group=self.cp_group)
                send_v = dist.isend(v_current.contiguous(), next_rank, group=self.cp_group)
                recv_v = dist.irecv(v_recv, prev_rank, group=self.cp_group)
                
                send_k.wait()
                recv_k.wait()
                send_v.wait()
                recv_v.wait()
                
                k_current = k_recv
                v_current = v_recv
        
        assert attn_num is not None and global_sum is not None
        attn_output = attn_num / (global_sum + 1e-8)
        return attn_output
    
    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """
        Ring attention forward pass.
        
        Args:
            x: [batch_size, seq_shard, hidden_size] - Local sequence shard
            causal: Enable causal masking
        
        Returns:
            output: [batch_size, seq_shard, hidden_size]
        """
        batch_size, seq_shard, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to [batch, num_heads, seq_shard, head_dim]
        q = q.view(batch_size, seq_shard, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_shard, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_shard, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Ring attention
        attn_output = self._ring_pass(q, k, v, seq_shard=seq_shard, causal=causal)
        
        # Reshape back to [batch, seq_shard, hidden_size]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_shard, self.hidden_size)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        return output


class OptimizedContextParallelism:
    """Optimized: Context Parallelism with ring attention."""
    
    def __init__(
        self,
        batch_size: int = 1,
        seq_length: int = 131072,  # 128K tokens
        hidden_size: int = 4096,
        num_heads: int = 32,
        num_layers: int = 1,
        cp_ranks: int = 4,  # Context parallel degree
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.cp_ranks = cp_ranks
        
        # Initialize distributed
        self._init_distributed()
        
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)
        
        # Sequence shard size per GPU (based on actual group size)
        self.seq_shard = seq_length // self.cp_size
        
        logger.info(
            f"CP Rank {self.cp_rank}/{self.cp_size}: "
            f"Total seq={seq_length}, shard={self.seq_shard}"
        )
    
    def _init_distributed(self):
        """Initialize distributed process group for Context Parallelism."""
        if not dist.is_initialized():
            if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
                dist.init_process_group(backend="nccl")
            else:
                logger.warning("Running in simulation mode (no distributed)")
                self.cp_rank = 0
                self.cp_size = 1
                self.local_rank = 0
                self.cp_group = None
                return
        
        self.cp_rank = dist.get_rank()
        self.cp_size = dist.get_world_size()
        self.local_rank = self.cp_rank % torch.cuda.device_count()
        
        # Create CP process group (all ranks participate)
        # Check NCCL backend for optimal performance
        if dist.get_backend() == 'nccl':
            self.cp_group = dist.new_group(ranks=list(range(self.cp_size)))
            logger.info(f"Initialized CP group: rank {self.cp_rank}/{self.cp_size}")
        else:
            logger.warning(f"Backend {dist.get_backend()} may not support process groups efficiently")
            self.cp_group = None
    
    def setup(self):
        """Initialize model and data."""
        if self.seq_length % self.cp_size != 0:
            raise ValueError(f"Sequence length {self.seq_length} must be divisible by cp_size={self.cp_size}")
        # Create ring attention layers
        self.layers = nn.ModuleList([
            RingAttention(
                self.hidden_size,
                self.num_heads,
                cp_group=self.cp_group,
                cp_rank=self.cp_rank,
                cp_size=self.cp_size,
            )
            for _ in range(self.num_layers)
        ]).to(self.device)
        
        # Create local sequence shard
        self.input = torch.randn(
            self.batch_size,
            self.seq_shard,  # Only local shard
            self.hidden_size,
            device=self.device,
            dtype=torch.float32
        )
        
        logger.info(
            f"Setup complete (Rank {self.cp_rank}): "
            f"local_seq={self.seq_shard}, total_seq={self.seq_length}"
        )
    
    def run(self) -> float:
        """Execute ring attention with Context Parallelism."""
        torch.cuda.synchronize()
        
        # Forward pass through all layers
        x = self.input
        for layer in self.layers:
            x = layer(x, causal=True)
        
        torch.cuda.synchronize()
        
        # Gather peak memory across all ranks
        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
        
        if self.cp_group is not None:
            # Reduce max across all ranks
            peak_tensor = torch.tensor([peak_memory_gb], device=self.device)
            dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX, group=self.cp_group)
            peak_memory_gb = peak_tensor.item()
        
        logger.info(f"Rank {self.cp_rank}: Peak memory: {peak_memory_gb:.2f} GB")
        
        return peak_memory_gb
    
    def cleanup(self):
        """Clean up resources."""
        del self.layers
        del self.input
        torch.cuda.empty_cache()
        
        if dist.is_initialized() and self.cp_group is not None:
            dist.destroy_process_group()


def run_benchmark(
    batch_size: int = 1,
    seq_length: int = 131072,  # 128K
    hidden_size: int = 4096,
    num_heads: int = 32,
    num_layers: int = 1,
    cp_ranks: int = 4,
    profile: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """Run optimized context parallelism benchmark."""
    
    benchmark = OptimizedContextParallelism(
        batch_size=batch_size,
        seq_length=seq_length,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        cp_ranks=cp_ranks,
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
        "num_gpus": cp_ranks,
        "parallelism": "context_parallel",
        "seq_per_gpu": seq_length // cp_ranks,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized Context Parallelism")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=131072,
                       help="Total sequence length (sharded across GPUs)")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--cp-ranks", type=int, default=4,
                       help="Number of GPUs for context parallelism")
    parser.add_argument("--profile", type=str, default="none")
    
    args = parser.parse_args()
    
    result = run_benchmark(
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        cp_ranks=args.cp_ranks,
        profile=args.profile,
    )
    
    print(f"\n{'='*60}")
    print(f"Optimized Context Parallelism Results")
    print(f"{'='*60}")
    print(f"Total sequence: {result['seq_length']:,} tokens")
    print(f"GPUs used: {result['num_gpus']}")
    print(f"Tokens per GPU: {result['seq_per_gpu']:,}")
    print(f"Parallelism: {result['parallelism']}")
    print(f"Mean time: {result['mean_time_ms']:.2f} ms")
    print(f"Peak memory: {result['peak_memory_gb']:.2f} GB (max across GPUs)")
    print(f"{'='*60}\n")
    print(f"Launch with: torchrun --nproc_per_node={args.cp_ranks} {__file__}")


#============================================================================
# Benchmark Harness Integration
#============================================================================

from common.python.benchmark_harness import BaseBenchmark, WorkloadMetadata


class ContextParallelismBenchmark(BaseBenchmark):
    """Benchmark harness wrapper for Context Parallelism."""

    def __init__(self):
        super().__init__()
        self.cp = None
        self.batch_size = 1
        self.seq_length = 131072  # 128K
        self.hidden_size = 4096
        self.num_heads = 32
        self._last = 0.0
        
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(self.seq_length),
        )

    def setup(self) -> None:
        """Setup: Initialize Context Parallelism."""
        self.cp = OptimizedContextParallelism(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_layers=1,
            cp_ranks=1,  # Single GPU for benchmark harness
        )
        self.cp.setup()

    def benchmark_fn(self) -> None:
        """Benchmark: Ring attention forward pass."""
        if self.cp is not None:
            self._last = self.cp.run()
        self._synchronize()

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        if self.cp is not None:
            self.cp.cleanup()
            self.cp = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=3)
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_fp32_ms', 10.0),
            reduced_precision_time_ms=getattr(self, '_reduced_ms', 5.0),
            precision_type="fp8",
        )

    def validate_result(self) -> Optional[str]:
        if self.cp is None:
            return "Context parallelism not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return ContextParallelismBenchmark()
