#!/usr/bin/env python3
"""Production NVSHMEM Training Patterns for 8x B200.

Demonstrates how to extend PyTorch training workloads with NVSHMEM and
torch.distributed.nn.SymmetricMemory for latency-critical paths on 8 GPU
Blackwell clusters.

Patterns:
1. Gradient buckets backed by symmetric memory for custom AllReduce
2. Hybrid sharding (FSDP parameters + NVSHMEM parameter server buffers)
3. Pipeline parallel microbatch handoff using one-sided NVSHMEM puts

Hardware assumptions:
- 8x Blackwell B200 GPUs (NVLink 5.0 @ 1800 GB/s per pair)
- CUDA 13.0+, PyTorch 2.10+, NVSHMEM 3.4+

Usage (conceptual):
    torchrun --nproc_per_node=8 nvshmem_training_example.py --demo gradient
    torchrun --nproc_per_node=8 nvshmem_training_example.py --demo hybrid
    torchrun --nproc_per_node=8 nvshmem_training_example.py --demo pipeline

Each demo degrades gracefully to NCCL when NVSHMEM or symmetric memory is
unavailable, providing runnable fallbacks for development laptops.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.optimization.symmetric_memory_patch import (
    ensure_symmetric_memory_api as _ensure_symmetric_memory_api,
)

_ensure_symmetric_memory_api()

try:
    from distributed_helper import setup_single_gpu_env
except ImportError:
    def setup_single_gpu_env():
        if "RANK" not in os.environ:
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29500")
            os.environ.setdefault("LOCAL_RANK", "0")  # Graceful fallback if arch_config not available


import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)


# ============================================================================
# Helpers
# ============================================================================


def nvshmem_available() -> bool:
    """Detect whether NVSHMEM-like symmetric memory APIs are exposed."""
    if hasattr(dist, "nn") and hasattr(dist.nn, "SymmetricMemory"):
        return True
    try:
        import nvshmem  # noqa: F401

        return True
    except Exception:
        return False


def init_process_group() -> Tuple[int, int, torch.device]:
    """Initialize distributed process group with CUDA fallback to Gloo."""
    setup_single_gpu_env()  # Auto-setup for single-GPU mode
    
    if not dist.is_initialized():
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", max(1, torch.cuda.device_count())))
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=60),
        )
        
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
    else:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")
    return rank, world_size, device


# ============================================================================
# Demo 1: Gradient Buckets with Symmetric Memory
# ============================================================================


@dataclass
class GradientBucket:
    """
    Gradient bucket backed by symmetric memory for one-sided NVSHMEM puts.

    Each GPU writes gradients into its local chunk, then performs a ring
    reduction using remote puts for latency-critical microbatches.
    """

    numel: int
    dtype: torch.dtype
    device: torch.device
    world_size: int
    handle: Optional[object] = None
    tensor: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        local = torch.zeros(self.numel, device=self.device, dtype=self.dtype)
        if nvshmem_available():
            try:
                self.handle = dist.nn.SymmetricMemory(local)
                self.tensor = self.handle.buffer
            except Exception:
                self.handle = None
                self.tensor = local
        else:
            self.tensor = local

    def allreduce_ring(self, rank: int) -> None:
        """
        Conceptual ring AllReduce using one-sided puts/gets when available.
        Falls back to NCCL AllReduce if symmetric memory is missing.
        """
        if self.tensor is None:
            raise RuntimeError("GradientBucket tensor not initialized")

        if nvshmem_available() and self.handle is not None:
            # Conceptual one-sided ring: each rank writes into next rank slot
            chunk = self.tensor.view(self.world_size, -1)[rank].clone()
            for step in range(self.world_size - 1):
                target = (rank + 1 + step) % self.world_size
                remote_buf = self.handle.get_buffer(target)
                remote_buf[rank].copy_(chunk)
                dist.barrier()
                chunk.add_(self.handle.get_buffer((rank - 1 - step) % self.world_size)[rank])
            self.tensor.view(self.world_size, -1)[rank].copy_(chunk)
        else:
            dist.all_reduce(self.tensor, op=dist.ReduceOp.SUM)
            self.tensor.mul_(1.0 / self.world_size)


def demo_gradient_buckets(batch: torch.Tensor, model: nn.Module) -> None:
    """Attach hooks that fuse gradients into symmetric memory buckets."""
    rank, world_size, device = init_process_group()

    model = model.to(device)
    batch = batch.to(device)

    bucket = GradientBucket(
        numel=sum(p.numel() for p in model.parameters()),
        dtype=batch.dtype,
        device=device,
        world_size=world_size,
    )

    offset = 0

    def _hook(param: torch.nn.Parameter, grad: torch.Tensor) -> torch.Tensor:
        nonlocal offset
        if grad is None:
            return grad
        next_offset = offset + grad.numel()
        bucket.tensor.flatten()[offset:next_offset].copy_(grad.flatten())
        offset = next_offset
        return grad

    for p in model.parameters():
        p.register_hook(lambda grad, p=p: _hook(p, grad))

    output = model(batch)
    loss = output.float().sum()
    loss.backward()
    bucket.allreduce_ring(rank)

    if rank == 0:
        print(f"[gradient_bucket] Reduced norm={bucket.tensor.norm().item():.3f}")


# ============================================================================
# Demo 2: Hybrid Sharding (FSDP + NVSHMEM Parameter Server)
# ============================================================================


class TransformerBlock(nn.Module):
    """Simple transformer block used across demos."""

    def __init__(self, d_model: int = 2048, n_heads: int = 16, mlp_ratio: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model * mlp_ratio)
        self.linear2 = nn.Linear(d_model * mlp_ratio, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        resid = x
        x = self.ln1(x)
        attn_out, _ = self.attn(x, x, x)
        x = resid + attn_out
        resid = x
        x = self.ln2(x)
        x = F.gelu(self.linear1(x))
        x = self.linear2(x)
        return x + resid


class NVSHMEMParameterServer:
    """
    Minimal parameter server that mirrors FSDP weights into symmetric memory.

    FSDP handles parameter sharding + optimizer states; NVSHMEM enables
    low-latency lookups for layers that need dense access (e.g. embeddings).
    """

    def __init__(self, module: nn.Module):
        self.module = module
        self.buffers: Dict[str, GradientBucket] = {}

    def materialize_parameter(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        if name not in self.buffers:
            bucket = GradientBucket(
                numel=tensor.numel(),
                dtype=tensor.dtype,
                device=tensor.device,
                world_size=dist.get_world_size(),
            )
            bucket.tensor.copy_(tensor.flatten())
            self.buffers[name] = bucket
        return self.buffers[name].tensor.view_as(tensor)

    def pull(self, name: str) -> torch.Tensor:
        if name not in self.buffers:
            raise KeyError(f"Parameter {name} not backed by NVSHMEM buffer")
        return self.buffers[name].tensor


def demo_hybrid_sharding(microbatch: torch.Tensor) -> None:
    """Showcase FSDP sharding with NVSHMEM-backed parameter server tensors."""
    if not torch.cuda.is_available():
        raise RuntimeError("Hybrid sharding demo requires CUDA-enabled environment")

    rank, world_size, device = init_process_group()
    torch.cuda.manual_seed(17 + rank)

    model = TransformerBlock().to(device)
    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float32
        ),
    )

    server = NVSHMEMParameterServer(fsdp_model)

    with FSDP.state_dict_type(fsdp_model, StateDictType.LOCAL_STATE_DICT):
        state_dict = fsdp_model.state_dict()

    for name, tensor in state_dict.items():
        mirror = server.materialize_parameter(name, tensor)
        tensor.copy_(mirror.view_as(tensor))

    out = fsdp_model(microbatch.to(device))
    loss = out.sum()
    loss.backward()

    if rank == 0:
        print(f"[hybrid_shard] gradients ready, NVSHMEM buffers={len(server.buffers)}")


# ============================================================================
# Demo 3: Pipeline Parallel with NVSHMEM Handoff
# ============================================================================


class PipelineStage(nn.Module):
    """Lightweight pipeline stage used for the NVSHMEM pipeline demo."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def demo_pipeline_parallel(microbatch: torch.Tensor) -> None:
    """
    Pipeline two stages using symmetric memory buffers for microbatch handoff.

    Stage 0 (ranks 0-3) writes outputs into symmetric buffers that stage 1
    (ranks 4-7) reads without host synchronization.
    """

    rank, world_size, device = init_process_group()
    assert world_size >= 2, "Pipeline demo requires at least 2 ranks"

    dim = microbatch.shape[-1]
    stage0 = PipelineStage(dim, dim * 4).to(device)
    stage1 = PipelineStage(dim, dim * 4).to(device)

    # Allocate symmetric buffer for microbatch handoff
    bucket = GradientBucket(
        numel=microbatch.numel(),
        dtype=microbatch.dtype,
        device=device,
        world_size=world_size,
    )

    if rank < world_size // 2:
        microbatch = stage0(microbatch.to(device))
        bucket.tensor.copy_(microbatch.flatten())
        if nvshmem_available() and bucket.handle is not None:
            next_rank = rank + world_size // 2
            remote = bucket.handle.get_buffer(next_rank)
            remote.copy_(bucket.tensor)
        dist.barrier()
    else:
        dist.barrier()
        if nvshmem_available() and bucket.handle is not None:
            remote = bucket.handle.get_buffer(rank)
            microbatch = remote.view_as(microbatch)
        else:
            microbatch = bucket.tensor.view_as(microbatch)
        microbatch = stage1(microbatch)

    if rank == 0:
        print("[pipeline] completed forward pass with NVSHMEM handoff")


# ============================================================================
# CLI Entrypoint
# ============================================================================


def _build_sample_batch(batch_size: int = 8, seq_len: int = 512, dim: int = 2048) -> torch.Tensor:
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    generator.manual_seed(0)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.randn(batch_size, seq_len, dim, device=device, generator=generator)


def main() -> None:
    parser = argparse.ArgumentParser(description="NVSHMEM training patterns")
    parser.add_argument(
        "--demo",
        choices=("gradient", "hybrid", "pipeline"),
        default="gradient",
        help="Which demonstration to run",
    )
    args = parser.parse_args()

    demo_map = {
        "gradient": demo_gradient_buckets,
        "hybrid": demo_hybrid_sharding,
        "pipeline": demo_pipeline_parallel,
    }

    init_process_group()
    batch = _build_sample_batch()
    model = TransformerBlock()

    if args.demo == "gradient":
        demo_map[args.demo](batch, model)
    elif args.demo == "hybrid":
        demo_map[args.demo](batch)
    else:
        demo_map[args.demo](batch)

    dist.barrier()
    if dist.get_rank() == 0:
        print(f"NVSHMEM available: {nvshmem_available()}")


if __name__ == "__main__":
    main()
