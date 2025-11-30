#!/usr/bin/env python3
"""
NVSHMEM + PyTorch Integration Guide for 8x B200
================================================

Demonstrates when and how to use NVSHMEM with PyTorch for low-latency
GPU-to-GPU communication on 8x B200 configurations.

NVSHMEM vs NCCL/PyTorch Collectives:
- NVSHMEM: Best for small, latency-sensitive, one-sided operations
- NCCL: Best for large collectives (AllReduce, AllGather)
- PyTorch: Best for general-purpose, high-level operations

Use NVSHMEM when:
  - Small message sizes (<1MB)
  - Irregular communication patterns
  - One-sided put/get operations
  - Fine-grained synchronization
  - Custom multi-GPU algorithms

Use NCCL/PyTorch when:
  - Large message sizes (>10MB)
  - Standard collectives (AllReduce, Broadcast)
  - High bandwidth utilization
  - Regular communication patterns
  - Training workloads

Requirements:
- PyTorch 2.10+ with NVSHMEM support
- NVSHMEM 3.4+ (CUDA 13)
- 8x B200 GPUs

Note: As of PyTorch 2.10, NVSHMEM support is experimental.
This module demonstrates concepts and provides PyTorch alternatives.
"""
import os
import sys
from typing import Optional

import torch
import torch.distributed as dist


def setup_single_gpu_env() -> None:
    """Setup environment for single-GPU testing."""
    if "RANK" not in os.environ:
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("LOCAL_RANK", "0")


def check_nvshmem_availability() -> bool:
    """
    Check if NVSHMEM is available in PyTorch.

    Returns:
        True if NVSHMEM support is available
    """
    if hasattr(torch.distributed, 'nn') and hasattr(torch.distributed.nn, 'SymmetricMemory'):
        return True
    return False


NVSHMEM_AVAILABLE = check_nvshmem_availability()


class SymmetricMemoryBuffer:
    """
    Wrapper for PyTorch 2.10 Symmetric Memory API.
    
    Provides NVSHMEM-like semantics using PyTorch's symmetric memory,
    which internally may use NVSHMEM on supported hardware.
    """
    
    def __init__(self, tensor: torch.Tensor, group=None):
        """
        Initialize symmetric memory buffer.
        
        Args:
            tensor: Local tensor to make symmetric
            group: Process group (default: WORLD)
        """
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = tensor.device
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.local_tensor = tensor
        
        self.sym_mem = None
        self.backend = "fallback"
        
        if NVSHMEM_AVAILABLE:
            self.sym_mem = torch.distributed.nn.SymmetricMemory(tensor, group=group)
            self.backend = "symmetric_memory"
    
    def put(self, data: torch.Tensor, target_rank: int) -> None:
        """
        One-sided put operation (write to remote GPU).
        
        Args:
            data: Data to write
            target_rank: Target GPU rank
        """
        if self.backend == "symmetric_memory" and self.sym_mem is not None:
            remote_buffer = self.sym_mem.get_buffer(target_rank)
            remote_buffer.copy_(data)
        elif self.rank != target_rank:
            dist.send(data, dst=target_rank)
    
    def get(self, source_rank: int) -> torch.Tensor:
        """
        One-sided get operation (read from remote GPU).
        
        Args:
            source_rank: Source GPU rank
            
        Returns:
            Data from remote GPU
        """
        result = torch.empty_like(self.local_tensor)
        
        if self.backend == "symmetric_memory" and self.sym_mem is not None:
            remote_buffer = self.sym_mem.get_buffer(source_rank)
            result.copy_(remote_buffer)
        elif self.rank != source_rank:
            dist.recv(result, src=source_rank)
        else:
            result.copy_(self.local_tensor)
        
        return result
    
    def barrier(self) -> None:
        """Synchronization barrier."""
        dist.barrier()


def benchmark_put_latency(
    size_bytes: int = 4096,
    num_iterations: int = 1000,
    target_rank: int = 1
) -> dict:
    """
    Benchmark one-sided put latency.
    
    Args:
        size_bytes: Message size in bytes
        num_iterations: Number of iterations
        target_rank: Target GPU rank
        
    Returns:
        Latency measurements
    """
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    
    num_elements = size_bytes // 4
    tensor = torch.randn(num_elements, device=device, dtype=torch.float32)
    
    results = {}
    
    if rank == 0:
        sym_buf = SymmetricMemoryBuffer(tensor)
        
        # Warmup
        for _ in range(10):
            sym_buf.put(tensor, target_rank)
        torch.cuda.synchronize()
        
        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(num_iterations):
            sym_buf.put(tensor, target_rank)
        end.record()
        end.synchronize()
        
        sym_time = start.elapsed_time(end) / num_iterations
        results["symmetric_memory_us"] = sym_time * 1000
    
    dist.barrier()
    return results


def print_performance_guide() -> None:
    """Print performance guide for NVSHMEM vs NCCL."""
    print("\n" + "=" * 80)
    print("NVSHMEM vs NCCL Performance Guide")
    print("=" * 80)
    print("\nNVSHMEM (via Symmetric Memory):")
    print("  ✓ Latency: 1-5 μs (ultra-low)")
    print("  ✓ Small messages: <1 MB")
    print("  ✓ One-sided operations")
    print("  ✗ Large messages: >10 MB (slower than NCCL)")
    print("\nNCCL:")
    print("  ✓ Throughput: 700-800 GB/s (8x B200)")
    print("  ✓ Large messages: >10 MB")
    print("  ✓ Optimized collectives (AllReduce, AllGather)")
    print("  ✗ Small message latency: 10-50 μs")
    print("=" * 80)


def main() -> None:
    """Main demonstration."""
    if not dist.is_initialized():
        setup_single_gpu_env()
        dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print("=" * 80)
        print("NVSHMEM + PyTorch Integration Demo")
        print("=" * 80)
        print(f"World size: {world_size} GPUs")
        print(f"NVSHMEM available: {NVSHMEM_AVAILABLE}")
        print_performance_guide()
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
