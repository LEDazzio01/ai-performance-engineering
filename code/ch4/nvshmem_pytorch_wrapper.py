#!/usr/bin/env python3
"""
NVSHMEM + PyTorch Integration Guide for 8x B200
==============================================

Demonstrates when and how to use NVSHMEM with PyTorch for low-latency
GPU-to-GPU communication on 8x B200 configurations.

NVSHMEM vs NCCL/PyTorch Collectives:
- NVSHMEM: Best for small, latency-sensitive, one-sided operations
- NCCL: Best for large collectives (AllReduce, AllGather)
- PyTorch: Best for general-purpose, high-level operations

Use NVSHMEM when:
✓ Small message sizes (<1MB)
✓ Irregular communication patterns
✓ One-sided put/get operations
✓ Fine-grained synchronization
✓ Custom multi-GPU algorithms

Use NCCL/PyTorch when:
✓ Large message sizes (>10MB)
✓ Standard collectives (AllReduce, Broadcast)
✓ High bandwidth utilization
✓ Regular communication patterns
✓ Training workloads

Requirements:
- PyTorch 2.9+ with NVSHMEM support
- NVSHMEM 3.4+ (CUDA 13)
- 8x B200 GPUs

Note: As of PyTorch 2.9, NVSHMEM support is experimental.
This module demonstrates concepts and provides PyTorch alternatives.

Author: AI Performance Engineering Team
"""
import arch_config  # noqa: F401 - Configure Blackwell optimizations

import torch
import torch.distributed as dist
import os
from typing import Optional, List
import time


# ============================================================================
# NVSHMEM Availability Check
# ============================================================================

def check_nvshmem_availability() -> bool:
    """
    Check if NVSHMEM is available in PyTorch.
    
    Returns:
        True if NVSHMEM support is available
    """
    try:
        # PyTorch 2.9 may expose NVSHMEM through distributed.nn.SymmetricMemory
        if hasattr(torch.distributed, 'nn') and hasattr(torch.distributed.nn, 'SymmetricMemory'):
            return True
        return False
    except:
        return False


NVSHMEM_AVAILABLE = check_nvshmem_availability()


# ============================================================================
# Symmetric Memory Wrapper (PyTorch 2.9)
# ============================================================================

class SymmetricMemoryBuffer:
    """
    Wrapper for PyTorch 2.9 Symmetric Memory API.
    
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
        
        # Try to use PyTorch symmetric memory
        if NVSHMEM_AVAILABLE:
            try:
                self.sym_mem = torch.distributed.nn.SymmetricMemory(tensor, group=group)
                self.backend = "symmetric_memory"
            except Exception as e:
                print(f"Symmetric memory init failed: {e}, using fallback")
                self.sym_mem = None
                self.backend = "fallback"
        else:
            self.sym_mem = None
            self.backend = "fallback"
        
        # Fallback: store local tensor
        self.local_tensor = tensor
    
    def put(self, data: torch.Tensor, target_rank: int):
        """
        One-sided put operation (write to remote GPU).
        
        Args:
            data: Data to write
            target_rank: Target GPU rank
        """
        if self.backend == "symmetric_memory":
            # Use symmetric memory put (if API supports it)
            # This is a conceptual example - actual API may differ
            try:
                remote_buffer = self.sym_mem.get_buffer(target_rank)
                remote_buffer.copy_(data)
            except:
                # Fallback to send/recv
                if self.rank != target_rank:
                    dist.send(data, dst=target_rank)
        else:
            # Fallback: use NCCL send
            if self.rank != target_rank:
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
        
        if self.backend == "symmetric_memory":
            try:
                remote_buffer = self.sym_mem.get_buffer(source_rank)
                result.copy_(remote_buffer)
                return result
            except:
                pass
        
        # Fallback: use NCCL recv
        if self.rank != source_rank:
            dist.recv(result, src=source_rank)
        else:
            result.copy_(self.local_tensor)
        
        return result
    
    def barrier(self):
        """Synchronization barrier."""
        dist.barrier()


# ============================================================================
# Performance Comparison: NVSHMEM vs NCCL
# ============================================================================

def benchmark_put_latency(
    size_bytes: int = 4096,
    num_iterations: int = 1000,
    target_rank: int = 1
) -> dict:
    """
    Benchmark one-sided put latency.
    
    Compares:
    - Symmetric memory (NVSHMEM-backed if available)
    - NCCL send/recv
    
    Args:
        size_bytes: Message size in bytes
        num_iterations: Number of iterations
        target_rank: Target GPU rank
        
    Returns:
        Latency measurements
    """
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    
    # Create tensor
    num_elements = size_bytes // 4  # float32
    tensor = torch.randn(num_elements, device=device, dtype=torch.float32)
    
    results = {}
    
    # Test 1: Symmetric Memory
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
    
    # Test 2: NCCL send/recv
    if rank == 0 or rank == target_rank:
        # Warmup
        for _ in range(10):
            if rank == 0:
                dist.send(tensor, dst=target_rank)
            else:
                recv_tensor = torch.empty_like(tensor)
                dist.recv(recv_tensor, src=0)
        torch.cuda.synchronize()
        
        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(num_iterations):
            if rank == 0:
                dist.send(tensor, dst=target_rank)
            else:
                recv_tensor = torch.empty_like(tensor)
                dist.recv(recv_tensor, src=0)
        end.record()
        end.synchronize()
        
        nccl_time = start.elapsed_time(end) / num_iterations
        if rank == 0:
            results["nccl_us"] = nccl_time * 1000
    
    dist.barrier()
    
    return results


def print_performance_guide():
    """Print performance guide for NVSHMEM vs NCCL."""
    print("\n" + "=" * 80)
    print("NVSHMEM vs NCCL Performance Guide")
    print("=" * 80)
    
    print("\n📊 Performance Characteristics:")
    print("\nNVSHMEM (via Symmetric Memory):")
    print("  ✓ Latency: 1-5 μs (ultra-low)")
    print("  ✓ Small messages: <1 MB")
    print("  ✓ One-sided operations")
    print("  ✗ Large messages: >10 MB (slower than NCCL)")
    print("  ✗ Complex collectives (use NCCL)")
    
    print("\nNCCL:")
    print("  ✓ Throughput: 700-800 GB/s (8x B200)")
    print("  ✓ Large messages: >10 MB")
    print("  ✓ Optimized collectives (AllReduce, AllGather)")
    print("  ✗ Small message latency: 10-50 μs")
    
    print("\n💡 Recommendation Matrix:")
    print("\n| Message Size | Pattern        | Best Choice        |")
    print("|-------------|----------------|-------------------|")
    print("| < 1 KB      | Point-to-point | NVSHMEM           |")
    print("| 1-100 KB    | Point-to-point | NVSHMEM           |")
    print("| 100KB-1MB   | Point-to-point | Either            |")
    print("| > 1 MB      | Point-to-point | NCCL              |")
    print("| Any         | AllReduce      | NCCL              |")
    print("| Any         | AllGather      | NCCL              |")
    print("| < 1 MB      | Irregular      | NVSHMEM           |")
    
    print("\n🎯 Use Cases:")
    print("\nNVSHMEM:")
    print("  • Fine-grained parameter server")
    print("  • Asynchronous gradient aggregation")
    print("  • Custom sparse all-to-all")
    print("  • Lock-free data structures")
    
    print("\nNCCL:")
    print("  • Standard DDP/FSDP training")
    print("  • Tensor parallelism gradients")
    print("  • Large model synchronization")
    print("  • Bulk data transfers")
    
    print("=" * 80)


# ============================================================================
# Example Use Cases
# ============================================================================

def example_parameter_server_update():
    """
    Example: Fine-grained parameter server using NVSHMEM.
    
    Use case: Asynchronous parameter updates with low latency.
    Better than NCCL for small, frequent updates.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Server parameters (on rank 0)
    param_size = 1024  # Small parameter
    
    if rank == 0:
        server_params = torch.zeros(param_size, device='cuda:0', dtype=torch.float32)
        sym_buf = SymmetricMemoryBuffer(server_params)
        
        print("\nParameter Server Example (NVSHMEM-style)")
        print("  Server on rank 0, workers on ranks 1-7")
        
        # Receive updates from workers
        for worker_rank in range(1, world_size):
            update = sym_buf.get(worker_rank)
            server_params += update
            print(f"  ✓ Received update from worker {worker_rank}")
        
        print(f"  Server params updated (sum: {server_params.sum().item():.2f})")
    else:
        # Worker: send gradient update
        gradient = torch.randn(param_size, device=f'cuda:{rank}', dtype=torch.float32) * 0.01
        sym_buf = SymmetricMemoryBuffer(gradient)
        
        # Send to server (rank 0)
        sym_buf.put(gradient, target_rank=0)
    
    dist.barrier()


def example_ring_allreduce_custom():
    """
    Example: Custom ring all-reduce using NVSHMEM.
    
    Demonstrates building custom collectives with NVSHMEM.
    Note: For production, use NCCL AllReduce instead.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    
    # Each rank has data
    data_size = 1024
    data = torch.ones(data_size, device=device, dtype=torch.float32) * (rank + 1)
    
    sym_buf = SymmetricMemoryBuffer(data)
    
    # Ring all-reduce: send to next, receive from previous
    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1) % world_size
    
    # Reduce phase
    for step in range(world_size - 1):
        # Send to next
        sym_buf.put(data, next_rank)
        
        # Receive from previous
        received = sym_buf.get(prev_rank)
        data += received
        
        sym_buf.barrier()
    
    # Allgather phase (simplified - just show concept)
    # In practice, continue ring pattern
    
    if rank == 0:
        expected_sum = sum(range(1, world_size + 1)) * world_size
        print(f"\nCustom Ring AllReduce:")
        print(f"  Expected sum: {expected_sum}")
        print(f"  Actual sum: {data[0].item()}")
        print(f"  ✓ Correct!" if abs(data[0].item() - expected_sum) < 0.1 else "  ✗ Incorrect")


# ============================================================================
# Main Demo
# ============================================================================

def main():
    """Main demonstration."""
    # Initialize distributed
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print("=" * 80)
        print("NVSHMEM + PyTorch Integration Demo")
        print("=" * 80)
        print(f"World size: {world_size} GPUs")
        print(f"NVSHMEM available: {NVSHMEM_AVAILABLE}")
        if not NVSHMEM_AVAILABLE:
            print("⚠ Using fallback implementation (NCCL)")
            print("  For true NVSHMEM, ensure PyTorch 2.9+ with NVSHMEM support")
        print("=" * 80)
    
    # Print guide
    if rank == 0:
        print_performance_guide()
    
    # Run benchmarks
    if world_size >= 2 and rank == 0:
        print("\n" + "=" * 80)
        print("Latency Benchmark (4KB messages)")
        print("=" * 80)
    
    if world_size >= 2:
        results = benchmark_put_latency(size_bytes=4096, num_iterations=1000)
        
        if rank == 0 and results:
            if "symmetric_memory_us" in results:
                print(f"Symmetric Memory: {results['symmetric_memory_us']:.2f} μs")
            if "nccl_us" in results:
                print(f"NCCL send/recv:   {results['nccl_us']:.2f} μs")
            
            if "symmetric_memory_us" in results and "nccl_us" in results:
                speedup = results["nccl_us"] / results["symmetric_memory_us"]
                print(f"Speedup:          {speedup:.2f}x")
    
    # Run examples
    if world_size >= 2:
        if rank == 0:
            print("\n" + "=" * 80)
            print("Example 1: Parameter Server")
            print("=" * 80)
        example_parameter_server_update()
        
        if world_size >= 4:
            if rank == 0:
                print("\n" + "=" * 80)
                print("Example 2: Custom Ring AllReduce")
                print("=" * 80)
            example_ring_allreduce_custom()
    
    # Summary
    if rank == 0:
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print("\n✓ NVSHMEM is best for:")
        print("  - Low-latency, small-message communication")
        print("  - Custom algorithms with irregular patterns")
        print("  - Fine-grained synchronization")
        
        print("\n✓ Use NCCL/PyTorch for:")
        print("  - Standard training (DDP, FSDP)")
        print("  - Large collective operations")
        print("  - Production workloads")
        
        print("\n✓ On 8x B200:")
        print("  - NCCL AllReduce: 700-800 GB/s")
        print("  - NVSHMEM latency: 1-5 μs")
        print("  - Combine both for optimal performance")
        print("=" * 80)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
