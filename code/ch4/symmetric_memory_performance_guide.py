#!/usr/bin/env python3
"""
Symmetric Memory Performance Guide and Best Practices for 8x B200
================================================================

Production-grade performance tuning guide and decision trees for NVSHMEM
and PyTorch symmetric memory on Blackwell B200 GPUs.

This file provides:
1. Decision trees: When to use NVSHMEM vs NCCL vs symmetric memory
2. Message size thresholds for optimal performance
3. Code examples for common performance patterns
4. Performance pitfalls and solutions
5. Profiling integration with Nsight Systems/Compute
6. Tuning parameters and recommendations

Hardware Context:
- 8x NVIDIA Blackwell B200 GPUs
- NVLink 5.0: 1800 GB/s per link, 18 links per GPU
- HBM3e: 8 TB/s bandwidth per GPU
- L2 Cache: 256 MB per GPU
- CUDA 13.0+, PyTorch 2.9+

Performance Characteristics:
- NVSHMEM/symmetric memory: < 1μs base latency, best for < 1MB
- NCCL: ~10-50μs base latency, best for > 1MB  
- P2P: ~5-20μs latency, good for point-to-point < 10MB

Usage:
    # Run decision tree analyzer
    python symmetric_memory_performance_guide.py --analyze

    # Profile application
    torchrun --nproc_per_node=8 symmetric_memory_performance_guide.py --profile

    # Benchmark message sizes
    torchrun --nproc_per_node=8 symmetric_memory_performance_guide.py --benchmark

Educational Notes:
------------------
Performance Optimization Philosophy:

1. Measure First:
   - Profile with Nsight Systems to find bottlenecks
   - Measure actual latency/bandwidth, not theoretical
   - Compare against baseline (NCCL)

2. Choose Right Tool:
   - Latency-bound (< 100μs): NVSHMEM/symmetric memory
   - Bandwidth-bound (> 1ms): NCCL AllReduce
   - Point-to-point: Consider P2P or symmetric memory

3. Optimize Holistically:
   - Overlap communication with computation
   - Batch small messages when possible
   - Use appropriate data types (FP16/BF16/FP8)

4. Validate Correctness:
   - Numerical precision checks
   - Synchronization correctness
   - Memory safety (no race conditions)
"""

from __future__ import annotations
import arch_config  # noqa: F401 - Configure Blackwell optimizations

import argparse
import datetime
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist


# ============================================================================
# Utilities
# ============================================================================


def symmetric_memory_available() -> bool:
    """Check if symmetric memory is available."""
    return hasattr(dist, "nn") and hasattr(dist.nn, "SymmetricMemory")


def init_distributed() -> Tuple[int, int, int]:
    """Initialize distributed process group."""
    if not dist.is_initialized():
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=60),
        )
        torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), torch.cuda.current_device()


# ============================================================================
# Decision Tree for Communication Method Selection
# ============================================================================


class CommunicationType(Enum):
    """Types of communication patterns."""
    ALLREDUCE = "allreduce"
    BROADCAST = "broadcast"
    P2P = "point_to_point"
    ALLGATHER = "allgather"
    REDUCE_SCATTER = "reduce_scatter"


class CommunicationMethod(Enum):
    """Communication methods available."""
    SYMMETRIC_MEMORY = "symmetric_memory"
    NVSHMEM = "nvshmem"
    NCCL = "nccl"
    P2P_CUDA = "p2p_cuda"


@dataclass
class PerformanceRecommendation:
    """Performance recommendation with rationale."""
    method: CommunicationMethod
    expected_latency_us: float
    expected_bandwidth_gbps: float
    rationale: str
    code_example: str


class PerformanceDecisionTree:
    """
    Decision tree for selecting optimal communication method.
    
    Based on empirical measurements on Blackwell B200.
    """
    
    # Message size thresholds (bytes)
    SMALL_MESSAGE = 1024  # 1 KB
    MEDIUM_MESSAGE = 1024 * 1024  # 1 MB
    LARGE_MESSAGE = 10 * 1024 * 1024  # 10 MB
    
    # Latency estimates (microseconds)
    SYMMETRIC_MEMORY_BASE_LATENCY = 0.5
    NVSHMEM_BASE_LATENCY = 1.0
    NCCL_BASE_LATENCY = 15.0
    P2P_BASE_LATENCY = 5.0
    
    # Bandwidth estimates (GB/s)
    NVLINK_5_BANDWIDTH = 1800.0
    NCCL_EFFECTIVE_BANDWIDTH = 1400.0
    
    def recommend(
        self,
        comm_type: CommunicationType,
        message_size_bytes: int,
        num_gpus: int,
        is_latency_critical: bool = False,
    ) -> PerformanceRecommendation:
        """
        Recommend best communication method based on use case.
        
        Args:
            comm_type: Type of communication pattern
            message_size_bytes: Size of message in bytes
            num_gpus: Number of GPUs involved
            is_latency_critical: Whether latency is critical (< 100μs required)
        
        Returns:
            PerformanceRecommendation with method and rationale
        """
        # Small messages + latency critical -> Symmetric Memory
        if message_size_bytes < self.SMALL_MESSAGE and is_latency_critical:
            if comm_type == CommunicationType.P2P:
                return self._recommend_symmetric_memory_p2p(message_size_bytes)
            elif comm_type == CommunicationType.ALLREDUCE:
                return self._recommend_symmetric_memory_allreduce(message_size_bytes, num_gpus)
        
        # Medium messages -> consider symmetric memory vs NCCL
        if message_size_bytes < self.MEDIUM_MESSAGE:
            if comm_type == CommunicationType.P2P:
                return self._recommend_symmetric_memory_p2p(message_size_bytes)
            elif comm_type in (CommunicationType.ALLREDUCE, CommunicationType.ALLGATHER):
                # For medium-sized AllReduce/AllGather, NCCL may be better
                return self._recommend_nccl(comm_type, message_size_bytes, num_gpus)
        
        # Large messages -> NCCL
        if message_size_bytes >= self.MEDIUM_MESSAGE:
            return self._recommend_nccl(comm_type, message_size_bytes, num_gpus)
        
        # Default: NCCL (safest choice)
        return self._recommend_nccl(comm_type, message_size_bytes, num_gpus)
    
    def _recommend_symmetric_memory_p2p(self, message_size_bytes: int) -> PerformanceRecommendation:
        """Recommend symmetric memory for P2P communication."""
        latency_us = self.SYMMETRIC_MEMORY_BASE_LATENCY + (message_size_bytes / (self.NVLINK_5_BANDWIDTH * 1e9)) * 1e6
        bandwidth_gbps = (message_size_bytes / (latency_us * 1e-6)) / 1e9
        
        code_example = """
# Symmetric memory P2P transfer
handle = dist.nn.SymmetricMemory(local_tensor)
remote_buffer = handle.get_buffer(target_rank)
remote_buffer.copy_(local_tensor, non_blocking=True)
"""
        
        return PerformanceRecommendation(
            method=CommunicationMethod.SYMMETRIC_MEMORY,
            expected_latency_us=latency_us,
            expected_bandwidth_gbps=bandwidth_gbps,
            rationale=(
                f"Message size {message_size_bytes} bytes is small. "
                f"Symmetric memory provides < 1μs base latency vs ~5-20μs for P2P. "
                f"Best for frequent small transfers in pipeline parallel or tensor parallel."
            ),
            code_example=code_example.strip(),
        )
    
    def _recommend_symmetric_memory_allreduce(self, message_size_bytes: int, num_gpus: int) -> PerformanceRecommendation:
        """Recommend symmetric memory for AllReduce."""
        # Ring AllReduce: (num_gpus - 1) steps
        num_steps = num_gpus - 1
        per_step_latency = self.SYMMETRIC_MEMORY_BASE_LATENCY + (message_size_bytes / num_gpus) / (self.NVLINK_5_BANDWIDTH * 1e9) * 1e6
        total_latency_us = num_steps * per_step_latency
        bandwidth_gbps = (message_size_bytes / (total_latency_us * 1e-6)) / 1e9
        
        code_example = """
# Custom ring AllReduce with symmetric memory
bucket = GradientBucket(numel=tensor.numel(), ...)
bucket.allreduce_ring(rank)  # See nvshmem_training_patterns.py
"""
        
        return PerformanceRecommendation(
            method=CommunicationMethod.SYMMETRIC_MEMORY,
            expected_latency_us=total_latency_us,
            expected_bandwidth_gbps=bandwidth_gbps,
            rationale=(
                f"Small message ({message_size_bytes} bytes) with {num_gpus} GPUs. "
                f"Custom ring AllReduce via symmetric memory: ~{total_latency_us:.1f}μs "
                f"vs ~{self.NCCL_BASE_LATENCY + (message_size_bytes / (self.NCCL_EFFECTIVE_BANDWIDTH * 1e9)) * 1e6:.1f}μs with NCCL. "
                f"Best for gradient sync in small models."
            ),
            code_example=code_example.strip(),
        )
    
    def _recommend_nccl(self, comm_type: CommunicationType, message_size_bytes: int, num_gpus: int) -> PerformanceRecommendation:
        """Recommend NCCL for larger messages."""
        latency_us = self.NCCL_BASE_LATENCY + (message_size_bytes / (self.NCCL_EFFECTIVE_BANDWIDTH * 1e9)) * 1e6
        bandwidth_gbps = min(self.NCCL_EFFECTIVE_BANDWIDTH, (message_size_bytes / (latency_us * 1e-6)) / 1e9)
        
        code_examples = {
            CommunicationType.ALLREDUCE: "dist.all_reduce(tensor, op=dist.ReduceOp.SUM)",
            CommunicationType.BROADCAST: "dist.broadcast(tensor, src=0)",
            CommunicationType.ALLGATHER: "dist.all_gather(tensor_list, tensor)",
            CommunicationType.REDUCE_SCATTER: "dist.reduce_scatter(output, input_list)",
            CommunicationType.P2P: "dist.send(tensor, dst=rank) / dist.recv(tensor, src=rank)",
        }
        
        return PerformanceRecommendation(
            method=CommunicationMethod.NCCL,
            expected_latency_us=latency_us,
            expected_bandwidth_gbps=bandwidth_gbps,
            rationale=(
                f"Message size {message_size_bytes} bytes is medium/large. "
                f"NCCL is highly optimized for bandwidth and provides ~{bandwidth_gbps:.0f} GB/s. "
                f"Best for large gradient sync, parameter broadcasts, and high-bandwidth operations."
            ),
            code_example=code_examples.get(comm_type, "dist.all_reduce(tensor)"),
        )


# ============================================================================
# Performance Pitfalls and Solutions
# ============================================================================


class PerformancePitfall:
    """Common performance pitfall with solution."""
    
    def __init__(self, name: str, description: str, symptom: str, solution: str, code_fix: str):
        self.name = name
        self.description = description
        self.symptom = symptom
        self.solution = solution
        self.code_fix = code_fix


COMMON_PITFALLS = [
    PerformancePitfall(
        name="Excessive Synchronization",
        description=(
            "Calling dist.barrier() or torch.cuda.synchronize() too frequently "
            "destroys overlap opportunities and adds ~10-100μs per call."
        ),
        symptom="High idle time in Nsight Systems, frequent synchronization gaps",
        solution=(
            "Remove unnecessary barriers. Use non-blocking operations and only "
            "synchronize when results are actually needed."
        ),
        code_fix="""
# BAD: Synchronize after every operation
for step in range(100):
    output = model(input)
    dist.barrier()  # Unnecessary!
    
# GOOD: Only synchronize when needed
for step in range(100):
    output = model(input)
# Synchronize once at end if needed
dist.barrier()
""",
    ),
    
    PerformancePitfall(
        name="Wrong Communication Primitive",
        description=(
            "Using NCCL for very small messages (< 1KB) or symmetric memory "
            "for very large messages (> 10MB)."
        ),
        symptom="High latency for small messages, low bandwidth for large messages",
        solution="Use decision tree to select appropriate method based on message size",
        code_fix="""
# BAD: NCCL for tiny gradients (< 1KB)
dist.all_reduce(tiny_grad)  # ~15μs latency

# GOOD: Symmetric memory for tiny gradients
sym_handle = dist.nn.SymmetricMemory(tiny_grad)
# Custom ring allreduce: ~1μs latency

# BAD: Symmetric memory for huge tensors (> 10MB)
remote_buf.copy_(huge_tensor)  # May timeout

# GOOD: NCCL for huge tensors
dist.all_reduce(huge_tensor)  # Optimized bandwidth
""",
    ),
    
    PerformancePitfall(
        name="Memory Alignment Issues",
        description=(
            "Unaligned memory accesses cause additional transactions and "
            "reduce effective bandwidth by 30-50%."
        ),
        symptom="Lower than expected bandwidth, high L2 cache misses",
        solution="Ensure tensors are aligned to 256-byte boundaries (HBM3e optimal burst)",
        code_fix="""
# BAD: Unaligned allocation
tensor = torch.randn(1023, device='cuda')  # Odd size

# GOOD: Aligned allocation
aligned_size = ((1023 + 255) // 256) * 256
tensor = torch.randn(aligned_size, device='cuda')[:1023]
""",
    ),
    
    PerformancePitfall(
        name="Blocking on Remote Access",
        description=(
            "Immediately using data from symmetric memory without checking "
            "if transfer is complete causes pipeline stalls."
        ),
        symptom="GPU idle time, waiting for remote data",
        solution="Use double buffering and overlap remote access with computation",
        code_fix="""
# BAD: Blocking remote access
remote_data = sym_handle.get_buffer(peer_rank)
result = compute(remote_data)  # May stall

# GOOD: Double buffering
buffer_A = sym_handle_A.get_buffer(peer_rank)
buffer_B = sym_handle_B.get_buffer(peer_rank)

for i in range(N):
    # Compute on buffer A while fetching into buffer B
    result = compute(buffer_A if i % 2 == 0 else buffer_B)
""",
    ),
    
    PerformancePitfall(
        name="Small Batch Sizes",
        description=(
            "Using very small batches (< 8) doesn't saturate GPU compute "
            "and makes communication overhead dominant."
        ),
        symptom="Low GPU utilization (< 50%), high % time in communication",
        solution="Increase batch size or use gradient accumulation",
        code_fix="""
# BAD: Tiny batches
batch_size = 4  # GPU only 20% utilized

# GOOD: Larger batches or gradient accumulation
batch_size = 32  # GPU 80%+ utilized

# OR: Gradient accumulation
for microbatch in microbatches:
    loss = model(microbatch)
    loss.backward()  # Accumulate
# Synchronize gradients once for all microbatches
""",
    ),
]


# ============================================================================
# Profiling Integration
# ============================================================================


class ProfilerIntegration:
    """
    Integration with Nsight Systems and Nsight Compute for profiling.
    
    Provides helpers for annotating symmetric memory operations.
    """
    
    @staticmethod
    def annotate_symmetric_memory_op(name: str, message_size_bytes: int):
        """
        Annotate symmetric memory operation for profiling.
        
        Usage:
            with ProfilerIntegration.annotate_symmetric_memory_op("AllReduce", 1024):
                bucket.allreduce_ring(rank)
        """
        class AnnotationContext:
            def __enter__(self):
                if torch.cuda.is_available():
                    torch.cuda.nvtx.range_push(f"SymMem:{name}:{message_size_bytes}B")
                return self
            
            def __exit__(self, *args):
                if torch.cuda.is_available():
                    torch.cuda.nvtx.range_pop()
        
        return AnnotationContext()
    
    @staticmethod
    def print_profiling_command():
        """Print Nsight Systems profiling command."""
        cmd = """
# Profile with Nsight Systems
nsys profile -t cuda,nvtx,osrt,cudnn,cublas \\
    -o symmetric_memory_profile \\
    --capture-range=cudaProfilerApi \\
    python your_script.py

# Analyze NVSHMEM/symmetric memory metrics with Nsight Compute
ncu --set full --target-processes all \\
    --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\\
lts__t_sector_hit_rate.pct,\\
sm__throughput.avg.pct_of_peak_sustained_elapsed \\
    python your_script.py
"""
        print(cmd.strip())


# ============================================================================
# Benchmark Suite
# ============================================================================


def benchmark_message_sizes() -> None:
    """
    Benchmark different message sizes to validate decision tree.
    
    Measures actual latency and bandwidth for NCCL vs symmetric memory.
    """
    rank, world_size, device = init_distributed()
    
    message_sizes = [1024, 4096, 16384, 65536, 262144, 1048576, 4194304]  # 1KB to 4MB
    
    results = []
    
    for size_bytes in message_sizes:
        numel = size_bytes // 4  # float32
        tensor = torch.randn(numel, device=device)
        
        # Benchmark NCCL
        dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        nccl_latency_us = (time.perf_counter() - start) * 1e6 / 100
        
        # Benchmark symmetric memory (if available)
        sym_latency_us = None
        if symmetric_memory_available():
            try:
                handle = dist.nn.SymmetricMemory(tensor)
                dist.barrier()
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(100):
                    next_rank = (rank + 1) % world_size
                    remote = handle.get_buffer(next_rank)
                    remote.copy_(tensor, non_blocking=True)
                torch.cuda.synchronize()
                sym_latency_us = (time.perf_counter() - start) * 1e6 / 100
            except Exception:
                pass
        
        results.append((size_bytes, nccl_latency_us, sym_latency_us))
    
    if rank == 0:
        print("\n" + "="*70)
        print("Message Size Benchmark Results")
        print("="*70)
        print(f"{'Size':>12} | {'NCCL (μs)':>12} | {'SymMem (μs)':>14} | {'Speedup':>10}")
        print("-"*70)
        for size_bytes, nccl_lat, sym_lat in results:
            size_str = f"{size_bytes // 1024}KB" if size_bytes >= 1024 else f"{size_bytes}B"
            sym_str = f"{sym_lat:.2f}" if sym_lat else "N/A"
            speedup = f"{nccl_lat / sym_lat:.2f}x" if sym_lat else "N/A"
            print(f"{size_str:>12} | {nccl_lat:>12.2f} | {sym_str:>14} | {speedup:>10}")
        print("="*70)


# ============================================================================
# Decision Tree Analysis
# ============================================================================


def analyze_use_case() -> None:
    """Run decision tree analysis for common use cases."""
    tree = PerformanceDecisionTree()
    
    use_cases = [
        ("Small gradient sync (100KB)", CommunicationType.ALLREDUCE, 100 * 1024, 8, True),
        ("Large gradient sync (10MB)", CommunicationType.ALLREDUCE, 10 * 1024 * 1024, 8, False),
        ("Pipeline activation (1MB)", CommunicationType.P2P, 1024 * 1024, 2, True),
        ("Parameter broadcast (50MB)", CommunicationType.BROADCAST, 50 * 1024 * 1024, 8, False),
        ("Tiny gradient (1KB)", CommunicationType.ALLREDUCE, 1024, 8, True),
    ]
    
    print("\n" + "="*80)
    print("Performance Decision Tree Analysis")
    print("="*80)
    
    for name, comm_type, size, num_gpus, latency_critical in use_cases:
        print(f"\n{name}")
        print("-" * 80)
        print(f"  Pattern: {comm_type.value}")
        print(f"  Message size: {size // 1024}KB")
        print(f"  GPUs: {num_gpus}")
        print(f"  Latency critical: {latency_critical}")
        
        rec = tree.recommend(comm_type, size, num_gpus, latency_critical)
        
        print(f"\n  Recommendation: {rec.method.value.upper()}")
        print(f"  Expected latency: {rec.expected_latency_us:.2f}μs")
        print(f"  Expected bandwidth: {rec.expected_bandwidth_gbps:.1f} GB/s")
        print(f"\n  Rationale: {rec.rationale}")
        print(f"\n  Code example:")
        for line in rec.code_example.split('\n'):
            print(f"    {line}")
    
    print("\n" + "="*80)


# ============================================================================
# CLI Entrypoint
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="Symmetric memory performance guide")
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run decision tree analysis",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run message size benchmarks",
    )
    parser.add_argument(
        "--pitfalls",
        action="store_true",
        help="Show common performance pitfalls",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Show profiling commands",
    )
    args = parser.parse_args()
    
    if args.analyze:
        analyze_use_case()
    
    if args.benchmark:
        init_distributed()
        benchmark_message_sizes()
    
    if args.pitfalls:
        print("\n" + "="*80)
        print("Common Performance Pitfalls and Solutions")
        print("="*80)
        for pitfall in COMMON_PITFALLS:
            print(f"\n{pitfall.name}")
            print("-" * 80)
            print(f"Description: {pitfall.description}")
            print(f"Symptom: {pitfall.symptom}")
            print(f"Solution: {pitfall.solution}")
            print("\nCode fix:")
            for line in pitfall.code_fix.strip().split('\n'):
                print(f"  {line}")
        print("\n" + "="*80)
    
    if args.profile:
        print("\n" + "="*80)
        print("Profiling Commands")
        print("="*80)
        ProfilerIntegration.print_profiling_command()
        print("="*80)
    
    if not any([args.analyze, args.benchmark, args.pitfalls, args.profile]):
        print("Symmetric Memory Performance Guide")
        print("="*80)
        print("Use --help to see available options")
        print("\nQuick start:")
        print("  --analyze    : Show decision tree recommendations")
        print("  --benchmark  : Measure actual performance")
        print("  --pitfalls   : Learn common mistakes")
        print("  --profile    : Get profiling commands")


if __name__ == "__main__":
    main()
