#!/usr/bin/env python3
"""
NVSHMEM vs NCCL Benchmark (Conceptual)
=====================================

Micro-benchmark comparing NCCL collectives against PyTorch 2.9 symmetric
memory (NVSHMEM-backed) operations on 8x Blackwell B200 GPUs.

Measurements:
- Latency (µs) for small message sizes (1 KB - 1 MB)
- Bandwidth (GB/s) for large message sizes (16 MB - 512 MB)

The script degrades gracefully when NVSHMEM/symmetric memory is missing,
reporting NCCL numbers only so it can run on non-Blackwell hardware.

Usage:
    torchrun --nproc_per_node=8 nvshmem_vs_nccl_benchmark.py \
        --min-bytes 1024 --max-bytes 67108864 --steps 6
"""

from __future__ import annotations
import arch_config  # noqa: F401 - Configure Blackwell optimizations

import argparse
import datetime
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.distributed as dist


# ============================================================================
# Utilities
# ============================================================================


def symmetric_memory_available() -> bool:
    return hasattr(dist, "nn") and hasattr(dist.nn, "SymmetricMemory")


def init_distributed() -> int:
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
    else:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    return rank


@dataclass
class BenchmarkResult:
    bytes: int
    latency_us: float
    bandwidth_gbps: float


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(num_bytes)
    unit = 0
    while value >= 1024 and unit < len(units) - 1:
        value /= 1024.0
        unit += 1
    return f"{value:.1f} {units[unit]}"


def _measure_nccl(bytes_per_rank: int, iterations: int) -> BenchmarkResult:
    device = torch.cuda.current_device()
    dtype = torch.float16
    numel = bytes_per_rank // torch.tensor([], dtype=dtype).element_size()
    numel = max(1, numel)

    tensor = torch.randn(numel, device=device, dtype=dtype)
    tensor2 = torch.empty_like(tensor)

    # Warmup
    for _ in range(5):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    end.record()
    torch.cuda.synchronize(device)

    elapsed_ms = start.elapsed_time(end)
    latency_us = (elapsed_ms * 1000.0) / iterations
    total_bytes = bytes_per_rank * dist.get_world_size()
    bandwidth_gbps = (total_bytes / (elapsed_ms / iterations / 1000.0)) / 1e9
    _ = tensor2
    return BenchmarkResult(bytes=bytes_per_rank, latency_us=latency_us, bandwidth_gbps=bandwidth_gbps)


def _measure_symmetric_memory(bytes_per_rank: int, iterations: int) -> Optional[BenchmarkResult]:
    if not symmetric_memory_available():
        return None

    device = torch.cuda.current_device()
    dtype = torch.float16
    numel = bytes_per_rank // torch.tensor([], dtype=dtype).element_size()
    numel = max(1, numel)

    local = torch.randn(numel, device=device, dtype=dtype)
    try:
        sym = dist.nn.SymmetricMemory(local)
        buffer = sym.buffer
    except Exception:
        return None

    neighbor = (dist.get_rank() + 1) % dist.get_world_size()
    remote = sym.get_buffer(neighbor)

    for _ in range(5):
        remote.copy_(buffer)
        torch.cuda.current_stream().synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)

    start.record()
    for _ in range(iterations):
        remote.copy_(buffer)
        torch.cuda.current_stream().synchronize()
    end.record()
    torch.cuda.synchronize(device)

    elapsed_ms = start.elapsed_time(end)
    latency_us = (elapsed_ms * 1000.0) / iterations
    bandwidth_gbps = (bytes_per_rank / (elapsed_ms / iterations / 1000.0)) / 1e9
    return BenchmarkResult(bytes=bytes_per_rank, latency_us=latency_us, bandwidth_gbps=bandwidth_gbps)


def sweep_sizes(min_bytes: int, max_bytes: int, steps: int) -> List[int]:
    if steps <= 1:
        return [min_bytes]
    ratios = torch.logspace(
        start=math.log10(min_bytes),
        end=math.log10(max_bytes),
        steps=steps,
        base=10.0,
    )
    return [int(r.item()) for r in ratios]


def benchmark(args: argparse.Namespace) -> Dict[str, List[BenchmarkResult]]:
    results: Dict[str, List[BenchmarkResult]] = {"nccl": [], "nvshmem": []}
    sizes = sweep_sizes(args.min_bytes, args.max_bytes, args.steps)

    for message_size in sizes:
        dist.barrier()
        nccl = _measure_nccl(message_size, args.iterations)
        results["nccl"].append(nccl)
        dist.barrier()
        nvshmem_res = _measure_symmetric_memory(message_size, args.iterations)
        if nvshmem_res is not None:
            results["nvshmem"].append(nvshmem_res)
        dist.barrier()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare NVSHMEM vs NCCL latency/bandwidth")
    parser.add_argument("--min-bytes", type=int, default=1024, help="Smallest message size per rank")
    parser.add_argument("--max-bytes", type=int, default=64 * 1024 * 1024, help="Largest message size per rank")
    parser.add_argument("--steps", type=int, default=6, help="Number of message sizes to sample")
    parser.add_argument("--iterations", type=int, default=50, help="Iterations per measurement")
    args = parser.parse_args()

    rank = init_distributed()
    results = benchmark(args)

    if rank == 0:
        print("\nNVSHMEM vs NCCL Benchmark (conceptual)")
        print("------------------------------------------------------")
        print(f"Symmetric memory available: {symmetric_memory_available()}")
        print("Message Size | NCCL Latency (us) | NCCL BW (GB/s) | NVSHMEM Latency (us) | NVSHMEM BW (GB/s)")
        print("-------------------------------------------------------------------------------------------")

        nvshmem_dict = {res.bytes: res for res in results["nvshmem"]}
        for res in results["nccl"]:
            nv = nvshmem_dict.get(res.bytes)
            nv_lat = f"{nv.latency_us:8.2f}" if nv else "   n/a "
            nv_bw = f"{nv.bandwidth_gbps:8.2f}" if nv else "   n/a "
            print(
                f"{_format_bytes(res.bytes):>12} | {res.latency_us:16.2f} | {res.bandwidth_gbps:13.2f} | "
                f"{nv_lat:>18} | {nv_bw:>15}"
            )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
