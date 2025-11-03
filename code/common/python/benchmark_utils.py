"""benchmark_utils.py - Shared benchmarking utilities for PyTorch examples."""

import time
import torch
import statistics
from typing import Callable, List, Optional, Dict, Any


def warmup_cuda(func: Callable, iterations: int = 10) -> None:
    """Warmup function to stabilize GPU clocks and caches."""
    for _ in range(iterations):
        func()
    torch.cuda.synchronize()


def benchmark_function(
    func: Callable,
    iterations: int = 100,
    warmup: int = 10,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """
    Benchmark a function with proper warmup and synchronization.
    
    Args:
        func: Function to benchmark
        iterations: Number of benchmark iterations
        warmup: Number of warmup iterations
        device: CUDA device (if None, will detect)
    
    Returns:
        Dictionary with timing statistics (mean, std, min, max, median)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    is_cuda = device.type == "cuda"
    
    # Warmup
    for _ in range(warmup):
        func()
    if is_cuda:
        torch.cuda.synchronize(device)
    
    # Benchmark
    times: List[float] = []
    for _ in range(iterations):
        if is_cuda:
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        func()
        if is_cuda:
            torch.cuda.synchronize(device)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds
    
    return {
        "mean_ms": statistics.mean(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_ms": min(times),
        "max_ms": max(times),
        "median_ms": statistics.median(times),
    }


def compare_implementations(
    baseline: Callable,
    optimized: Callable,
    name: str = "Comparison",
    iterations: int = 100,
    warmup: int = 10
) -> None:
    """
    Compare baseline vs optimized implementations and print results.
    
    Args:
        baseline: Baseline function
        optimized: Optimized function
        name: Name for the comparison
        iterations: Number of benchmark iterations
        warmup: Number of warmup iterations
    """
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"{'='*70}")
    
    print("\nBaseline:")
    baseline_stats = benchmark_function(baseline, iterations, warmup)
    print(f"  Mean: {baseline_stats['mean_ms']:.3f} ms")
    print(f"  Std:  {baseline_stats['std_ms']:.3f} ms")
    print(f"  Min:  {baseline_stats['min_ms']:.3f} ms")
    print(f"  Max:  {baseline_stats['max_ms']:.3f} ms")
    
    print("\nOptimized:")
    optimized_stats = benchmark_function(optimized, iterations, warmup)
    print(f"  Mean: {optimized_stats['mean_ms']:.3f} ms")
    print(f"  Std:  {optimized_stats['std_ms']:.3f} ms")
    print(f"  Min:  {optimized_stats['min_ms']:.3f} ms")
    print(f"  Max:  {optimized_stats['max_ms']:.3f} ms")
    
    speedup = baseline_stats['mean_ms'] / optimized_stats['mean_ms']
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"{'='*70}\n")


def calculate_bandwidth_gbs(bytes_transferred: int, time_ms: float) -> float:
    """Calculate memory bandwidth in GB/s."""
    return (bytes_transferred / (1024**3)) / (time_ms / 1000)


def calculate_tflops(flops: int, time_ms: float) -> float:
    """Calculate TFLOPS (trillion floating-point operations per second)."""
    return (flops / 1e12) / (time_ms / 1000)


def print_gpu_info(device: int = 0) -> None:
    """Print GPU information for the specified device."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    prop = torch.cuda.get_device_properties(device)
    print(f"\nGPU {device}: {prop.name}")
    print(f"  Compute capability: {prop.major}.{prop.minor}")
    print(f"  Total memory: {prop.total_memory / (1024**3):.2f} GB")
    print(f"  Multi-processors: {prop.multi_processor_count}")
    print(f"  CUDA cores: ~{prop.multi_processor_count * 128}")  # Approximate
    print()


def format_comparison_table(results: Dict[str, Dict[str, float]]) -> str:
    """
    Format comparison results as a markdown table.
    
    Args:
        results: Dictionary mapping implementation name to stats dict
    
    Returns:
        Formatted markdown table string
    """
    lines = []
    lines.append("| Implementation | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |")
    lines.append("|---------------|-----------|----------|----------|----------|")
    
    for name, stats in results.items():
        lines.append(
            f"| {name:13s} | {stats['mean_ms']:9.3f} | "
            f"{stats['std_ms']:8.3f} | {stats['min_ms']:8.3f} | "
            f"{stats['max_ms']:8.3f} |"
        )
    
    return "\n".join(lines)

