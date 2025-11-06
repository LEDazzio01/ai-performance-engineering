#!/usr/bin/env python3

import pathlib
import sys

_EXTRAS_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_EXTRAS_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTRAS_REPO_ROOT))

from pathlib import Path

"""Chapter 2: Hardware Topology Inspection

Analyze CPU/GPU topology, memory bandwidth, and interconnect characteristics
for NVIDIA Blackwell-based systems.
"""
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    pass
except Exception:  # pragma: no cover - optional per-environment tweak
    arch_config = None  # Graceful fallback if arch_config is absent


import time
from typing import Any, Dict

import GPUtil
import psutil
import torch
import torch.cuda.nvtx as nvtx

BENCHMARK_QUICK = os.environ.get("BENCHMARK_QUICK", "0") == "1"


def get_architecture() -> str:
    """Detect and return the current GPU architecture."""
    if not torch.cuda.is_available():
        return "cpu"

    device_props = torch.cuda.get_device_properties(0)
    compute_capability = f"{device_props.major}.{device_props.minor}"

    if compute_capability == "10.0":
        return "blackwell"

    if device_props.major == 12:
        return "grace_blackwell"

    return "other"


def get_architecture_info() -> Dict[str, Any]:
    """Get detailed architecture information."""
    arch = get_architecture()
    if arch == "blackwell":
        return {
            "name": "Blackwell B200/B300",
            "compute_capability": "10.0",
            "sm_version": "sm_100",
            "memory_bandwidth": "up to ~8 TB/s",
            "tensor_cores": "5th Gen",
            "features": ["HBM3e", "TMA", "NVLink-C2C"],
        }
    if arch == "grace_blackwell":
        return {
            "name": "Grace-Blackwell GB10",
            "compute_capability": "12.x",
            "sm_version": "sm_121",
            "memory_bandwidth": "Unknown",
            "tensor_cores": "Next Gen",
            "features": ["Grace-Blackwell coherence fabric"],
        }

    return {
        "name": "Other",
        "compute_capability": "Unknown",
        "sm_version": "Unknown",
        "memory_bandwidth": "Unknown",
        "tensor_cores": "Unknown",
        "features": [],
    }


def get_gpu_info() -> Dict[str, Any]:
    """Get comprehensive GPU information."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    device_props = torch.cuda.get_device_properties(0)
    compute_capability = f"{device_props.major}.{device_props.minor}"

    major = device_props.major
    architecture = "Unsupported"
    memory_bandwidth_tbps = None
    tensor_cores = "Unknown"
    hbm3e_memory = False
    tma_support = False
    nvlink_c2c = False
    max_unified_memory_tb = None
    sm_version = "sm_unknown"

    if major == 10:
        architecture = "Blackwell B200/B300"
        memory_bandwidth_tbps = 8.0
        tensor_cores = "5th Generation"
        hbm3e_memory = True
        tma_support = True
        nvlink_c2c = True
        max_unified_memory_tb = 30
        sm_version = "sm_100"
    elif major == 12:
        architecture = "Grace-Blackwell GB10"
        tensor_cores = "Next Generation"
        tma_support = True
        nvlink_c2c = True
        sm_version = "sm_121"

    return {
        "name": device_props.name,
        "compute_capability": compute_capability,
        "total_memory_gb": device_props.total_memory / (1024 ** 3),
        "memory_bandwidth_gbps": (memory_bandwidth_tbps * 1000) if memory_bandwidth_tbps else None,
        "max_threads_per_block": getattr(device_props, 'max_threads_per_block', 1024),
        "max_threads_per_sm": device_props.max_threads_per_multi_processor,
        "num_sms": device_props.multi_processor_count,
        "warp_size": device_props.warp_size,
        "max_shared_memory_per_block": device_props.shared_memory_per_block,
        "max_shared_memory_per_sm": device_props.shared_memory_per_multiprocessor,
        "l2_cache_size": device_props.L2_cache_size,
        "architecture": architecture,
        "sm_version": sm_version,
        "hbm3e_memory": hbm3e_memory,
        "memory_bandwidth_tbps": memory_bandwidth_tbps,
        "tma_support": tma_support,
        "nvlink_c2c": nvlink_c2c,
        "tensor_cores": tensor_cores,
        "unified_memory": True,
        "max_unified_memory_tb": max_unified_memory_tb,
    }


def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information."""
    core_util = psutil.cpu_percent(interval=0.1, percpu=True)
    overall_util = sum(core_util) / len(core_util) if core_util else psutil.cpu_percent(interval=None)

    cpu_info = {
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "cpu_percent": core_util,
        "overall_percent": overall_util,
    }

    memory = psutil.virtual_memory()
    memory_info = {
        "total_gb": memory.total / 1e9,
        "available_gb": memory.available / 1e9,
        "used_gb": memory.used / 1e9,
        "percent": memory.percent,
    }

    gpu_info: Dict[str, Dict[str, Any]] = {}
    try:
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            gpu_info[f"gpu_{i}"] = {
                "name": gpu.name,
                "load": gpu.load,
                "memory_used": gpu.memoryUsed,
                "memory_total": gpu.memoryTotal,
                "temperature": gpu.temperature,
                "uuid": gpu.uuid,
            }
    except Exception:
        pass

    return {
        "cpu": cpu_info,
        "memory": memory_info,
        "gpus": gpu_info,
    }


def demonstrate_blackwell_features() -> None:
    """Demonstrate detected GPU architecture features."""
    print("\n=== Blackwell/Grace-Blackwell Features Demonstration ===\n")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping Blackwell features")
        return

    gpu_info = get_gpu_info()

    architecture = gpu_info.get("architecture")
    if architecture == "Blackwell B200/B300":
        print(" This is a Blackwell B200/B300 GPU")
        print(f" Compute Capability: {gpu_info['compute_capability']} (SM100)")
        print(f" Memory: {gpu_info['total_memory_gb']:.1f} GB")
        print(f" Memory Bandwidth: {gpu_info['memory_bandwidth_tbps']} TB/s")
        print(" 5th Generation Tensor Cores")
        print(" TMA (Tensor Memory Accelerator)")
        print(" NVLink-C2C (Grace CPU ↔ Blackwell GPU coherent link)")
        print(" Unified Memory Architecture")
        print(f" Max Unified Memory: {gpu_info['max_unified_memory_tb']} TB")
        print(" Grace LPDDR5X bandwidth: single-Grace provides up to ~500 GB/s")
        print(" Dual-Grace \"Grace CPU Superchip\" reaches ~1 TB/s (not part of GB200)")
    elif architecture == "Grace-Blackwell GB10":
        print(" This is a Grace-Blackwell GB10 GPU")
        print(f" Compute Capability: {gpu_info['compute_capability']} ({gpu_info.get('sm_version', 'unknown SM')})")
        print(f" Memory: {gpu_info['total_memory_gb']:.1f} GB")
        bandwidth = gpu_info.get("memory_bandwidth_gbps")
        bandwidth_str = f"{bandwidth:.1f} GB/s" if bandwidth else "Unknown"
        print(f" Memory Bandwidth: {bandwidth_str}")
        print(f" Tensor Cores: {gpu_info['tensor_cores']}")
        if gpu_info.get("nvlink_c2c"):
            print(" NVLink-C2C fabric available")
        print(" Stream-ordered memory APIs supported")
    else:
        print("This is not a Blackwell B200/B300 GPU")
        print(f"GPU: {gpu_info['name']}")
        print(f"Compute Capability: {gpu_info['compute_capability']}")
        bandwidth = gpu_info.get("memory_bandwidth_gbps")
        bandwidth_str = f"{bandwidth:.1f} GB/s" if bandwidth else "Unknown"
        print(f"Memory Bandwidth: {bandwidth_str}")


def benchmark_memory_bandwidth() -> None:
    """Benchmark memory bandwidth to demonstrate HBM3e performance."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory benchmark")
        return

    print("\n=== Memory Bandwidth Benchmark ===")

    copy_sizes_gb = [2] if BENCHMARK_QUICK else [2, 4, 8]
    for size_gb in copy_sizes_gb:
        size_bytes = int(size_gb * (1024 ** 3))
        num_elements = size_bytes // 2  # float16
        src = torch.empty(num_elements, device="cuda", dtype=torch.float16)
        dst = torch.empty_like(src)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        iterations = 5 if BENCHMARK_QUICK else 50

        start.record()
        for _ in range(iterations):
            dst.copy_(src)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        bandwidth_gbps = (size_bytes * iterations / elapsed_ms) / 1e6
        print(f"Copy {size_gb:2d} GB: {bandwidth_gbps / 1024:.2f} TB/s")

    sizes = [1024, 4096] if BENCHMARK_QUICK else [1024, 2048, 4096, 8192, 16384]

    for size in sizes:
        try:
            a = torch.randn(size, size, device="cuda")
            b = torch.randn(size, size, device="cuda")

            warmup_iters = 2 if BENCHMARK_QUICK else 5
            for _ in range(warmup_iters):
                _ = torch.mm(a, b)

            torch.cuda.synchronize()
            start_time = time.time()

            with nvtx.range(f"gemm_{size}"):
                active_iters = 3 if BENCHMARK_QUICK else 10
                for _ in range(active_iters):
                    _ = torch.mm(a, b)

            torch.cuda.synchronize()
            end_time = time.time()

            avg_time = (end_time - start_time) / active_iters
            bytes_transferred = 2 * size * size * 4
            bandwidth_gbps = (bytes_transferred / avg_time) / 1e9

            print(f"GEMM {size}x{size}: {avg_time:.4f}s, {bandwidth_gbps:.1f} GB/s")

        except RuntimeError as err:
            if "out of memory" in str(err):
                print(f"GEMM {size}x{size}: OOM")
            else:
                print(f"GEMM {size}x{size}: Error - {err}")


def benchmark_tensor_operations() -> None:
    """Benchmark various tensor operations to demonstrate Blackwell optimizations."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tensor benchmark")
        return

    print("\n=== Tensor Operations Benchmark ===")

    ops = [
        ("Matrix Multiplication", lambda x, y: torch.mm(x, y)),
        ("Element-wise Addition", torch.add),
        ("Element-wise Multiplication", torch.mul),
        ("Matrix Transpose", lambda x, _: x.t()),
        ("Reduction Sum", lambda x, _: torch.sum(x)),
    ]

    size = 1024
    a = torch.randn(size, size, device="cuda")
    b = torch.randn(size, size, device="cuda")

    for op_name, op_func in ops:
        try:
            warmup_iters = 1 if BENCHMARK_QUICK else 2
            for _ in range(warmup_iters):
                op_func(a, b)

            torch.cuda.synchronize()
            start_time = time.time()

            with nvtx.range(f"tensor_op_{op_name.lower().replace(' ', '_')}"):
                active_iters = 3 if BENCHMARK_QUICK else 10
                for _ in range(active_iters):
                    op_func(a, b)

            torch.cuda.synchronize()
            end_time = time.time()

            avg_time = (end_time - start_time) / active_iters
            print(f"{op_name:25}: {avg_time:.6f}s")

        except Exception as exc:
            print(f"{op_name:25}: Error - {exc}")


def demonstrate_memory_hierarchy() -> None:
    """Demonstrate memory hierarchy analysis."""
    print("\n=== Memory Hierarchy Analysis ===")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory hierarchy")
        return

    gpu_info = get_gpu_info()

    print("Memory Hierarchy (from fastest to slowest):")
    print("1. Registers (per-thread)")
    print("2. Shared Memory (per-block)")
    print("3. L1 Cache (per-SM)")
    print("4. L2 Cache (global)")
    print("5. HBM3e Memory (global)")
    print("6. CPU Memory (via unified memory)")

    print("\nMemory Specifications:")
    print(f"• Shared Memory per Block: {gpu_info['max_shared_memory_per_block'] / 1024:.1f} KB")
    print(f"• Shared Memory per SM: {gpu_info['max_shared_memory_per_sm'] / 1024:.1f} KB")
    print(f"• L2 Cache Size: {gpu_info['l2_cache_size'] / 1024:.1f} KB")
    print(f"• Global Memory: {gpu_info['total_memory_gb']:.1f} GB")

    if gpu_info.get("hbm3e_memory"):
        print(f"• HBM3e Bandwidth: {gpu_info['memory_bandwidth_tbps']} TB/s")
        print("• Unified Memory: 30 TB total")


def demonstrate_profiling_capabilities() -> None:
    """Demonstrate the latest profiling capabilities."""
    print("\n=== Latest Profiling Capabilities ===")

    print("Available Profiling Tools:")
    print("1. Nsight Systems (nsys) - Timeline analysis")
    print("   Command: nsys profile -t cuda,nvtx,osrt -o timeline_profile python script.py")

    print("\n2. Nsight Compute (ncu) - Kernel-level analysis")
    print("   Command: ncu --metrics achieved_occupancy,warp_execution_efficiency -o kernel_profile python script.py")

    print("\n3. PyTorch Profiler - Framework-level analysis")
    print('   Command: python -c "import torch.profiler; print(\'Available\')"')

    print("\n4. HTA (Holistic Tracing Analysis) - Multi-GPU analysis")
    print("   Command: nsys profile -t cuda,nvtx,osrt,cudnn,cublas,nccl -o hta_profile python script.py")

    print("\n5. Perf - System-level analysis")
    print("   Command: perf record -g -p $(pgrep python) -o perf.data")

    print("\n6. Enhanced PyTorch Profiler - Memory, FLOPs, modules")
    print("   Features: record_shapes=True, with_stack=True, with_flops=True, profile_memory=True")

    try:
        from torch.profiler import ProfilerActivity, profile  # noqa: F401
        print("\n PyTorch Profiler is available")
    except ImportError:
        print("\n PyTorch Profiler not available")


def demonstrate_system_monitoring() -> None:
    """Demonstrate comprehensive system monitoring."""
    print("\n=== System Monitoring Demo ===")

    system_info = get_system_info()

    print("CPU Information:")
    print(f"• Physical Cores: {system_info['cpu']['physical_cores']}")
    print(f"• Logical Cores: {system_info['cpu']['logical_cores']}")
    print(f"• Overall Usage: {system_info['cpu']['overall_percent']:.1f}%")

    print("\nMemory Information:")
    print(f"• Total Memory: {system_info['memory']['total_gb']:.1f} GB")
    print(f"• Available Memory: {system_info['memory']['available_gb']:.1f} GB")
    print(f"• Memory Usage: {system_info['memory']['percent']:.1f}%")

    if system_info.get("gpus"):
        print("\nGPU Information:")
        for gpu_id, gpu_data in system_info['gpus'].items():
            print(f"• {gpu_id}: {gpu_data['name']}")
            print(f"  - Utilization: {gpu_data['load'] * 100:.1f}%")
            print(f"  - Memory Used: {gpu_data['memory_used']} MB")
            print(f"  - Memory Total: {gpu_data['memory_total']} MB")
            print(f"  - Temperature: {gpu_data['temperature']}°C")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        print("\nPyTorch GPU Memory:")
        print(f"• Allocated: {allocated:.2f} GB")
        print(f"• Cached: {cached:.2f} GB")


def demonstrate_blackwell_optimizations() -> None:
    """Demonstrate Blackwell B200/B300 specific optimizations."""
    print("\n=== Blackwell B200/B300 Optimizations ===")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping Blackwell optimizations")
        return

    gpu_info = get_gpu_info()

    if gpu_info.get("architecture") == "Blackwell B200/B300":
        print("Blackwell B200/B300 Optimizations:")
        print("1. HBM3e Memory Optimizations")
        print("   • High-bandwidth memory (7.8 TB/s)")
        print("   • Optimized memory access patterns")
        print("   • Unified memory architecture")

        print("\n2. Tensor Core Optimizations")
        print("   • 5th Generation Tensor Cores")
        print("   • FP8/FP4 precision support")
        print("   • Enhanced matrix operations")

        print("\n3. TMA (Tensor Memory Accelerator)")
        print("   • Efficient data movement")
        print("   • Reduced memory latency")
        print("   • Optimized memory bandwidth")

        print("\n4. NVLink-C2C")
        print("   • Direct GPU-to-GPU communication")
        print("   • High-speed data transfer")
        print("   • Reduced communication overhead")

        print("\n5. Unified Memory")
        print("   • 30 TB total unified memory")
        print("   • Seamless CPU-GPU memory access")
        print("   • Optimized memory management")
    else:
        print("This GPU does not support Blackwell B200/B300 optimizations")


def _format_bandwidth(bandwidth_gbps: Any) -> str:
    if isinstance(bandwidth_gbps, (int, float)):
        return f"{bandwidth_gbps:.1f} GB/s"
    return "Unknown"


def main() -> None:
    """Main function to demonstrate hardware analysis."""
    print("=== Chapter 2: AI System Hardware Overview ===")
    print("Hardware Information and Benchmarking Demo")
    print("=" * 50)

    gpu_info = get_gpu_info()
    system_info = get_system_info()

    print("\n1. GPU Information:")
    print("-" * 20)
    if "error" not in gpu_info:
        print(f"GPU: {gpu_info['name']}")
        print(f"Compute Capability: {gpu_info['compute_capability']}")
        print(f"Memory: {gpu_info['total_memory_gb']:.1f} GB")
        print(f"Memory Bandwidth: {_format_bandwidth(gpu_info.get('memory_bandwidth_gbps'))}")
        print(f"Number of SMs: {gpu_info['num_sms']}")
        print(f"Max Threads per Block: {gpu_info['max_threads_per_block']}")
        print(f"Warp Size: {gpu_info['warp_size']}")
    else:
        print("CUDA not available")

    print("\n2. Blackwell B200/B300 Specific Information:")
    print("-" * 40)
    if "error" not in gpu_info:
        if gpu_info.get("architecture") == "Blackwell B200/B300":
            print(" This is a Blackwell B200/B300 GPU")
            print(f" Compute Capability: {gpu_info['compute_capability']} (SM100)")
            print(f" Memory: {gpu_info['total_memory_gb']:.1f} GB HBM3e")
            print(f" Memory Bandwidth: {gpu_info['memory_bandwidth_tbps']} TB/s")
            print(" 5th Generation Tensor Cores")
            print(" TMA (Tensor Memory Accelerator)")
            print(" NVLink-C2C (Direct GPU-to-GPU communication)")
            print(" Unified Memory Architecture")
            print(f" Max Unified Memory: {gpu_info['max_unified_memory_tb']} TB")
        else:
            print("This is not a Blackwell B200/B300 GPU")
            print(f"GPU: {gpu_info['name']}")
            print(f"Compute Capability: {gpu_info['compute_capability']}")
            print(f"Memory: {gpu_info['total_memory_gb']:.1f} GB")
            print(f"Memory Bandwidth: {_format_bandwidth(gpu_info.get('memory_bandwidth_gbps'))}")

    print("\n3. System Information:")
    print("-" * 20)
    print(f"CPU Cores: {system_info['cpu']['physical_cores']} physical, {system_info['cpu']['logical_cores']} logical")
    print(f"CPU Usage: {system_info['cpu']['overall_percent']:.1f}%")
    print(f"Memory: {system_info['memory']['total_gb']:.1f} GB total, {system_info['memory']['available_gb']:.1f} GB available")
    print(f"Memory Usage: {system_info['memory']['percent']:.1f}%")

    demonstrate_blackwell_features()
    benchmark_memory_bandwidth()
    benchmark_tensor_operations()
    demonstrate_memory_hierarchy()
    demonstrate_profiling_capabilities()
    demonstrate_system_monitoring()
    demonstrate_blackwell_optimizations()

    print("\n=== Summary ===")
    print("This demo shows hardware analysis capabilities:")
    print("1. GPU architecture detection and capabilities")
    print("2. Memory hierarchy analysis")
    print("3. Blackwell B200/B300 specific features")
    print("4. Performance benchmarking")
    print("5. System monitoring")
    print("6. Latest profiling tools integration")
    print("7. Memory bandwidth testing")
    print("8. Tensor operation benchmarking")
    print("9. Comprehensive system analysis")
    print("10. Blackwell-specific optimizations")


if __name__ == "__main__":
    main()

# Note: Architecture-specific optimizations are handled in arch_config.py
# The following configuration options are not available in the current PyTorch version:
# - torch._inductor.config.triton.use_hopper_optimizations
# - torch._inductor.config.triton.hbm3_optimizations
# - torch._inductor.config.triton.use_blackwell_optimizations
# - torch._inductor.config.triton.hbm3e_optimizations
# - torch._inductor.config.triton.tma_support
# - torch._inductor.config.triton.autotune_mode
# - torch._dynamo.config.automatic_dynamic_shapes
