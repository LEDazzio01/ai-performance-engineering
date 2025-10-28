"""
GB200/GB300 Topology-Aware Tensor Placement
===========================================

Demonstrates NUMA-aware tensor placement and CPU-GPU affinity for
Grace-Blackwell superchips.

Grace CPU Architecture:
- 72 ARM Neoverse V2 cores (144 threads)
- LPDDR5X memory (480GB-1TB)
- Multiple NUMA nodes (typically 2-4)
- NVLink-C2C: 900 GB/s to each Blackwell GPU

Key Features:
1. Detect Grace CPU topology and NUMA configuration
2. CPU-GPU affinity binding for optimal data transfer
3. Automatic tensor placement strategies
4. Memory allocation hints for coherent access

Requirements:
- PyTorch 2.9+
- GB200/GB300 superchip (graceful fallback on other systems)
"""
import arch_config  # noqa: F401 - Configure Blackwell optimizations

import os
import platform
import subprocess
import torch
import psutil
from typing import Dict, List, Optional, Tuple


def detect_grace_cpu() -> Dict[str, any]:
    """
    Detect if running on Grace CPU (ARM Neoverse V2).
    
    Returns:
        Dictionary with Grace CPU information
    """
    info = {
        "is_grace": False,
        "architecture": platform.machine(),
        "num_cores": psutil.cpu_count(logical=False),
        "num_threads": psutil.cpu_count(logical=True),
        "numa_nodes": 0,
        "cpu_memory_gb": psutil.virtual_memory().total / (1024**3),
    }
    
    # Check if ARM
    if info["architecture"] not in ["aarch64", "arm64"]:
        return info
    
    # Check for Neoverse (Grace indicator)
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            if 'ARM' in cpuinfo or 'Neoverse' in cpuinfo:
                info["is_grace"] = True
    except:
        pass
    
    # Detect NUMA nodes
    try:
        result = subprocess.run(['numactl', '--hardware'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'available:' in line:
                    info["numa_nodes"] = int(line.split(':')[1].split()[0])
    except:
        info["numa_nodes"] = 1  # Assume 1 if detection fails
    
    return info


def detect_gb200_gb300_topology() -> Dict[str, any]:
    """
    Detect complete GB200/GB300 topology.
    
    Returns:
        Dictionary with system topology information
    """
    grace_info = detect_grace_cpu()
    
    topology = {
        "is_gb200_gb300": False,
        "grace_cpu": grace_info,
        "num_gpus": 0,
        "gpu_memory_per_device_gb": 0,
        "total_system_memory_gb": 0,
        "gpus": [],
        "numa_gpu_mapping": {},
    }
    
    if not torch.cuda.is_available():
        return topology
    
    topology["num_gpus"] = torch.cuda.device_count()
    
    # Check GPUs
    all_blackwell = True
    for i in range(topology["num_gpus"]):
        props = torch.cuda.get_device_properties(i)
        is_blackwell = (props.major == 10 and props.minor == 0)
        all_blackwell = all_blackwell and is_blackwell
        
        topology["gpus"].append({
            "id": i,
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "memory_gb": props.total_memory / (1024**3),
            "is_blackwell": is_blackwell,
        })
        
        if i == 0:
            topology["gpu_memory_per_device_gb"] = props.total_memory / (1024**3)
    
    # GB200/GB300 = Grace + Blackwell
    topology["is_gb200_gb300"] = grace_info["is_grace"] and all_blackwell
    
    # Calculate total system memory (CPU + GPU)
    topology["total_system_memory_gb"] = (
        grace_info["cpu_memory_gb"] + 
        topology["gpu_memory_per_device_gb"] * topology["num_gpus"]
    )
    
    # Detect NUMA-GPU affinity
    if grace_info["numa_nodes"] > 0:
        # On GB200/GB300, GPUs are typically distributed across NUMA nodes
        # For 8 GPUs and 2 NUMA nodes: GPUs 0-3 on NUMA 0, GPUs 4-7 on NUMA 1
        gpus_per_numa = topology["num_gpus"] // grace_info["numa_nodes"]
        for i in range(topology["num_gpus"]):
            numa_node = i // gpus_per_numa if gpus_per_numa > 0 else 0
            topology["numa_gpu_mapping"][i] = numa_node
    
    return topology


def print_topology_info(topology: Dict[str, any]) -> None:
    """Print detailed topology information."""
    print("=" * 80)
    print("GB200/GB300 System Topology")
    print("=" * 80)
    
    grace = topology["grace_cpu"]
    print(f"\nCPU Configuration:")
    print(f"  Architecture: {grace['architecture']}")
    print(f"  Is Grace CPU: {'✓ Yes' if grace['is_grace'] else '✗ No'}")
    print(f"  Physical Cores: {grace['num_cores']}")
    print(f"  Logical Threads: {grace['num_threads']}")
    print(f"  NUMA Nodes: {grace['numa_nodes']}")
    print(f"  CPU Memory: {grace['cpu_memory_gb']:.1f} GB")
    
    print(f"\nGPU Configuration:")
    print(f"  Number of GPUs: {topology['num_gpus']}")
    for gpu in topology["gpus"]:
        marker = " (Blackwell)" if gpu["is_blackwell"] else ""
        print(f"  GPU {gpu['id']}: {gpu['name']}{marker}")
        print(f"    Memory: {gpu['memory_gb']:.1f} GB")
    
    if topology["is_gb200_gb300"]:
        print(f"\n✓ GB200/GB300 Grace-Blackwell Superchip Detected!")
        print(f"  Total System Memory: {topology['total_system_memory_gb']:.1f} GB")
        print(f"  CPU Memory: {grace['cpu_memory_gb']:.1f} GB")
        print(f"  GPU Memory: {topology['gpu_memory_per_device_gb'] * topology['num_gpus']:.1f} GB")
        print(f"  NVLink-C2C: 900 GB/s per GPU")
        
        if topology["numa_gpu_mapping"]:
            print(f"\n  NUMA-GPU Mapping:")
            for gpu_id, numa_node in topology["numa_gpu_mapping"].items():
                print(f"    GPU {gpu_id} → NUMA Node {numa_node}")
    else:
        print(f"\n⚠ Not a GB200/GB300 system")
    
    print("=" * 80)


def set_cpu_affinity_for_gpu(gpu_id: int, topology: Dict[str, any]) -> bool:
    """
    Set CPU affinity for current process based on GPU assignment.
    
    On GB200/GB300, bind to the NUMA node closest to the GPU.
    
    Args:
        gpu_id: GPU device ID
        topology: System topology dict
        
    Returns:
        True if affinity was set successfully
    """
    if not topology["is_gb200_gb300"]:
        print(f"⚠ Not GB200/GB300, skipping CPU affinity")
        return False
    
    numa_node = topology["numa_gpu_mapping"].get(gpu_id, 0)
    
    try:
        # Get CPUs in this NUMA node
        result = subprocess.run(
            ['numactl', '--hardware'],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            return False
        
        # Parse NUMA node CPU list
        cpu_list = []
        for line in result.stdout.split('\n'):
            if f'node {numa_node} cpus:' in line:
                cpu_str = line.split(':', 1)[1].strip()
                # Parse CPU ranges (e.g., "0-17 36-53")
                for part in cpu_str.split():
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        cpu_list.extend(range(start, end + 1))
                    else:
                        cpu_list.append(int(part))
                break
        
        if cpu_list:
            # Set affinity
            p = psutil.Process()
            p.cpu_affinity(cpu_list)
            print(f"✓ Set CPU affinity for GPU {gpu_id}")
            print(f"  NUMA Node: {numa_node}")
            print(f"  CPUs: {cpu_list[:8]}{'...' if len(cpu_list) > 8 else ''}")
            return True
    except Exception as e:
        print(f"⚠ Failed to set CPU affinity: {e}")
    
    return False


def allocate_tensor_with_numa_hint(
    shape: Tuple[int, ...],
    device: str = "cuda:0",
    dtype: torch.dtype = torch.float32,
    on_cpu_memory: bool = False,
) -> torch.Tensor:
    """
    Allocate tensor with NUMA-aware placement.
    
    Args:
        shape: Tensor shape
        device: Target device
        dtype: Data type
        on_cpu_memory: If True, allocate on CPU memory with GPU access
        
    Returns:
        Allocated tensor
    """
    if on_cpu_memory:
        # Allocate on CPU with GPU access (unified memory)
        # This is optimal on GB200/GB300 with NVLink-C2C
        tensor = torch.empty(shape, dtype=dtype, device='cpu', pin_memory=True)
        print(f"  Allocated {tensor.numel() * tensor.element_size() / 1e6:.1f} MB on CPU (pinned)")
    else:
        # Standard GPU allocation
        tensor = torch.empty(shape, dtype=dtype, device=device)
        print(f"  Allocated {tensor.numel() * tensor.element_size() / 1e6:.1f} MB on {device}")
    
    return tensor


def demonstrate_hybrid_placement(topology: Dict[str, any]) -> None:
    """
    Demonstrate optimal tensor placement for GB200/GB300.
    """
    if not topology["is_gb200_gb300"]:
        print("⚠ Hybrid placement optimized for GB200/GB300")
        print("  Running in fallback mode")
    
    print("\n=== Hybrid CPU-GPU Tensor Placement ===\n")
    
    # Scenario 1: Large model training
    print("Scenario 1: Large Model Training (7B parameters)")
    print("-" * 60)
    
    # Parameters on GPU (frequently accessed)
    params = allocate_tensor_with_numa_hint(
        (7_000_000_000 // 1000,),  # Simplified
        device="cuda:0",
        dtype=torch.float16,
        on_cpu_memory=False
    )
    print("  ✓ Model parameters on GPU")
    
    # Optimizer states on CPU (less frequently accessed)
    optimizer_state_m = allocate_tensor_with_numa_hint(
        (7_000_000_000 // 1000,),
        device="cuda:0",
        dtype=torch.float32,
        on_cpu_memory=True  # On CPU with GPU access
    )
    print("  ✓ Optimizer momentum on CPU (via NVLink-C2C)")
    
    optimizer_state_v = allocate_tensor_with_numa_hint(
        (7_000_000_000 // 1000,),
        device="cuda:0",
        dtype=torch.float32,
        on_cpu_memory=True
    )
    print("  ✓ Optimizer variance on CPU (via NVLink-C2C)")
    
    total_gpu = params.numel() * params.element_size() / 1e9
    total_cpu = (optimizer_state_m.numel() * optimizer_state_m.element_size() +
                 optimizer_state_v.numel() * optimizer_state_v.element_size()) / 1e9
    
    print(f"\nMemory Distribution:")
    print(f"  GPU: {total_gpu:.2f} GB (parameters)")
    print(f"  CPU: {total_cpu:.2f} GB (optimizer states)")
    print(f"  Total: {total_gpu + total_cpu:.2f} GB")
    print(f"  GPU memory saved: {total_cpu:.2f} GB ({total_cpu / (total_gpu + total_cpu) * 100:.1f}%)")
    
    if topology["is_gb200_gb300"]:
        print(f"\n  NVLink-C2C enables ~800 GB/s CPU↔GPU transfers")
        print(f"  Optimizer update overhead: <5% vs GPU-only")
    
    # Cleanup
    del params, optimizer_state_m, optimizer_state_v
    
    # Scenario 2: Inference with large KV cache
    print("\n\nScenario 2: Inference with Large KV Cache")
    print("-" * 60)
    
    # Model on GPU
    print("  Model weights: GPU")
    
    # KV cache on CPU (can be much larger)
    print("  KV cache: CPU memory (via NVLink-C2C)")
    print("    → Store thousands of sequences")
    print("    → Access with ~800 GB/s bandwidth")
    print("    → Transparent to model code")


def demonstrate_multi_gpu_placement(topology: Dict[str, any]) -> None:
    """
    Demonstrate NUMA-aware placement for multi-GPU training.
    """
    if topology["num_gpus"] < 2:
        print("\nMulti-GPU placement requires 2+ GPUs")
        return
    
    print("\n=== NUMA-Aware Multi-GPU Placement ===\n")
    
    for gpu_id in range(min(topology["num_gpus"], 8)):
        numa_node = topology["numa_gpu_mapping"].get(gpu_id, 0)
        print(f"Process for GPU {gpu_id}:")
        print(f"  → Bind to NUMA node {numa_node}")
        print(f"  → Allocate CPU tensors from NUMA node {numa_node}")
        print(f"  → DataLoader workers use NUMA node {numa_node} CPUs")
        print(f"  → Optimal for: CPU preprocessing + GPU training pipeline")
        print()


def main():
    """Main demonstration."""
    print("=" * 80)
    print("GB200/GB300 Topology-Aware Tensor Placement")
    print("=" * 80)
    print()
    
    # Detect topology
    topology = detect_gb200_gb300_topology()
    print_topology_info(topology)
    
    # Demonstrate placement strategies
    demonstrate_hybrid_placement(topology)
    
    # Multi-GPU placement
    if topology["num_gpus"] >= 2:
        demonstrate_multi_gpu_placement(topology)
    
    # Best practices
    print("\n=== Best Practices for GB200/GB300 ===\n")
    print("1. Place frequently-accessed data on GPU memory")
    print("2. Place optimizer states and KV cache on CPU memory")
    print("3. Use pinned memory for CPU allocations (enables NVLink-C2C)")
    print("4. Bind processes to NUMA nodes matching their GPUs")
    print("5. Use torch.cuda.Stream for overlapping CPU↔GPU transfers")
    print()
    print("Expected Performance:")
    print("  - CPU→GPU: ~800 GB/s (NVLink-C2C)")
    print("  - GPU→GPU: ~800-900 GB/s (NVLink 5.0)")
    print("  - Memory capacity: 480GB-1TB CPU + 1.44TB GPU (8 GPUs)")
    print("=" * 80)


if __name__ == "__main__":
    main()
