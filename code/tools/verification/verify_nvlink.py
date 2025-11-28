#!/usr/bin/env python3
"""
Verify NVLink connectivity and configuration on multi-GPU systems.
Tests NVLink topology, bandwidth, and NCCL configuration.
"""

import subprocess
import sys
import re
import os
from pathlib import Path

try:
    import torch
    import torch.distributed as dist
except ImportError:
    print("ERROR: PyTorch not installed. Please install PyTorch first.")
    sys.exit(1)


def print_section(title):
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print('='*80)


def ensure_nccl_defaults() -> None:
    """Populate recommended NCCL envs if they are not already set."""
    defaults = {
        "NCCL_P2P_LEVEL": "NVL",
        "NCCL_P2P_DISABLE": "0",
        "NCCL_IB_DISABLE": "1",
        "NCCL_SHM_DISABLE": "0",
        "NCCL_NVLS_ENABLE": "1",
        "NCCL_SOCKET_IFNAME": "lo",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def check_nvlink_topology():
    """Check NVLink topology using nvidia-smi."""
    print_section("NVLink Topology Check")
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            check=True
        )
        
        output = result.stdout
        
        # Check for NVLink connections
        if "NV" not in output:
            print("ERROR: No NVLink connections found!")
            print("   System appears to be using PCIe only.")
            return False
        
        # Count NVLink connections
        nv_matches = re.findall(r'NV(\d+)', output)
        if nv_matches:
            nv_counts = {}
            for match in nv_matches:
                count = int(match)
                nv_counts[count] = nv_counts.get(count, 0) + 1
            
            print("[OK] NVLink connections detected:")
            for nv_count, occurrences in sorted(nv_counts.items()):
                print(f"   NV{nv_count}: {occurrences} connections")
                bandwidth_per_link = 50  # GB/s for NVLink 4.0/5.0
                total_bw = nv_count * bandwidth_per_link
                print(f"   → {nv_count} links @ {bandwidth_per_link} GB/s = {total_bw} GB/s per GPU")
            
            # Check for full mesh (NV18 for 8 GPUs)
            if 18 in nv_counts:
                print("\n🏆 FULL NV18 MESH DETECTED!")
                print("   This is the BEST possible configuration for 8x B200 GPUs")
                print("   Total bandwidth: 900 GB/s per GPU")
            
            return True
        else:
            print("WARNING: NVLink detected but unable to parse connection count")
            return True
            
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Error running nvidia-smi: {e}")
        return False
    except FileNotFoundError:
        print("ERROR: nvidia-smi not found. Is NVIDIA driver installed?")
        return False


def check_nvlink_status():
    """Check NVLink status and link speeds."""
    print_section("NVLink Status Check")
    
    try:
        result = subprocess.run(
            ["nvidia-smi", "nvlink", "--status"],
            capture_output=True,
            text=True,
            check=True
        )
        
        output = result.stdout
        
        # Count active links per GPU
        gpu_links = {}
        current_gpu = None
        
        for line in output.split('\n'):
            if 'GPU' in line and 'NVIDIA' in line:
                match = re.search(r'GPU (\d+):', line)
                if match:
                    current_gpu = int(match.group(1))
                    gpu_links[current_gpu] = 0
            elif 'Link' in line and 'GB/s' in line and current_gpu is not None:
                gpu_links[current_gpu] += 1
        
        if gpu_links:
            print("[OK] Active NVLink connections per GPU:")
            for gpu, link_count in sorted(gpu_links.items()):
                print(f"   GPU {gpu}: {link_count} active links @ 50 GB/s each")
            
            avg_links = sum(gpu_links.values()) / len(gpu_links)
            print(f"\n   Average: {avg_links:.1f} links per GPU")
            
            if avg_links >= 18:
                print("   🏆 Full mesh configuration confirmed!")
            elif avg_links >= 12:
                print("   [OK] Good connectivity (partial mesh)")
            else:
                print("   WARNING: Limited NVLink connectivity")
            
            return True
        else:
            print("WARNING: Unable to parse NVLink status")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Error checking NVLink status: {e}")
        return False


def check_nvls_support():
    """Check for NVLS (NVLink Switch) support."""
    print_section("NVLS Multicast Support")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"Found {gpu_count} CUDA devices")
    
    # NVLS is supported on Hopper and newer
    has_nvls = False
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        compute_cap = f"{props.major}.{props.minor}"
        
        # Hopper (9.0) and Blackwell (10.0) support NVLS
        if props.major >= 9:
            has_nvls = True
            print(f"[OK] GPU {i} ({props.name}): Compute {compute_cap} - NVLS supported")
        else:
            print(f"WARNING: GPU {i} ({props.name}): Compute {compute_cap} - NVLS not supported")
    
    if has_nvls:
        print("\n[OK] NVLS multicast support available")
        print("   Enables optimized collective operations")
    
    return has_nvls


def test_p2p_access():
    """Test P2P access between GPUs."""
    print_section("P2P Access Test")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return False
    
    gpu_count = torch.cuda.device_count()
    
    if gpu_count < 2:
        print("WARNING: Less than 2 GPUs, skipping P2P test")
        return True
    
    print(f"Testing P2P access between {gpu_count} GPUs...")
    
    p2p_matrix = []
    for i in range(gpu_count):
        row = []
        for j in range(gpu_count):
            if i == j:
                row.append('-')
            else:
                can_access = torch.cuda.can_device_access_peer(i, j)
                row.append('' if can_access else '✗')
        p2p_matrix.append(row)
    
    # Print matrix
    print("\nP2P Access Matrix:")
    print("     " + "  ".join(f"GPU{i}" for i in range(gpu_count)))
    for i, row in enumerate(p2p_matrix):
        print(f"GPU{i}  " + "   ".join(row))
    
    # Check if all pairs can access
    all_accessible = all(
        cell == '' or cell == '-' 
        for row in p2p_matrix 
        for cell in row
    )
    
    if all_accessible:
        print("\n[OK] All GPU pairs have P2P access enabled")
        print("   NVLink communication fully functional")
        return True
    else:
        print("\nERROR: Some GPU pairs cannot access each other via P2P")
        print("   Check NCCL configuration and ACS settings")
        return False


def check_nccl_config():
    """Check NCCL environment configuration."""
    print_section("NCCL Configuration Check")
    
    import os
    
    config_items = [
        ("NCCL_P2P_LEVEL", "NVL", "Force NVLink usage"),
        ("NCCL_P2P_DISABLE", "0", "Enable P2P (0=enabled, 1=disabled)"),
        ("NCCL_IB_DISABLE", "1", "Disable InfiniBand"),
        ("NCCL_SHM_DISABLE", "0", "Enable shared memory"),
        ("NCCL_NVLS_ENABLE", "1", "Enable NVLS multicast"),
    ]
    
    all_good = True
    
    for env_var, recommended, description in config_items:
        current = os.environ.get(env_var, "not set")
        
        if current == recommended:
            print(f"[OK] {env_var}={current}")
            print(f"   {description}")
        elif current == "not set":
            print(f"WARNING: {env_var} not set (recommended: {recommended})")
            print(f"   {description}")
            all_good = False
        else:
            print(f"ERROR: {env_var}={current} (recommended: {recommended})")
            print(f"   {description}")
            all_good = False
    
    if all_good:
        print("\n[OK] NCCL configuration optimal for NVLink")
    else:
        print("\nWARNING: NCCL configuration could be optimized")
        print("\nRecommended configuration:")
        print("export NCCL_P2P_LEVEL=NVL")
        print("export NCCL_P2P_DISABLE=0")
        print("export NCCL_IB_DISABLE=1")
        print("export NCCL_SHM_DISABLE=0")
        print("export NCCL_NVLS_ENABLE=1")
    
    return all_good


def quick_bandwidth_test():
    """Quick bandwidth test between GPU pairs."""
    print_section("Quick Bandwidth Test")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return False
    
    gpu_count = torch.cuda.device_count()
    
    if gpu_count < 2:
        print("WARNING: Less than 2 GPUs, skipping bandwidth test")
        return True
    
    # Test first pair only for quick verification
    size_mb = 256
    size = size_mb * 1024 * 1024 // 4  # float32
    
    print(f"Testing bandwidth between GPU 0 and GPU 1 ({size_mb} MB)...")
    
    try:
        # Create tensors
        with torch.cuda.device(0):
            src = torch.randn(size, device='cuda:0')
        
        with torch.cuda.device(1):
            dst = torch.empty(size, device='cuda:1')
        
        # Warmup
        for _ in range(3):
            dst.copy_(src)
        torch.cuda.synchronize()
        
        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        iterations = 10
        start.record()
        for _ in range(iterations):
            dst.copy_(src)
        end.record()
        end.synchronize()
        
        elapsed_ms = start.elapsed_time(end) / iterations
        bandwidth_gbs = (size_mb / 1024) / (elapsed_ms / 1000)
        
        print(f"\n[OK] Measured bandwidth: {bandwidth_gbs:.2f} GB/s")
        
        if bandwidth_gbs > 200:
            print("   🏆 Excellent! NVLink is working properly")
        elif bandwidth_gbs > 100:
            print("   [OK] Good bandwidth, NVLink appears functional")
        elif bandwidth_gbs > 20:
            print("   WARNING: Moderate bandwidth, may be using PCIe")
        else:
            print("   ERROR: Low bandwidth, check NVLink configuration")
        
        return bandwidth_gbs > 100
        
    except Exception as e:
        print(f"ERROR: Error during bandwidth test: {e}")
        return False


def main():
    """Run all NVLink verification checks."""
    print("\n" + "="*80)
    print("  NVLink Verification Suite for Multi-GPU Systems")
    print("="*80)
    ensure_nccl_defaults()

    results = {}
    
    # Run all checks
    results['topology'] = check_nvlink_topology()
    results['status'] = check_nvlink_status()
    results['nvls'] = check_nvls_support()
    results['p2p'] = test_p2p_access()
    results['nccl'] = check_nccl_config()
    results['bandwidth'] = quick_bandwidth_test()
    
    # Summary
    print_section("Verification Summary")
    
    checks = [
        ('NVLink Topology', results['topology']),
        ('NVLink Status', results['status']),
        ('NVLS Support', results['nvls']),
        ('P2P Access', results['p2p']),
        ('NCCL Config', results['nccl']),
        ('Bandwidth Test', results['bandwidth']),
    ]
    
    for check_name, passed in checks:
        status = "[OK] PASS" if passed else "ERROR: FAIL"
        print(f"{status}  {check_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All NVLink verification checks PASSED!")
        print("   Your system has optimal NVLink configuration.")
        return 0
    else:
        print("\nWARNING: Some checks failed or need attention.")
        print("   Review the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
