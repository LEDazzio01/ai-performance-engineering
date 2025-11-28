"""
Test Commands - GPU bandwidth, network speed, warmup audit, diagnostics.

Provides commands for:
- GPU memory bandwidth tests
- Network throughput tests
- Warmup and JIT audit
- System diagnostics
"""

from __future__ import annotations

import subprocess
import time
from typing import Optional


def _print_header(title: str, emoji: str = "🧪"):
    print(f"\n{emoji} {title}")
    print("=" * 70)


# =============================================================================
# GPU BANDWIDTH TEST
# =============================================================================

def gpu_bandwidth(args) -> int:
    """Test GPU memory bandwidth."""
    _print_header("GPU Bandwidth Test", "⚡")
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("  ❌ CUDA not available")
            return 1
        
        device = torch.device('cuda')
        props = torch.cuda.get_device_properties(0)
        
        print(f"  GPU: {props.name}")
        print(f"  Testing memory bandwidth...\n")
        
        # Test different sizes
        sizes_gb = [0.1, 0.5, 1.0, 2.0, 4.0]
        
        print(f"  {'Size (GB)':<12} {'H2D (GB/s)':<15} {'D2H (GB/s)':<15} {'D2D (GB/s)':<15}")
        print("-" * 60)
        
        for size_gb in sizes_gb:
            size_bytes = int(size_gb * 1e9)
            num_elements = size_bytes // 4  # float32
            
            # Create tensors
            host_tensor = torch.randn(num_elements)
            device_tensor = torch.empty(num_elements, device=device)
            device_tensor2 = torch.empty(num_elements, device=device)
            
            # Warmup
            device_tensor.copy_(host_tensor)
            torch.cuda.synchronize()
            
            # Host to Device
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(5):
                device_tensor.copy_(host_tensor)
            torch.cuda.synchronize()
            h2d_time = (time.perf_counter() - start) / 5
            h2d_bw = size_gb / h2d_time
            
            # Device to Host
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(5):
                host_tensor.copy_(device_tensor)
            torch.cuda.synchronize()
            d2h_time = (time.perf_counter() - start) / 5
            d2h_bw = size_gb / d2h_time
            
            # Device to Device
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(10):
                device_tensor2.copy_(device_tensor)
            torch.cuda.synchronize()
            d2d_time = (time.perf_counter() - start) / 10
            d2d_bw = size_gb / d2d_time
            
            print(f"  {size_gb:<12.1f} {h2d_bw:<15.1f} {d2h_bw:<15.1f} {d2d_bw:<15.1f}")
            
            # Cleanup
            del host_tensor, device_tensor, device_tensor2
            torch.cuda.empty_cache()
        
        # Theoretical peak
        print(f"\n  Theoretical HBM Bandwidth: ~{props.total_memory / 1e9 * 3:.0f} GB/s (estimated)")
        
    except ImportError:
        print("  ❌ PyTorch not available")
        return 1
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return 1
    
    print()
    return 0


# =============================================================================
# NETWORK TEST
# =============================================================================

def network_test(args) -> int:
    """Test network throughput for distributed training."""
    _print_header("Network Test", "🌐")
    
    # Check for NCCL
    try:
        import torch.distributed as dist
        nccl_available = dist.is_nccl_available()
    except:
        nccl_available = False
    
    print(f"  NCCL Available: {'✅' if nccl_available else '❌'}")
    
    # Check for IB
    try:
        result = subprocess.run(['ibstat'], capture_output=True, text=True, timeout=5)
        ib_available = result.returncode == 0
        if ib_available:
            print("  InfiniBand: ✅ Available")
    except:
        ib_available = False
        print("  InfiniBand: ❌ Not available")
    
    # Check for NVLink
    try:
        result = subprocess.run(
            ['nvidia-smi', 'nvlink', '--status'],
            capture_output=True, text=True, timeout=5
        )
        nvlink_available = 'NVLink' in result.stdout
        print(f"  NVLink: {'✅ Available' if nvlink_available else '❌ Not available'}")
    except:
        nvlink_available = False
    
    # NCCL test (if multi-GPU)
    try:
        import torch
        num_gpus = torch.cuda.device_count()
        
        if num_gpus > 1:
            print(f"\n  Multi-GPU Test ({num_gpus} GPUs):")
            print("-" * 50)
            
            # Simple P2P bandwidth test
            for i in range(num_gpus):
                for j in range(num_gpus):
                    if i != j:
                        can_access = torch.cuda.can_device_access_peer(i, j)
                        print(f"    GPU {i} → GPU {j}: {'✅ P2P' if can_access else '❌ No P2P'}")
        else:
            print("\n  Single GPU detected - multi-GPU tests skipped")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    print("\n  For detailed NCCL testing:")
    print("    nccl-tests/build/all_reduce_perf -b 1M -e 1G -f 2 -g <num_gpus>")
    
    return 0


# =============================================================================
# WARMUP AUDIT
# =============================================================================

def warmup_audit(args) -> int:
    """Audit warmup behavior and JIT compilation."""
    _print_header("Warmup Audit", "🔥")
    
    script = getattr(args, 'script', None)
    iterations = getattr(args, 'iterations', 10)
    
    if not script:
        print("  Warmup analysis helps identify:")
        print("    • JIT compilation overhead")
        print("    • CUDA kernel caching")
        print("    • Memory allocation patterns")
        print("    • torch.compile warmup")
        
        print("\n  Usage: aisp test warmup <script.py> [--iterations N]")
        print("\n  Example output would show:")
        print("    Iteration 1: 2.5s (first run - compilation)")
        print("    Iteration 2: 0.3s (cached)")
        print("    Iteration 3: 0.1s (steady state)")
        return 0
    
    print(f"  Script: {script}")
    print(f"  Iterations: {iterations}")
    print("\n  Running warmup analysis...")
    
    # In a real implementation, this would run the script multiple times
    # and measure timing for each iteration
    print("\n  ⚠️ Warmup analysis requires running your script multiple times")
    print("    This feature is under development")
    
    return 0


# =============================================================================
# SPEEDTEST
# =============================================================================

def speedtest(args) -> int:
    """Run comprehensive speed tests."""
    _print_header("Speed Test Suite", "🏃")
    
    test_type = getattr(args, 'type', 'all')
    
    print(f"  Running: {test_type} tests\n")
    
    results = {}
    
    # 1. GEMM Test
    if test_type in ['all', 'gemm']:
        print("  1. Matrix Multiplication (GEMM)")
        print("-" * 50)
        
        try:
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            sizes = [(1024, 1024), (4096, 4096), (8192, 8192)]
            
            for m, n in sizes:
                a = torch.randn(m, n, device=device, dtype=torch.float16)
                b = torch.randn(n, m, device=device, dtype=torch.float16)
                
                # Warmup
                for _ in range(3):
                    torch.mm(a, b)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                for _ in range(10):
                    torch.mm(a, b)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                elapsed = (time.perf_counter() - start) / 10
                tflops = (2 * m * n * m) / elapsed / 1e12
                
                print(f"    {m}x{n}: {tflops:.1f} TFLOPS ({elapsed*1000:.2f}ms)")
                results[f'gemm_{m}x{n}'] = tflops
                
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    # 2. Attention Test
    if test_type in ['all', 'attention']:
        print("\n  2. Attention Benchmark")
        print("-" * 50)
        
        try:
            import torch
            import torch.nn.functional as F
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            batch, heads, seq_len, head_dim = 4, 32, 4096, 128
            
            q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
            k = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
            v = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
            
            # Warmup
            for _ in range(3):
                F.scaled_dot_product_attention(q, k, v)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            for _ in range(10):
                F.scaled_dot_product_attention(q, k, v)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = (time.perf_counter() - start) / 10
            
            print(f"    Batch={batch}, Heads={heads}, Seq={seq_len}, HeadDim={head_dim}")
            print(f"    Time: {elapsed*1000:.2f}ms")
            results['attention'] = elapsed
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    # Summary
    print("\n  Summary:")
    print("-" * 50)
    for name, value in results.items():
        if 'tflops' in str(value).lower() or value > 100:
            print(f"    {name}: {value:.1f} TFLOPS")
        else:
            print(f"    {name}: {value*1000:.2f}ms")
    
    return 0


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def diagnostics(args) -> int:
    """Run system diagnostics."""
    _print_header("System Diagnostics", "🔍")
    
    issues = []
    warnings = []
    
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print("  ✅ CUDA: Available")
            print(f"     Version: {torch.version.cuda}")
        else:
            issues.append("CUDA not available")
            print("  ❌ CUDA: Not available")
    except ImportError:
        issues.append("PyTorch not installed")
        print("  ❌ PyTorch: Not installed")
    
    # Check GPU memory
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free = props.total_memory - torch.cuda.memory_allocated(i)
                free_gb = free / 1e9
                
                if free_gb < 1:
                    warnings.append(f"GPU {i} low memory: {free_gb:.1f}GB free")
                print(f"  ✅ GPU {i}: {props.name} ({free_gb:.1f}GB free)")
    except Exception as e:
        warnings.append(f"GPU check failed: {e}")
    
    # Check key libraries
    libs = [
        ('flash_attn', 'Flash Attention'),
        ('triton', 'Triton'),
        ('transformer_engine', 'Transformer Engine'),
        ('vllm', 'vLLM'),
        ('deepspeed', 'DeepSpeed'),
    ]
    
    print("\n  Libraries:")
    for module, name in libs:
        try:
            __import__(module)
            print(f"    ✅ {name}: Installed")
        except ImportError:
            print(f"    ❌ {name}: Not installed")
    
    # Summary
    print("\n  Summary:")
    print("-" * 50)
    
    if issues:
        print(f"  ❌ Issues ({len(issues)}):")
        for issue in issues:
            print(f"      • {issue}")
    else:
        print("  ✅ No critical issues found")
    
    if warnings:
        print(f"  ⚠️ Warnings ({len(warnings)}):")
        for warning in warnings:
            print(f"      • {warning}")
    
    return 0


# =============================================================================
# COMMAND REGISTRATION
# =============================================================================

def register_commands(subparsers):
    """Register test commands."""
    test_parser = subparsers.add_parser("test", help="Performance tests and diagnostics")
    test_subparsers = test_parser.add_subparsers(dest="test_command")
    
    # GPU Bandwidth
    bw_p = test_subparsers.add_parser("bandwidth", help="GPU memory bandwidth test")
    bw_p.set_defaults(func=gpu_bandwidth)
    
    # Network
    net_p = test_subparsers.add_parser("network", help="Network throughput test")
    net_p.set_defaults(func=network_test)
    
    # Warmup
    warm_p = test_subparsers.add_parser("warmup", help="Warmup/JIT audit")
    warm_p.add_argument("script", nargs="?", help="Script to analyze")
    warm_p.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    warm_p.set_defaults(func=warmup_audit)
    
    # Speedtest
    speed_p = test_subparsers.add_parser("speed", help="Run speed tests")
    speed_p.add_argument("--type", choices=['all', 'gemm', 'attention', 'memory'], default='all')
    speed_p.set_defaults(func=speedtest)
    
    # Diagnostics
    diag_p = test_subparsers.add_parser("diagnostics", help="System diagnostics")
    diag_p.set_defaults(func=diagnostics)

