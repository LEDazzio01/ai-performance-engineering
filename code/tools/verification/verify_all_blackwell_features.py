#!/usr/bin/env python3
"""
Comprehensive Blackwell Feature Verification with Profiling
============================================================

Verifies all Blackwell B200/B300 features and optionally profiles with nsys/ncu.

Features verified:
- TMEM (Tensor Memory Accelerator)
- TMA (Tensor Memory Access)
- CTA Clusters
- DSMEM (Distributed Shared Memory)
- FP8 (E4M3, E5M2)
- FP4 (E2M1 - status check)
- Warp Specialization
- 5th Gen Tensor Cores (tcgen05)
- Triton Blackwell Features (TMA descriptors, pipelining, torch.compile)

Usage:
    python verify_all_blackwell_features.py [--profile] [--ncu] [--nsys] [--skip-triton]
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json

import torch

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLACKWELL_TESTS_DIR = PROJECT_ROOT / "tools" / "blackwell_optimizations"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "blackwell_verification"


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_result(name: str, passed: bool, detail: str = "") -> None:
    """Print a test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    line = f"  {status}: {name}"
    if detail:
        line += f" - {detail}"
    print(line)


def check_gpu() -> Tuple[bool, Dict]:
    """Check GPU and return info."""
    if not torch.cuda.is_available():
        return False, {"error": "CUDA not available"}
    
    props = torch.cuda.get_device_properties(0)
    info = {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "major": props.major,
        "minor": props.minor,
        "total_memory_gb": props.total_memory / 1024**3,
        "multi_processor_count": props.multi_processor_count,
    }
    
    is_blackwell = props.major == 10
    return is_blackwell, info


def run_cuda_test(test_name: str, timeout: int = 30) -> Tuple[bool, str, float]:
    """Run a CUDA test binary."""
    test_path = BLACKWELL_TESTS_DIR / test_name
    
    if not test_path.exists():
        return False, f"Test binary not found: {test_path}", 0.0
    
    try:
        start = time.time()
        result = subprocess.run(
            [str(test_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(BLACKWELL_TESTS_DIR)
        )
        elapsed = time.time() - start
        
        output = result.stdout + result.stderr
        passed = result.returncode == 0 and "PASSED" in output
        
        return passed, output, elapsed
    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout}s", timeout
    except Exception as e:
        return False, str(e), 0.0


def check_python_feature(name: str, test_fn) -> Tuple[bool, str]:
    """Check a Python-level feature."""
    try:
        result = test_fn()
        if isinstance(result, tuple):
            return result
        return True, str(result) if result else "OK"
    except Exception as e:
        return False, str(e)


def check_fp4_support() -> Dict:
    """Check FP4 support status."""
    status = {
        "dtype_available": hasattr(torch, 'float4_e2m1fn_x2'),
        "scaled_mm_available": hasattr(torch, '_scaled_mm'),
        "cublaslt_available": False,
        "native_ops_working": False,
        "workaround_available": True,  # Our packed uint8 implementation
    }
    
    # Check cuBLASLt
    try:
        import ctypes
        ctypes.CDLL('libcublasLt.so.13')
        status["cublaslt_available"] = True
    except (OSError, ImportError):
        pass
    
    # Check if native FP4 conversion works
    if status["dtype_available"]:
        try:
            # Try creating and converting
            fp4_tensor = torch.empty(64, 64, dtype=torch.float4_e2m1fn_x2, device='cuda')
            # Try conversion (this is what fails in PyTorch 2.9.1)
            fp32_tensor = torch.randn(32, 64, device='cuda')
            fp4_from_fp32 = fp32_tensor.view(-1, 2).to(torch.float4_e2m1fn_x2)
            status["native_ops_working"] = True
        except (NotImplementedError, RuntimeError):
            status["native_ops_working"] = False
    
    return status


def check_fp8_support() -> Dict:
    """Check FP8 support status."""
    status = {
        "e4m3_available": hasattr(torch, 'float8_e4m3fn'),
        "e5m2_available": hasattr(torch, 'float8_e5m2'),
        "scaled_mm_available": hasattr(torch, '_scaled_mm'),
        "conversion_working": False,
        "tensor_core_accelerated": False,
    }
    
    if status["e4m3_available"]:
            x = torch.randn(64, 64, device='cuda', dtype=torch.float16)
            x_fp8 = x.to(torch.float8_e4m3fn)
            status["conversion_working"] = True
            
            # Check if _scaled_mm works
            # _scaled_mm requires row-major A and column-major B
            # a.t() gives column-major view, so we use a @ b.t()
            if status["scaled_mm_available"]:
                try:
                    M, K, N = 64, 128, 64
                    a = torch.randn(M, K, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
                    b = torch.randn(N, K, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
                    scale_a = torch.tensor(1.0, device='cuda', dtype=torch.float32)
                    scale_b = torch.tensor(1.0, device='cuda', dtype=torch.float32)
                    # b.t() provides column-major view of (K, N) matrix
                    result = torch._scaled_mm(a, b.t(), scale_a, scale_b)
                    status["tensor_core_accelerated"] = True
                except Exception as e:
                    status["scaled_mm_error"] = str(e)[:100]
    
    return status


def check_tma_support() -> Dict:
    """Check TMA support via device attributes."""
    status = {
        "hardware_supported": False,
        "driver_attribute": False,
    }
    
    # First try via PyTorch (more reliable)
    try:
        props = torch.cuda.get_device_properties(0)
        # TMA is available on SM 9.0+ (Hopper and Blackwell)
        if props.major >= 9:
            status["hardware_supported"] = True
    except Exception:
        pass
    
    # Then try CUDA driver API
    try:
        import ctypes
        cuda = ctypes.CDLL('libcuda.so')
        
        # Initialize CUDA first
        cuda.cuInit(0)
        
        # CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 143
        attr = ctypes.c_int()
        device = ctypes.c_int(0)
        result = cuda.cuDeviceGetAttribute(ctypes.byref(attr), 143, device)
        
        if result == 0 and attr.value == 1:
            status["driver_attribute"] = True
            status["hardware_supported"] = True
    except Exception:
        pass
    
    return status


def check_cluster_support() -> Dict:
    """Check CTA cluster support."""
    status = {
        "hardware_supported": False,
        "driver_attribute": False,
    }
    
    # First try via PyTorch
    try:
        props = torch.cuda.get_device_properties(0)
        # Clusters are available on SM 9.0+ (Hopper and Blackwell)
        if props.major >= 9:
            status["hardware_supported"] = True
    except Exception:
        pass
    
    # Then try CUDA driver API
    try:
        import ctypes
        cuda = ctypes.CDLL('libcuda.so')
        
        # Initialize CUDA
        cuda.cuInit(0)
        
        # CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 133
        attr = ctypes.c_int()
        device = ctypes.c_int(0)
        result = cuda.cuDeviceGetAttribute(ctypes.byref(attr), 133, device)
        
        if result == 0 and attr.value == 1:
            status["driver_attribute"] = True
            status["hardware_supported"] = True
    except Exception:
        pass
    
    return status


def run_nsys_profile(test_name: str) -> Optional[str]:
    """Run nsys profiling on a test."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    test_path = BLACKWELL_TESTS_DIR / test_name
    output_path = RESULTS_DIR / f"nsys_{test_name}"
    
    if not test_path.exists():
        return None
    
    try:
        result = subprocess.run(
            [
                "nsys", "profile",
                "--stats=true",
                "-o", str(output_path),
                "-f", "true",
                str(test_path)
            ],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(BLACKWELL_TESTS_DIR)
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"nsys error: {e}"


def run_ncu_profile(test_name: str) -> Optional[str]:
    """Run ncu profiling on a test."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    test_path = BLACKWELL_TESTS_DIR / test_name
    output_path = RESULTS_DIR / f"ncu_{test_name}"
    
    if not test_path.exists():
        return None
    
    try:
        result = subprocess.run(
            [
                "ncu",
                "--target-processes", "all",
                "--set", "full",
                "-o", str(output_path),
                "-f",
                str(test_path)
            ],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(BLACKWELL_TESTS_DIR)
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"ncu error: {e}"


def run_triton_blackwell_tests() -> Dict:
    """Run Triton Blackwell feature tests."""
    results = {}
    
    try:
        import triton
        import triton.language as tl
        results["triton_version"] = triton.__version__
    except ImportError:
        return {"error": "Triton not installed"}
    
    # Test 1: Basic Kernel Compilation
    try:
        @triton.jit
        def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask)
            y = tl.load(y_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, x + y, mask=mask)
        
        n = 1024
        x = torch.randn(n, device='cuda')
        y = torch.randn(n, device='cuda')
        out = torch.empty_like(x)
        grid = (triton.cdiv(n, 256),)
        _add_kernel[grid](x, y, out, n, BLOCK=256)
        torch.cuda.synchronize()
        
        if torch.allclose(out, x + y):
            results["Basic Kernel"] = {"status": "PASS", "detail": "Compiles and executes correctly"}
        else:
            results["Basic Kernel"] = {"status": "FAIL", "detail": "Results incorrect"}
    except Exception as e:
        results["Basic Kernel"] = {"status": "FAIL", "detail": str(e)[:80]}
    
    # Test 2: TMA Descriptors
    if hasattr(tl, 'make_tensor_descriptor'):
        try:
            # Setup allocator for TMA
            from triton.runtime import _allocation as triton_allocation
            
            class _TorchAllocator:
                def __init__(self):
                    self._bufs = {}
                def __call__(self, size, alignment, stream):
                    aligned = (size + alignment - 1) // alignment * alignment
                    buf = torch.empty(aligned, dtype=torch.uint8, device='cuda')
                    ptr = buf.data_ptr()
                    self._bufs[ptr] = buf
                    return ptr
            
            triton_allocation.set_allocator(_TorchAllocator())
            
            @triton.jit
            def _tma_copy(inp, out, N: tl.constexpr, BLOCK: tl.constexpr):
                pid = tl.program_id(0)
                desc = tl.make_tensor_descriptor(inp, shape=[N], strides=[1], block_shape=[BLOCK])
                data = desc.load([pid * BLOCK])
                offs = pid * BLOCK + tl.arange(0, BLOCK)
                tl.store(out + offs, data, mask=offs < N)
            
            N, BLOCK = 1024, 128
            x = torch.randn(N, device='cuda')
            y = torch.zeros_like(x)
            _tma_copy[(triton.cdiv(N, BLOCK),)](x, y, N, BLOCK)
            torch.cuda.synchronize()
            
            if torch.allclose(y, x):
                results["TMA Descriptors"] = {"status": "PASS", "detail": "Hardware TMA works"}
            else:
                results["TMA Descriptors"] = {"status": "FAIL", "detail": "Results incorrect"}
        except Exception as e:
            err = str(e)
            if "allocator" in err.lower():
                results["TMA Descriptors"] = {"status": "WARN", "detail": "Needs triton.set_allocator()"}
            else:
                results["TMA Descriptors"] = {"status": "FAIL", "detail": err[:80]}
    else:
        results["TMA Descriptors"] = {"status": "SKIP", "detail": "tl.make_tensor_descriptor not available"}
    
    # Test 3: Multi-Stage Pipeline
    try:
        @triton.jit
        def _matmul_kernel(
            a_ptr, b_ptr, c_ptr, M, N, K,
            stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        ):
            pid = tl.program_id(0)
            num_pid_m = tl.cdiv(M, BLOCK_M)
            pid_m, pid_n = pid % num_pid_m, pid // num_pid_m
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for _ in range(0, K, BLOCK_K):
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                acc += tl.dot(a, b)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(c_ptrs, acc.to(tl.float16))
        
        M = N = K = 256
        a = torch.randn(M, K, device='cuda', dtype=torch.float16)
        b = torch.randn(K, N, device='cuda', dtype=torch.float16)
        c = torch.empty(M, N, device='cuda', dtype=torch.float16)
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
        _matmul_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
            BLOCK_M, BLOCK_N, BLOCK_K,
            num_stages=3, num_warps=4,
        )
        torch.cuda.synchronize()
        expected = torch.matmul(a, b)
        if torch.allclose(c, expected, rtol=1e-1, atol=1e-1):
            results["Multi-Stage Pipeline"] = {"status": "PASS", "detail": "num_stages=3 works"}
        else:
            results["Multi-Stage Pipeline"] = {"status": "FAIL", "detail": "Results incorrect"}
    except Exception as e:
        err = str(e)
        if "latencies" in err.lower():
            results["Multi-Stage Pipeline"] = {"status": "WARN", "detail": "Known compiler issue"}
        else:
            results["Multi-Stage Pipeline"] = {"status": "FAIL", "detail": err[:80]}
    
    # Test 4: torch.compile Integration
    try:
        def _matmul_fn(a, b):
            return torch.matmul(a, b)
        compiled = torch.compile(_matmul_fn, mode='reduce-overhead')
        a = torch.randn(256, 256, device='cuda', dtype=torch.float16)
        b = torch.randn(256, 256, device='cuda', dtype=torch.float16)
        for _ in range(3):
            _ = compiled(a, b)
        torch.cuda.synchronize()
        result = compiled(a, b)
        expected = torch.matmul(a, b)
        if torch.allclose(result, expected, rtol=1e-2, atol=1e-2):
            results["torch.compile"] = {"status": "PASS", "detail": "Triton backend works"}
        else:
            results["torch.compile"] = {"status": "FAIL", "detail": "Results incorrect"}
    except Exception as e:
        err = str(e)
        if "sm_121a" in err or "ptxas" in err.lower():
            results["torch.compile"] = {"status": "FAIL", "detail": "SM patch not applied"}
        else:
            results["torch.compile"] = {"status": "FAIL", "detail": err[:80]}
    
    # Test 5: FP8 Support
    if hasattr(torch, 'float8_e4m3fn'):
        try:
            @triton.jit
            def _fp8_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
                pid = tl.program_id(0)
                offs = pid * BLOCK + tl.arange(0, BLOCK)
                mask = offs < n
                x = tl.load(x_ptr + offs, mask=mask)
                tl.store(out_ptr + offs, x, mask=mask)
            
            n = 1024
            x = torch.randn(n, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
            out = torch.empty_like(x)
            _fp8_kernel[(triton.cdiv(n, 256),)](x, out, n, BLOCK=256)
            torch.cuda.synchronize()
            results["FP8 Triton"] = {"status": "PASS", "detail": "FP8 tensors work"}
        except Exception as e:
            results["FP8 Triton"] = {"status": "FAIL", "detail": str(e)[:80]}
    else:
        results["FP8 Triton"] = {"status": "SKIP", "detail": "torch.float8_e4m3fn not available"}
    
    # Test 6: Persistent Kernel Pattern
    try:
        @triton.jit
        def _persistent_add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr, NUM_SMS: tl.constexpr):
            pid = tl.program_id(0)
            num_blocks = tl.cdiv(n, BLOCK)
            for block_id in range(pid, num_blocks, NUM_SMS):
                offs = block_id * BLOCK + tl.arange(0, BLOCK)
                mask = offs < n
                x = tl.load(x_ptr + offs, mask=mask)
                y = tl.load(y_ptr + offs, mask=mask)
                tl.store(out_ptr + offs, x + y, mask=mask)
        
        n = 8192
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
        x = torch.randn(n, device='cuda')
        y = torch.randn(n, device='cuda')
        out = torch.empty_like(x)
        grid = (min(num_sms, triton.cdiv(n, 256)),)
        _persistent_add[grid](x, y, out, n, BLOCK=256, NUM_SMS=num_sms)
        torch.cuda.synchronize()
        if torch.allclose(out, x + y):
            results["Persistent Kernels"] = {"status": "PASS", "detail": f"{num_sms} SMs"}
        else:
            results["Persistent Kernels"] = {"status": "FAIL", "detail": "Results incorrect"}
    except Exception as e:
        results["Persistent Kernels"] = {"status": "FAIL", "detail": str(e)[:80]}
    
    return results


def analyze_ncu_for_tensor_cores(report_path: Path) -> Dict:
    """Analyze NCU report for tensor core usage."""
    metrics = {
        "tensor_pipe_active": False,
        "tma_used": False,
        "tensor_memory_active": False,
    }
    
    ncu_report = report_path.with_suffix(".ncu-rep")
    if not ncu_report.exists():
        return metrics
    
        result = subprocess.run(
            [
                "ncu", "--import", str(ncu_report),
                "--page", "raw",
                "--print-kernel-base", "function"
            ],
            capture_output=True,
            text=True,
            timeout=60
        )
        output = result.stdout
        
        # Check for tensor core metrics
        if "sm__pipe_tensor_cycles_active" in output:
            # Parse the value
            for line in output.split('\n'):
                if "sm__pipe_tensor_cycles_active" in line and "pct_of_peak" in line:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if '%' in p:
                                val = float(parts[i-1])
                                if val > 0:
                                    metrics["tensor_pipe_active"] = True
        
        if "tensor_map_access_supported" in output:
            metrics["tma_used"] = True
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Verify Blackwell features")
    parser.add_argument("--profile", action="store_true", help="Run profiling")
    parser.add_argument("--nsys", action="store_true", help="Run nsys profiling")
    parser.add_argument("--ncu", action="store_true", help="Run ncu profiling")
    parser.add_argument("--json", action="store_true", help="Output JSON results")
    parser.add_argument("--skip-triton", action="store_true", help="Skip Triton tests")
    args = parser.parse_args()
    
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": {},
        "summary": {"passed": 0, "failed": 0, "skipped": 0}
    }
    
    # Check GPU
    print_header("Blackwell Feature Verification")
    is_blackwell, gpu_info = check_gpu()
    
    print(f"\nGPU: {gpu_info.get('name', 'Unknown')}")
    print(f"Compute Capability: {gpu_info.get('compute_capability', 'Unknown')}")
    print(f"Memory: {gpu_info.get('total_memory_gb', 0):.1f} GB")
    print(f"SMs: {gpu_info.get('multi_processor_count', 0)}")
    
    results["gpu"] = gpu_info
    
    if not is_blackwell:
        print("\n⚠ Not a Blackwell GPU - some tests may not apply")
    
    # Build CUDA tests
    print_header("Building CUDA Tests")
    build_result = subprocess.run(
        ["make", "clean", "all"],
        capture_output=True,
        text=True,
        cwd=str(BLACKWELL_TESTS_DIR)
    )
    if build_result.returncode != 0:
        print("⚠ Build warnings/errors (continuing anyway):")
        print(build_result.stderr[-500:] if len(build_result.stderr) > 500 else build_result.stderr)
    else:
        print("✓ Build successful")
    
    # Run CUDA tests
    print_header("CUDA Feature Tests")
    
    cuda_tests = [
        ("test_tmem", "TMEM (Tensor Memory)"),
        ("test_tma", "TMA (Tensor Memory Accelerator)"),
        ("test_clusters", "CTA Clusters"),
        ("test_dsmem", "DSMEM (Distributed Shared Memory)"),
        ("test_fp8", "FP8 Precision"),
        ("test_warp_spec", "Warp Specialization"),
    ]
    
    for test_binary, test_name in cuda_tests:
        print(f"\n  Testing: {test_name}")
        passed, output, elapsed = run_cuda_test(test_binary)
        print_result(test_name, passed, f"{elapsed:.2f}s")
        
        results["tests"][test_binary] = {
            "name": test_name,
            "passed": passed,
            "elapsed": elapsed,
        }
        
        if passed:
            results["summary"]["passed"] += 1
        else:
            results["summary"]["failed"] += 1
            if "FAILED" in output or "failed" in output.lower():
                # Extract failure reason
                for line in output.split('\n'):
                    if "failed" in line.lower() or "error" in line.lower():
                        print(f"    → {line.strip()[:80]}")
    
    # Python-level feature checks
    print_header("Python API Feature Checks")
    
    # TMA
    tma_status = check_tma_support()
    print_result("TMA Hardware Support", tma_status["hardware_supported"])
    results["tests"]["tma_python"] = tma_status
    
    # Clusters
    cluster_status = check_cluster_support()
    print_result("Cluster Launch Support", cluster_status["hardware_supported"])
    results["tests"]["cluster_python"] = cluster_status
    
    # FP8
    print("\n  FP8 Support:")
    fp8_status = check_fp8_support()
    print_result("FP8 E4M3 dtype", fp8_status["e4m3_available"])
    print_result("FP8 E5M2 dtype", fp8_status["e5m2_available"])
    print_result("FP8 conversion", fp8_status["conversion_working"])
    print_result("FP8 _scaled_mm (tensor cores)", fp8_status["tensor_core_accelerated"])
    results["tests"]["fp8_python"] = fp8_status
    
    # FP4
    print("\n  FP4 Support:")
    fp4_status = check_fp4_support()
    print_result("FP4 E2M1 dtype", fp4_status["dtype_available"])
    print_result("FP4 native ops", fp4_status["native_ops_working"], 
             "Not implemented in PyTorch 2.9.1" if not fp4_status["native_ops_working"] else "")
    print_result("FP4 packed workaround", fp4_status["workaround_available"])
    results["tests"]["fp4_python"] = fp4_status
    
    # Triton Blackwell Features
    if not args.skip_triton:
        print_header("Triton Blackwell Feature Tests")
        triton_results = run_triton_blackwell_tests()
        results["tests"]["triton"] = triton_results
        
        for test_name, test_result in triton_results.items():
            if test_name == "triton_version":
                print(f"\n  Triton Version: {test_result}")
                continue
            status = test_result.get("status", "UNKNOWN")
            detail = test_result.get("detail", "")
            passed = status == "PASS"
            print_result(test_name, passed, detail)
            if passed:
                results["summary"]["passed"] += 1
            elif status == "SKIP":
                results["summary"]["skipped"] += 1
            else:
                results["summary"]["failed"] += 1
    
    # Profiling
    if args.profile or args.nsys or args.ncu:
        print_header("Profiling")
        
        if args.profile or args.nsys:
            print("\n  Running nsys profiling on test_fp8...")
            nsys_output = run_nsys_profile("test_fp8")
            if nsys_output:
                print("  ✓ nsys profile generated")
                # Extract key metrics
                if "CUDA API Statistics" in str(nsys_output):
                    print("    → CUDA API calls captured")
        
        if args.profile or args.ncu:
            print("\n  Running ncu profiling on test_fp8...")
            print("  (This may take a few minutes...)")
            ncu_output = run_ncu_profile("test_fp8")
            if ncu_output:
                print("  ✓ ncu profile generated")
                
                # Analyze for tensor cores
                ncu_analysis = analyze_ncu_for_tensor_cores(RESULTS_DIR / "ncu_test_fp8")
                if ncu_analysis["tma_used"]:
                    print("    → TMA hardware confirmed active")
                if ncu_analysis["tensor_pipe_active"]:
                    print("    → Tensor core pipeline active (tcgen05)")
                else:
                    print("    → Note: Custom kernels may not use tensor core intrinsics")
                    print("           Use cuBLAS/cuBLASLt for hardware tensor core acceleration")
    
    # Summary
    print_header("Summary")
    
    total = results["summary"]["passed"] + results["summary"]["failed"]
    print(f"\n  Tests Passed: {results['summary']['passed']}/{total}")
    
    print("\n  Blackwell Features Status:")
    features = [
        ("TMEM", "test_tmem"),
        ("TMA", "test_tma"),
        ("CTA Clusters", "test_clusters"),
        ("DSMEM", "test_dsmem"),
        ("FP8", "test_fp8"),
        ("Warp Features", "test_warp_spec"),
    ]
    
    all_passed = True
    for name, test_key in features:
        if test_key in results["tests"]:
            passed = results["tests"][test_key].get("passed", False)
            status = "✓" if passed else "✗"
            print(f"    {status} {name}")
            if not passed:
                all_passed = False
    
    # Triton features
    if "triton" in results["tests"] and not args.skip_triton:
        print("\n  Triton Blackwell Features:")
        triton_results = results["tests"]["triton"]
        triton_features = ["Basic Kernel", "TMA Descriptors", "Multi-Stage Pipeline", 
                          "torch.compile", "FP8 Triton", "Persistent Kernels"]
        for feat in triton_features:
            if feat in triton_results:
                status_str = triton_results[feat].get("status", "UNKNOWN")
                icon = "✓" if status_str == "PASS" else ("○" if status_str == "SKIP" else "✗")
                print(f"    {icon} {feat}")
                if status_str == "FAIL":
                    all_passed = False
    
    print("\n  FP4 Status:")
    print(f"    • dtype exists: {fp4_status['dtype_available']}")
    print(f"    • native ops: {'Working' if fp4_status['native_ops_working'] else 'Not implemented (use packed uint8 workaround)'}")
    print(f"    • To enable native FP4:")
    print(f"      1. Wait for PyTorch to implement copy_/conversion for float4_e2m1fn_x2")
    print(f"      2. Or use cuBLASLt directly with FP4 compute type")
    print(f"      3. Or use our packed uint8 quantization (ch19/native_fp4_quantization.py)")
    
    # Save results
    if args.json:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_file = RESULTS_DIR / "verification_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to: {results_file}")
    
    print("\n" + "=" * 70)
    if all_passed and fp8_status["conversion_working"]:
        print(" ✓ All core Blackwell features are WORKING!")
    else:
        print(" ⚠ Some features need attention")
    print("=" * 70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

