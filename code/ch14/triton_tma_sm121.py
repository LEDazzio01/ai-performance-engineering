#!/usr/bin/env python3
"""
Triton TMA Kernels for Grace-Blackwell GB10
============================================

Demonstrates TMA (Tensor Memory Accelerator) usage in Triton 3.5 for GB10 (SM 12.1).

Key Features:
- TMA descriptor-based memory access
- Hardware-accelerated bulk transfers
- Optimized for Grace-Blackwell coherence fabric
- Conservative configurations to avoid Triton 3.5 compiler bugs

Requirements:
- Grace-Blackwell GB10 (SM 12.1)
- CUDA 13.0+
- PyTorch 2.9+
- Triton 3.5+

Usage:
    python triton_tma_sm121.py

Note: Due to Triton 3.5 compiler bugs with aggressive TMA configurations,
      this implementation uses conservative block sizes and pipeline stages.
      See ch14/triton_tma_blackwell.py for detailed bug documentation.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple
import time

# Import architecture configuration
try:
    from arch_config import configure_optimizations
    configure_optimizations()
except ImportError:
    print("⚠️  Warning: Could not import arch_config")


def check_sm121_support() -> Tuple[bool, str]:
    """Check if running on GB10 with TMA support."""
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    props = torch.cuda.get_device_properties(0)
    cc = f"{props.major}.{props.minor}"
    
    # GB10 is SM 12.1, but also support Hopper (9.0+) and Blackwell (10.0+)
    if props.major >= 12 or props.major >= 9:
        return True, f"TMA supported (SM {cc})"
    else:
        return False, f"TMA requires SM 9.0+, found SM {cc}"


# ============================================================================
# TMA Copy Kernel (1D)
# ============================================================================

@triton.jit
def tma_copy_1d_kernel(
    input_ptr, output_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simple 1D copy using TMA descriptors.
    
    This demonstrates the basic TMA pattern:
    1. Create tensor descriptor
    2. Load data using descriptor
    3. Store data (can also use descriptor for stores)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create TMA descriptor for input
    input_desc = tl.make_tensor_descriptor(
        input_ptr,
        shape=[N],
        strides=[1],
        block_shape=[BLOCK_SIZE],
    )
    
    # Load using TMA
    data = input_desc.load([block_start])
    
    # Store (using standard store for simplicity)
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    tl.store(output_ptr + offsets, data, mask=mask)


def tma_copy_1d(src: torch.Tensor, dst: torch.Tensor) -> None:
    """Copy tensor using TMA."""
    assert src.is_contiguous() and dst.is_contiguous()
    assert src.shape == dst.shape
    assert src.device == dst.device
    
    N = src.numel()
    BLOCK_SIZE = 256
    
    grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE),)
    tma_copy_1d_kernel[grid](src, dst, N, BLOCK_SIZE)


# ============================================================================
# TMA Vector Add Kernel
# ============================================================================

@triton.jit
def tma_vector_add_kernel(
    a_ptr, b_ptr, c_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Vector addition using TMA descriptors.
    
    Demonstrates TMA with computation:
    1. Load two vectors using TMA
    2. Perform element-wise addition
    3. Store result
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create TMA descriptors
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[N],
        strides=[1],
        block_shape=[BLOCK_SIZE],
    )
    
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N],
        strides=[1],
        block_shape=[BLOCK_SIZE],
    )
    
    # Load using TMA
    a = a_desc.load([block_start])
    b = b_desc.load([block_start])
    
    # Compute
    c = a + b
    
    # Store result
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    tl.store(c_ptr + offsets, c, mask=mask)


def tma_vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vector addition using TMA."""
    assert a.shape == b.shape
    assert a.is_contiguous() and b.is_contiguous()
    
    N = a.numel()
    BLOCK_SIZE = 256
    
    c = torch.empty_like(a)
    
    grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE),)
    tma_vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE)
    
    return c


# ============================================================================
# TMA GEMM Kernel (Conservative Configuration)
# ============================================================================

@triton.jit
def tma_gemm_conservative_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Conservative TMA GEMM kernel.
    
    Uses small block sizes to avoid Triton 3.5 compiler bug.
    See ch14/triton_tma_blackwell.py for detailed bug documentation.
    
    Configuration:
    - BLOCK_K=32 (instead of optimal 128+)
    - No autotune (to avoid compiler crash)
    - Single pipeline stage
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N
    
    # Create TMA descriptors
    A_desc = tl.make_tensor_descriptor(
        A_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    
    B_desc = tl.make_tensor_descriptor(
        B_ptr,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        block_shape=[BLOCK_K, BLOCK_N],
    )
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Main loop
    for k0 in range(0, K, BLOCK_K):
        # Load tiles using TMA
        a = A_desc.load([m0, k0])
        b = B_desc.load([k0, n0])
        
        # Compute
        acc += tl.dot(a, b, out_dtype=tl.float32)
    
    # Store result (no TMA descriptor to simplify)
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def tma_gemm_conservative(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Conservative TMA GEMM.
    
    Uses fixed configuration to avoid Triton 3.5 compiler bugs.
    """
    assert A.ndim == 2 and B.ndim == 2
    assert A.shape[1] == B.shape[0]
    
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    
    # Conservative configuration
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    tma_gemm_conservative_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    
    return C


# ============================================================================
# Benchmarking and Testing
# ============================================================================

def benchmark_tma_copy(sizes=[1024, 4096, 16384, 65536], num_iters=100):
    """Benchmark TMA copy vs standard copy."""
    print("\n" + "="*80)
    print("  TMA Copy Benchmark")
    print("="*80 + "\n")
    
    results = []
    
    for size in sizes:
        # Allocate tensors
        src = torch.randn(size, device='cuda', dtype=torch.float32)
        dst_tma = torch.zeros_like(src)
        dst_std = torch.zeros_like(src)
        
        # Warmup
        for _ in range(10):
            tma_copy_1d(src, dst_tma)
            dst_std.copy_(src)
        torch.cuda.synchronize()
        
        # Benchmark TMA
        start = time.perf_counter()
        for _ in range(num_iters):
            tma_copy_1d(src, dst_tma)
        torch.cuda.synchronize()
        tma_time = (time.perf_counter() - start) / num_iters
        
        # Benchmark standard
        start = time.perf_counter()
        for _ in range(num_iters):
            dst_std.copy_(src)
        torch.cuda.synchronize()
        std_time = (time.perf_counter() - start) / num_iters
        
        # Calculate bandwidth
        bytes_transferred = size * 4 * 2  # read + write
        tma_bw = bytes_transferred / tma_time / 1e9  # GB/s
        std_bw = bytes_transferred / std_time / 1e9  # GB/s
        
        speedup = std_time / tma_time
        
        results.append({
            'size': size,
            'tma_time': tma_time * 1e6,  # us
            'std_time': std_time * 1e6,  # us
            'tma_bw': tma_bw,
            'std_bw': std_bw,
            'speedup': speedup,
        })
        
        print(f"Size: {size:6d} elements")
        print(f"  TMA:      {tma_time*1e6:8.2f} us  ({tma_bw:6.2f} GB/s)")
        print(f"  Standard: {std_time*1e6:8.2f} us  ({std_bw:6.2f} GB/s)")
        print(f"  Speedup:  {speedup:6.2f}x")
        print()
        
        # Verify correctness
        if not torch.allclose(dst_tma, dst_std):
            print("  ⚠️  Warning: TMA results differ from standard copy!")
    
    return results


def benchmark_tma_gemm(sizes=[512, 1024, 2048], num_iters=10):
    """Benchmark TMA GEMM vs PyTorch."""
    print("\n" + "="*80)
    print("  TMA GEMM Benchmark")
    print("="*80 + "\n")
    
    results = []
    
    for size in sizes:
        M = N = K = size
        
        # Allocate tensors
        A = torch.randn(M, K, device='cuda', dtype=torch.float32)
        B = torch.randn(K, N, device='cuda', dtype=torch.float32)
        
        # Warmup
        for _ in range(3):
            C_tma = tma_gemm_conservative(A, B)
            C_torch = torch.matmul(A, B)
        torch.cuda.synchronize()
        
        # Benchmark TMA GEMM
        start = time.perf_counter()
        for _ in range(num_iters):
            C_tma = tma_gemm_conservative(A, B)
        torch.cuda.synchronize()
        tma_time = (time.perf_counter() - start) / num_iters
        
        # Benchmark PyTorch
        start = time.perf_counter()
        for _ in range(num_iters):
            C_torch = torch.matmul(A, B)
        torch.cuda.synchronize()
        torch_time = (time.perf_counter() - start) / num_iters
        
        # Calculate TFLOPS
        flops = 2 * M * N * K
        tma_tflops = flops / tma_time / 1e12
        torch_tflops = flops / torch_time / 1e12
        
        results.append({
            'size': size,
            'tma_time': tma_time * 1e3,  # ms
            'torch_time': torch_time * 1e3,  # ms
            'tma_tflops': tma_tflops,
            'torch_tflops': torch_tflops,
        })
        
        print(f"Size: {M}x{K} @ {K}x{N}")
        print(f"  TMA GEMM:     {tma_time*1e3:8.2f} ms  ({tma_tflops:6.2f} TFLOPS)")
        print(f"  PyTorch:      {torch_time*1e3:8.2f} ms  ({torch_tflops:6.2f} TFLOPS)")
        print(f"  Efficiency:   {(tma_tflops/torch_tflops)*100:6.2f}%")
        print()
        
        # Verify correctness
        if not torch.allclose(C_tma, C_torch, rtol=1e-3, atol=1e-3):
            max_diff = torch.max(torch.abs(C_tma - C_torch)).item()
            print(f"  ⚠️  Warning: Max difference: {max_diff}")
    
    return results


def test_tma_operations():
    """Test TMA operations for correctness."""
    print("\n" + "="*80)
    print("  TMA Operations Test")
    print("="*80 + "\n")
    
    all_passed = True
    
    # Test 1: TMA Copy
    print("Test 1: TMA Copy")
    try:
        N = 1024
        src = torch.randn(N, device='cuda', dtype=torch.float32)
        dst = torch.zeros_like(src)
        tma_copy_1d(src, dst)
        
        if torch.allclose(src, dst):
            print("  ✓ PASSED\n")
        else:
            print("  ❌ FAILED: Results don't match\n")
            all_passed = False
    except Exception as e:
        print(f"  ❌ FAILED: {e}\n")
        all_passed = False
    
    # Test 2: TMA Vector Add
    print("Test 2: TMA Vector Add")
    try:
        N = 1024
        a = torch.randn(N, device='cuda', dtype=torch.float32)
        b = torch.randn(N, device='cuda', dtype=torch.float32)
        c_tma = tma_vector_add(a, b)
        c_ref = a + b
        
        if torch.allclose(c_tma, c_ref):
            print("  ✓ PASSED\n")
        else:
            print("  ❌ FAILED: Results don't match\n")
            all_passed = False
    except Exception as e:
        print(f"  ❌ FAILED: {e}\n")
        all_passed = False
    
    # Test 3: TMA GEMM
    print("Test 3: TMA GEMM (Conservative)")
    try:
        M, N, K = 256, 256, 256
        A = torch.randn(M, K, device='cuda', dtype=torch.float32)
        B = torch.randn(K, N, device='cuda', dtype=torch.float32)
        C_tma = tma_gemm_conservative(A, B)
        C_ref = torch.matmul(A, B)
        
        if torch.allclose(C_tma, C_ref, rtol=1e-3, atol=1e-3):
            print("  ✓ PASSED\n")
        else:
            max_diff = torch.max(torch.abs(C_tma - C_ref)).item()
            print(f"  ❌ FAILED: Max difference: {max_diff}\n")
            all_passed = False
    except Exception as e:
        print(f"  ❌ FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        all_passed = False
    
    return all_passed


def main():
    """Main entry point."""
    print("="*80)
    print("  Triton TMA for Grace-Blackwell GB10")
    print("="*80)
    
    # Check support
    supported, msg = check_sm121_support()
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"TMA Support: {msg}")
    
    if not supported:
        print("\n❌ TMA not supported on this device")
        return 1
    
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton: {triton.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    
    # Run tests
    print("\n" + "="*80)
    print("  Running TMA Tests")
    print("="*80)
    
    all_passed = test_tma_operations()
    
    if all_passed:
        print("\n✓ All tests passed!")
        
        # Run benchmarks
        print("\n" + "="*80)
        print("  Running Benchmarks")
        print("="*80)
        
        benchmark_tma_copy(sizes=[1024, 4096, 16384, 65536], num_iters=100)
        benchmark_tma_gemm(sizes=[512, 1024, 2048], num_iters=10)
        
        print("\n" + "="*80)
        print("  Summary")
        print("="*80)
        print("\n✓ TMA is working on your Grace-Blackwell GB10!")
        print("✓ Hardware-accelerated bulk transfers engaged")
        print("✓ Descriptor-based memory access operational")
        print("\nNote: Using conservative configurations due to Triton 3.5 compiler bug.")
        print("      See ch14/triton_tma_blackwell.py for details.")
        
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

