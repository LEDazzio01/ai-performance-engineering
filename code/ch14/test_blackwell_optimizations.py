#!/usr/bin/env python3
"""
Test script to validate Blackwell optimizations maintain correctness.

This script verifies that all optimizations produce numerically correct results
compared to PyTorch baseline implementations.
"""
import arch_config  # noqa: F401 - Configure Blackwell optimizations

import torch
import sys

print("Testing Blackwell Optimizations...")
print("=" * 70)

# Check CUDA availability
if not torch.cuda.is_available():
    print("CUDA not available - skipping tests")
    sys.exit(0)

# Check device
props = torch.cuda.get_device_properties(0)
is_blackwell = props.major == 10 and props.minor == 0
print(f"GPU: {props.name}")
print(f"Compute Capability: {props.major}.{props.minor}")
print(f"Blackwell: {'YES' if is_blackwell else 'NO (tests will still run)'}")
print("=" * 70)

# Import optimized kernels
try:
    from triton_tma_blackwell import tma_copy_2d, tma_gemm
    from triton_examples import (
        tiled_matmul,
        persistent_matmul_descriptor,
        persistent_matmul_queue,
    )
    print("\n✓ Successfully imported optimized kernels")
except ImportError as e:
    print(f"\n✗ Failed to import kernels: {e}")
    sys.exit(1)

# Test parameters
test_sizes = [512, 1024, 2048]
dtype = torch.float16
tolerance = 1e-2

print(f"\nRunning correctness tests...")
print(f"Tolerance: {tolerance}")
print(f"Sizes: {test_sizes}")
print("=" * 70)

all_passed = True

# Test 1: TMA Copy
print("\n[1/5] Testing TMA Copy...")
for size in test_sizes:
    src = torch.randn(size, size, device='cuda', dtype=dtype)
    dst = torch.empty_like(src)
    
    tma_copy_2d(src, dst)
    
    max_diff = torch.abs(src - dst).max().item()
    passed = max_diff < tolerance
    all_passed = all_passed and passed
    
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  Size {size}x{size}: {status} (max_diff={max_diff:.2e})")

# Test 2: TMA GEMM
print("\n[2/5] Testing TMA GEMM...")
for size in test_sizes:
    A = torch.randn(size, size, device='cuda', dtype=dtype)
    B = torch.randn(size, size, device='cuda', dtype=dtype)
    
    C_tma = tma_gemm(A, B)
    C_torch = torch.matmul(A.float(), B.float())
    
    max_diff = torch.abs(C_tma - C_torch).max().item()
    passed = max_diff < tolerance
    all_passed = all_passed and passed
    
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  Size {size}x{size}: {status} (max_diff={max_diff:.2e})")

# Test 3: Tiled Matmul
print("\n[3/5] Testing Tiled Matmul...")
for size in test_sizes:
    A = torch.randn(size, size, device='cuda', dtype=dtype)
    B = torch.randn(size, size, device='cuda', dtype=dtype)
    
    C_triton = tiled_matmul(A, B)
    C_torch = torch.matmul(A.float(), B.float())
    
    max_diff = torch.abs(C_triton - C_torch).max().item()
    passed = max_diff < tolerance
    all_passed = all_passed and passed
    
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  Size {size}x{size}: {status} (max_diff={max_diff:.2e})")

# Test 4: Persistent Matmul (Descriptor)
print("\n[4/5] Testing Persistent Matmul (Descriptor)...")
for size in test_sizes:
    A = torch.randn(size, size, device='cuda', dtype=dtype)
    B = torch.randn(size, size, device='cuda', dtype=dtype)
    
    C_triton = persistent_matmul_descriptor(A, B)
    C_torch = torch.matmul(A.float(), B.float())
    
    max_diff = torch.abs(C_triton - C_torch).max().item()
    passed = max_diff < tolerance
    all_passed = all_passed and passed
    
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  Size {size}x{size}: {status} (max_diff={max_diff:.2e})")

# Test 5: Persistent Matmul (Queue)
print("\n[5/5] Testing Persistent Matmul (Queue)...")
for size in test_sizes:
    A = torch.randn(size, size, device='cuda', dtype=dtype)
    B = torch.randn(size, size, device='cuda', dtype=dtype)
    
    C_triton = persistent_matmul_queue(A, B)
    C_torch = torch.matmul(A.float(), B.float())
    
    max_diff = torch.abs(C_triton - C_torch).max().item()
    passed = max_diff < tolerance
    all_passed = all_passed and passed
    
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  Size {size}x{size}: {status} (max_diff={max_diff:.2e})")

# Summary
print("\n" + "=" * 70)
if all_passed:
    print("✓ ALL TESTS PASSED")
    print("\nAll Blackwell optimizations maintain numerical correctness!")
    if is_blackwell:
        print("Run full benchmarks to measure performance gains:")
        print("  python triton_tma_blackwell.py")
        print("  python triton_examples.py")
else:
    print("✗ SOME TESTS FAILED")
    print("\nPlease review the failed tests above.")
    sys.exit(1)

print("=" * 70)
