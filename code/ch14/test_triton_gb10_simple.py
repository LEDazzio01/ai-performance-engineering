#!/usr/bin/env python3
"""
Test if Triton works on GB10 (SM 12.1) with simple kernels (no TMA)
"""

# CRITICAL: Import arch_config first to apply Triton SM 12.1 patch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import arch_config  # noqa: F401 - Patches Triton to support SM 12.1

import torch
import triton
import triton.language as tl

print("=" * 80)
print("Testing Triton on GB10 (SM 12.1) - Simple Kernels (No TMA)")
print("=" * 80)

# Check environment
props = torch.cuda.get_device_properties(0)
print(f"GPU:      {props.name}")
print(f"CC:       {props.major}.{props.minor}")
print(f"PyTorch:  {torch.__version__}")
print(f"Triton:   {triton.__version__}")
print()

# Test 1: Simple addition kernel
print("Test 1: Simple vector addition")
print("-" * 80)

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

try:
    size = 1024
    x = torch.randn(size, device='cuda')
    y = torch.randn(size, device='cuda')
    output = torch.empty(size, device='cuda')
    
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, size, BLOCK_SIZE=256)
    torch.cuda.synchronize()
    
    expected = x + y
    if torch.allclose(output, expected):
        print("✅ Simple addition kernel WORKS!")
    else:
        print(f"❌ Mismatch: max diff = {(output - expected).abs().max().item()}")
except Exception as e:
    print(f"❌ FAILED: {e}")

print()

# Test 2: Matrix multiplication (no TMA, just regular loads/stores)
print("Test 2: Simple matrix multiplication")
print("-" * 80)

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c = accumulator.to(tl.float16)
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

try:
    M, N, K = 512, 512, 512
    A = torch.randn((M, K), device='cuda', dtype=torch.float16)
    B = torch.randn((K, N), device='cuda', dtype=torch.float16)
    C = torch.empty((M, N), device='cuda', dtype=torch.float16)
    
    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_SIZE_M']),
            triton.cdiv(N, META['BLOCK_SIZE_N']),
        )
    
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
    )
    torch.cuda.synchronize()
    
    C_torch = torch.matmul(A, B)
    max_diff = (C - C_torch).abs().max().item()
    
    if max_diff < 0.1:  # FP16 has lower precision
        print(f"✅ Matrix multiplication WORKS! (max diff: {max_diff:.6f})")
    else:
        print(f"❌ Large mismatch: max diff = {max_diff}")
except Exception as e:
    print(f"❌ FAILED: {e}")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print("If both tests passed, Triton 3.5.0 + arch_config patch WORKS on GB10!")
print("TMA features are B200-specific and not available on GB10 (SM 12.1)")
print("=" * 80)

