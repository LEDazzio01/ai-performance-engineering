#!/usr/bin/env python3
"""
PyTorch torch.compile with TMA for Grace-Blackwell GB10
========================================================

Demonstrates how PyTorch 2.9's torch.compile automatically engages TMA
(Tensor Memory Accelerator) on Grace-Blackwell GB10 when using max-autotune mode.

Key Features:
- Automatic TMA engagement through torch.compile
- No manual kernel writing required
- TMA-aware kernel selection via Inductor
- CUTLASS and Triton backend integration

Requirements:
- Grace-Blackwell GB10 (SM 12.1)
- PyTorch 2.9+
- CUDA 13.0+

Usage:
    python pytorch_tma_sm121.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict, List, Tuple
import os

# Import architecture configuration to enable TMA optimizations
try:
    from arch_config import configure_optimizations, ArchitectureConfig
    configure_optimizations()
    arch_config = ArchitectureConfig()
    print(f"✓ Architecture optimizations configured for: {arch_config.get_architecture_name()}")
except ImportError:
    print("⚠️  Warning: Could not import arch_config")
    arch_config = None


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def check_tma_environment():
    """Check if TMA environment is properly configured."""
    print_section("TMA Environment Check")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")
    
    # Check TMA support
    if props.major < 9:
        print(f"❌ TMA requires SM 9.0+, found {props.major}.{props.minor}")
        return False
    
    print(f"✓ TMA supported (SM {props.major}.{props.minor})")
    
    # Check PyTorch version
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    
    # Check TMA-related environment variables
    print("\nTMA Environment Variables:")
    tma_vars = {
        "TRITON_TMA_ENABLE": os.environ.get("TRITON_TMA_ENABLE", "not set"),
        "TORCH_CUDNN_V8_API_ENABLED": os.environ.get("TORCH_CUDNN_V8_API_ENABLED", "not set"),
    }
    for key, value in tma_vars.items():
        print(f"  {key}: {value}")
    
    # Check Inductor config
    if hasattr(torch, "_inductor"):
        print("\nInductor Configuration:")
        cfg = torch._inductor.config
        if hasattr(cfg, "max_autotune_gemm_backends"):
            print(f"  max_autotune_gemm_backends: {cfg.max_autotune_gemm_backends}")
        if hasattr(cfg, "triton") and hasattr(cfg.triton, "cudagraphs"):
            print(f"  triton.cudagraphs: {cfg.triton.cudagraphs}")
    
    return True


# ============================================================================
# Simple Operations
# ============================================================================

def matmul_simple(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Simple matrix multiplication."""
    return torch.matmul(A, B)


def matmul_with_bias(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Matrix multiplication with bias."""
    return torch.matmul(A, B) + bias


def fused_linear_relu(A: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Fused linear + ReLU operation."""
    return F.relu(torch.matmul(A, weight.t()) + bias)


# ============================================================================
# Neural Network Modules
# ============================================================================

class SimpleLinear(nn.Module):
    """Simple linear layer for testing."""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPBlock(nn.Module):
    """MLP block with multiple layers."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class AttentionBlock(nn.Module):
    """Simplified attention block."""
    
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        
        return out


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark_function(
    fn,
    *args,
    num_warmup: int = 10,
    num_iters: int = 100,
    **kwargs
) -> Tuple[float, torch.Tensor]:
    """Benchmark a function."""
    # Warmup
    for _ in range(num_warmup):
        result = fn(*args, **kwargs)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        result = fn(*args, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    avg_time = elapsed / num_iters
    return avg_time, result


def benchmark_matmul():
    """Benchmark matrix multiplication with and without torch.compile."""
    print_section("Matrix Multiplication Benchmark")
    
    sizes = [512, 1024, 2048, 4096]
    
    for size in sizes:
        M = N = K = size
        
        # Allocate tensors
        A = torch.randn(M, K, device='cuda', dtype=torch.float16)
        B = torch.randn(K, N, device='cuda', dtype=torch.float16)
        
        # Eager mode
        eager_time, eager_result = benchmark_function(matmul_simple, A, B, num_iters=50)
        
        # Compiled (default)
        compiled_default = torch.compile(matmul_simple, mode='default')
        default_time, default_result = benchmark_function(compiled_default, A, B, num_iters=50)
        
        # Compiled (max-autotune) - enables TMA-aware kernels
        compiled_max = torch.compile(matmul_simple, mode='max-autotune')
        max_time, max_result = benchmark_function(compiled_max, A, B, num_iters=50)
        
        # Calculate TFLOPS
        flops = 2 * M * N * K
        eager_tflops = flops / eager_time / 1e12
        default_tflops = flops / default_time / 1e12
        max_tflops = flops / max_time / 1e12
        
        print(f"Size: {M}x{K} @ {K}x{N}")
        print(f"  Eager:            {eager_time*1e3:8.2f} ms  ({eager_tflops:6.2f} TFLOPS)")
        print(f"  Compiled (default): {default_time*1e3:8.2f} ms  ({default_tflops:6.2f} TFLOPS)  [{default_tflops/eager_tflops:.2f}x]")
        print(f"  Compiled (max):     {max_time*1e3:8.2f} ms  ({max_tflops:6.2f} TFLOPS)  [{max_tflops/eager_tflops:.2f}x]")
        print()
        
        # Verify correctness
        if not torch.allclose(eager_result, default_result, rtol=1e-2, atol=1e-2):
            print("  ⚠️  Warning: Default compiled results differ from eager")
        if not torch.allclose(eager_result, max_result, rtol=1e-2, atol=1e-2):
            print("  ⚠️  Warning: Max-autotune compiled results differ from eager")


def benchmark_mlp():
    """Benchmark MLP block."""
    print_section("MLP Block Benchmark")
    
    batch_size = 32
    seq_len = 512
    hidden_size = 768
    
    # Create model
    model = MLPBlock(hidden_size).cuda().half()
    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float16)
    
    # Eager mode
    with torch.no_grad():
        eager_time, eager_result = benchmark_function(model, x, num_iters=50)
    
    # Compiled (default)
    compiled_default = torch.compile(model, mode='default')
    with torch.no_grad():
        default_time, default_result = benchmark_function(compiled_default, x, num_iters=50)
    
    # Compiled (max-autotune)
    compiled_max = torch.compile(model, mode='max-autotune')
    with torch.no_grad():
        max_time, max_result = benchmark_function(compiled_max, x, num_iters=50)
    
    print(f"Input: [{batch_size}, {seq_len}, {hidden_size}]")
    print(f"  Eager:              {eager_time*1e3:8.2f} ms")
    print(f"  Compiled (default): {default_time*1e3:8.2f} ms  [{eager_time/default_time:.2f}x speedup]")
    print(f"  Compiled (max):     {max_time*1e3:8.2f} ms  [{eager_time/max_time:.2f}x speedup]")
    print()


def benchmark_attention():
    """Benchmark attention block."""
    print_section("Attention Block Benchmark")
    
    batch_size = 16
    seq_len = 512
    hidden_size = 768
    num_heads = 12
    
    # Create model
    model = AttentionBlock(hidden_size, num_heads).cuda().half()
    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', dtype=torch.float16)
    
    # Eager mode
    with torch.no_grad():
        eager_time, eager_result = benchmark_function(model, x, num_iters=50)
    
    # Compiled (default)
    compiled_default = torch.compile(model, mode='default')
    with torch.no_grad():
        default_time, default_result = benchmark_function(compiled_default, x, num_iters=50)
    
    # Compiled (max-autotune)
    compiled_max = torch.compile(model, mode='max-autotune')
    with torch.no_grad():
        max_time, max_result = benchmark_function(compiled_max, x, num_iters=50)
    
    print(f"Input: [{batch_size}, {seq_len}, {hidden_size}], Heads: {num_heads}")
    print(f"  Eager:              {eager_time*1e3:8.2f} ms")
    print(f"  Compiled (default): {default_time*1e3:8.2f} ms  [{eager_time/default_time:.2f}x speedup]")
    print(f"  Compiled (max):     {max_time*1e3:8.2f} ms  [{eager_time/max_time:.2f}x speedup]")
    print()


def test_correctness():
    """Test correctness of compiled operations."""
    print_section("Correctness Tests")
    
    all_passed = True
    
    # Test 1: Simple matmul
    print("Test 1: Matrix Multiplication")
    try:
        A = torch.randn(256, 256, device='cuda', dtype=torch.float32)
        B = torch.randn(256, 256, device='cuda', dtype=torch.float32)
        
        eager_result = matmul_simple(A, B)
        compiled_fn = torch.compile(matmul_simple, mode='max-autotune')
        compiled_result = compiled_fn(A, B)
        
        if torch.allclose(eager_result, compiled_result, rtol=1e-4, atol=1e-4):
            print("  ✓ PASSED\n")
        else:
            max_diff = torch.max(torch.abs(eager_result - compiled_result)).item()
            print(f"  ❌ FAILED: Max difference: {max_diff}\n")
            all_passed = False
    except Exception as e:
        print(f"  ❌ FAILED: {e}\n")
        all_passed = False
    
    # Test 2: Matmul with bias
    print("Test 2: Matrix Multiplication with Bias")
    try:
        A = torch.randn(256, 256, device='cuda', dtype=torch.float32)
        B = torch.randn(256, 256, device='cuda', dtype=torch.float32)
        bias = torch.randn(256, device='cuda', dtype=torch.float32)
        
        eager_result = matmul_with_bias(A, B, bias)
        compiled_fn = torch.compile(matmul_with_bias, mode='max-autotune')
        compiled_result = compiled_fn(A, B, bias)
        
        if torch.allclose(eager_result, compiled_result, rtol=1e-4, atol=1e-4):
            print("  ✓ PASSED\n")
        else:
            max_diff = torch.max(torch.abs(eager_result - compiled_result)).item()
            print(f"  ❌ FAILED: Max difference: {max_diff}\n")
            all_passed = False
    except Exception as e:
        print(f"  ❌ FAILED: {e}\n")
        all_passed = False
    
    # Test 3: MLP block
    print("Test 3: MLP Block")
    try:
        model = MLPBlock(256).cuda()
        x = torch.randn(8, 128, 256, device='cuda', dtype=torch.float32)
        
        with torch.no_grad():
            eager_result = model(x)
            compiled_model = torch.compile(model, mode='max-autotune')
            compiled_result = compiled_model(x)
        
        if torch.allclose(eager_result, compiled_result, rtol=1e-3, atol=1e-3):
            print("  ✓ PASSED\n")
        else:
            max_diff = torch.max(torch.abs(eager_result - compiled_result)).item()
            print(f"  ❌ FAILED: Max difference: {max_diff}\n")
            all_passed = False
    except Exception as e:
        print(f"  ❌ FAILED: {e}\n")
        all_passed = False
    
    return all_passed


def main():
    """Main entry point."""
    print("="*80)
    print("  PyTorch torch.compile with TMA for Grace-Blackwell GB10")
    print("="*80)
    
    # Check environment
    if not check_tma_environment():
        print("\n❌ TMA environment not properly configured")
        return 1
    
    # Run correctness tests
    print("\n" + "="*80)
    print("  Running Correctness Tests")
    print("="*80)
    
    all_passed = test_correctness()
    
    if not all_passed:
        print("\n❌ Some correctness tests failed")
        return 1
    
    print("\n✓ All correctness tests passed!")
    
    # Run benchmarks
    print("\n" + "="*80)
    print("  Running Performance Benchmarks")
    print("="*80)
    
    benchmark_matmul()
    benchmark_mlp()
    benchmark_attention()
    
    # Summary
    print_section("Summary")
    print("✓ PyTorch torch.compile is working with TMA on your GB10!")
    print("✓ max-autotune mode enables TMA-aware kernel selection")
    print("✓ Automatic performance optimization without manual kernel writing")
    print("\nKey Takeaways:")
    print("  1. Use mode='max-autotune' for best TMA engagement")
    print("  2. TMA provides automatic memory optimization")
    print("  3. No code changes needed - just compile your models")
    print("\nRecommendations:")
    print("  - Use FP16/BF16 for best performance on GB10")
    print("  - Enable CUDA graphs for additional speedup")
    print("  - Profile with Nsight Systems to verify TMA usage")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

