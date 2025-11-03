#!/usr/bin/env python3
"""
Test if PyTorch's torch.compile with Triton backend can use TMA on GB10
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import arch_config  # Patches Triton for SM 12.1

import torch
import torch._dynamo
import torch._inductor.config as inductor_config

print("=" * 80)
print("Testing PyTorch torch.compile with Triton backend on GB10 (SM 12.1)")
print("=" * 80)

props = torch.cuda.get_device_properties(0)
print(f"GPU:      {props.name}")
print(f"CC:       {props.major}.{props.minor}")
print(f"PyTorch:  {torch.__version__}")
print()

# Check inductor Triton config
print("PyTorch Inductor Triton Config:")
print(f"  triton.cudagraph_trees: {inductor_config.triton.cudagraph_trees}")
print(f"  triton.dense_indexing: {inductor_config.triton.dense_indexing}")
print(f"  triton.max_tiles: {inductor_config.triton.max_tiles}")
print()

# Try a simple matmul with torch.compile
print("Test 1: Simple matmul with torch.compile + Triton backend")
print("-" * 80)

def matmul(x, y):
    return torch.matmul(x, y)

try:
    # Compile with Triton backend
    compiled_matmul = torch.compile(matmul, backend="inductor", mode="max-autotune")
    
    # Test
    x = torch.randn(512, 512, device='cuda', dtype=torch.float16)
    y = torch.randn(512, 512, device='cuda', dtype=torch.float16)
    
    # First run compiles
    result = compiled_matmul(x, y)
    torch.cuda.synchronize()
    
    # Check correctness
    expected = torch.matmul(x, y)
    if torch.allclose(result, expected, rtol=1e-3, atol=1e-3):
        print("✅ torch.compile with Triton backend WORKS!")
        print(f"   Result shape: {result.shape}")
    else:
        max_diff = (result - expected).abs().max().item()
        print(f"⚠️  Result mismatch: max diff = {max_diff}")
        
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

print()

# Try with TMA-specific hint (if available)
print("Test 2: Check if PyTorch has TMA-specific settings")
print("-" * 80)

# Check for TMA-related configs
tma_attrs = [attr for attr in dir(inductor_config) if 'tma' in attr.lower()]
if tma_attrs:
    print(f"Found TMA-related settings: {tma_attrs}")
    for attr in tma_attrs:
        print(f"  {attr}: {getattr(inductor_config, attr)}")
else:
    print("No TMA-specific settings found in PyTorch inductor config")

print()

# Check triton config
print("Triton-specific inductor config:")
triton_config = inductor_config.triton
for attr in dir(triton_config):
    if not attr.startswith('_'):
        print(f"  {attr}: {getattr(triton_config, attr)}")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print("PyTorch uses the same Triton 3.5.0 as standalone.")
print("torch.compile generates Triton code through the same backend.")
print("If TMA doesn't work in standalone Triton, it won't work in PyTorch either.")
print("=" * 80)

