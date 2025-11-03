#!/usr/bin/env python3
"""
Quick GPT-style model test - NO HEAVY COMPILATION
Shows realistic torch.compile speedup on B200
"""
import arch_config  # noqa: F401 - Configure Blackwell optimizations

import torch
import torch.nn as nn
import time
import warnings
from tqdm import tqdm

# Suppress deprecated warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class SimpleGPTBlock(nn.Module):
    def __init__(self, d_model=4096, n_heads=32):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x

def benchmark_quick(model, x, name, num_iters=100):
    """Quick benchmark"""
    # Warmup
    for _ in range(20):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    avg_ms = (elapsed / num_iters) * 1000
    tokens_per_sec = (x.shape[0] * x.shape[1] * num_iters) / elapsed
    
    print(f"\n{name}:")
    print(f"  Time: {avg_ms:.2f} ms")
    print(f"  Throughput: {tokens_per_sec/1000:.1f}K tokens/sec")
    
    return avg_ms

def main():
    # NEW PyTorch 2.9 API (fixes warnings!)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    
    print("=" * 80)
    print("QUICK GPT TEST ON B200")
    print("=" * 80)
    
    # Test different sizes to find sweet spot
    configs = [
        (32, 4096, 8, 2048),    # ~10B params - minimum for B200
        (48, 5120, 4, 2048),    # ~15B params
        (64, 6144, 2, 2048),    # ~24B params - good B200 showcase
    ]
    
    for n_layers, d_model, batch, seq_len in configs:
        print(f"\n{'=' * 80}")
        print(f"Config: {n_layers} layers, d_model={d_model}, batch={batch}, seq={seq_len}")
        print(f"{'=' * 80}")
        
        # Create model
        blocks = [SimpleGPTBlock(d_model=d_model) for _ in range(n_layers)]
        model = nn.Sequential(*blocks).cuda().eval()
        
        params = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"Parameters: {params:.2f}B")
        
        # Input
        x = torch.randn(batch, seq_len, d_model, device='cuda')
        mem = x.numel() * 4 / 1e9
        print(f"Input size: {mem:.2f} GB")
        
        # Eager
        eager_time = benchmark_quick(model, x, "Eager Mode")
        
        # Compiled (reduce-overhead mode - faster compilation)
        print("\n[Compiling... this may take 30-60 seconds]")
        model_compiled = torch.compile(model, mode='reduce-overhead')
        print("[Compilation done, now benchmarking...]")
        compiled_time = benchmark_quick(model_compiled, x, "Compiled Mode")
        
        speedup = eager_time / compiled_time
        print(f"\n>>> Speedup: {speedup:.2f}x")
        
        if speedup > 1.2:
            print("✅ GOOD speedup!")
            break
        else:
            print("⚠️ Too small, trying larger...")
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

if __name__ == "__main__":
    main()
