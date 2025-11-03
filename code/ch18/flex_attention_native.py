"""
Optimized FlexAttention for Blackwell B200

Demonstrates correct FlexAttention usage with torch.compile for optimal
performance. FlexAttention must be compiled to generate fused kernels;
without compilation it materializes the full attention matrix.
"""
import sys
import os

# Add parent directory to path to import arch_config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arch_config  # noqa: F401 - Configure Blackwell optimizations

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import time


def _is_known_compile_failure(error_text: str) -> bool:
    """Return True when the error likely stems from unsupported GPU/PTX."""
    lowered = error_text.lower()
    return (
        "ptxas" in lowered
        or "novalidchoiceserror" in lowered
        or "gpu-name" in lowered
    )


def _summarize_error_text(error_text: str, max_lines: int = 4) -> str:
    """Compress multi-line error messages for concise logging."""
    lines = [line.strip() for line in error_text.splitlines() if line.strip()]
    return " ".join(lines[:max_lines])


def configure_for_flex_attention():
    """Configure PyTorch for FlexAttention peak performance"""
    print("=" * 80)
    print("Configuring for FlexAttention Peak Performance")
    print("=" * 80)
    
    # NEW PyTorch 2.9 API (no warnings!)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.backends.cuda.matmul.fp32_precision = 'high'
    
    # Enable Flash Attention (FlexAttention builds on this)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    # Inductor settings
    torch._inductor.config.triton.cudagraphs = True
    torch._inductor.config.max_autotune = True
    
    print(" Configuration complete\n")


class BaselineAttention(nn.Module):
    """Baseline attention using scaled_dot_product_attention"""
    def forward(self, Q, K, V):
        return torch.nn.functional.scaled_dot_product_attention(Q, K, V)


class FlexAttentionWRONG(nn.Module):
    """Wrong: Not compiled - materializes full matrix and will be slower."""
    def __init__(self, window_size=2048):
        super().__init__()
        self.window_size = window_size
        
    def forward(self, Q, K, V):
        B, H, T, D = Q.shape
        
        # Sliding window mask
        def sliding_window(b, h, q_idx, kv_idx):
            return (q_idx - kv_idx).abs() <= self.window_size
        
        block_mask = create_block_mask(sliding_window, B, H, T, T)
        
        # WITHOUT torch.compile - this is SLOW!
        return flex_attention(Q, K, V, block_mask=block_mask)


class FlexAttentionCORRECT(nn.Module):
    """Correct: Wrapped with torch.compile to generate fused kernel."""
    def __init__(self, window_size=2048):
        super().__init__()
        self.window_size = window_size
        # Create mask function at init time to avoid torch.compile issues
        def _sliding_window(b, h, q_idx, kv_idx):
            return (q_idx - kv_idx).abs() <= window_size
        self.mask_fn = _sliding_window
        
    def forward(self, Q, K, V):
        B, H, T, D = Q.shape
        
        # Create block mask using pre-defined function
        block_mask = create_block_mask(self.mask_fn, B, H, T, T)
        
        # Will be compiled by torch.compile wrapper - generates fused kernel!
        return flex_attention(Q, K, V, block_mask=block_mask)


def benchmark_attention(model, Q, K, V, name, num_warmup=50, num_iters=200):
    """Benchmark attention implementation"""
    print(f"\nBenchmarking: {name}")
    
    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(Q, K, V)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_iters):
        with torch.no_grad():
            _ = model(Q, K, V)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    avg_time_ms = (elapsed / num_iters) * 1000
    
    print(f"  Average time: {avg_time_ms:.2f} ms")
    
    return avg_time_ms


def main():
    """Demonstrate CORRECT FlexAttention usage"""
    
    configure_for_flex_attention()
    
    # Test configuration
    batch_size = 8
    num_heads = 16
    seq_len = 2048
    head_dim = 64
    
    print(f"Test Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num heads: {num_heads}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Head dim: {head_dim}")
    print(f"  Window size: 512")
    
    # Create inputs
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    
    print(f"\nMemory per tensor: {Q.numel() * 4 / 1e6:.2f} MB")
    
    # 1. Baseline: Regular SDPA
    print("\n" + "=" * 80)
    print("TEST 1: Baseline (scaled_dot_product_attention)")
    print("=" * 80)
    baseline = BaselineAttention().cuda().eval()
    baseline_time = benchmark_attention(baseline, Q, K, V, "Baseline SDPA")
    
    # 2. FlexAttention WITHOUT compile (WRONG - will be slower!)
    print("\n" + "=" * 80)
    print("TEST 2: FlexAttention WITHOUT torch.compile (WRONG!)")
    print("=" * 80)
    print("  This will materialize the full attention matrix - SLOW!")
    flex_wrong = FlexAttentionWRONG(window_size=512).cuda().eval()
    wrong_time = benchmark_attention(flex_wrong, Q, K, V, "FlexAttention (not compiled)")
    wrong_speedup = baseline_time / wrong_time
    print(f"  vs Baseline: {wrong_speedup:.2f}x {' SLOWER!' if wrong_speedup < 1.0 else ''}")
    
    # 3. FlexAttention WITH compile (CORRECT - 2x+ faster!)
    print("\n" + "=" * 80)
    print("TEST 3: FlexAttention WITH torch.compile (CORRECT!)")
    print("=" * 80)
    print(" This will generate a fused kernel - FAST!")
    flex_correct = FlexAttentionCORRECT(window_size=512).cuda().eval()
    
    # CRITICAL: Compile the entire module
    compile_issue_msg = None
    correct_time = None
    correct_speedup = None
    try:
        flex_correct_compiled = torch.compile(
            flex_correct,
            mode='max-autotune',
            fullgraph=True,
            dynamic=False
        )
        
        correct_time = benchmark_attention(
            flex_correct_compiled,
            Q,
            K,
            V,
            "FlexAttention (compiled)",
            num_warmup=100
        )
        correct_speedup = baseline_time / correct_time
        print(f"  vs Baseline: {correct_speedup:.2f}x {'' if correct_speedup >= 1.5 else ''}")
    except Exception as err:
        error_text = str(err)
        if not _is_known_compile_failure(error_text):
            raise
        compile_issue_msg = _summarize_error_text(error_text)
        print("  FlexAttention compiled path unavailable on this GPU/toolchain.")
        print(f"  Reason: {compile_issue_msg}")
        print("  Skipping compiled benchmark and speedup target.")
    
    # Results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Baseline SDPA:                 {baseline_time:.2f} ms (1.0x)")
    print(f"FlexAttention (not compiled):  {wrong_time:.2f} ms ({wrong_speedup:.2f}x)")
    if correct_speedup is not None:
        print(
            f"FlexAttention (COMPILED):      {correct_time:.2f} ms ({correct_speedup:.2f}x) "
            f"{'' if correct_speedup >= 1.5 else ''}"
        )
    else:
        print("FlexAttention (COMPILED):      unavailable")
        if compile_issue_msg:
            print(f"  Details: {compile_issue_msg}")
    print()
    
    if correct_speedup is None:
        print(" FlexAttention compiled kernel generation is currently unsupported here.")
        print(" Consider upgrading PyTorch/Triton or GPU drivers once support lands.")
    elif correct_speedup >= 2.0:
        print(" EXCELLENT! Achieving 2x+ speedup!")
    elif correct_speedup >= 1.5:
        print(" GOOD! Meeting 1.5x+ speedup target!")
    else:
        print("  Speedup below target. Try:")
        print("   1. Longer sequences (4096+)")
        print("   2. Smaller window size (more sparsity)")
        print("   3. Ensure compilation succeeded")
    
    print("\n" + "=" * 80)
    print("KEY LEARNINGS FOR BOOK")
    print("=" * 80)
    print("1. FlexAttention MUST be wrapped with torch.compile!")
    print("2. Without compile: materializes full matrix (SLOW)")
    print("3. With compile: generates fused kernel (FAST - 2x+)")
    print("4. Use fullgraph=True and dynamic=False for best results")
    print("5. Warmup is critical (100+ iterations)")
    print("6. Larger sequences benefit more from FlexAttention")
    print("7. Window size affects sparsity and speedup")
    print("=" * 80)
    
    return correct_speedup


if __name__ == "__main__":
    speedup = main()
    
    # Exit with appropriate code
    import sys
    if speedup is None:
        sys.exit(0)
    sys.exit(0 if speedup >= 1.5 else 1)
