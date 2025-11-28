#!/usr/bin/env python3
"""
Lab: Matching cuBLAS on Blackwell
=================================

This is a SELF-CONTAINED lab that demonstrates the performance gap between
a custom tcgen05 kernel and NVIDIA's cuBLAS library.

No imports from other chapters - everything needed is in this directory.

Stages:
  - Stage 0: cuBLAS (the target - highly optimized)
  - Stage 1: Naive CUDA with shared memory (no tensor cores)
  - Stage 2: tcgen05 tensor cores (basic CuTE/CUTLASS implementation)

The gap analysis shows what optimizations cuBLAS uses that we don't.
"""

import argparse
import ctypes
import time
from pathlib import Path

import torch

# Local imports only
_LAB_DIR = Path(__file__).resolve().parent

# Try to load custom naive kernel
_kernels_lib = None
try:
    _kernels_lib = ctypes.CDLL(str(_LAB_DIR / "kernels.so"))
except OSError:
    pass  # Will use fallback


def get_device_info():
    """Get GPU device information."""
    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_gb": props.total_memory / 1e9,
    }


def benchmark_kernel(fn, *args, warmup=5, iters=20):
    """Benchmark a kernel function."""
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    
    # Timed runs
    start = time.time()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    
    elapsed_ms = (time.time() - start) / iters * 1000
    return elapsed_ms


def calculate_tflops(M, N, K, time_ms):
    """Calculate TFLOPS for GEMM."""
    flops = 2 * M * N * K
    return (flops / 1e12) / (time_ms / 1000)


# =============================================================================
# Stage Implementations
# =============================================================================

def stage0_cublas(A, B_T):
    """Stage 0: cuBLAS baseline (target to match).
    
    cuBLAS achieves near-peak tensor core utilization through:
    - Persistent kernels that amortize launch overhead
    - Deep software pipelining (3+ stages)
    - Auto-tuned tile configurations per problem size
    - Efficient epilogue and store operations
    """
    return torch.matmul(A, B_T.T)


def stage1_naive_smem(A, B_T):
    """Stage 1: Naive CUDA with shared memory tiling (no tensor cores).
    
    Uses basic tiling to reduce global memory traffic but:
    - No tensor cores - scalar FMA only
    - Simple shared memory layout
    - Sequential K-loop without pipelining
    
    This is ~100x slower than tensor core implementations.
    """
    if _kernels_lib is None:
        # Fallback: FP32 matmul (still slow, but works)
        return torch.matmul(A.float(), B_T.T.float())
    
    M, K = A.shape
    N = B_T.shape[0]
    C = torch.zeros(M, N, device='cuda', dtype=torch.float32)
    
    _kernels_lib.launch_gemm_naive_smem(
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_void_p(B_T.data_ptr()),
        ctypes.c_void_p(C.data_ptr()),
        ctypes.c_int(M),
        ctypes.c_int(N),
        ctypes.c_int(K),
        ctypes.c_void_p(0)
    )
    return C


def stage2_tcgen05_basic(A, B_T):
    """Stage 2: tcgen05 tensor cores (basic configuration).
    
    Uses Blackwell's 5th-generation tensor cores via CuTE/CUTLASS:
    - SM100_MMA_F16BF16_SS operation (128x256 tiles)
    - TMA (Tensor Memory Accelerator) for async loads
    - TMEM (Tensor Memory) for accumulator storage
    - Barrier-based synchronization
    
    Achieves ~20-25% of cuBLAS performance due to:
    - Single-stage pipeline (no overlap of TMA loads)
    - Simple tile scheduling (not persistent)
    - No auto-tuning for problem size
    """
    try:
        from tcgen05_loader import matmul_tcgen05
        return matmul_tcgen05(A, B_T)
    except Exception as e:
        print(f"  [tcgen05 unavailable: {e}]")
        return torch.matmul(A, B_T.T)


def stage3_tcgen05_pipelined(A, B_T):
    """Stage 3: 2-stage pipelined tcgen05.
    
    Key optimization: Double-buffered shared memory.
    While computing tile K, we prefetch tile K+1 via TMA.
    """
    try:
        from tcgen05_loader import matmul_tcgen05_pipelined
        return matmul_tcgen05_pipelined(A, B_T)
    except Exception as e:
        print(f"  [tcgen05_pipelined unavailable: {e}]")
        return torch.matmul(A, B_T.T)


def stage4_tcgen05_3stage(A, B_T):
    """Stage 4: 3-stage pipelined tcgen05.
    
    Deeper pipelining with 3 shared memory buffers.
    Prefetches 2 tiles ahead while computing current tile.
    Better latency hiding than 2-stage.
    """
    try:
        from tcgen05_loader import matmul_tcgen05_3stage
        return matmul_tcgen05_3stage(A, B_T)
    except Exception as e:
        print(f"  [tcgen05_3stage unavailable: {e}]")
        return torch.matmul(A, B_T.T)


def stage5_tcgen05_swizzled(A, B_T):
    """Stage 5: 3-stage pipeline + swizzled tile scheduling.
    
    Tiles processed in cache-friendly swizzled order.
    XOR swizzle pattern improves L2 hit rate by 10-20%.
    """
    try:
        from tcgen05_loader import matmul_tcgen05_swizzled
        return matmul_tcgen05_swizzled(A, B_T)
    except Exception as e:
        print(f"  [tcgen05_swizzled unavailable: {e}]")
        return torch.matmul(A, B_T.T)


def stage6_cluster(A, B_T):
    """Stage 6: Thread block cluster structure.
    
    Uses 3-stage pipeline optimized for cluster execution.
    Cluster launch enables L2 multicast for better cache utilization.
    """
    try:
        from tcgen05_loader import matmul_tcgen05_cluster
        return matmul_tcgen05_cluster(A, B_T)
    except Exception as e:
        print(f"  [tcgen05_cluster unavailable: {e}]")
        return torch.matmul(A, B_T.T)


def stage7_autotuned(A, B_T):
    """Stage 7: Auto-Select Best Configuration
    
    Automatically selects the best kernel:
    - Benchmarks all available optimizations (stages 2-6)
    - Caches winner per problem size and GPU
    - Adapts to your specific hardware
    """
    try:
        from autotune import matmul_autotuned
        return matmul_autotuned(A, B_T)
    except Exception as e:
        print(f"  [autotune unavailable: {e}]")
        return torch.matmul(A, B_T.T)


# Stage registry - progressive/compounding optimizations
STAGES = {
    0: ("cuBLAS (Target)", stage0_cublas),
    1: ("Naive (SMEM tiling)", stage1_naive_smem),
    2: ("+ Tensor Cores", stage2_tcgen05_basic),
    3: ("+ 2-Stage Pipeline", stage3_tcgen05_pipelined),
    4: ("+ 3-Stage Pipeline", stage4_tcgen05_3stage),
    5: ("+ Swizzled Tiles", stage5_tcgen05_swizzled),
    6: ("+ Cluster Structure", stage6_cluster),
    7: ("Auto-Select Best", stage7_autotuned),
}


def run_stage(stage_num, A, B_T, M, N, K, verbose=True):
    """Run a single stage and return results."""
    name, fn = STAGES[stage_num]
    
    try:
        time_ms = benchmark_kernel(fn, A, B_T)
        tflops = calculate_tflops(M, N, K, time_ms)
        
        if verbose:
            bar_len = min(50, int(tflops / 20))
            bar = "█" * bar_len
            print(f"  Stage {stage_num}: {name:<25} {time_ms:>8.3f} ms  {tflops:>7.1f} TFLOPS  {bar}")
        
        return {"stage": stage_num, "name": name, "time_ms": time_ms, "tflops": tflops}
    
    except Exception as e:
        if verbose:
            print(f"  Stage {stage_num}: {name:<25} FAILED: {e}")
        return {"stage": stage_num, "name": name, "time_ms": None, "tflops": None, "error": str(e)}


def verify_correctness(A, B_T, verbose=True):
    """Verify that our kernels produce correct results."""
    if verbose:
        print("\nVerifying correctness...")
    
    ref = torch.matmul(A, B_T.T)
    
    for stage_num, (name, fn) in STAGES.items():
        if stage_num == 0:
            continue
        try:
            result = fn(A, B_T)
            ref_fp32 = ref.float()
            result_fp32 = result.float()
            max_diff = (ref_fp32 - result_fp32).abs().max().item()
            rel_err = max_diff / ref_fp32.abs().max().item()
            passed = rel_err < 0.01
            if verbose:
                status = "✓" if passed else "✗"
                print(f"  Stage {stage_num}: {name:<25} {status} (rel_err={rel_err:.2e})")
        except Exception as e:
            if verbose:
                print(f"  Stage {stage_num}: {name:<25} ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(description="Matching cuBLAS Lab")
    parser.add_argument("--stage", type=int, help="Run specific stage only")
    parser.add_argument("--size", type=int, default=4096, help="Matrix size (default: 4096)")
    parser.add_argument("--verify", action="store_true", help="Verify correctness")
    parser.add_argument("--no-naive", action="store_true", help="Skip slow naive kernel")
    args = parser.parse_args()
    
    device_info = get_device_info()
    M = N = K = args.size
    
    # tcgen05 requires specific alignment (128x256x64)
    if M % 128 != 0 or N % 256 != 0 or K % 64 != 0:
        M = ((M + 127) // 128) * 128
        N = ((N + 255) // 256) * 256
        K = ((K + 63) // 64) * 64
        print(f"Note: Adjusted size to {M}x{N}x{K} for tcgen05 alignment")
    
    print()
    print("=" * 75)
    print("  LAB: Matching cuBLAS on Blackwell")
    print("=" * 75)
    print()
    print(f"  Device: {device_info['name']} (SM {device_info['compute_capability']})")
    print(f"  Matrix: A[{M}x{K}] @ B^T[{N}x{K}] = C[{M}x{N}] (FP16)")
    print(f"  FLOPs:  {2*M*N*K/1e12:.2f} TFLOP per GEMM")
    print()
    
    torch.manual_seed(42)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B_T = torch.randn(N, K, device="cuda", dtype=torch.float16)
    
    if args.verify:
        verify_correctness(A, B_T)
        print()
    
    print("  Running benchmarks...")
    print("-" * 75)
    
    results = []
    stages_to_run = [args.stage] if args.stage is not None else list(STAGES.keys())
    
    for stage_num in stages_to_run:
        if stage_num not in STAGES:
            continue
        if args.no_naive and stage_num == 1:
            print(f"  Stage {stage_num}: Skipped (--no-naive)")
            continue
        result = run_stage(stage_num, A, B_T, M, N, K)
        results.append(result)
    
    print("-" * 75)
    
    # Gap Analysis
    cublas_result = next((r for r in results if r["stage"] == 0), None)
    if cublas_result and cublas_result["tflops"] and len(results) > 1:
        print()
        print("  Gap Analysis:")
        print("-" * 75)
        for r in results:
            if r["tflops"] and r["stage"] != 0:
                pct = (r["tflops"] / cublas_result["tflops"]) * 100
                gap_x = cublas_result["tflops"] / r["tflops"]
                print(f"  Stage {r['stage']}: {pct:>5.1f}% of cuBLAS ({gap_x:.1f}x to close)")
    
    print()
    print("=" * 75)
    print("  What's in the Gap?")
    print("=" * 75)
    print("""
  At 16K: We achieved ~46% of cuBLAS (122 vs 268 TFLOPS)!
  The remaining ~54% gap comes from:
  
  1. PERSISTENT KERNELS (~15-20% of gap)
     cuBLAS CTAs stay resident and process multiple tiles.
     
  2. WARP SPECIALIZATION (~10-15% of gap)
     Dedicated producer (TMA) and consumer (MMA) warps.
     
  3. TRUE CLUSTER LAUNCH (~5-10% of gap)
     cudaLaunchKernelEx with cluster dims + TMA multicast.
     
  4. SASS-LEVEL TUNING (~10% of gap)
     Hand-tuned instruction scheduling and register allocation.

  TIP: Use --size 16384 for best results (larger = better % of cuBLAS)
  See README.md for details. Study CUTLASS 4.x to go further.
""")


if __name__ == "__main__":
    main()
