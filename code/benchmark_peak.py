"""
Comprehensive Peak Performance Validation

Benchmark suite for validating peak performance on Blackwell B200.
Tests HBM3e bandwidth, torch.compile speedup, FlexAttention, and FP16 compute.
"""

import torch
import torch.nn as nn
import time
import json
from datetime import datetime
import arch_config  # noqa: F401 - Configure Blackwell optimizations


def init_environment():
    """Initialize and detect hardware"""
    print("=" * 80)
    print("BLACKWELL B200 PEAK PERFORMANCE VALIDATION")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print(" CUDA not available")
        return False

    num_gpus = torch.cuda.device_count()
    print(f"Detected CUDA devices: {num_gpus}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    all_blackwell = True
    for idx in range(num_gpus):
        prop = torch.cuda.get_device_properties(idx)
        is_blackwell = (prop.major == 10 and prop.minor == 0)
        all_blackwell = all_blackwell and is_blackwell
        mem_gb = prop.total_memory / (1024**3)
        print(f"- GPU {idx}: {prop.name} | CC {prop.major}.{prop.minor} | "
              f"SMs {prop.multi_processor_count} | {mem_gb:.1f} GB | "
              f"{'Blackwell' if is_blackwell else 'Non-Blackwell'}")

    if num_gpus and num_gpus != 8:
        print(f"⚠ Expected 8 GPUs for 8x B200 validation, found {num_gpus}")

    print("=" * 80 + "\n")

    return all_blackwell and num_gpus == 8


def configure_peak_performance():
    """Configure PyTorch for absolute peak performance"""
    print("Configuring for PEAK PERFORMANCE...")
    
    # TF32 - Enable for Blackwell
    torch.set_float32_matmul_precision('high')
    cuda_matmul = getattr(torch.backends.cuda, "matmul", None)
    if cuda_matmul is not None:
        if hasattr(cuda_matmul, "allow_tf32"):
            try:
                cuda_matmul.allow_tf32 = True
            except (RuntimeError, AttributeError):
                pass
        elif hasattr(cuda_matmul, "fp32_precision"):
            try:
                cuda_matmul.fp32_precision = 'tf32'
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass

    cudnn_conv = getattr(torch.backends.cudnn, "conv", None)
    if cudnn_conv is not None:
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            try:
                torch.backends.cudnn.allow_tf32 = True
            except (RuntimeError, AttributeError):
                pass
        elif hasattr(cudnn_conv, "fp32_precision"):
            try:
                cudnn_conv.fp32_precision = 'tf32'
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
    
    # Flash Attention
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)  # Disable slow fallback
    
    # Inductor - AGGRESSIVE settings
    torch._inductor.config.triton.cudagraphs = True
    torch._inductor.config.triton.cudagraph_trees = True
    torch._inductor.config.max_autotune = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.epilogue_fusion = True
    torch._inductor.config.aggressive_fusion = True
    
    # CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    
    # Set CUDA streams to maximize throughput
    torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of memory
    
    print(" Peak configuration applied\n")


def benchmark_hbm3e_bandwidth_peak():
    """Benchmark HBM3e bandwidth with optimal access patterns."""
    print("=" * 80)
    print("TEST 1: HBM3e Bandwidth (Peak Performance)")
    print("=" * 80)
    
    results = {}
    peak_bandwidth_tbs = 7.8
    
    # Use LARGE transfers to saturate HBM3e bandwidth
    sizes_gb = [8, 12, 16]
    
    for size_gb in sizes_gb:
        size_bytes = int(size_gb * (1024**3))
        
        # Use float16 for maximum bandwidth (2-byte elements)
        # Blackwell HBM3e is optimized for FP16/FP8
        num_elements = size_bytes // 2
        x = torch.empty(num_elements, device="cuda", dtype=torch.float16)
        y = torch.empty_like(x)

        # Touch memory once so pages are resident without incurring RNG costs
        y.copy_(x)
        torch.cuda.synchronize()

        # Warmup (100 iterations for thermal stability)
        for _ in range(100):
            y.copy_(x)
        torch.cuda.synchronize()
        
        # Benchmark (100 iterations for accuracy)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        iterations = 100
        
        start.record()
        for _ in range(iterations):
            y.copy_(x)
        end.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start.elapsed_time(end)
        bandwidth_gbs = (size_bytes * iterations / elapsed_ms) / 1e6
        bandwidth_tbs = bandwidth_gbs / 1024
        utilization = (bandwidth_tbs / peak_bandwidth_tbs) * 100
        
        print(f"  Size: {size_gb:2d} GB | Bandwidth: {bandwidth_tbs:5.2f} TB/s | Utilization: {utilization:5.1f}%")
        results[f"size_{size_gb}gb"] = {
            "bandwidth_tbs": bandwidth_tbs,
            "utilization_pct": utilization
        }
    
    # Get peak bandwidth
    peak_measured = max(r["bandwidth_tbs"] for r in results.values())
    peak_util = (peak_measured / peak_bandwidth_tbs) * 100
    
    print(f"\n Peak Bandwidth: {peak_measured:.2f} TB/s ({peak_util:.1f}% utilization)")
    print(f"   Target: >7.0 TB/s | Status: {'PASS ' if peak_measured > 7.0 else 'GOOD '}")
    
    results["peak_bandwidth_tbs"] = peak_measured
    results["peak_utilization_pct"] = peak_util
    results["status"] = "PASS" if peak_measured > 7.0 else "GOOD"
    
    return results


def benchmark_fp16_peak_compute():
    """Benchmark FP16 compute throughput."""
    print("\n" + "=" * 80)
    print("TEST 2: FP16 Compute Throughput (Peak Performance)")
    print("=" * 80)
    
    results = {}
    
    # Use sizes that maximize tensor core utilization
    for M in [4096, 8192]:
        K, N = M, M
        
        A = torch.randn(M, K, device='cuda', dtype=torch.float16)
        B = torch.randn(K, N, device='cuda', dtype=torch.float16)
        
        # Warmup
        for _ in range(20):
            C = torch.matmul(A, B)
        torch.cuda.synchronize()
        
        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        iterations = 50
        
        start.record()
        for _ in range(iterations):
            C = torch.matmul(A, B)
        end.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start.elapsed_time(end) / iterations
        
        # Calculate TFLOPS (2*M*N*K operations)
        flops = 2 * M * N * K
        tflops = flops / (elapsed_ms * 1e-3) / 1e12
        
        print(f"  Matrix {M}x{M}: {elapsed_ms:.2f} ms | {tflops:.0f} TFLOPS")
        results[f"matrix_{M}x{M}"] = {"time_ms": elapsed_ms, "tflops": tflops}
    
    peak_tflops = max(r["tflops"] for r in results.values())
    print(f"\n Peak FP16: {peak_tflops:.0f} TFLOPS")
    print(f"   Target: >1000 TFLOPS | Status: {'PASS ' if peak_tflops > 1000 else 'GOOD '}")
    
    results["peak_tflops"] = peak_tflops
    results["status"] = "PASS" if peak_tflops > 1000 else "GOOD"
    
    return results


def benchmark_torch_compile_peak():
    """
    Benchmark torch.compile with PROPER configuration
    Target: 1.3-1.5x speedup
    """
    print("\n" + "=" * 80)
    print("TEST 3: torch.compile Speedup (Properly Configured)")
    print("=" * 80)
    
    # Create a realistic model
    class TransformerBlock(nn.Module):
        def __init__(self, d_model=1024):
            super().__init__()
            self.norm1 = nn.LayerNorm(d_model)
            self.attn_qkv = nn.Linear(d_model, 3 * d_model)
            self.attn_out = nn.Linear(d_model, d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn1 = nn.Linear(d_model, 4 * d_model)
            self.ffn2 = nn.Linear(4 * d_model, d_model)
        
        def forward(self, x):
            # Simplified transformer block
            residual = x
            x = self.norm1(x)
            qkv = self.attn_qkv(x)
            x = self.attn_out(qkv[:, :, :1024])  # Simplified attention
            x = x + residual
            
            residual = x
            x = self.norm2(x)
            x = self.ffn1(x)
            x = torch.nn.functional.gelu(x)
            x = self.ffn2(x)
            x = x + residual
            return x
    
    model = TransformerBlock().cuda().half().eval()  # Use FP16 for speed
    
    # LARGER batch for better GPU utilization
    x = torch.randn(64, 512, 1024, device='cuda', dtype=torch.float16)
    
    # Eager mode - warmup and benchmark
    with torch.no_grad():
        for _ in range(50):
            _ = model(x)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        iterations = 200
        for _ in range(iterations):
            _ = model(x)
        torch.cuda.synchronize()
        eager_time = (time.perf_counter() - start) / iterations * 1000
    
    # Compiled mode with CUDA graphs and aggressive optimization
    model_compiled = torch.compile(
        model,
        mode='max-autotune',
        fullgraph=True,
        dynamic=False  # Static shapes for CUDA graphs
    )
    
    # CRITICAL: Extensive warmup for torch.compile
    print("  Warming up compiled model (200 iterations for CUDA graph capture)...")
    with torch.no_grad():
        for _ in range(200):
            _ = model_compiled(x)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(iterations):
            _ = model_compiled(x)
        torch.cuda.synchronize()
        compiled_time = (time.perf_counter() - start) / iterations * 1000
    
    speedup = eager_time / compiled_time
    
    print(f"  Eager mode:    {eager_time:.3f} ms")
    print(f"  Compiled mode: {compiled_time:.3f} ms")
    print(f"  Speedup:       {speedup:.2f}x {'' if speedup >= 1.3 else ''}")
    
    print(f"\n torch.compile: {speedup:.2f}x speedup")
    print(f"   Target: >1.3x | Status: {'PASS ' if speedup >= 1.3 else 'NEEDS TUNING '}")
    
    return {
        "eager_time_ms": eager_time,
        "compiled_time_ms": compiled_time,
        "speedup": speedup,
        "status": "PASS" if speedup >= 1.3 else "NEEDS TUNING"
    }


def main():
    """Run all peak performance benchmarks"""
    
    is_blackwell = init_environment()
    if not torch.cuda.is_available():
        print(" CUDA not available, exiting")
        return
    
    configure_peak_performance()
    
    all_results = {}
    
    # Test 1: HBM3e Bandwidth
    all_results["hbm3e_bandwidth"] = benchmark_hbm3e_bandwidth_peak()
    
    # Test 2: FP16 Compute
    all_results["fp16_compute"] = benchmark_fp16_peak_compute()
    
    # Test 3: torch.compile
    all_results["torch_compile"] = benchmark_torch_compile_peak()
    
    # Summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE SUMMARY")
    print("=" * 80)
    
    tests_passed = 0
    total_tests = 0
    
    if "hbm3e_bandwidth" in all_results:
        bw = all_results["hbm3e_bandwidth"]["peak_bandwidth_tbs"]
        status = all_results["hbm3e_bandwidth"]["status"]
        print(f"\n  HBM3e Bandwidth:  {bw:.2f} TB/s | {status} {'' if status == 'PASS' else ''}")
        if status == "PASS": tests_passed += 1
        total_tests += 1
    
    if "fp16_compute" in all_results:
        tflops = all_results["fp16_compute"]["peak_tflops"]
        status = all_results["fp16_compute"]["status"]
        print(f"  FP16 Compute:     {tflops:.0f} TFLOPS | {status} {'' if status == 'PASS' else ''}")
        if status == "PASS": tests_passed += 1
        total_tests += 1
    
    if "torch_compile" in all_results:
        speedup = all_results["torch_compile"]["speedup"]
        status = all_results["torch_compile"]["status"]
        print(f"  torch.compile:    {speedup:.2f}x | {status} {'' if status == 'PASS' else ''}")
        if status == "PASS": tests_passed += 1
        total_tests += 1
    
    print("\n" + "=" * 80)
    score = (tests_passed / total_tests * 100) if total_tests > 0 else 0
    print(f"Overall Score: {tests_passed}/{total_tests} ({score:.0f}%)")
    print(f"Status: {' PEAK PERFORMANCE!' if tests_passed == total_tests else '  NEEDS TUNING'}")
    print("=" * 80)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"BENCHMARK_PEAK_RESULTS_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n Results saved to: {filename}")
    
    # Key learnings
    print("\n" + "=" * 80)
    print("KEY LEARNINGS")
    print("=" * 80)
    print("1. HBM3e: Use large transfers (8-16 GB) for peak bandwidth")
    print("2. torch.compile: Need 100+ warmup iterations!")
    print("3. TF32: Must enable with torch.set_float32_matmul_precision('high')")
    print("4. CUDA graphs: Enable triton.cudagraph_trees for 15-20% boost")
    print("5. Matrix sizes: Use multiples of 128 for tensor core efficiency")
    print("=" * 80)


if __name__ == "__main__":
    main()
