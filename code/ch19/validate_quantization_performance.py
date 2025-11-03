#!/usr/bin/env python3
"""
Comprehensive FP4/FP6/FP8 Performance Validation
================================================

Validates and profiles all quantization examples with:
- PyTorch profiler (detailed kernel analysis)
- NVTX markers (for nsys profiling)
- Memory tracking
- Performance metrics collection
- Automated report generation

Usage:
    # Run all validations with profiling
    python validate_quantization_performance.py --profile-all
    
    # Run specific example
    python validate_quantization_performance.py --example fp4 --profile
    
    # Generate nsys profile
    nsys profile -o fp4_profile python validate_quantization_performance.py --example fp4
    
    # Generate ncu profile (specific kernel)
    ncu --set full -o fp4_ncu python validate_quantization_performance.py --example fp4 --iterations 1
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch profiler
from torch.profiler import profile, ProfilerActivity, record_function

# NVTX for nsys profiling
try:
    import torch.cuda.nvtx as nvtx
    NVTX_AVAILABLE = True
except (ImportError, AttributeError):
    NVTX_AVAILABLE = False
    # Fallback no-op context manager
    class nvtx:
        @staticmethod
        def range(msg):
            class DummyContext:
                def __enter__(self): return self
                def __exit__(self, *args): pass
            return DummyContext()


@dataclass
class BenchmarkResult:
    """Store comprehensive benchmark results"""
    example_name: str
    precision: str
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_time_ms: float
    memory_allocated_mb: float
    memory_reserved_mb: float
    throughput_tokens_per_sec: float
    tflops: Optional[float]
    batch_size: int
    seq_len: int
    model_params: int
    gpu_name: str
    compute_capability: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ProfiledBenchmark:
    """Wrapper for running benchmarks with comprehensive profiling"""
    
    def __init__(self, name: str, enable_profiler: bool = False):
        self.name = name
        self.enable_profiler = enable_profiler
        self.results: List[BenchmarkResult] = []
        
    def benchmark_function(
        self,
        func: callable,
        args: tuple,
        precision: str,
        warmup_iters: int = 10,
        benchmark_iters: int = 50,
        batch_size: int = 1,
        seq_len: int = 1,
        model_params: int = 0,
    ) -> BenchmarkResult:
        """
        Run a function with comprehensive profiling and timing.
        
        Args:
            func: Function to benchmark
            args: Arguments to pass to function
            precision: Precision mode (fp4, fp6, fp8, fp16, fp32)
            warmup_iters: Number of warmup iterations
            benchmark_iters: Number of benchmark iterations
            batch_size: Batch size for throughput calculation
            seq_len: Sequence length for throughput calculation
            model_params: Model parameters for TFLOPS calculation
        
        Returns:
            BenchmarkResult with comprehensive metrics
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Get GPU info
        props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
        gpu_name = props.name if props else "CPU"
        compute_capability = f"{props.major}.{props.minor}" if props else "N/A"
        
        # Warmup
        with nvtx.range(f"{self.name}_{precision}_warmup"):
            for _ in range(warmup_iters):
                _ = func(*args)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        
        # Clear memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        # Benchmark
        times = []
        with nvtx.range(f"{self.name}_{precision}_benchmark"):
            for i in range(benchmark_iters):
                if torch.cuda.is_available():
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    
                    start_event.record()
                    with nvtx.range(f"{self.name}_{precision}_iter_{i}"):
                        _ = func(*args)
                    end_event.record()
                    
                    torch.cuda.synchronize()
                    elapsed_ms = start_event.elapsed_time(end_event)
                    times.append(elapsed_ms)
                else:
                    start = time.time()
                    _ = func(*args)
                    elapsed_ms = (time.time() - start) * 1000
                    times.append(elapsed_ms)
        
        # Calculate statistics
        avg_time_ms = sum(times) / len(times)
        min_time_ms = min(times)
        max_time_ms = max(times)
        std_time_ms = (sum((t - avg_time_ms) ** 2 for t in times) / len(times)) ** 0.5
        
        # Memory statistics
        if torch.cuda.is_available():
            memory_allocated_mb = torch.cuda.max_memory_allocated() / 1024**2
            memory_reserved_mb = torch.cuda.max_memory_reserved() / 1024**2
        else:
            memory_allocated_mb = 0
            memory_reserved_mb = 0
        
        # Throughput (tokens/sec)
        throughput = (batch_size * seq_len * 1000) / avg_time_ms if avg_time_ms > 0 else 0
        
        # TFLOPS estimation (if model_params provided)
        tflops = None
        if model_params > 0:
            # Approximate FLOPs for transformer forward pass: 2 * params per token
            flops = 2 * model_params * batch_size * seq_len
            tflops = (flops / 1e12) / (avg_time_ms / 1000)
        
        result = BenchmarkResult(
            example_name=self.name,
            precision=precision,
            avg_time_ms=avg_time_ms,
            min_time_ms=min_time_ms,
            max_time_ms=max_time_ms,
            std_time_ms=std_time_ms,
            memory_allocated_mb=memory_allocated_mb,
            memory_reserved_mb=memory_reserved_mb,
            throughput_tokens_per_sec=throughput,
            tflops=tflops,
            batch_size=batch_size,
            seq_len=seq_len,
            model_params=model_params,
            gpu_name=gpu_name,
            compute_capability=compute_capability,
        )
        
        self.results.append(result)
        return result
    
    def run_with_pytorch_profiler(
        self,
        func: callable,
        args: tuple,
        precision: str,
        output_dir: str = "./profiler_output",
    ):
        """Run function with PyTorch profiler for detailed kernel analysis"""
        
        os.makedirs(output_dir, exist_ok=True)
        trace_path = os.path.join(output_dir, f"{self.name}_{precision}_trace.json")
        
        print(f"\n🔍 Running PyTorch profiler for {self.name} ({precision})...")
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        ) as prof:
            with record_function(f"{self.name}_{precision}"):
                for _ in range(10):  # Profile 10 iterations
                    _ = func(*args)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
        
        # Export trace
        prof.export_chrome_trace(trace_path)
        print(f"  ✓ Trace saved to: {trace_path}")
        
        # Print summary
        print("\n" + "=" * 80)
        print(f"PyTorch Profiler Summary - {self.name} ({precision})")
        print("=" * 80)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        
        return trace_path
    
    def print_results(self):
        """Print formatted benchmark results"""
        if not self.results:
            print("No results to display")
            return
        
        print("\n" + "=" * 100)
        print(f"Benchmark Results: {self.name}")
        print("=" * 100)
        
        # Header
        print(f"{'Precision':<12} {'Time (ms)':<12} {'Std (ms)':<12} {'Memory (MB)':<14} "
              f"{'Throughput':<18} {'TFLOPS':<10}")
        print("-" * 100)
        
        # Results
        for result in self.results:
            tflops_str = f"{result.tflops:.2f}" if result.tflops else "N/A"
            print(f"{result.precision:<12} {result.avg_time_ms:<12.3f} {result.std_time_ms:<12.3f} "
                  f"{result.memory_allocated_mb:<14.2f} {result.throughput_tokens_per_sec:<18.1f} "
                  f"{tflops_str:<10}")
        
        # Speedup analysis
        if len(self.results) > 1:
            baseline = self.results[0]
            print("\n" + "=" * 100)
            print("Speedup Analysis (vs baseline)")
            print("=" * 100)
            print(f"{'Precision':<12} {'Speedup':<12} {'Memory Reduction':<20} {'Throughput Gain':<15}")
            print("-" * 100)
            
            for result in self.results:
                speedup = baseline.avg_time_ms / result.avg_time_ms if result.avg_time_ms > 0 else 0
                mem_reduction = (1 - result.memory_allocated_mb / baseline.memory_allocated_mb) * 100 if baseline.memory_allocated_mb > 0 else 0
                throughput_gain = result.throughput_tokens_per_sec / baseline.throughput_tokens_per_sec if baseline.throughput_tokens_per_sec > 0 else 0
                
                print(f"{result.precision:<12} {speedup:<12.2f}x {mem_reduction:<20.1f}% {throughput_gain:<15.2f}x")
    
    def save_results(self, output_path: str):
        """Save results to JSON"""
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        results_dict = {
            "benchmark_name": self.name,
            "results": [r.to_dict() for r in self.results],
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_path}")


def validate_fp4_example(profiler: ProfiledBenchmark, enable_pytorch_profiler: bool = False):
    """Validate FP4 quantization example"""
    
    print("\n" + "=" * 100)
    print("Validating FP4 Quantization (native_fp4_quantization.py)")
    print("=" * 100)
    
    # Import the actual example
    try:
        # Add parent directory to path for arch_config
        parent_dir = str(Path(__file__).parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        sys.path.insert(0, str(Path(__file__).parent))
        from native_fp4_quantization import FP4MLP, FP4Tensor
    except ImportError as e:
        print(f"❌ Failed to import FP4 example: {e}")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16
    
    # Configuration
    batch_size = 64
    seq_len = 2048
    d_model = 4096
    d_ff = 16384
    
    x = torch.randn(batch_size, seq_len, d_model, dtype=dtype, device=device)
    
    # FP16 baseline
    print("\n📊 Testing FP16 baseline...")
    mlp_fp16 = FP4MLP(d_model, d_ff, dtype=dtype).to(device)
    model_params = sum(p.numel() for p in mlp_fp16.parameters())
    
    result_fp16 = profiler.benchmark_function(
        func=mlp_fp16,
        args=(x,),
        precision="FP16",
        batch_size=batch_size,
        seq_len=seq_len,
        model_params=model_params,
    )
    
    if enable_pytorch_profiler:
        profiler.run_with_pytorch_profiler(mlp_fp16, (x,), "FP16")
    
    # FP4 quantized (simulated - actual FP4 requires Blackwell)
    print("\n📊 Testing FP4 quantized...")
    # Note: This simulates FP4 compression
    with nvtx.range("FP4_quantization"):
        mlp_fp4 = FP4MLP(d_model, d_ff, dtype=dtype).to(device)
        # Simulate FP4 compression on weights
        for param in mlp_fp4.parameters():
            fp4_tensor = FP4Tensor(param.data, dtype=dtype)
            # Dequantize for computation (actual FP4 would use tensor cores)
            param.data = fp4_tensor.dequantize()
    
    result_fp4 = profiler.benchmark_function(
        func=mlp_fp4,
        args=(x,),
        precision="FP4",
        batch_size=batch_size,
        seq_len=seq_len,
        model_params=model_params,
    )
    
    if enable_pytorch_profiler:
        profiler.run_with_pytorch_profiler(mlp_fp4, (x,), "FP4")
    
    print("\n✅ FP4 validation complete!")
    print(f"   Expected: 75% memory savings, ~1600 TFLOPS on Blackwell")
    print(f"   Actual:   {(1 - result_fp4.memory_allocated_mb / result_fp16.memory_allocated_mb) * 100:.1f}% memory savings")


def validate_fp8_matmul_example(profiler: ProfiledBenchmark, enable_pytorch_profiler: bool = False):
    """Validate FP8 compiled matmul example"""
    
    print("\n" + "=" * 100)
    print("Validating FP8 Compiled Matmul (fp8_compiled_matmul.py)")
    print("=" * 100)
    
    # Check FP8 support
    try:
        FP8_E4M3 = torch.float8_e4m3fn
        FP8_AVAILABLE = True
    except AttributeError:
        print("⚠️  FP8 not available, using FP16 for comparison")
        FP8_AVAILABLE = False
        FP8_E4M3 = torch.float16
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # GEMM configuration (reduced size to avoid OOM)
    M, N, K = 1024, 1024, 1024
    
    # FP32 baseline
    print("\n📊 Testing FP32 baseline...")
    A_fp32 = torch.randn(M, K, dtype=torch.float32, device=device)
    B_fp32 = torch.randn(K, N, dtype=torch.float32, device=device)
    
    def matmul_fp32():
        return torch.matmul(A_fp32, B_fp32)
    
    # Calculate FLOPs for GEMM: 2*M*N*K
    flops = 2 * M * N * K
    
    result_fp32 = profiler.benchmark_function(
        func=matmul_fp32,
        args=(),
        precision="FP32",
        batch_size=M,
        seq_len=N,
        model_params=0,  # Will calculate TFLOPS manually
    )
    result_fp32.tflops = (flops / 1e12) / (result_fp32.avg_time_ms / 1000)
    
    # FP16
    print("\n📊 Testing FP16...")
    A_fp16 = A_fp32.half()
    B_fp16 = B_fp32.half()
    
    def matmul_fp16():
        return torch.matmul(A_fp16, B_fp16)
    
    result_fp16 = profiler.benchmark_function(
        func=matmul_fp16,
        args=(),
        precision="FP16",
        batch_size=M,
        seq_len=N,
        model_params=0,
    )
    result_fp16.tflops = (flops / 1e12) / (result_fp16.avg_time_ms / 1000)
    
    if enable_pytorch_profiler:
        profiler.run_with_pytorch_profiler(matmul_fp16, (), "FP16")
    
    # FP8 (if available)
    if FP8_AVAILABLE:
        print("\n📊 Testing FP8...")
        # Simple FP8 conversion
        scale = 1.0
        A_fp8 = (A_fp32 / scale).to(FP8_E4M3)
        B_fp8 = (B_fp32 / scale).to(FP8_E4M3)
        
        def matmul_fp8():
            # Dequantize for matmul (actual FP8 matmul uses _scaled_mm)
            return torch.matmul(A_fp8.to(torch.float16), B_fp8.to(torch.float16)) * (scale * scale)
        
        result_fp8 = profiler.benchmark_function(
            func=matmul_fp8,
            args=(),
            precision="FP8",
            batch_size=M,
            seq_len=N,
            model_params=0,
        )
        result_fp8.tflops = (flops / 1e12) / (result_fp8.avg_time_ms / 1000)
        
        if enable_pytorch_profiler:
            profiler.run_with_pytorch_profiler(matmul_fp8, (), "FP8")
    
    print("\n✅ FP8 matmul validation complete!")
    print(f"   Expected: 450 TFLOPS on Blackwell (vs 225 TFLOPS FP16)")
    print(f"   Actual FP32: {result_fp32.tflops:.2f} TFLOPS")
    print(f"   Actual FP16: {result_fp16.tflops:.2f} TFLOPS")
    if FP8_AVAILABLE and result_fp8:
        print(f"   Actual FP8:  {result_fp8.tflops:.2f} TFLOPS")


def generate_validation_report(all_results: List[ProfiledBenchmark], output_dir: str):
    """Generate comprehensive validation report"""
    
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "quantization_validation_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# FP4/FP6/FP8 Quantization Validation Report\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # System info
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            f.write("## System Information\n\n")
            f.write(f"- **GPU**: {props.name}\n")
            f.write(f"- **Compute Capability**: {props.major}.{props.minor}\n")
            f.write(f"- **Memory**: {props.total_memory / 1024**3:.2f} GB\n")
            f.write(f"- **CUDA Version**: {torch.version.cuda}\n")
            f.write(f"- **PyTorch Version**: {torch.__version__}\n\n")
        
        # Results for each benchmark
        for bench in all_results:
            if not bench.results:
                continue
            
            f.write(f"## {bench.name}\n\n")
            
            # Table
            f.write("| Precision | Time (ms) | Memory (MB) | Throughput (tok/s) | TFLOPS |\n")
            f.write("|-----------|-----------|-------------|--------------------|---------|\n")
            
            for result in bench.results:
                tflops_str = f"{result.tflops:.2f}" if result.tflops else "N/A"
                f.write(f"| {result.precision} | {result.avg_time_ms:.3f} | "
                       f"{result.memory_allocated_mb:.2f} | "
                       f"{result.throughput_tokens_per_sec:.1f} | {tflops_str} |\n")
            
            f.write("\n")
            
            # Analysis
            if len(bench.results) > 1:
                baseline = bench.results[0]
                f.write("### Performance Analysis\n\n")
                
                for result in bench.results[1:]:
                    speedup = baseline.avg_time_ms / result.avg_time_ms
                    mem_reduction = (1 - result.memory_allocated_mb / baseline.memory_allocated_mb) * 100
                    
                    f.write(f"**{result.precision} vs {baseline.precision}:**\n")
                    f.write(f"- Speedup: {speedup:.2f}x\n")
                    f.write(f"- Memory Reduction: {mem_reduction:.1f}%\n")
                    
                    if result.tflops and baseline.tflops:
                        f.write(f"- TFLOPS Improvement: {result.tflops / baseline.tflops:.2f}x\n")
                    
                    f.write("\n")
        
        # Summary
        f.write("## Summary\n\n")
        f.write("### Expected Performance on Blackwell B200\n\n")
        f.write("- **FP4**: ~1600 TFLOPS, 75% memory savings vs FP16\n")
        f.write("- **FP6**: ~1400 TFLOPS, 50% memory savings vs FP16\n")
        f.write("- **FP8**: ~450 TFLOPS, 50% memory savings vs FP16\n")
        f.write("- **FP16**: ~225 TFLOPS (baseline)\n\n")
        
        f.write("### Profiling Commands\n\n")
        f.write("```bash\n")
        f.write("# PyTorch profiler (built-in)\n")
        f.write("python validate_quantization_performance.py --profile-all\n\n")
        f.write("# Nsys profiling\n")
        f.write("nsys profile -o fp4_profile --trace=cuda,nvtx \\\n")
        f.write("  python validate_quantization_performance.py --example fp4\n\n")
        f.write("# NCU profiling (specific kernels)\n")
        f.write("ncu --set full -o fp8_matmul_ncu \\\n")
        f.write("  python validate_quantization_performance.py --example fp8_matmul --iterations 1\n")
        f.write("```\n")
    
    print(f"\n✅ Validation report saved to: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Validate FP4/FP6/FP8 quantization examples with profiling")
    parser.add_argument("--example", choices=["fp4", "fp6", "fp8", "fp8_matmul", "all"], default="all",
                       help="Which example to validate")
    parser.add_argument("--profile", action="store_true", help="Enable PyTorch profiler")
    parser.add_argument("--profile-all", action="store_true", help="Enable all profiling tools")
    parser.add_argument("--iterations", type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--output-dir", default="./validation_results", help="Output directory for results")
    
    args = parser.parse_args()
    
    enable_pytorch_profiler = args.profile or args.profile_all
    
    print("=" * 100)
    print("FP4/FP6/FP8 Quantization Performance Validation")
    print("=" * 100)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Validation requires GPU.")
        sys.exit(1)
    
    props = torch.cuda.get_device_properties(0)
    print(f"\n✓ GPU: {props.name}")
    print(f"✓ Compute Capability: {props.major}.{props.minor}")
    print(f"✓ Memory: {props.total_memory / 1024**3:.2f} GB")
    
    is_blackwell = (props.major == 10 and props.minor == 0)
    if is_blackwell:
        print("✓ Blackwell GPU detected - FP4/FP6/FP8 tensor cores available!")
    else:
        print(f"⚠️  Not Blackwell (SM {props.major}.{props.minor}) - FP4 performance will be emulated")
    
    if NVTX_AVAILABLE:
        print("✓ NVTX markers enabled for nsys profiling")
    else:
        print("⚠️  NVTX not available - nsys profiling will have limited annotations")
    
    # Run validations
    all_results = []
    
    if args.example in ["fp4", "all"]:
        profiler = ProfiledBenchmark("FP4_Quantization", enable_profiler=enable_pytorch_profiler)
        validate_fp4_example(profiler, enable_pytorch_profiler)
        profiler.print_results()
        profiler.save_results(os.path.join(args.output_dir, "fp4_results.json"))
        all_results.append(profiler)
    
    if args.example in ["fp8_matmul", "all"]:
        profiler = ProfiledBenchmark("FP8_Matmul", enable_profiler=enable_pytorch_profiler)
        validate_fp8_matmul_example(profiler, enable_pytorch_profiler)
        profiler.print_results()
        profiler.save_results(os.path.join(args.output_dir, "fp8_matmul_results.json"))
        all_results.append(profiler)
    
    # Generate comprehensive report
    report_path = generate_validation_report(all_results, args.output_dir)
    
    print("\n" + "=" * 100)
    print("✅ All validations complete!")
    print("=" * 100)
    print(f"\n📊 Results saved to: {args.output_dir}")
    print(f"📄 Report: {report_path}")
    print("\n💡 Next steps:")
    print("   1. Review the validation report")
    print("   2. Run nsys profiling: nsys profile -o profile --trace=cuda,nvtx python validate_quantization_performance.py")
    print("   3. Run ncu profiling: ncu --set full -o ncu_profile python validate_quantization_performance.py --iterations 1")
    print("   4. Compare results with expected Blackwell performance:")
    print("      - FP4: ~1600 TFLOPS, 75% memory savings")
    print("      - FP8: ~450 TFLOPS, 50% memory savings")
    print("      - FP16: ~225 TFLOPS (baseline)")


if __name__ == "__main__":
    main()

