"""roofline_analysis_ilp.py - Roofline analysis for ILP-optimized kernels.

Demonstrates roofline analysis comparing baseline vs optimized ILP kernels.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import pathlib
import sys

_EXTRAS_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_EXTRAS_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTRAS_REPO_ROOT))

from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import time
from typing import Optional, Dict
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)
from ch6.baseline_gemm_ilp import BaselineGEMMILPBenchmark
from ch6.optimized_gemm_tensor_cores import OptimizedGEMMTensorCoresBenchmark


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch6")
    return torch.device("cuda")


class RooflineAnalyzer:
    """Roofline model analyzer for ILP kernels."""
    
    def __init__(self, peak_bandwidth_gbs: float = 8000, peak_compute_tflops: float = 2000):
        """Initialize roofline analyzer."""
        self.peak_bandwidth = peak_bandwidth_gbs  # GB/s
        self.peak_compute = peak_compute_tflops  # TFLOPS
        self.ridge_point = self.peak_compute / self.peak_bandwidth  # FLOP/Byte
    
    def calculate_ai(self, flops: float, bytes_accessed: float) -> float:
        """Calculate arithmetic intensity."""
        return flops / bytes_accessed if bytes_accessed > 0 else 0
    
    def predict_performance(self, ai: float) -> float:
        """Predict maximum achievable performance given AI."""
        memory_bound = self.peak_bandwidth * ai / 1000  # Convert to TFLOPS
        return min(memory_bound, self.peak_compute)
    
    def analyze_kernel(self, benchmark: Benchmark, iterations: int = 50) -> Dict:
        """Analyze a kernel's performance on roofline model."""
        harness = BenchmarkHarness(
            mode=BenchmarkMode.CUSTOM,
            config=BenchmarkConfig(iterations=iterations, warmup=10)
        )
        result = harness.benchmark(benchmark)
        
        # Get roofline metrics if available
        if hasattr(benchmark, 'get_roofline_metrics'):
            metrics = benchmark.get_roofline_metrics()
            flops = metrics.get('flops', 0)
            bytes_accessed = metrics.get('bytes', 0)
            ai = metrics.get('ai', 0)
        else:
            # Fallback: estimate from benchmark
            ai = 0
            flops = 0
            bytes_accessed = 0
        
        # Calculate achieved performance
        time_sec = result.mean_ms / 1000.0
        achieved_tflops = (flops / time_sec) / 1e12 if time_sec > 0 else 0
        max_achievable = self.predict_performance(ai) if ai > 0 else 0
        efficiency = (achieved_tflops / max_achievable * 100) if max_achievable > 0 else 0
        
        # Determine bottleneck
        if ai < self.ridge_point:
            bottleneck = "Memory-bound"
        else:
            bottleneck = "Compute-bound"
        
        return {
            'name': benchmark.__class__.__name__,
            'ai': ai,
            'achieved_tflops': achieved_tflops,
            'max_achievable_tflops': max_achievable,
            'efficiency': efficiency,
            'bottleneck': bottleneck,
            'time_ms': result.mean_ms,
        }


class RooflineAnalysisILPBenchmark(Benchmark):
    """Benchmark for roofline analysis of ILP kernels."""
    
    def __init__(self):
        self.device = resolve_device()
        self.analyzer = None
        self.results = None
    
    def setup(self) -> None:
        """Setup: Initialize roofline analyzer."""
        # B200 specs: 8 TB/s bandwidth, 2000 TFLOPS FP16
        self.analyzer = RooflineAnalyzer(
            peak_bandwidth_gbs=8000,
            peak_compute_tflops=2000
        )
    
    def benchmark_fn(self) -> None:
        """Benchmark: Run roofline analysis."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("roofline_analysis_ilp", enable=enable_nvtx):
            # Analyze baseline GEMM
            baseline = BaselineGEMMILPBenchmark()
            baseline.setup()
            baseline_result = self.analyzer.analyze_kernel(baseline)
            baseline.teardown()
            
            # Analyze optimized GEMM with tensor cores
            optimized = OptimizedGEMMTensorCoresBenchmark()
            optimized.setup()
            optimized_result = self.analyzer.analyze_kernel(optimized)
            optimized.teardown()
            
            self.results = {
                'baseline': baseline_result,
                'optimized': optimized_result,
                'ridge_point': self.analyzer.ridge_point,
            }

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.results = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=1,  # Analysis is deterministic
            warmup=0,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.results is None:
            return "Results not generated"
        if 'baseline' not in self.results or 'optimized' not in self.results:
            return "Incomplete results"
        return None
    
    def print_results(self) -> None:
        """Print roofline analysis results."""
        if self.results is None:
            return
        
        print("\n" + "="*80)
        print("ROOFLINE ANALYSIS FOR ILP-OPTIMIZED KERNELS")
        print("="*80)
        print(f"\nRidge Point: {self.results['ridge_point']:.1f} FLOP/Byte")
        print(f" • AI < {self.results['ridge_point']:.0f}: Memory-bound")
        print(f" • AI > {self.results['ridge_point']:.0f}: Compute-bound\n")
        
        print(f"{'Kernel':<40} {'AI':<12} {'Achieved':<15} {'Efficiency':<12} {'Bottleneck':<15}")
        print("-"*80)
        for key in ['baseline', 'optimized']:
            r = self.results[key]
            print(f"{r['name']:<40} {r['ai']:<12.2f} {r['achieved_tflops']:<7.2f} TFLOPS {r['efficiency']:<12.1f}% {r['bottleneck']:<15}")
        print("="*80)
        
        # Calculate speedup
        baseline_time = self.results['baseline']['time_ms']
        optimized_time = self.results['optimized']['time_ms']
        speedup = baseline_time / optimized_time if optimized_time > 0 else 0
        print(f"\nSpeedup: {speedup:.2f}x ({baseline_time:.2f} ms → {optimized_time:.2f} ms)")
        print("\n  Tip: Tensor cores provide significant speedup for compute-bound GEMM operations")


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return RooflineAnalysisILPBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    benchmark.print_results()
