"""roofline_analysis_ilp.py - Roofline analysis for ILP-optimized kernels."""

from __future__ import annotations

import pathlib
import sys
from typing import Dict, Optional

import torch

_EXTRAS_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_EXTRAS_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTRAS_REPO_ROOT))

from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from ch6.baseline_gemm_ilp import BaselineGEMMILPBenchmark
from ch6.optimized_gemm_tensor_cores import OptimizedGEMMTensorCoresBenchmark


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
    
    def analyze_kernel(self, benchmark: BaseBenchmark, iterations: int = 50) -> Dict:
        """Analyze a kernel's performance on roofline model."""
        from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
        harness = BenchmarkHarness(
            mode=BenchmarkMode.CUSTOM,
            config=BenchmarkConfig(iterations=iterations, warmup=10)
        )
        result = harness.benchmark(benchmark)
        
        if hasattr(benchmark, "get_roofline_metrics"):
            metrics = benchmark.get_roofline_metrics()
            flops = metrics.get("flops", 0)
            bytes_accessed = metrics.get("bytes", 0)
            ai = metrics.get("ai", 0)
        else:
            ai = 0
            flops = 0
            bytes_accessed = 0
        
        time_ms = result.timing.mean_ms if result.timing else 0.0
        time_sec = time_ms / 1000.0 if time_ms else 0.0
        achieved_tflops = (flops / time_sec) / 1e12 if time_sec > 0 else 0
        max_achievable = self.predict_performance(ai) if ai > 0 else 0
        efficiency = (achieved_tflops / max_achievable * 100) if max_achievable > 0 else 0
        
        bottleneck = "Memory-bound" if ai < self.ridge_point else "Compute-bound"
        
        return {
            "name": benchmark.__class__.__name__,
            "ai": ai,
            "achieved_tflops": achieved_tflops,
            "max_achievable_tflops": max_achievable,
            "efficiency": efficiency,
            "bottleneck": bottleneck,
            "time_ms": time_ms,
            "flops": flops,
            "bytes_accessed": bytes_accessed,
        }


class RooflineAnalysisILPBenchmark(BaseBenchmark):
    """Benchmark for roofline analysis of ILP kernels."""
    
    def __init__(self):
        super().__init__()
        self.analyzer: Optional[RooflineAnalyzer] = None
        self.results: Optional[Dict] = None
    
    def setup(self) -> None:
        """Setup: Initialize roofline analyzer."""
        self.analyzer = RooflineAnalyzer(
            peak_bandwidth_gbs=8000,
            peak_compute_tflops=2000,
        )
    
    def benchmark_fn(self) -> None:
        """Benchmark: Run roofline analysis."""
        assert self.analyzer is not None
        with self._nvtx_range("roofline_analysis_ilp"):
            baseline = BaselineGEMMILPBenchmark()
            baseline.setup()
            baseline_result = self.analyzer.analyze_kernel(baseline)
            baseline.teardown()
            
            optimized = OptimizedGEMMTensorCoresBenchmark()
            optimized.setup()
            optimized_result = self.analyzer.analyze_kernel(optimized)
            optimized.teardown()
            
            self.results = {
                "baseline": baseline_result,
                "optimized": optimized_result,
                "ridge_point": self.analyzer.ridge_point,
            }

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.results = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=1,  # Analysis is deterministic
            warmup=5,
        )
    
    def get_workload_metadata(self):
        return None
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.results is None:
            return "Results not generated"
        if "baseline" not in self.results or "optimized" not in self.results:
            return "Incomplete results"
        return None

    def get_custom_metrics(self) -> Optional[dict]:
        """Return roofline analysis metrics for ILP optimization comparison."""
        if self.results is None:
            return None
        baseline = self.results.get("baseline", {})
        optimized = self.results.get("optimized", {})
        baseline_time = baseline.get("time_ms", 0.0)
        optimized_time = optimized.get("time_ms", 0.0)
        speedup = baseline_time / optimized_time if optimized_time > 0 else 0.0
        return {
            "roofline.ridge_point": self.results.get("ridge_point", 0.0),
            "roofline.baseline_ai": baseline.get("ai", 0.0),
            "roofline.baseline_tflops": baseline.get("achieved_tflops", 0.0),
            "roofline.baseline_efficiency_pct": baseline.get("efficiency", 0.0),
            "roofline.optimized_ai": optimized.get("ai", 0.0),
            "roofline.optimized_tflops": optimized.get("achieved_tflops", 0.0),
            "roofline.optimized_efficiency_pct": optimized.get("efficiency", 0.0),
            "roofline.speedup": speedup,
        }

    def print_results(self) -> None:
        """Print roofline analysis results."""
        if self.results is None:
            return
        
        print("\n" + "=" * 80)
        print("ROOFLINE ANALYSIS FOR ILP-OPTIMIZED KERNELS")
        print("=" * 80)
        print(f"\nRidge Point: {self.results['ridge_point']:.1f} FLOP/Byte")
        print(f" • AI < {self.results['ridge_point']:.0f}: Memory-bound")
        print(f" • AI > {self.results['ridge_point']:.0f}: Compute-bound\n")
        
        print(f"{'Kernel':<40} {'AI':<12} {'Achieved':<15} {'Efficiency':<12} {'Bottleneck':<15}")
        print("-" * 80)
        for key in ["baseline", "optimized"]:
            r = self.results[key]
            print(f"{r['name']:<40} {r['ai']:<12.2f} {r['achieved_tflops']:<7.2f} TFLOPS {r['efficiency']:<12.1f}% {r['bottleneck']:<15}")
        print("=" * 80)
        
        baseline_time = self.results["baseline"]["time_ms"]
        optimized_time = self.results["optimized"]["time_ms"]
        speedup = baseline_time / optimized_time if optimized_time > 0 else 0
        print(f"\nSpeedup: {speedup:.2f}x ({baseline_time:.2f} ms → {optimized_time:.2f} ms)")
        print("\n  Tip: Tensor cores provide significant speedup for compute-bound GEMM operations")


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return RooflineAnalysisILPBenchmark()
