#!/usr/bin/env python3
"""
Roofline Analysis Tool for NVIDIA B200

Measures kernel performance and plots on roofline model to identify
memory-bound vs compute-bound bottlenecks.

Usage:
    python3 roofline_analysis.py
"""

import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Callable


class RooflineAnalyzer:
    """Roofline model analyzer for GPU kernels."""
    
    def __init__(self, peak_bandwidth_gbs: float = 8000, peak_compute_tflops: float = 2000):
        """
        Initialize roofline analyzer.
        
        Args:
            peak_bandwidth_gbs: Peak memory bandwidth in GB/s (B200: 8000)
            peak_compute_tflops: Peak compute in TFLOPS (B200 FP16: 2000)
        """
        self.peak_bandwidth = peak_bandwidth_gbs  # GB/s
        self.peak_compute = peak_compute_tflops   # TFLOPS
        self.ridge_point = peak_compute / peak_bandwidth  # FLOP/Byte
        
        print(f"Roofline Analyzer initialized for NVIDIA B200:")
        print(f"  Peak Memory Bandwidth: {self.peak_bandwidth} GB/s")
        print(f"  Peak Compute (FP16): {self.peak_compute} TFLOPS")
        print(f"  Ridge Point: {self.ridge_point:.1f} FLOP/Byte")
        print()
    
    def measure_kernel(self, fn: Callable, *args, warmup: int = 10, iterations: int = 100) -> float:
        """
        Measure kernel execution time.
        
        Args:
            fn: Function to benchmark
            args: Arguments to pass to function
            warmup: Number of warmup iterations
            iterations: Number of measurement iterations
            
        Returns:
            Average execution time in seconds
        """
        # Warmup
        for _ in range(warmup):
            fn(*args)
        
        # Measure
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            fn(*args)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        return elapsed / iterations
    
    def calculate_ai(self, flops: float, bytes_accessed: float) -> float:
        """
        Calculate arithmetic intensity.
        
        Args:
            flops: Total floating point operations
            bytes_accessed: Total bytes transferred
            
        Returns:
            Arithmetic intensity in FLOP/Byte
        """
        return flops / bytes_accessed
    
    def predict_performance(self, ai: float) -> float:
        """
        Predict maximum achievable performance given AI.
        
        Args:
            ai: Arithmetic intensity in FLOP/Byte
            
        Returns:
            Maximum performance in TFLOPS
        """
        # Memory-bound ceiling
        memory_bound = self.peak_bandwidth * ai / 1000  # Convert to TFLOPS
        
        # Actual ceiling is minimum of memory and compute bounds
        return min(memory_bound, self.peak_compute)
    
    def plot_roofline(self, kernels: List[Dict], output_file: str = 'roofline_analysis.png'):
        """
        Plot kernels on roofline model.
        
        Args:
            kernels: List of dicts with keys: 'name', 'ai', 'achieved_tflops'
            output_file: Output filename for plot
        """
        # AI range for plotting
        ai_range = np.logspace(-2, 3, 1000)
        
        # Memory-bound ceiling: Performance = Bandwidth × AI
        memory_bound = self.peak_bandwidth * ai_range / 1000  # Convert to TFLOPS
        
        # Compute ceiling: Constant peak
        compute_bound = np.full_like(ai_range, self.peak_compute)
        
        # Actual roofline: Minimum of both ceilings
        roofline = np.minimum(memory_bound, compute_bound)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot roofline
        ax.loglog(ai_range, roofline, 'k-', linewidth=2.5, label='B200 Roofline')
        
        # Plot ridge point
        ax.axvline(x=self.ridge_point, color='gray', linestyle='--', linewidth=1.5,
                   label=f'Ridge Point ({self.ridge_point:.0f} FLOP/Byte)')
        
        # Add shaded regions
        ax.axvspan(0.01, self.ridge_point, alpha=0.1, color='red', 
                   label='Memory-Bound Region')
        ax.axvspan(self.ridge_point, 1000, alpha=0.1, color='green',
                   label='Compute-Bound Region')
        
        # Plot kernels
        colors = plt.cm.tab10(np.linspace(0, 1, len(kernels)))
        for i, kernel in enumerate(kernels):
            ai = kernel['ai']
            achieved = kernel['achieved_tflops']
            name = kernel['name']
            
            # Plot point
            ax.loglog(ai, achieved, 'o', markersize=12, color=colors[i], 
                     markeredgecolor='black', markeredgewidth=1.5)
            
            # Calculate efficiency
            max_achievable = self.predict_performance(ai)
            efficiency = (achieved / max_achievable * 100) if max_achievable > 0 else 0
            
            # Add label with efficiency
            label_text = f"{name}\n({efficiency:.1f}% of peak)"
            ax.text(ai * 1.2, achieved, label_text, fontsize=9, 
                   verticalalignment='center')
        
        # Formatting
        ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Performance (TFLOPS)', fontsize=13, fontweight='bold')
        ax.set_title('Roofline Model - NVIDIA B200 GPU\n'
                    'Memory Bandwidth: 8 TB/s | Peak Compute (FP16): 2000 TFLOPS',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim([0.01, self.peak_compute * 1.5])
        ax.set_xlim([0.01, 1000])
        
        # Save
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Roofline plot saved to {output_file}")
        plt.close()


def benchmark_vector_operations(analyzer: RooflineAnalyzer) -> List[Dict]:
    """Benchmark simple vector operations."""
    results = []
    
    N = 10_000_000
    print(f"Benchmarking vector operations (N={N:,})...")
    
    # Vector Add: out = a + b
    print("  • Vector Add...")
    a = torch.randn(N, device='cuda', dtype=torch.float32)
    b = torch.randn(N, device='cuda', dtype=torch.float32)
    
    time_add = analyzer.measure_kernel(lambda: a + b)
    flops_add = N  # 1 FLOP per element
    bytes_add = 3 * N * 4  # 2 reads + 1 write, 4 bytes each
    ai_add = analyzer.calculate_ai(flops_add, bytes_add)
    achieved_tflops_add = (flops_add / time_add) / 1e12
    
    results.append({
        'name': 'Vector Add',
        'ai': ai_add,
        'achieved_tflops': achieved_tflops_add,
        'time_ms': time_add * 1000
    })
    
    # Vector Multiply: out = a * b
    print("  • Vector Multiply...")
    time_mul = analyzer.measure_kernel(lambda: a * b)
    flops_mul = N
    bytes_mul = 3 * N * 4
    ai_mul = analyzer.calculate_ai(flops_mul, bytes_mul)
    achieved_tflops_mul = (flops_mul / time_mul) / 1e12
    
    results.append({
        'name': 'Vector Multiply',
        'ai': ai_mul,
        'achieved_tflops': achieved_tflops_mul,
        'time_ms': time_mul * 1000
    })
    
    # Fused: out = (a + b) * (a - b)
    print("  • Fused Add-Sub-Mul...")
    time_fused = analyzer.measure_kernel(lambda: (a + b) * (a - b))
    flops_fused = 3 * N  # 1 add + 1 sub + 1 mul
    bytes_fused = 3 * N * 4  # 2 reads + 1 write (intermediate results in registers!)
    ai_fused = analyzer.calculate_ai(flops_fused, bytes_fused)
    achieved_tflops_fused = (flops_fused / time_fused) / 1e12
    
    results.append({
        'name': 'Fused (Add-Sub-Mul)',
        'ai': ai_fused,
        'achieved_tflops': achieved_tflops_fused,
        'time_ms': time_fused * 1000
    })
    
    return results


def benchmark_matrix_operations(analyzer: RooflineAnalyzer) -> List[Dict]:
    """Benchmark matrix operations."""
    results = []
    
    # Small matrix multiply
    M = 512
    print(f"\nBenchmarking matrix operations (M={M})...")
    print("  • MatMul (512×512)...")
    A_small = torch.randn(M, M, device='cuda', dtype=torch.float16)
    B_small = torch.randn(M, M, device='cuda', dtype=torch.float16)
    
    time_mm_small = analyzer.measure_kernel(lambda: torch.mm(A_small, B_small))
    flops_mm_small = 2 * M**3  # 2N³ for matrix multiply
    bytes_mm_small = 3 * M**2 * 2  # 3 matrices, 2 bytes (FP16)
    ai_mm_small = analyzer.calculate_ai(flops_mm_small, bytes_mm_small)
    achieved_tflops_mm_small = (flops_mm_small / time_mm_small) / 1e12
    
    results.append({
        'name': 'MatMul (512)',
        'ai': ai_mm_small,
        'achieved_tflops': achieved_tflops_mm_small,
        'time_ms': time_mm_small * 1000
    })
    
    # Medium matrix multiply
    M = 2048
    print(f"  • MatMul (2048×2048)...")
    A_med = torch.randn(M, M, device='cuda', dtype=torch.float16)
    B_med = torch.randn(M, M, device='cuda', dtype=torch.float16)
    
    time_mm_med = analyzer.measure_kernel(lambda: torch.mm(A_med, B_med), iterations=20)
    flops_mm_med = 2 * M**3
    bytes_mm_med = 3 * M**2 * 2
    ai_mm_med = analyzer.calculate_ai(flops_mm_med, bytes_mm_med)
    achieved_tflops_mm_med = (flops_mm_med / time_mm_med) / 1e12
    
    results.append({
        'name': 'MatMul (2048)',
        'ai': ai_mm_med,
        'achieved_tflops': achieved_tflops_mm_med,
        'time_ms': time_mm_med * 1000
    })
    
    # Large matrix multiply (closer to ridge point)
    M = 4096
    print(f"  • MatMul (4096×4096)...")
    A_large = torch.randn(M, M, device='cuda', dtype=torch.float16)
    B_large = torch.randn(M, M, device='cuda', dtype=torch.float16)
    
    time_mm_large = analyzer.measure_kernel(lambda: torch.mm(A_large, B_large), iterations=10)
    flops_mm_large = 2 * M**3
    bytes_mm_large = 3 * M**2 * 2
    ai_mm_large = analyzer.calculate_ai(flops_mm_large, bytes_mm_large)
    achieved_tflops_mm_large = (flops_mm_large / time_mm_large) / 1e12
    
    results.append({
        'name': 'MatMul (4096)',
        'ai': ai_mm_large,
        'achieved_tflops': achieved_tflops_mm_large,
        'time_ms': time_mm_large * 1000
    })
    
    return results


def print_results(results: List[Dict], analyzer: RooflineAnalyzer):
    """Print detailed results."""
    print("\n" + "="*80)
    print("ROOFLINE ANALYSIS RESULTS")
    print("="*80)
    print(f"{'Kernel':<25} {'AI (FLOP/B)':<15} {'Achieved':<15} {'Time':<12} {'Status':<15}")
    print("-"*80)
    
    for r in results:
        ai = r['ai']
        achieved = r['achieved_tflops']
        max_achievable = analyzer.predict_performance(ai)
        efficiency = (achieved / max_achievable * 100) if max_achievable > 0 else 0
        
        # Determine bottleneck
        if ai < analyzer.ridge_point:
            status = "Memory-bound"
        else:
            status = "Compute-bound"
        
        print(f"{r['name']:<25} {ai:<15.4f} {achieved:<7.2f} TFLOPS {r['time_ms']:<8.2f} ms {status:<15}")
        print(f"{'':25} {'':15} ({efficiency:.1f}% of peak)")
    
    print("="*80)
    print(f"\nRidge Point: {analyzer.ridge_point:.1f} FLOP/Byte")
    print(f"  • AI < {analyzer.ridge_point:.0f}: Memory-bound → Optimize memory traffic")
    print(f"  • AI > {analyzer.ridge_point:.0f}: Compute-bound → Optimize compute")


def main():
    """Main roofline analysis."""
    print("="*80)
    print("ROOFLINE ANALYSIS FOR NVIDIA B200")
    print("="*80)
    print()
    
    # Initialize analyzer
    analyzer = RooflineAnalyzer(
        peak_bandwidth_gbs=8000,   # 8 TB/s
        peak_compute_tflops=2000   # 2000 TFLOPS FP16
    )
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    print(f"Using GPU: {torch.cuda.get_device_name(0)}\n")
    
    # Benchmark operations
    vector_results = benchmark_vector_operations(analyzer)
    matrix_results = benchmark_matrix_operations(analyzer)
    
    all_results = vector_results + matrix_results
    
    # Print results
    print_results(all_results, analyzer)
    
    # Plot roofline
    print("\nGenerating roofline plot...")
    analyzer.plot_roofline(all_results)
    
    print("\n" + "="*80)
    print("✅ Analysis complete! View roofline_analysis.png for visualization.")
    print("="*80)


if __name__ == '__main__':
    main()

