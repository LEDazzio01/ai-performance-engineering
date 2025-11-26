#!/usr/bin/env python3
"""
GPU Profiling Suite Examples

Demonstrates how to use the profiling tools for performance analysis.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn


# =============================================================================
# Example 1: Basic Profiling with UnifiedProfiler
# =============================================================================

def example_unified_profiler():
    """Basic profiling of GPU code."""
    from tools.profiling import UnifiedProfiler
    
    print("=" * 60)
    print("Example 1: UnifiedProfiler")
    print("=" * 60)
    
    # Skip if no GPU
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping GPU profiling")
        return
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(1024, 4096),
        nn.ReLU(),
        nn.Linear(4096, 1024),
        nn.LayerNorm(1024),
    ).cuda()
    
    input_tensor = torch.randn(32, 1024, device='cuda')
    
    # Create profiler
    profiler = UnifiedProfiler(
        output_dir=Path("/tmp/profiles"),
        enable_trace=True,
        enable_memory=True,
        warmup_iterations=3,
        profile_iterations=10,
    )
    
    # Profile the model
    print("\nProfiling model forward pass...")
    
    with profiler.profile("model_forward") as session:
        for _ in range(10):
            output = model(input_tensor)
    
    # Print results
    print(f"\nResults:")
    print(f"  Total time: {session.total_time_ms:.2f}ms")
    print(f"  GPU time: {session.cuda_time_ms:.2f}ms")
    print(f"  CPU time: {session.cpu_time_ms:.2f}ms")
    print(f"  Peak memory: {session.peak_memory_mb:.1f}MB")
    
    print(f"\nTop 5 kernels:")
    for kernel in session.kernels[:5]:
        print(f"  {kernel.name}: {kernel.cuda_time_us:.0f}μs ({kernel.call_count} calls)")
    
    if session.bottlenecks:
        print(f"\nBottlenecks identified:")
        for b in session.bottlenecks:
            print(f"  ⚠️ {b}")
    
    if session.recommendations:
        print(f"\nRecommendations:")
        for r in session.recommendations:
            print(f"  💡 {r}")
    
    if session.trace_path:
        print(f"\nTrace saved to: {session.trace_path}")


# =============================================================================
# Example 2: Profile a Function
# =============================================================================

def example_profile_function():
    """Profile a specific function with warmup."""
    from tools.profiling import UnifiedProfiler
    
    print("\n" + "=" * 60)
    print("Example 2: Profile Function")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping")
        return
    
    # Define a function to profile
    def matmul_operation(a, b):
        return torch.matmul(a, b)
    
    # Create inputs
    a = torch.randn(1024, 1024, device='cuda')
    b = torch.randn(1024, 1024, device='cuda')
    
    # Profile
    profiler = UnifiedProfiler()
    session = profiler.profile_function(
        matmul_operation,
        a, b,
        name="matmul_1024x1024",
        warmup=5,
        iterations=20,
    )
    
    print(f"\nMatrix multiplication (1024x1024):")
    print(f"  Time per iteration: {session.total_time_ms:.3f}ms")
    print(f"  Peak memory: {session.peak_memory_mb:.1f}MB")


# =============================================================================
# Example 3: Memory Profiling
# =============================================================================

def example_memory_profiling():
    """Track memory usage over time."""
    from tools.profiling import MemoryProfiler
    
    print("\n" + "=" * 60)
    print("Example 3: Memory Profiling")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping")
        return
    
    mem_profiler = MemoryProfiler(
        sample_interval_ms=1.0,
        record_history=True,
    )
    
    print("\nTracking memory during model creation and inference...")
    
    with mem_profiler.track("model_lifecycle"):
        # Create model
        model = nn.Sequential(
            nn.Linear(2048, 8192),
            nn.ReLU(),
            nn.Linear(8192, 2048),
        ).cuda()
        mem_profiler.sample()
        
        # Create input
        x = torch.randn(64, 2048, device='cuda')
        mem_profiler.sample()
        
        # Forward pass
        for _ in range(5):
            output = model(x)
            mem_profiler.sample()
    
    # Analyze
    peak = mem_profiler.get_peak_analysis()
    timeline = mem_profiler.get_timeline()
    
    print(f"\nMemory Analysis:")
    print(f"  Peak allocated: {peak['peak_allocated_mb']:.1f}MB")
    print(f"  Final allocated: {peak['current_allocated_mb']:.1f}MB")
    print(f"  Potential leak: {peak['potential_leak']}")
    
    print(f"\nTimeline ({len(timeline)} samples):")
    for i, point in enumerate(timeline[:5]):
        print(f"  t={point['time_ms']:.1f}ms: {point['allocated_mb']:.1f}MB allocated")
    
    # Export
    mem_profiler.export(Path("/tmp/memory_profile.json"))
    print(f"\nExported to /tmp/memory_profile.json")


# =============================================================================
# Example 4: Flame Graph Generation
# =============================================================================

def example_flame_graph():
    """Generate a flame graph from profiling data."""
    from tools.profiling import FlameGraphGenerator
    
    print("\n" + "=" * 60)
    print("Example 4: Flame Graph Generation")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping")
        return
    
    # Create model and profile
    model = nn.TransformerEncoderLayer(d_model=512, nhead=8).cuda()
    x = torch.randn(32, 10, 512, device='cuda')
    
    print("\nProfiling transformer layer...")
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for _ in range(10):
            output = model(x)
    
    # Generate flame graph
    generator = FlameGraphGenerator(
        min_duration_us=10.0,
        group_small_kernels=True,
    )
    
    flame_data = generator.from_profiler(prof)
    
    # Export
    generator.export(flame_data, Path("/tmp/flame.html"), format="html")
    generator.export(flame_data, Path("/tmp/flame.json"), format="json")
    
    print(f"\nFlame graph generated:")
    print(f"  HTML: /tmp/flame.html")
    print(f"  JSON: /tmp/flame.json")
    print(f"\nOpen /tmp/flame.html in a browser for interactive visualization!")


# =============================================================================
# Example 5: CPU/GPU Timeline
# =============================================================================

def example_timeline():
    """Generate CPU/GPU timeline visualization."""
    from tools.profiling import TimelineGenerator
    
    print("\n" + "=" * 60)
    print("Example 5: CPU/GPU Timeline")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping")
        return
    
    # Create model
    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
    ).cuda()
    
    x = torch.randn(64, 1024, device='cuda')
    
    print("\nCapturing CPU/GPU timeline...")
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
        for _ in range(5):
            output = model(x)
            torch.cuda.synchronize()
    
    # Generate timeline
    generator = TimelineGenerator(min_duration_us=1.0)
    timeline = generator.from_profiler(prof)
    
    print(f"\nTimeline Statistics:")
    print(f"  Total time: {timeline.total_time_us/1000:.2f}ms")
    print(f"  CPU active: {timeline.cpu_active_time_us/1000:.2f}ms")
    print(f"  GPU active: {timeline.gpu_active_time_us/1000:.2f}ms")
    print(f"  CPU/GPU overlap: {timeline.overlap_time_us/1000:.2f}ms")
    print(f"  Streams: {len(timeline.streams)}")
    
    # Export
    generator.generate_html_viewer(timeline, Path("/tmp/timeline.html"))
    print(f"\nTimeline saved to: /tmp/timeline.html")


# =============================================================================
# Example 6: torch.compile Analysis
# =============================================================================

def example_torch_compile():
    """Analyze torch.compile behavior."""
    from tools.profiling.torch_compile import TorchCompileAnalyzer
    
    print("\n" + "=" * 60)
    print("Example 6: torch.compile Analysis")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping")
        return
    
    # Create a model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(1024, 4096)
            self.fc2 = nn.Linear(4096, 1024)
            self.ln = nn.LayerNorm(1024)
        
        def forward(self, x):
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            x = self.ln(x)
            return x
    
    model = SimpleModel().cuda()
    sample_input = torch.randn(32, 1024, device='cuda')
    
    print("\nAnalyzing torch.compile behavior...")
    
    analyzer = TorchCompileAnalyzer(
        backend="inductor",
        mode="default",
    )
    
    report = analyzer.analyze(
        model,
        sample_input,
        warmup=3,
        iterations=10,
    )
    
    print(f"\nResults:")
    print(f"  Speedup: {report.speedup:.2f}x")
    print(f"  Eager time: {report.eager_time_ms:.2f}ms")
    print(f"  Compiled time: {report.compiled_time_ms:.2f}ms")
    print(f"  Compile time: {report.compile_time_ms/1000:.1f}s")
    print(f"  Graph breaks: {report.total_graph_breaks}")
    print(f"  Fusion ratio: {report.fusion_ratio:.1f}x")
    
    if report.graph_breaks:
        print(f"\nGraph breaks:")
        for b in report.graph_breaks[:3]:
            print(f"  ⚠️ {b.reason}")
            print(f"     💡 {b.suggestion}")
    
    print(f"\nRecommendations:")
    for r in report.recommendations:
        print(f"  💡 {r}")
    
    # Export
    analyzer.export_report(report, Path("/tmp/compile_report.html"), format="html")
    print(f"\nReport saved to: /tmp/compile_report.html")


# =============================================================================
# Example 7: Compare Profiling Sessions
# =============================================================================

def example_compare_sessions():
    """Compare baseline vs optimized performance."""
    from tools.profiling import UnifiedProfiler
    
    print("\n" + "=" * 60)
    print("Example 7: Compare Profiling Sessions")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, skipping")
        return
    
    # Baseline model
    baseline_model = nn.Sequential(
        nn.Linear(1024, 4096),
        nn.ReLU(),
        nn.Linear(4096, 1024),
    ).cuda()
    
    # Optimized model (compiled)
    optimized_model = torch.compile(
        nn.Sequential(
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
        ).cuda(),
        mode="reduce-overhead"
    )
    
    x = torch.randn(64, 1024, device='cuda')
    
    profiler = UnifiedProfiler()
    
    print("\nProfiling baseline...")
    baseline = profiler.profile_function(baseline_model, x, name="baseline", warmup=5, iterations=20)
    
    print("Profiling optimized...")
    # Warmup compile
    for _ in range(3):
        optimized_model(x)
    optimized = profiler.profile_function(optimized_model, x, name="optimized", warmup=0, iterations=20)
    
    # Compare
    comparison = profiler.compare_sessions(baseline, optimized)
    
    print(f"\nComparison:")
    print(f"  Speedup: {comparison['speedup']:.2f}x")
    print(f"  Time saved: {comparison['timing_diff']['total_ms']:.2f}ms per iteration")
    print(f"  Memory diff: {comparison['memory_diff']['peak_mb']:.1f}MB")
    
    if comparison['kernel_changes']:
        print(f"\nKernel changes:")
        for change in comparison['kernel_changes'][:5]:
            if 'diff_us' in change:
                print(f"  {change['name']}: {change['diff_us']:.0f}μs saved")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("📊 GPU Profiling Suite Examples\n")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", type=int, choices=[1, 2, 3, 4, 5, 6, 7],
                       help="Run specific example (1-7)")
    parser.add_argument("--all", action="store_true", help="Run all examples")
    args = parser.parse_args()
    
    examples = [
        ("UnifiedProfiler", example_unified_profiler),
        ("Profile Function", example_profile_function),
        ("Memory Profiling", example_memory_profiling),
        ("Flame Graph", example_flame_graph),
        ("CPU/GPU Timeline", example_timeline),
        ("torch.compile Analysis", example_torch_compile),
        ("Compare Sessions", example_compare_sessions),
    ]
    
    if args.example:
        name, func = examples[args.example - 1]
        print(f"Running Example {args.example}: {name}\n")
        func()
    elif args.all:
        for i, (name, func) in enumerate(examples, 1):
            print(f"\n{'#' * 60}")
            print(f"# Example {i}: {name}")
            print(f"{'#' * 60}\n")
            try:
                func()
            except Exception as e:
                print(f"❌ Error: {e}")
    else:
        print("Usage:")
        print("  python profiling_examples.py --example N  # Run specific example")
        print("  python profiling_examples.py --all        # Run all examples")
        print("\nExamples:")
        for i, (name, _) in enumerate(examples, 1):
            print(f"  {i}. {name}")

