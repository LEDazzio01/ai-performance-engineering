"""Example: How to use comprehensive metric comparison.

This example shows how to compare all metrics (not just timing) between
baseline and optimized benchmark results.
"""

from core.benchmark.comparison import (
    compare_and_display_all_metrics,
    compare_all_metrics,
    format_metric_comparison_table,
    format_metric_comparison_summary,
)
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode, BenchmarkConfig


def example_basic_usage(baseline_result, optimized_result):
    """Basic usage - compare and display all metrics automatically."""
    
    # Simplest way: compare and display everything
    comprehensive = compare_and_display_all_metrics(
        baseline_result=baseline_result,
        optimized_result=optimized_result,
        name="My Benchmark",
        format_style="both"  # Shows both table and summary
    )
    
    return comprehensive


def example_with_raw_metrics(baseline_result, optimized_result):
    """Example: Enable raw profiler metrics (opt-in)."""
    
    # Include raw profiler metrics (NCU/NSYS/Torch raw counters)
    comprehensive = compare_and_display_all_metrics(
        baseline_result=baseline_result,
        optimized_result=optimized_result,
        name="My Benchmark",
        include_raw_metrics=True  # Enable raw metrics
    )
    
    return comprehensive


def example_with_chapter_config(baseline_result, optimized_result):
    """Example: Use chapter-specific metrics from performance_targets.py."""
    
    # Enable chapter-specific metrics (e.g., ch07 for memory access patterns)
    comprehensive = compare_and_display_all_metrics(
        baseline_result=baseline_result,
        optimized_result=optimized_result,
        name="My Benchmark",
        chapter="ch07"  # Loads metrics from performance_targets.py for ch07
    )
    
    return comprehensive


def example_with_both_parameters(baseline_result, optimized_result):
    """Example: Combine raw metrics and chapter config."""
    
    # Use both chapter-specific metrics and raw profiler metrics
    comprehensive = compare_and_display_all_metrics(
        baseline_result=baseline_result,
        optimized_result=optimized_result,
        name="My Benchmark",
        include_raw_metrics=True,  # Include raw profiler counters
        chapter="ch10"  # Chapter-specific metrics for Tensor Cores
    )
    
    return comprehensive


def example_custom_formatting(baseline_result, optimized_result):
    """Custom formatting - get comparison object and format yourself."""
    
    # Get comprehensive comparison with new parameters
    comprehensive = compare_all_metrics(
        baseline_result=baseline_result,
        optimized_result=optimized_result,
        regression_threshold_pct=5.0,
        improvement_threshold_pct=5.0,
        include_raw_metrics=False,  # Exclude raw metrics by default
        chapter=None  # No chapter-specific metrics
    )
    
    # Display full table
    print(format_metric_comparison_table(
        comprehensive,
        name="My Benchmark",
        show_only_significant=False  # Show all metrics
    ))
    
    # Display concise summary
    print(format_metric_comparison_summary(comprehensive, name="My Benchmark"))
    
    # Access individual comparisons programmatically
    all_comparisons = comprehensive.get_all_comparisons()
    for comp in all_comparisons:
        if comp.significant_change:
            print(f"{comp.display_name}: {comp.ratio:.2f}x")
    
    return comprehensive


def example_integration_with_harness(baseline_benchmark, optimized_benchmark):
    """Example: Integrate with benchmark harness."""
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(
            iterations=20,
            warmup=5,
            enable_memory_tracking=True  # Important: enables memory metrics
        )
    )
    
    # Run benchmarks
    baseline_result = harness.benchmark(baseline_benchmark)
    optimized_result = harness.benchmark(optimized_benchmark)
    
    # Compare all metrics with chapter-specific configuration
    comprehensive = compare_and_display_all_metrics(
        baseline_result=baseline_result,
        optimized_result=optimized_result,
        name=f"{baseline_benchmark.name} vs {optimized_benchmark.name}",
        format_style="table",
        show_only_significant=True,  # Only show metrics that changed significantly
        include_raw_metrics=False,  # Exclude raw metrics by default
        chapter=None  # Can specify chapter for chapter-specific metrics
    )
    
    # Check for regressions
    regressions = [c for c in comprehensive.get_all_comparisons() if c.regression]
    if regressions:
        print(f"\n⚠️  Warning: {len(regressions)} metric(s) regressed!")
        for r in regressions:
            print(f"  - {r.display_name}: {r.regression_pct:.1f}% worse")
    
    return comprehensive


def example_filter_metrics(baseline_result, optimized_result):
    """Example: Filter to specific metric categories."""
    
    comprehensive = compare_all_metrics(
        baseline_result=baseline_result,
        optimized_result=optimized_result,
        include_raw_metrics=False,  # Exclude raw metrics
        chapter=None  # No chapter-specific metrics
    )
    
    # Get only timing metrics
    timing_metrics = [
        c for c in comprehensive.get_all_comparisons()
        if c.metric_name.startswith("timing.")
    ]
    
    # Get only memory metrics
    memory_metrics = [
        c for c in comprehensive.get_all_comparisons()
        if c.metric_name.startswith("memory.")
    ]
    
    # Get only profiler metrics
    profiler_metrics = [
        c for c in comprehensive.get_all_comparisons()
        if c.metric_name.startswith("profiler_metrics.")
    ]
    
    print(f"Found {len(timing_metrics)} timing metrics")
    print(f"Found {len(memory_metrics)} memory metrics")
    print(f"Found {len(profiler_metrics)} profiler metrics")
    
    return comprehensive

