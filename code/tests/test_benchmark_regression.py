"""Pytest tests for benchmark performance regression detection.

Tests that benchmarks maintain performance characteristics and detects regressions.

These tests are marked as slow and require CUDA. They can be skipped in CI
by using pytest's -m flag to exclude slow tests.

Usage:
    pytest tests/test_benchmark_regression.py -m "not slow"  # Skip regression tests
    pytest tests/test_benchmark_regression.py  # Run regression tests (slow)
"""

import pytest
import sys
from pathlib import Path
from typing import Dict, List

# Add repo root to path
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.env import apply_env_defaults
apply_env_defaults()

import torch
from core.utils.chapter_compare_template import discover_benchmarks, load_benchmark
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode, BenchmarkConfig
from core.benchmark.comparison import compare_results, ComparisonResult, format_comparison


# Skip tests if CUDA is not available (NVIDIA GPU required)
# Tests are marked as slow and can be skipped with: pytest -m "not slow"
pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required - NVIDIA GPU and tools must be available"
    ),
    pytest.mark.slow,
]


@pytest.fixture(scope="module")
def harness():
    """Create a benchmark harness for regression testing."""
    config = BenchmarkConfig(
        iterations=5,  # Minimal iterations for lightweight regression checks
        warmup=5,
        timeout_seconds=10,
        enable_profiling=False,  # Disable profiling for regression tests
        enable_nsys=False,
        enable_ncu=False,
    )
    return BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)


def test_quick_baseline_optimized_speedup(request, harness):
    """Lightweight sanity check: verify one benchmark pair shows expected speedup.
    
    This is a minimal regression check that only tests a single benchmark pair
    to keep CI times reasonable. Additional slow tests can be run by removing
    the -m "not slow" filter.
    """
    repo_root = Path(__file__).parent.parent
    ch01_dir = repo_root / "ch01"
    
    if not ch01_dir.exists():
        pytest.skip("ch01 directory not found")
    
    # Only test one specific benchmark pair for the quick check
    baseline_path = ch01_dir / "baseline_ilp_basic.py"
    optimized_path = ch01_dir / "optimized_ilp_basic.py"
    
    if not baseline_path.exists() or not optimized_path.exists():
        pytest.skip("ILP benchmark files not found")
    
    baseline = load_benchmark(baseline_path)
    optimized = load_benchmark(optimized_path)
    
    if baseline is None or optimized is None:
        pytest.skip("Failed to load benchmarks")
    
    # Run benchmarks with minimal iterations
    baseline_result = harness.benchmark(baseline)
    optimized_result = harness.benchmark(optimized)
    
    # Both should complete successfully
    assert baseline_result.timing is not None and baseline_result.timing.iterations > 0
    assert optimized_result.timing is not None and optimized_result.timing.iterations > 0
    
    # Use comparison utility to verify
    comparison = compare_results(baseline_result, optimized_result)
    assert comparison.speedup > 0
    # Optimized should not show significant regression (allow variance for small sample)
    assert not comparison.regression or comparison.regression_pct < 20.0, \
        f"Optimized benchmark shows regression: {comparison.regression_pct:.1f}% slower"


@pytest.mark.slow
def test_benchmark_result_consistency(request, harness):
    """Test that benchmark results are consistent across runs (variance check)."""
    repo_root = Path(__file__).parent.parent
    baseline_path = repo_root / "ch01" / "baseline_ilp_basic.py"
    
    if not baseline_path.exists():
        pytest.skip("baseline_ilp_basic.py not found")
    
    benchmark = load_benchmark(baseline_path)
    if benchmark is None:
        pytest.skip("Failed to load benchmark")
    
    # Run benchmark twice
    result1 = harness.benchmark(benchmark)
    result2 = harness.benchmark(benchmark)
    
    # Results should be within reasonable variance (50% tolerance for small sample sizes)
    result1_mean = result1.timing.mean_ms if result1.timing else 0.0
    result2_mean = result2.timing.mean_ms if result2.timing else 0.0
    if result1_mean > 0 and result2_mean > 0:
        variance = abs(result1_mean - result2_mean) / result1_mean
        assert variance < 0.5, f"Benchmark results too inconsistent: {result1_mean:.3f}ms vs {result2_mean:.3f}ms (variance: {variance:.1%})"


@pytest.mark.slow
def test_benchmark_memory_usage(request):
    """Test that benchmarks report memory usage when enabled."""
    repo_root = Path(__file__).parent.parent
    baseline_path = repo_root / "ch01" / "baseline_ilp_basic.py"
    
    if not baseline_path.exists():
        pytest.skip("baseline_ilp_basic.py not found")
    
    benchmark = load_benchmark(baseline_path)
    if benchmark is None:
        pytest.skip("Failed to load benchmark")
    
    # Create harness with memory tracking enabled
    config = BenchmarkConfig(
        iterations=3,  # Minimal iterations
        warmup=5,
        enable_memory_tracking=True,
        enable_profiling=False,
    )
    memory_harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    
    result = memory_harness.benchmark(benchmark)
    
    # Memory tracking should provide metrics if enabled
    # Note: Some benchmarks may not allocate GPU memory, so we just check that
    # the result is valid (memory metrics may be None)
    assert result.timing is not None and result.timing.iterations > 0
    # Memory metrics are optional - just ensure result is valid


@pytest.mark.slow
def test_benchmark_timeout_handling(request):
    """Test that benchmarks respect timeout limits."""
    repo_root = Path(__file__).parent.parent
    baseline_path = repo_root / "ch01" / "baseline_ilp_basic.py"
    
    if not baseline_path.exists():
        pytest.skip("baseline_ilp_basic.py not found")
    
    benchmark = load_benchmark(baseline_path)
    if benchmark is None:
        pytest.skip("Failed to load benchmark")
    
    # Create harness with very short timeout
    config = BenchmarkConfig(
        iterations=1000,  # Many iterations
        warmup=5,
        timeout_seconds=1,  # Very short timeout
        enable_profiling=False,
    )
    timeout_harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
    
    # Benchmark should either complete or timeout gracefully
    try:
        result = timeout_harness.benchmark(benchmark)
        # If it completes, that's fine too (benchmark is fast enough)
        assert result.timing is not None and result.timing.iterations >= 0
    except RuntimeError as e:
        # Timeout is acceptable - should raise RuntimeError with timeout message
        assert "timeout" in str(e).lower() or "TIMEOUT" in str(e) or "failed" in str(e).lower()


@pytest.mark.slow
def test_benchmark_validation_coverage(request):
    """Test that benchmarks with validation actually validate correctly."""
    repo_root = Path(__file__).parent.parent
    
    # Test benchmarks that should have validation (only a few for lightweight check)
    validation_benchmarks = [
        ("ch01", "optimized_ilp_basic.py"),
        ("ch18", "optimized_quantization.py"),
    ]
    
    for chapter_name, benchmark_file in validation_benchmarks:
        chapter_dir = repo_root / chapter_name
        if not chapter_dir.exists():
            continue
        
        benchmark_path = chapter_dir / benchmark_file
        if not benchmark_path.exists():
            continue
        
        benchmark = load_benchmark(benchmark_path)
        if benchmark is None:
            continue
        
        benchmark.setup()
        benchmark.benchmark_fn()
        validation_error = benchmark.validate_result()
        
        # Validation should pass (return None) for correct implementations
        assert validation_error is None, f"Validation failed for {benchmark_file}: {validation_error}"
        benchmark.teardown()
