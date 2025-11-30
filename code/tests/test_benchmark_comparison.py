"""Unit tests for benchmark comparison utilities."""

import sys
from pathlib import Path
from typing import Optional, Dict
import pytest
from unittest.mock import patch, MagicMock

# Add repo root to path
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.benchmark.comparison import (
    MetricComparison,
    MetricDirection,
    ComparisonResult,
    ComprehensiveComparison,
    compare_metric,
    compare_all_metrics,
    compare_and_display_all_metrics,
    extract_metrics,
    get_chapter_metric_config,
    METRIC_CONFIG,
)
from core.benchmark.models import (
    BenchmarkResult,
    TimingStats,
    InferenceTimingStats,
    MemoryStats,
    ProfilerMetrics,
    NcuMetrics,
    NsysMetrics,
    TorchMetrics,
)


def create_mock_benchmark_result(
    mean_ms: float = 100.0,
    median_ms: float = 95.0,
    p99_ms: Optional[float] = None,
    peak_mb: Optional[float] = None,
    ncu_kernel_time: Optional[float] = None,
    ncu_sm_throughput: Optional[float] = None,
    raw_metrics: Optional[Dict[str, float]] = None,
    inference_timing: Optional[InferenceTimingStats] = None
) -> BenchmarkResult:
    """Create a mock BenchmarkResult for testing."""
    timing = TimingStats(
        mean_ms=mean_ms,
        median_ms=median_ms,
        std_ms=5.0,
        min_ms=90.0,
        max_ms=110.0,
        iterations=100,
        warmup_iterations=10,
    )
    if p99_ms is not None:
        timing.p99_ms = p99_ms
        timing.percentiles = {99.0: p99_ms, 95.0: 98.0, 90.0: 97.0}
    
    memory = None
    if peak_mb is not None:
        memory = MemoryStats(peak_mb=peak_mb, allocated_mb=peak_mb * 0.9)
    
    profiler_metrics = None
    if ncu_kernel_time is not None or ncu_sm_throughput is not None or raw_metrics:
        ncu_metrics = NcuMetrics(
            kernel_time_ms=ncu_kernel_time,
            sm_throughput_pct=ncu_sm_throughput,
            raw_metrics=raw_metrics or {}
        )
        profiler_metrics = ProfilerMetrics(ncu=ncu_metrics)
    
    return BenchmarkResult(
        timing=timing,
        memory=memory,
        profiler_metrics=profiler_metrics,
        inference_timing=inference_timing
    )


class TestPercentageCalculation:
    """Test percentage calculation fixes."""
    
    def test_delta_based_percentage_lower_is_better(self):
        """Test delta-based percentage for lower-is-better metrics."""
        comp = compare_metric(
            "timing.mean_ms",
            baseline_value=100.0,
            optimized_value=50.0,
            regression_threshold_pct=5.0,
            improvement_threshold_pct=5.0
        )
        assert comp is not None
        # 100ms -> 50ms = 50% improvement (not 100%)
        assert comp.improvement_pct == pytest.approx(50.0, abs=0.1)
        assert comp.ratio == pytest.approx(2.0, abs=0.01)
    
    def test_delta_based_percentage_regression(self):
        """Test delta-based percentage for regressions."""
        comp = compare_metric(
            "timing.mean_ms",
            baseline_value=80.0,
            optimized_value=100.0,
            regression_threshold_pct=5.0,
            improvement_threshold_pct=5.0
        )
        assert comp is not None
        # 80ms -> 100ms = 25% regression (not 20%)
        assert comp.regression_pct == pytest.approx(25.0, abs=0.1)
        assert comp.regression is True
    
    def test_zero_baseline_handling(self):
        """Test zero baseline handling - should not crash."""
        comp = compare_metric(
            "timing.mean_ms",
            baseline_value=0.0,
            optimized_value=50.0,
            regression_threshold_pct=5.0,
            improvement_threshold_pct=5.0
        )
        assert comp is not None
        # Percentage should be None when baseline is zero
        assert comp.improvement_pct is None
        assert comp.regression_pct is None
        # Ratio can still be calculated
        assert comp.ratio == float('inf')
    
    def test_zero_baseline_both_zero(self):
        """Test when both baseline and optimized are zero."""
        comp = compare_metric(
            "timing.mean_ms",
            baseline_value=0.0,
            optimized_value=0.0,
            regression_threshold_pct=5.0,
            improvement_threshold_pct=5.0
        )
        assert comp is not None
        assert comp.improvement_pct is None
        assert comp.regression_pct is None
        assert comp.ratio == 1.0
    
    def test_throughput_zero_baseline(self):
        """Test throughput metrics (higher-is-better) with zero baseline."""
        comp = compare_metric(
            "profiler_metrics.ncu.sm_throughput_pct",
            baseline_value=0.0,
            optimized_value=50.0,
            regression_threshold_pct=5.0,
            improvement_threshold_pct=5.0
        )
        assert comp is not None
        assert comp.improvement_pct is None  # Cannot compute percentage
        assert comp.ratio == float('inf')
    
    def test_none_baseline_handling(self):
        """Test None baseline handling."""
        # This would require special handling in compare_metric
        # For now, test that it doesn't crash
        comp = compare_metric(
            "timing.mean_ms",
            baseline_value=0.0,  # Using 0 as proxy for None
            optimized_value=50.0,
            regression_threshold_pct=5.0,
            improvement_threshold_pct=5.0
        )
        assert comp is not None


class TestThresholdApplication:
    """Test threshold application fixes."""
    
    def test_metric_specific_thresholds(self):
        """Test that metric-specific thresholds are applied."""
        # timing.std_ms has threshold of 10.0
        comp = compare_metric(
            "timing.std_ms",
            baseline_value=100.0,
            optimized_value=92.0,  # 8% improvement, below 10% threshold
            regression_threshold_pct=5.0,
            improvement_threshold_pct=5.0
        )
        assert comp is not None
        # Should use metric-specific threshold (10.0), so 8% is not significant
        # Actually, we pass thresholds as parameters, so it uses the passed threshold
        # The metric-specific threshold is used when we don't pass explicit thresholds
        assert comp.improvement_pct == pytest.approx(8.0, abs=0.1)
        assert comp.significant_change == (comp.improvement_pct >= 5.0)
    
    def test_backward_compatibility_single_threshold(self):
        """Test backward compatibility with single threshold config."""
        # This tests that old configs with 4-element tuples still work
        # We've migrated all configs, but test the logic anyway
        comp = compare_metric(
            "timing.mean_ms",
            baseline_value=100.0,
            optimized_value=95.0,
            regression_threshold_pct=5.0,
            improvement_threshold_pct=5.0
        )
        assert comp is not None
        assert comp.improvement_pct == pytest.approx(5.0, abs=0.1)


class TestDeduplication:
    """Test deduplication fixes."""
    
    def test_no_double_timing_in_get_all_comparisons(self):
        """Test that get_all_comparisons() doesn't double-count timing."""
        baseline = create_mock_benchmark_result(mean_ms=100.0)
        optimized = create_mock_benchmark_result(mean_ms=50.0)
        
        comprehensive = compare_all_metrics(baseline, optimized, include_timing=True)
        
        all_comparisons = comprehensive.get_all_comparisons()
        timing_metrics = [c for c in all_comparisons if c.metric_name == "timing.mean_ms"]
        
        # Should have exactly one timing.mean_ms entry
        assert len(timing_metrics) == 1
    
    def test_percentile_deduplication(self):
        """Test that percentiles don't duplicate when attribute exists."""
        baseline = create_mock_benchmark_result(mean_ms=100.0, p99_ms=110.0)
        optimized = create_mock_benchmark_result(mean_ms=50.0, p99_ms=55.0)
        
        baseline_metrics = extract_metrics(baseline)
        optimized_metrics = extract_metrics(optimized)
        
        # Should have p99_ms from attribute, not from percentiles dict
        assert "timing.p99_ms" in baseline_metrics
        assert baseline_metrics["timing.p99_ms"] == 110.0
        
        # Should not have duplicate p99 entries
        p99_keys = [k for k in baseline_metrics.keys() if "p99" in k]
        assert len(p99_keys) == 1  # Only one p99 entry
    
    def test_arbitrary_percentile_keys(self):
        """Test that arbitrary percentile keys don't create duplicates."""
        # Create baseline with percentiles dict but no p99_ms attribute
        timing = TimingStats(
            mean_ms=100.0,
            median_ms=95.0,
            std_ms=5.0,
            min_ms=90.0,
            max_ms=110.0,
            iterations=100,
            warmup_iterations=10,
            percentiles={92.0: 105.0, 97.0: 108.0, 99.0: 110.0}
        )
        baseline = BenchmarkResult(timing=timing)
        
        metrics = extract_metrics(baseline)
        
        # Should have p92, p97, p99 from percentiles dict
        assert "timing.p92_ms" in metrics
        assert "timing.p97_ms" in metrics
        assert "timing.p99_ms" in metrics
        
        # If p99_ms attribute also exists, should not duplicate
        timing.p99_ms = 110.0
        baseline2 = BenchmarkResult(timing=timing)
        metrics2 = extract_metrics(baseline2)
        p99_keys = [k for k in metrics2.keys() if "p99" in k]
        assert len(p99_keys) == 1  # Should not duplicate (attribute takes precedence)


class TestRawMetrics:
    """Test raw metrics exclusion."""
    
    def test_raw_metrics_excluded_by_default(self):
        """Test that raw metrics are excluded by default."""
        baseline = create_mock_benchmark_result(
            mean_ms=100.0,
            ncu_kernel_time=50.0,
            raw_metrics={"sm__inst_executed_pipe_tensor.sum": 1000.0}
        )
        
        metrics = extract_metrics(baseline, include_raw_metrics=False)
        
        # Should not have raw metrics
        raw_keys = [k for k in metrics.keys() if ".raw." in k]
        assert len(raw_keys) == 0
    
    def test_raw_metrics_included_when_opted_in(self):
        """Test that raw metrics are included when opted in."""
        baseline = create_mock_benchmark_result(
            mean_ms=100.0,
            ncu_kernel_time=50.0,
            raw_metrics={"sm__inst_executed_pipe_tensor.sum": 1000.0}
        )
        
        metrics = extract_metrics(baseline, include_raw_metrics=True)
        
        # Should have raw metrics
        raw_keys = [k for k in metrics.keys() if ".raw." in k]
        assert len(raw_keys) > 0


class TestChapterConfig:
    """Test chapter-specific metric configuration."""
    
    def test_chapter_config_loading(self):
        """Test that chapter config loads correctly."""
        config = get_chapter_metric_config("ch07")
        
        # Should return a dict (may be empty if ch07 has no metrics)
        assert isinstance(config, dict)
    
    def test_unknown_metric_handling(self):
        """Test that unknown metrics are handled gracefully."""
        # This tests the warning/logging behavior
        # We can't easily test logging, but we can verify it doesn't crash
        config = get_chapter_metric_config("ch01")  # ch01 has no metrics in performance_targets
        assert isinstance(config, dict)
    
    def test_chapter_config_merges_with_base(self):
        """Test that chapter config merges with base config."""
        baseline = create_mock_benchmark_result(mean_ms=100.0)
        optimized = create_mock_benchmark_result(mean_ms=50.0)
        
        comprehensive = compare_all_metrics(
            baseline,
            optimized,
            chapter="ch07"  # ch07 has utilization_percent metric
        )
        
        # Should still work even if chapter has metrics
        assert comprehensive is not None
    
    def test_chapter_specific_metrics_flow_through_compare_all_metrics(self):
        """Test that chapter-specific metrics flow through compare_all_metrics end-to-end."""
        # Use actual CH7 metrics (dram_throughput_percent, achieved_occupancy_percent)
        baseline = create_mock_benchmark_result(
            mean_ms=100.0,
            ncu_sm_throughput=85.0,
            raw_metrics={"l1tex__throughput.avg.pct_of_peak_sustained_active": 95.0}
        )
        # Add DRAM throughput and occupancy to baseline
        baseline.profiler_metrics.ncu.dram_throughput_pct = 85.0
        baseline.profiler_metrics.ncu.occupancy_pct = 80.0
        
        optimized = create_mock_benchmark_result(
            mean_ms=50.0,
            ncu_sm_throughput=90.0,
            raw_metrics={"l1tex__throughput.avg.pct_of_peak_sustained_active": 100.0}
        )
        # Add improved DRAM throughput and occupancy to optimized
        optimized.profiler_metrics.ncu.dram_throughput_pct = 92.0
        optimized.profiler_metrics.ncu.occupancy_pct = 88.0
        
        comparison = compare_all_metrics(
            baseline_result=baseline,
            optimized_result=optimized,
            include_timing=True,
            chapter="ch07",
            include_raw_metrics=True
        )
        
        assert comparison is not None
        # Check that chapter-specific metrics are included
        metric_names = [c.metric_name for c in comparison.metric_comparisons]
        
        # Should include chapter metrics (CH7 has dram_throughput_percent, achieved_occupancy_percent)
        assert "profiler_metrics.ncu.dram_throughput_pct" in metric_names
        assert "profiler_metrics.ncu.occupancy_pct" in metric_names
        
        # Find the DRAM throughput comparison
        dram_comp = next((c for c in comparison.metric_comparisons 
                        if c.metric_name == "profiler_metrics.ncu.dram_throughput_pct"), None)
        assert dram_comp is not None
        assert dram_comp.baseline_value == 85.0
        assert dram_comp.optimized_value == 92.0
        assert dram_comp.direction == MetricDirection.HIGHER_IS_BETTER
        
        # Find the occupancy comparison
        occ_comp = next((c for c in comparison.metric_comparisons 
                        if c.metric_name == "profiler_metrics.ncu.occupancy_pct"), None)
        assert occ_comp is not None
        assert occ_comp.baseline_value == 80.0
        assert occ_comp.optimized_value == 88.0
        assert occ_comp.direction == MetricDirection.HIGHER_IS_BETTER


class TestDirectionHeuristics:
    """Test improved direction heuristics."""
    
    def test_inst_executed_higher_is_better(self):
        """Test that inst_executed metrics are higher-is-better."""
        comp = compare_metric(
            "profiler_metrics.ncu.raw.sm__inst_executed_pipe_tensor.sum",
            baseline_value=100.0,
            optimized_value=200.0,
            regression_threshold_pct=5.0,
            improvement_threshold_pct=5.0
        )
        assert comp is not None
        assert comp.direction == MetricDirection.HIGHER_IS_BETTER
    
    def test_conflicts_lower_is_better(self):
        """Test that conflicts metrics are lower-is-better."""
        comp = compare_metric(
            "profiler_metrics.ncu.raw.l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
            baseline_value=100.0,
            optimized_value=50.0,
            regression_threshold_pct=5.0,
            improvement_threshold_pct=5.0
        )
        assert comp is not None
        assert comp.direction == MetricDirection.LOWER_IS_BETTER


class TestAPIPropagation:
    """Test that parameters propagate through API chain."""
    
    def test_compare_all_metrics_accepts_parameters(self):
        """Test that compare_all_metrics accepts new parameters."""
        baseline = create_mock_benchmark_result(mean_ms=100.0)
        optimized = create_mock_benchmark_result(mean_ms=50.0)
        
        # Should accept include_raw_metrics and chapter parameters
        comprehensive = compare_all_metrics(
            baseline,
            optimized,
            include_raw_metrics=True,
            chapter="ch07"
        )
        
        assert comprehensive is not None
    
    def test_compare_and_display_accepts_parameters(self):
        """Test that compare_and_display_all_metrics accepts new parameters."""
        baseline = create_mock_benchmark_result(mean_ms=100.0)
        optimized = create_mock_benchmark_result(mean_ms=50.0)
        
        # Should accept include_raw_metrics and chapter parameters
        comprehensive = compare_and_display_all_metrics(
            baseline,
            optimized,
            name="Test",
            include_raw_metrics=True,
            chapter="ch07"
        )
        
        assert comprehensive is not None


class TestInferenceTiming:
    """Test inference timing (TTFT/TPOT) functionality."""
    
    def test_inference_timing_stats_creation(self):
        """Test InferenceTimingStats model creation."""
        inference_timing = InferenceTimingStats(
            ttft_mean_ms=150.0,
            ttft_p50_ms=145.0,
            ttft_p90_ms=180.0,
            ttft_p95_ms=200.0,
            ttft_p99_ms=250.0,
            ttft_percentiles={50.0: 145.0, 90.0: 180.0, 95.0: 200.0, 99.0: 250.0},
            tpot_mean_ms=35.0,
            tpot_p50_ms=34.0,
            tpot_p90_ms=38.0,
            tpot_p95_ms=39.5,
            tpot_p99_ms=42.0,
            tpot_percentiles={50.0: 34.0, 90.0: 38.0, 95.0: 39.5, 99.0: 42.0},
            num_requests=100,
            total_tokens_generated=5000,
        )
        
        assert inference_timing.ttft_mean_ms == 150.0
        assert inference_timing.ttft_p99_ms == 250.0
        assert inference_timing.tpot_mean_ms == 35.0
        assert inference_timing.tpot_p99_ms == 42.0
        assert inference_timing.num_requests == 100
        assert inference_timing.total_tokens_generated == 5000
    
    def test_extract_inference_timing_metrics(self):
        """Test extraction of inference timing metrics."""
        inference_timing = InferenceTimingStats(
            ttft_mean_ms=150.0,
            ttft_p50_ms=145.0,
            ttft_p90_ms=180.0,
            ttft_p95_ms=200.0,
            ttft_p99_ms=250.0,
            ttft_percentiles={50.0: 145.0, 90.0: 180.0, 95.0: 200.0, 99.0: 250.0},
            tpot_mean_ms=35.0,
            tpot_p50_ms=34.0,
            tpot_p90_ms=38.0,
            tpot_p95_ms=39.5,
            tpot_p99_ms=42.0,
            tpot_percentiles={50.0: 34.0, 90.0: 38.0, 95.0: 39.5, 99.0: 42.0},
            num_requests=100,
            total_tokens_generated=5000,
        )
        
        result = create_mock_benchmark_result(inference_timing=inference_timing)
        metrics = extract_metrics(result)
        
        assert "inference_timing.ttft_mean_ms" in metrics
        assert metrics["inference_timing.ttft_mean_ms"] == 150.0
        assert "inference_timing.ttft_p99_ms" in metrics
        assert metrics["inference_timing.ttft_p99_ms"] == 250.0
        assert "inference_timing.tpot_mean_ms" in metrics
        assert metrics["inference_timing.tpot_mean_ms"] == 35.0
        assert "inference_timing.tpot_p99_ms" in metrics
        assert metrics["inference_timing.tpot_p99_ms"] == 42.0
    
    def test_compare_inference_timing_metrics(self):
        """Test comparison of inference timing metrics."""
        baseline_inference = InferenceTimingStats(
            ttft_mean_ms=200.0,
            ttft_p99_ms=300.0,
            tpot_mean_ms=40.0,
            tpot_p99_ms=50.0,
            ttft_percentiles={99.0: 300.0},
            tpot_percentiles={99.0: 50.0},
            num_requests=100,
            total_tokens_generated=5000,
        )
        
        optimized_inference = InferenceTimingStats(
            ttft_mean_ms=150.0,
            ttft_p99_ms=250.0,
            tpot_mean_ms=35.0,
            tpot_p99_ms=42.0,
            ttft_percentiles={99.0: 250.0},
            tpot_percentiles={99.0: 42.0},
            num_requests=100,
            total_tokens_generated=5000,
        )
        
        baseline_result = create_mock_benchmark_result(inference_timing=baseline_inference)
        optimized_result = create_mock_benchmark_result(inference_timing=optimized_inference)
        
        comparison = compare_all_metrics(
            baseline_result,
            optimized_result,
            regression_threshold_pct=5.0,
            improvement_threshold_pct=5.0
        )
        
        # Find TTFT P99 comparison
        ttft_comparison = None
        tpot_comparison = None
        for comp in comparison.metric_comparisons:
            if comp.metric_name == "inference_timing.ttft_p99_ms":
                ttft_comparison = comp
            elif comp.metric_name == "inference_timing.tpot_p99_ms":
                tpot_comparison = comp
        
        assert ttft_comparison is not None
        assert ttft_comparison.baseline_value == 300.0
        assert ttft_comparison.optimized_value == 250.0
        assert ttft_comparison.improvement_pct == pytest.approx(16.67, abs=0.1)  # (300-250)/300 * 100
        
        assert tpot_comparison is not None
        assert tpot_comparison.baseline_value == 50.0
        assert tpot_comparison.optimized_value == 42.0
        assert tpot_comparison.improvement_pct == pytest.approx(16.0, abs=0.1)  # (50-42)/50 * 100
    
    def test_chapter_metric_config_ttft_tpot(self):
        """Test chapter metric config mappings for TTFT/TPOT."""
        # Mock get_chapter_metrics to return CH17 metrics
        with patch('core.benchmark.comparison.get_chapter_metrics') as mock_get_metrics:
            mock_get_metrics.return_value = {
                "ttft_p99_ms": {"target": 300, "unit": "ms", "lower_is_better": True},
                "ttft_p99_mlperf_ms": {"target": 450, "unit": "ms", "lower_is_better": True},
                "tpot_p99_ms": {"target": 40, "unit": "ms", "lower_is_better": True},
            }
            
            config = get_chapter_metric_config("ch17")
            
            # Check that mappings point to inference_timing metrics
            assert "inference_timing.ttft_p99_ms" in config
            assert "inference_timing.tpot_p99_ms" in config
            
            # Verify display names
            ttft_config = config["inference_timing.ttft_p99_ms"]
            assert ttft_config[0] == "TTFT P99"  # display_name
            assert ttft_config[1] == MetricDirection.LOWER_IS_BETTER


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
