#!/usr/bin/env python3
"""Unit tests for profiler_config.py.

Run with: pytest tests/test_profiler_config.py -v
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from common.python.profiler_config import (
    # Metric sets
    ROOFLINE_METRICS,
    DEEP_DIVE_METRICS,
    MINIMAL_METRICS,
    CH6_KERNEL_METRICS,
    CH7_MEMORY_METRICS,
    CH8_OPTIMIZATION_METRICS,
    CH9_COMPUTE_METRICS,
    CH10_PIPELINE_METRICS,
    CH11_STREAM_METRICS,
    CH12_GRAPH_METRICS,
    CH13_PYTORCH_METRICS,
    CHAPTER_METRICS,
    # Functions
    get_chapter_metrics,
    ProfilerConfig,
    set_default_profiler_metric_set,
    discover_nvtx_includes,
    build_profiler_config_from_benchmark,
)


class TestMetricSets:
    """Test metric set definitions."""
    
    def test_roofline_metrics_not_empty(self):
        """Roofline metrics should have content."""
        assert len(ROOFLINE_METRICS) > 0
        assert "gpu__time_duration.avg" in ROOFLINE_METRICS
    
    def test_deep_dive_includes_roofline(self):
        """Deep dive should be a superset of roofline."""
        for metric in ROOFLINE_METRICS:
            assert metric in DEEP_DIVE_METRICS
    
    def test_minimal_is_subset_of_roofline(self):
        """Minimal should be a subset of roofline."""
        for metric in MINIMAL_METRICS:
            assert metric in ROOFLINE_METRICS
    
    def test_chapter_metrics_include_roofline(self):
        """Chapter metrics should include roofline base."""
        for ch_metrics in [CH6_KERNEL_METRICS, CH7_MEMORY_METRICS, CH9_COMPUTE_METRICS]:
            for metric in ROOFLINE_METRICS:
                assert metric in ch_metrics, f"Missing {metric} in chapter metrics"
    
    def test_ch6_has_bank_conflict_metrics(self):
        """Ch6 should have bank conflict metrics."""
        bank_metrics = [m for m in CH6_KERNEL_METRICS if "bank_conflict" in m]
        assert len(bank_metrics) >= 1
    
    def test_ch7_has_coalescing_metrics(self):
        """Ch7 should have coalescing metrics."""
        coal_metrics = [m for m in CH7_MEMORY_METRICS if "sector" in m or "bytes_per_sector" in m]
        assert len(coal_metrics) >= 1
    
    def test_ch8_has_stall_metrics(self):
        """Ch8 should have warp stall metrics."""
        stall_metrics = [m for m in CH8_OPTIMIZATION_METRICS if "stalled" in m]
        assert len(stall_metrics) >= 3
    
    def test_ch9_has_tensor_core_metrics(self):
        """Ch9 should have tensor core metrics."""
        tensor_metrics = [m for m in CH9_COMPUTE_METRICS if "tensor" in m]
        assert len(tensor_metrics) >= 1


class TestGetChapterMetrics:
    """Test get_chapter_metrics function."""
    
    def test_known_chapters(self):
        """Known chapters should return specific metrics."""
        assert get_chapter_metrics(6) == CH6_KERNEL_METRICS
        assert get_chapter_metrics(7) == CH7_MEMORY_METRICS
        assert get_chapter_metrics(8) == CH8_OPTIMIZATION_METRICS
        assert get_chapter_metrics(9) == CH9_COMPUTE_METRICS
        assert get_chapter_metrics(10) == CH10_PIPELINE_METRICS
        assert get_chapter_metrics(11) == CH11_STREAM_METRICS
        assert get_chapter_metrics(12) == CH12_GRAPH_METRICS
        assert get_chapter_metrics(13) == CH13_PYTORCH_METRICS
        assert get_chapter_metrics(14) == CH13_PYTORCH_METRICS  # Same as 13
    
    def test_unknown_chapter_returns_roofline(self):
        """Unknown chapters should return roofline metrics."""
        assert get_chapter_metrics(1) == ROOFLINE_METRICS
        assert get_chapter_metrics(99) == ROOFLINE_METRICS
    
    def test_all_chapters_return_list(self):
        """All chapters should return a list."""
        for ch in range(1, 21):
            result = get_chapter_metrics(ch)
            assert isinstance(result, list)
            assert len(result) > 0


class TestProfilerConfig:
    """Test ProfilerConfig class."""
    
    def test_default_config(self):
        """Default config should have sensible defaults."""
        config = ProfilerConfig()
        assert config.metric_set == "minimal"
        assert config.preset == "minimal"
        assert config.ncu_replay_mode == "kernel"
    
    def test_nsys_command_generation(self):
        """nsys command should be properly formatted."""
        config = ProfilerConfig(preset="minimal")
        cmd = config.get_nsys_command(
            output_path="/tmp/test",
            python_script="test.py",
            python_executable="/usr/bin/python3"
        )
        
        assert "nsys" in cmd
        assert "profile" in cmd
        assert "-o" in cmd
        assert "/tmp/test" in cmd
        assert "test.py" in cmd
        assert "/usr/bin/python3" in cmd
    
    def test_nsys_command_with_nvtx_filters(self):
        """nsys command should include NVTX filters."""
        config = ProfilerConfig(nvtx_includes=["kernel1", "kernel2"])
        cmd = config.get_nsys_command(
            output_path="/tmp/test",
            python_script="test.py"
        )
        
        assert "--nvtx-include" in cmd
        assert "kernel1" in cmd
        assert "kernel2" in cmd
    
    def test_ncu_command_generation(self):
        """ncu command should be properly formatted."""
        config = ProfilerConfig(metric_set="roofline")
        cmd = config.get_ncu_command(
            output_path="/tmp/test",
            python_script="test.py",
            python_executable="/usr/bin/python3"
        )
        
        assert "ncu" in cmd
        assert "--set" in cmd
        assert "--metrics" in cmd
        assert "-o" in cmd
        assert "/tmp/test" in cmd
    
    def test_ncu_command_with_custom_metrics(self):
        """ncu command should use custom metrics when provided."""
        config = ProfilerConfig()
        custom_metrics = ["gpu__time_duration.avg", "sm__throughput.avg"]
        cmd = config.get_ncu_command(
            output_path="/tmp/test",
            python_script="test.py",
            metrics=custom_metrics
        )
        
        # Check metrics are joined
        metrics_str = ",".join(custom_metrics)
        assert metrics_str in cmd
    
    def test_torch_profiler_config_minimal(self):
        """Minimal preset should return lightweight config."""
        config = ProfilerConfig(preset="minimal")
        torch_cfg = config.get_torch_profiler_config()
        
        # May be empty if torch not available
        if torch_cfg:
            assert torch_cfg.get("record_shapes") is False
            assert torch_cfg.get("profile_memory") is False
    
    def test_torch_profiler_config_deep_dive(self):
        """Deep dive preset should return full config."""
        config = ProfilerConfig(preset="deep_dive")
        torch_cfg = config.get_torch_profiler_config()
        
        # May be empty if torch not available
        if torch_cfg:
            assert torch_cfg.get("record_shapes") is True
            assert torch_cfg.get("profile_memory") is True


class TestSetDefaultProfilerMetricSet:
    """Test set_default_profiler_metric_set function."""
    
    def test_valid_metric_sets(self):
        """Valid metric sets should be accepted."""
        for metric_set in ["deep_dive", "roofline", "minimal"]:
            set_default_profiler_metric_set(metric_set)
            # Should not raise
    
    def test_invalid_metric_set_raises(self):
        """Invalid metric sets should raise ValueError."""
        with pytest.raises(ValueError):
            set_default_profiler_metric_set("invalid")
    
    def test_case_insensitive(self):
        """Metric set names should be case-insensitive."""
        set_default_profiler_metric_set("DEEP_DIVE")
        set_default_profiler_metric_set("RoofLine")
        set_default_profiler_metric_set("MINIMAL")
        # Should not raise


class TestDiscoverNvtxIncludes:
    """Test discover_nvtx_includes function."""
    
    def test_with_benchmark_class(self):
        """Should include benchmark class name."""
        result = discover_nvtx_includes(
            benchmark_module=None,
            benchmark_class="MyBenchmark"
        )
        assert any("mybenchmark" in r.lower() for r in result)
    
    def test_with_explicit_includes(self):
        """Should include explicit filters."""
        result = discover_nvtx_includes(
            benchmark_module=None,
            explicit=["prefill", "decode"]
        )
        assert "prefill" in result
        assert "decode" in result
    
    def test_limit_applied(self):
        """Should respect limit parameter."""
        result = discover_nvtx_includes(
            benchmark_module=None,
            explicit=["a", "b", "c", "d", "e", "f", "g", "h"],
            limit=3
        )
        assert len(result) <= 3
    
    def test_priority_keywords(self):
        """Priority keywords should be ordered first."""
        result = discover_nvtx_includes(
            benchmark_module=None,
            explicit=["other", "prefill", "random", "decode"],
            limit=10
        )
        # prefill and decode should come before other/random
        if "prefill" in result and "other" in result:
            assert result.index("prefill") < result.index("other")


class TestBuildProfilerConfigFromBenchmark:
    """Test build_profiler_config_from_benchmark function."""
    
    def test_with_minimal_config(self):
        """Should build config from minimal harness config."""
        mock_config = MagicMock()
        mock_config.profile_type = "minimal"
        mock_config.ncu_metric_set = None
        mock_config.ncu_sampling_interval = 75000
        mock_config.nsys_nvtx_include = None
        
        result = build_profiler_config_from_benchmark(mock_config)
        
        assert isinstance(result, ProfilerConfig)
        assert result.preset == "minimal"
    
    def test_with_deep_dive_config(self):
        """Should build config from deep dive harness config."""
        mock_config = MagicMock()
        mock_config.profile_type = "deep_dive"
        mock_config.ncu_metric_set = "deep_dive"
        mock_config.ncu_sampling_interval = 50000
        mock_config.nsys_nvtx_include = ["kernel1"]
        
        result = build_profiler_config_from_benchmark(mock_config)
        
        assert result.preset == "deep_dive"
        assert result.metric_set == "deep_dive"


class TestMetricNaming:
    """Test that metric names follow expected patterns."""
    
    def test_all_metrics_have_valid_format(self):
        """All metrics should have valid ncu metric format."""
        all_metrics = set(
            ROOFLINE_METRICS + 
            DEEP_DIVE_METRICS + 
            MINIMAL_METRICS +
            CH6_KERNEL_METRICS +
            CH7_MEMORY_METRICS +
            CH8_OPTIMIZATION_METRICS +
            CH9_COMPUTE_METRICS +
            CH10_PIPELINE_METRICS +
            CH11_STREAM_METRICS +
            CH12_GRAPH_METRICS +
            CH13_PYTORCH_METRICS
        )
        
        for metric in all_metrics:
            # Should have at least one __ separator
            assert "__" in metric, f"Invalid metric format: {metric}"
            # Should not be empty
            assert len(metric) > 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

