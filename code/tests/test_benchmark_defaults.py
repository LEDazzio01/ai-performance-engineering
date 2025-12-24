"""Regression tests for BenchmarkDefaults and BenchmarkConfig integration."""

import os
import sys
from pathlib import Path
import pytest

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.benchmark.defaults import BenchmarkDefaults, get_defaults, set_defaults
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, ExecutionMode, LaunchVia


class TestBenchmarkDefaults:
    """Test BenchmarkDefaults functionality."""
    
    def test_default_values(self):
        """Test that defaults match expected values."""
        defaults = BenchmarkDefaults()
        assert defaults.iterations == 100
        assert defaults.warmup == 10
        assert defaults.enable_profiling is False
        assert defaults.enable_nsys is False
        assert defaults.enable_ncu is True
        assert defaults.use_subprocess is True
        assert defaults.measurement_timeout_seconds == 1200
        assert defaults.execution_mode is None
        assert defaults.launch_via == "python"
        assert defaults.detect_setup_precomputation is True
        assert defaults.graph_capture_cheat_ratio_threshold == 10.0
        assert defaults.graph_capture_memory_threshold_mb == 100.0
        assert defaults.ncu_replay_mode == "kernel"
    
    def test_from_env_returns_defaults(self):
        """Test that from_env() returns default values (env vars no longer supported)."""
        defaults = BenchmarkDefaults.from_env()
        # Should use declared defaults
        assert defaults.enable_profiling is False, "enable_profiling should default to False"
        assert defaults.enable_nsys is False, "enable_nsys should default to False"
        assert defaults.enable_ncu is True, "enable_ncu should default to True"
        assert defaults.use_subprocess is True, "use_subprocess should default to True"
        assert defaults.execution_mode is None, "execution_mode should default to None"
        assert defaults.iterations == 100, "iterations should default to 100"
        assert defaults.warmup == 10, "warmup should default to 10"
    

class TestBenchmarkConfigDefaults:
    """Test BenchmarkConfig uses BenchmarkDefaults correctly."""
    
    def test_config_uses_defaults(self):
        """Test that BenchmarkConfig uses BenchmarkDefaults."""
        config = BenchmarkConfig()
        # Should use defaults from BenchmarkDefaults
        assert config.iterations == 100
        assert config.warmup == 10
        assert config.enable_profiling is False, "enable_profiling should default to False"
        assert config.enable_nsys is False, "enable_nsys should default to False"
        assert config.enable_ncu is True, "enable_ncu should default to True"
        assert config.use_subprocess is True, "use_subprocess should default to True"
        assert config.execution_mode == ExecutionMode.SUBPROCESS
        assert config.launch_via == LaunchVia.PYTHON
        assert config.detect_setup_precomputation is True
        assert config.graph_capture_cheat_ratio_threshold == 10.0
        assert config.graph_capture_memory_threshold_mb == 100.0
        assert config.ncu_replay_mode == "kernel"
    


class TestBenchmarkTimeoutMultiplier:
    """Additional coverage for timeout multiplier behavior."""
    
    def setup_method(self):
        self._original_defaults = get_defaults()
    
    def teardown_method(self):
        set_defaults(self._original_defaults)
    
    def test_multiplier_scales_default_timeouts(self):
        """Default timeout fields should scale when multiplier > 1."""
        base_defaults = BenchmarkDefaults(timeout_multiplier=1.0)
        set_defaults(base_defaults)
        config = BenchmarkConfig(timeout_multiplier=2.0)
        assert config.setup_timeout_seconds == (base_defaults.setup_timeout_seconds or 0) * 2
        assert config.measurement_timeout_seconds == base_defaults.measurement_timeout_seconds * 2
        assert config.warmup_timeout_seconds == config.measurement_timeout_seconds
        assert config.nsys_timeout_seconds == base_defaults.nsys_timeout_seconds * 2
        assert config.ncu_timeout_seconds == base_defaults.ncu_timeout_seconds * 2
    
    def test_multiplier_does_not_override_explicit_values(self):
        """Explicit per-stage overrides must remain unchanged."""
        set_defaults(BenchmarkDefaults(timeout_multiplier=1.0))
        config = BenchmarkConfig(
            measurement_timeout_seconds=2,
            warmup_timeout_seconds=1,
            timeout_multiplier=5.0,
        )
        assert config.measurement_timeout_seconds == 2
        assert config.warmup_timeout_seconds == 1
    
    def test_config_explicit_override(self):
        """Test that explicit values override defaults."""
        config = BenchmarkConfig(iterations=999, warmup=99)
        assert config.iterations == 999
        assert config.warmup == 99

    def test_config_graph_capture_overrides(self):
        """Graph capture thresholds should be configurable."""
        config = BenchmarkConfig(
            graph_capture_cheat_ratio_threshold=3.5,
            graph_capture_memory_threshold_mb=42.0,
        )
        assert config.graph_capture_cheat_ratio_threshold == 3.5
        assert config.graph_capture_memory_threshold_mb == 42.0


class TestWarmupEnforcement:
    """Test warmup minimum enforcement - CRITICAL for accurate measurements.
    
    Low warmup causes JIT/compile overhead to be included in measurements,
    leading to incorrect speedup calculations. The harness must enforce
    minimum warmup to prevent this.
    """
    
    def test_minimum_warmup_enforced(self):
        """Test that warmup below minimum is auto-corrected to minimum."""
        import warnings
        from core.benchmark.defaults import MINIMUM_WARMUP_ITERATIONS
        
        # Setting warmup below minimum should trigger warning and auto-correct
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = BenchmarkConfig(warmup=0)
            
            # Check warmup was raised to minimum
            assert config.warmup >= MINIMUM_WARMUP_ITERATIONS, \
                f"Expected warmup >= {MINIMUM_WARMUP_ITERATIONS}, got {config.warmup}"
            
            # Check warning was issued
            warmup_warnings = [x for x in w if 'warmup' in str(x.message).lower()]
            assert len(warmup_warnings) > 0, \
                "Expected warning about low warmup"
    
    def test_warmup_at_minimum_is_accepted(self):
        """Test that warmup exactly at minimum doesn't trigger warning."""
        import warnings
        from core.benchmark.defaults import MINIMUM_WARMUP_ITERATIONS
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = BenchmarkConfig(warmup=MINIMUM_WARMUP_ITERATIONS)
            
            assert config.warmup == MINIMUM_WARMUP_ITERATIONS
            warmup_warnings = [x for x in w if 'warmup' in str(x.message).lower()]
            assert len(warmup_warnings) == 0, \
                "Should not warn when warmup is at minimum"
    
    def test_warmup_above_minimum_is_accepted(self):
        """Test that warmup above minimum is accepted unchanged."""
        import warnings
        from core.benchmark.defaults import MINIMUM_WARMUP_ITERATIONS
        
        high_warmup = MINIMUM_WARMUP_ITERATIONS + 10
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = BenchmarkConfig(warmup=high_warmup)
            
            assert config.warmup == high_warmup
            warmup_warnings = [x for x in w if 'warmup' in str(x.message).lower()]
            assert len(warmup_warnings) == 0, \
                "Should not warn when warmup is above minimum"
    
    def test_validate_warmup_function(self):
        """Test the validate_warmup helper function directly."""
        import warnings
        from core.benchmark.defaults import validate_warmup, MINIMUM_WARMUP_ITERATIONS
        
        # Test low warmup is corrected
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_warmup(2, context="test")
            assert result >= MINIMUM_WARMUP_ITERATIONS
            assert len(w) > 0  # Warning should be issued
        
        # Test valid warmup passes through
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_warmup(10, context="test")
            assert result == 10
            assert len(w) == 0  # No warning
    
