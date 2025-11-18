"""Regression tests for BenchmarkDefaults and BenchmarkConfig integration."""

import os
import sys
from pathlib import Path
import pytest

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_defaults import BenchmarkDefaults, get_defaults, set_defaults
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, ExecutionMode, LaunchVia


class TestBenchmarkDefaults:
    """Test BenchmarkDefaults functionality."""
    
    def test_default_values(self):
        """Test that defaults match expected values."""
        defaults = BenchmarkDefaults()
        assert defaults.iterations == 100
        assert defaults.warmup == 10
        assert defaults.enable_profiling is True
        assert defaults.enable_nsys is True
        assert defaults.enable_ncu is True
        assert defaults.use_subprocess is True
        assert defaults.measurement_timeout_seconds == 15
        assert defaults.execution_mode is None
        assert defaults.launch_via == "python"
    
    def test_from_env_returns_defaults(self):
        """Test that from_env() returns default values (env vars no longer supported)."""
        defaults = BenchmarkDefaults.from_env()
        # Should use declared defaults
        assert defaults.enable_profiling is True, "enable_profiling should default to True"
        assert defaults.enable_nsys is True, "enable_nsys should default to True"
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
        assert config.enable_profiling is True, "enable_profiling should default to True"
        assert config.enable_nsys is True, "enable_nsys should default to True"
        assert config.enable_ncu is True, "enable_ncu should default to True"
        assert config.use_subprocess is True, "use_subprocess should default to True"
        assert config.execution_mode == ExecutionMode.SUBPROCESS
        assert config.launch_via == LaunchVia.PYTHON
    


class TestBenchmarkTimeoutMultiplier:
    """Additional coverage for timeout multiplier behavior."""
    
    def setup_method(self):
        self._original_defaults = get_defaults()
    
    def teardown_method(self):
        set_defaults(self._original_defaults)
    
    def test_multiplier_scales_default_timeouts(self):
        """Default timeout fields should scale when multiplier > 1."""
        set_defaults(BenchmarkDefaults(timeout_multiplier=1.0))
        config = BenchmarkConfig(timeout_multiplier=2.0)
        # setup_timeout_seconds default 30 -> 60 after scaling
        assert config.setup_timeout_seconds == 60
        # measurement_timeout_seconds default 15 -> 30 after scaling
        assert config.measurement_timeout_seconds == 30
        # Derived warmup timeout should follow measurement timeout
        assert config.warmup_timeout_seconds == 30
        # Profiler-specific timeouts should also scale
        assert config.nsys_timeout_seconds == 240
        assert config.ncu_timeout_seconds == 360
    
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
    
