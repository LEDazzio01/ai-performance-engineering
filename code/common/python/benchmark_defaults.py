"""Centralized default values for benchmark configuration.

This module provides a single source of truth for all default values used
throughout the benchmark harness, enabling easy configuration via environment
variables and configuration files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BenchmarkDefaults:
    """Centralized default values for benchmark configuration.
    
    All defaults can be overridden by passing values directly to BenchmarkConfig
    or via CLI flags (e.g., --iterations, --warmup, --profile).
    
    Note: Environment variables are no longer supported. Use CLI flags instead.
    """
    
    # Execution defaults
    iterations: int = 100
    warmup: int = 10
    min_run_time_ms: float = 100.0
    
    # Statistics defaults
    percentiles: List[float] = field(default_factory=lambda: [25, 50, 75, 99])
    
    # Feature flags
    enable_memory_tracking: bool = False
    enable_profiling: bool = True
    enable_nsys: bool = True
    enable_ncu: bool = True
    enable_nvtx: Optional[bool] = None  # Auto-enabled if profiling enabled
    enable_cleanup: bool = False
    use_subprocess: bool = True
    execution_mode: Optional[str] = None
    launch_via: str = "python"  # python | torchrun
    nproc_per_node: Optional[int] = None
    nnodes: Optional[str] = None
    rdzv_backend: Optional[str] = None
    rdzv_endpoint: Optional[str] = None
    env_passthrough: List[str] = field(default_factory=lambda: ["CUDA_VISIBLE_DEVICES"])
    target_extra_args: dict = field(default_factory=dict)
    multi_gpu_required: bool = False
    profile_type: str = "minimal"
    nsys_nvtx_include: Optional[List[str]] = None
    
    # Reproducibility defaults
    # Default to nondeterministic for performance; users can opt-in if needed.
    deterministic: bool = False
    seed: Optional[int] = 1337
    
    # Timeout defaults (in seconds)
    setup_timeout_seconds: Optional[int] = 30
    warmup_timeout_seconds: Optional[int] = None  # Defaults to measurement_timeout
    measurement_timeout_seconds: int = 15
    profiling_timeout_seconds: Optional[int] = None  # Defaults to max(nsys, ncu)
    nsys_timeout_seconds: int = 120
    ncu_timeout_seconds: int = 180
    timeout_multiplier: float = 3.0
    ncu_sampling_interval: int = 75000
    
    # Legacy timeout (deprecated)
    timeout_seconds: int = 15
    
    # Output defaults
    profiling_output_dir: Optional[str] = None
    ncu_metric_set: str = "auto"  # 'auto', 'deep_dive', 'roofline', 'minimal'
    
    @classmethod
    def from_env(cls) -> BenchmarkDefaults:
        """Create BenchmarkDefaults with default values.
        
        Note: Environment variables are no longer supported. Use CLI flags instead.
        This method now simply returns default values for backward compatibility.
        """
        return cls()
    
    def to_dict(self) -> dict:
        """Convert defaults to dictionary."""
        return {
            "iterations": self.iterations,
            "warmup": self.warmup,
            "min_run_time_ms": self.min_run_time_ms,
            "percentiles": self.percentiles,
            "enable_memory_tracking": self.enable_memory_tracking,
            "enable_profiling": self.enable_profiling,
            "enable_nsys": self.enable_nsys,
            "enable_ncu": self.enable_ncu,
            "enable_nvtx": self.enable_nvtx,
            "enable_cleanup": self.enable_cleanup,
            "use_subprocess": self.use_subprocess,
            "execution_mode": self.execution_mode,
            "launch_via": self.launch_via,
            "nproc_per_node": self.nproc_per_node,
            "nnodes": self.nnodes,
            "rdzv_backend": self.rdzv_backend,
            "rdzv_endpoint": self.rdzv_endpoint,
            "env_passthrough": self.env_passthrough,
            "target_extra_args": self.target_extra_args,
            "multi_gpu_required": self.multi_gpu_required,
            "profile_type": self.profile_type,
            "nsys_nvtx_include": self.nsys_nvtx_include,
            "deterministic": self.deterministic,
            "seed": self.seed,
            "setup_timeout_seconds": self.setup_timeout_seconds,
            "warmup_timeout_seconds": self.warmup_timeout_seconds,
            "measurement_timeout_seconds": self.measurement_timeout_seconds,
            "profiling_timeout_seconds": self.profiling_timeout_seconds,
            "nsys_timeout_seconds": self.nsys_timeout_seconds,
            "ncu_timeout_seconds": self.ncu_timeout_seconds,
            "timeout_multiplier": self.timeout_multiplier,
            "timeout_seconds": self.timeout_seconds,
            "profiling_output_dir": self.profiling_output_dir,
            "ncu_metric_set": self.ncu_metric_set,
            "ncu_sampling_interval": self.ncu_sampling_interval,
        }


# Global instance - can be overridden for testing or custom configurations
_defaults = BenchmarkDefaults.from_env()


def get_defaults() -> BenchmarkDefaults:
    """Get the global BenchmarkDefaults instance."""
    return _defaults


def set_defaults(defaults: BenchmarkDefaults) -> None:
    """Set the global BenchmarkDefaults instance (useful for testing)."""
    global _defaults
    _defaults = defaults
