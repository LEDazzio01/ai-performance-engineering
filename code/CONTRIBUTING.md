# Contributing Guide

Welcome! This guide covers coding standards and best practices for the AI Performance Engineering codebase.

## Table of Contents

- [File Organization](#file-organization)
- [Benchmark Structure](#benchmark-structure)
- [Coding Standards](#coding-standards)
- [Metrics Guidelines](#metrics-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)

---

## File Organization

### Chapter Structure

```
ch{N}/
├── __init__.py                    # Module exports
├── baseline_{technique}.py        # Unoptimized reference
├── optimized_{technique}.py       # Optimized version
├── baseline_{technique}.cu        # CUDA kernel (if applicable)
├── optimized_{technique}.cu       # Optimized CUDA kernel
└── Makefile                       # CUDA compilation rules
```

### Naming Conventions

| Pattern | Purpose | Example |
|---------|---------|---------|
| `baseline_*.py` | Unoptimized reference | `baseline_memory_access.py` |
| `optimized_*.py` | Optimized implementation | `optimized_memory_access.py` |
| `*_bench.py` | Standalone benchmark | `matmul_bench.py` |
| `*_demo.py` | Demonstration/example | `flash_attention_demo.py` |

---

## Benchmark Structure

### Required Methods

Every benchmark inheriting from `BaseBenchmark` must implement:

```python
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig

class MyBenchmark(BaseBenchmark):
    """One-line description of what this benchmark measures."""
    
    def setup(self) -> None:
        """Initialize tensors, models, etc. Called once before benchmarking."""
        self.N = 1024
        self.tensor = torch.randn(self.N, device='cuda')
    
    def benchmark_fn(self) -> None:
        """The operation to benchmark. Called repeatedly for timing."""
        result = self.tensor.sum()
        torch.cuda.synchronize()
    
    def teardown(self) -> None:
        """Cleanup. Called once after benchmarking."""
        del self.tensor
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics. Use helper functions!"""
        return compute_memory_transfer_metrics(
            bytes_transferred=self.N * 4,
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
        )

def get_benchmark():
    """Factory function for benchmark discovery."""
    return MyBenchmark()
```

### Optional Methods

```python
def validate_result(self) -> bool:
    """Verify correctness. Return True if results are valid."""
    return self._result is not None

def _nvtx_range(self, name: str):
    """Context manager for NVTX profiling ranges."""
    return torch.cuda.nvtx.range(name)

def get_config(self) -> BenchmarkConfig:
    """Return custom benchmark configuration."""
    return BenchmarkConfig(
        warmup_iterations=10,
        benchmark_iterations=100,
    )
```

---

## Coding Standards

### Imports

```python
# Standard library
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Third-party
import torch
import torch.nn as nn

# Local
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.python.benchmark_harness import BaseBenchmark
from common.python.benchmark_metrics import compute_memory_transfer_metrics
```

### Type Hints

Always use type hints for function signatures:

```python
def compute_something(
    tensor: torch.Tensor,
    scale: float = 1.0,
    device: Optional[str] = None,
) -> Dict[str, float]:
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def compute_bandwidth(bytes_transferred: int, elapsed_ms: float) -> float:
    """Calculate achieved bandwidth in GB/s.
    
    Args:
        bytes_transferred: Total bytes moved
        elapsed_ms: Time elapsed in milliseconds
    
    Returns:
        Bandwidth in GB/s
    
    Raises:
        ValueError: If elapsed_ms is zero or negative
    """
    ...
```

### CUDA Synchronization

Always synchronize before timing:

```python
def benchmark_fn(self) -> None:
    # Do work
    result = self.model(self.input)
    
    # ALWAYS synchronize before timing ends
    torch.cuda.synchronize()
```

---

## Metrics Guidelines

### Use Helper Functions

Always use the standardized metric helpers from `common/python/benchmark_metrics.py`:

```python
from common.python.benchmark_metrics import (
    compute_memory_transfer_metrics,    # Ch2: bandwidth
    compute_kernel_fundamentals_metrics, # Ch6: bank conflicts
    compute_memory_access_metrics,       # Ch7: coalescing
    compute_optimization_metrics,        # Ch8: speedup
    compute_roofline_metrics,            # Ch9: arithmetic intensity
    compute_pipeline_metrics,            # Ch10: pipeline efficiency
    compute_stream_metrics,              # Ch11: overlap
    compute_graph_metrics,               # Ch12: launch overhead
    compute_precision_metrics,           # Ch13: FP8/FP16
    compute_triton_metrics,              # Ch14: Triton kernels
    compute_inference_metrics,           # Ch15-17: TTFT/TPOT
    compute_speculative_decoding_metrics, # Ch18: acceptance rate
    compute_distributed_metrics,         # Ch4: collective bandwidth
    compute_moe_metrics,                 # MoE: expert utilization
)
```

### Metric Naming Convention

Use `category.metric_name` format:

```python
# Good
{
    "transfer.achieved_gbps": 100.5,
    "transfer.efficiency_pct": 85.2,
    "memory.coalescing_pct": 95.0,
}

# Bad
{
    "bandwidth": 100.5,      # Missing category
    "efficiency": 85.2,      # Missing category
    "CoalescingPercent": 95.0,  # Wrong format
}
```

### Defensive Guards

Use conditional returns for metrics that depend on runtime data:

```python
def get_custom_metrics(self) -> Optional[dict]:
    # Guard against missing data
    if not hasattr(self, '_last_elapsed_ms'):
        return None
    
    return compute_memory_transfer_metrics(
        bytes_transferred=self.N * 4,
        elapsed_ms=self._last_elapsed_ms,
    )
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_benchmark_metrics.py -v

# Run with coverage
pytest tests/ --cov=common/python --cov-report=html
```

### Writing Tests

```python
import pytest
from common.python.benchmark_metrics import compute_memory_transfer_metrics

class TestMemoryTransferMetrics:
    def test_basic_transfer(self):
        """Test basic bandwidth calculation."""
        metrics = compute_memory_transfer_metrics(
            bytes_transferred=1e9,  # 1 GB
            elapsed_ms=100.0,       # 100 ms
        )
        
        # 1 GB in 100 ms = 10 GB/s
        assert abs(metrics["transfer.achieved_gbps"] - 10.0) < 0.01
    
    def test_zero_time_handling(self):
        """Should handle edge cases gracefully."""
        metrics = compute_memory_transfer_metrics(
            bytes_transferred=1e9,
            elapsed_ms=0.0,  # Edge case
        )
        assert metrics["transfer.achieved_gbps"] > 0  # Should not crash
```

---

## Documentation

### File Headers

Every Python file should have a docstring:

```python
#!/usr/bin/env python3
"""Optimized: Memory coalescing with vectorized loads.

Demonstrates 128-bit vectorized memory access patterns for
optimal memory bandwidth utilization on Blackwell GPUs.

Key optimizations:
- float4 vectorized loads (128-bit)
- Aligned memory access
- Coalesced access patterns

Expected speedup: 2-4× over baseline
"""
```

### README Files

Labs should have README.md with:
- Goal/purpose
- Key techniques demonstrated
- File descriptions
- Running instructions
- Configuration options
- Related chapters

---

## Tools

### Analyze Metrics Coverage

```bash
# Check get_custom_metrics() status
python scripts/update_custom_metrics.py --analyze

# Apply suggested improvements
python scripts/update_custom_metrics.py --apply

# Validate metric quality
python scripts/update_custom_metrics.py --validate
```

### Benchmark Comparison

```bash
# Compare baseline vs optimized
python -m tools.cli.benchmark_cli compare \
    ch7.baseline_memory_access \
    ch7.optimized_memory_access
```

### Profiling

```bash
# nsys profile
nsys profile -o output python ch7/optimized_memory_access.py

# ncu profile with chapter-specific metrics
ncu --set full --metrics $(python -c "from common.python.profiler_config import get_chapter_metrics; print(','.join(get_chapter_metrics(7)))") python ch7/optimized_memory_access.py
```

---

## Development Workflow

### Using the Makefile

```bash
# See all available targets
make help

# Run all checks before committing
make check

# Run tests with coverage
make test-cov

# Generate coverage report
make coverage
```

### CI/CD Integration

The following checks run automatically on push/PR:
- Unit tests for benchmark_metrics, profiler_config, update_custom_metrics
- Benchmark import validation
- Metrics coverage analysis
- Linting (flake8, mypy)

See `.github/workflows/benchmark-validation.yml` for details.

---

## Checklist

Before submitting changes:

- [ ] Benchmark inherits from `BaseBenchmark`
- [ ] Has `get_benchmark()` factory function
- [ ] Has `get_custom_metrics()` using helper functions
- [ ] Includes `torch.cuda.synchronize()` in `benchmark_fn()`
- [ ] Has docstring with description
- [ ] Uses type hints
- [ ] Follows `baseline_`/`optimized_` naming
- [ ] Tests pass (`make test`)
- [ ] Validation passes (`make check`)

