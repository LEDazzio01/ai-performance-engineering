# AI Systems Performance Engineering

Reference implementation of high-performance PyTorch, CUDA, and Triton workloads for NVIDIA Blackwell platforms.
The repository packages 20 focused chapters, advanced labs, and a shared benchmarking harness so you can profile baselines, apply optimizations, and capture artifacts that prove performance gains.

---

## Quick Start

```bash
# Setup
cd ai-performance-engineering/code
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Verify CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

# Run a benchmark
python ch7/optimized_memory_access.py

# Or use the CLI
python tools/cli/benchmark_cli.py run --targets ch7 --profile minimal
```

---

## Directory Layout

| Path | Description |
|------|-------------|
| `ch1` - `ch20` | One directory per chapter with baseline/optimized benchmarks |
| `labs/` | Deep-dive labs for matmul, MoE, FlexAttention, distributed training, etc. |
| `common/python/` | Shared benchmark harness, metrics, and profiling utilities |
| `tools/` | CLI, dashboard, and analysis utilities |
| `tests/` | pytest tests (28 files) |
| `scripts/` | Development utilities |

---

## Development Workflow

```bash
make help           # See all available commands
make check          # Run all validation checks
make test           # Run all tests
make coverage       # Generate benchmark coverage report
make metrics        # Check get_custom_metrics() status
```

---

## Running Benchmarks

### Direct Execution
```bash
python ch7/optimized_memory_access.py
```

### Using the CLI
```bash
# List available targets
python tools/cli/benchmark_cli.py list-targets --chapter ch7

# Run with profiling
python tools/cli/benchmark_cli.py run --targets ch7 --profile minimal

# Compare baseline vs optimized
python -m tools.cli.benchmark_cli compare ch7.baseline_memory_access ch7.optimized_memory_access
```

### Using the Harness Directly
```python
from common.python.benchmark_harness import BenchmarkHarness
from ch7.optimized_memory_access import get_benchmark

harness = BenchmarkHarness()
results = harness.run(get_benchmark())
print(results)
```

---

## Profiling

```bash
# Timeline profile (nsys)
nsys profile -o timeline python ch7/optimized_memory_access.py

# Kernel analysis (ncu)
ncu -o kernel_analysis python ch7/optimized_memory_access.py

# Open in NVIDIA Nsight
nsys-ui timeline.nsys-rep
ncu-ui kernel_analysis.ncu-rep
```

---

## Creating a New Benchmark

```python
#!/usr/bin/env python3
"""Optimized: Description of optimization."""

import torch
from common.python.benchmark_harness import BaseBenchmark
from common.python.benchmark_metrics import compute_memory_transfer_metrics

class OptimizedMyTechnique(BaseBenchmark):
    def setup(self):
        self.N = 1024 * 1024
        self.tensor = torch.randn(self.N, device='cuda')
    
    def benchmark_fn(self):
        result = self.tensor.sum()
        torch.cuda.synchronize()
    
    def get_custom_metrics(self):
        return compute_memory_transfer_metrics(
            bytes_transferred=self.N * 4,
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
        )

def get_benchmark():
    return OptimizedMyTechnique()

if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness
    BenchmarkHarness().run(get_benchmark())
```

See `CONTRIBUTING.md` for full coding standards.

---

## CLI Utilities

```bash
# KV cache sizing
python tools/cli/benchmark_cli.py utils --tool kv-cache -- \
    --layers 80 --hidden 8192 --tokens 4096 --batch 8 --dtype fp8

# Cost per token
python tools/cli/benchmark_cli.py utils --tool cost-per-token -- \
    --avg-power 800 --throughput 1500 --electricity-cost 0.16

# Hardware probe
python tools/cli/benchmark_cli.py utils --tool probe-hw
```

---

## Validation

```bash
# Run tests
pytest tests/ -v

# Validate benchmark imports
python scripts/validate_imports.py

# Check metrics coverage
python scripts/update_custom_metrics.py --analyze

# Full validation suite
make check
```

---

## Labs

| Lab | Description |
|-----|-------------|
| `async_input_pipeline/` | CPU→GPU data loading overlap |
| `blackwell_matmul/` | TMA, clusters, TCGEN05 matmul |
| `distributed_training/` | FSDP2 + FP8 communication |
| `kv_optimization/` | FP8/FP4 KV cache compression |
| `speculative_decode/` | Draft-verify decoding |
| `ultimate_moe_inference/` | **All 144 techniques** |

Each lab has a README.md with detailed instructions.

---

## Notes

- `setup.sh` installs system prerequisites (drivers, CUDA, Nsight)
- `python tools/testing/run_all_benchmarks.py --targets ch*` for regression suites
- Set `AIPERF_FAST_BENCH=1` for lightweight CI validation
- `artifacts/` holds run outputs; clean via `python cleanup.py`
