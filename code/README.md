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
python -m cli.aisp bench run --targets ch7 --profile minimal
```

> Canonical CLI: use `aisp` for everything. No other CLI wrappers are maintained.

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
python -m cli.aisp bench list-targets --chapter ch7

# Run with profiling
python -m cli.aisp bench run --targets ch7 --profile minimal

# Compare baseline vs optimized
python -m cli.aisp bench compare ch7.baseline_memory_access ch7.optimized_memory_access

# Quick verification (lightweight smoke test)
python -m cli.aisp bench verify --targets ch7
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

## MCP Integration (aisp MCP server)

- Start the server with `python -m tools.mcp.server --serve` (or via `mcp.json`); clients should consume the `application/json` content entry from MCP responses.
- `isError` mirrors the payload `status` field returned in the JSON envelope.
- Response trimming is generous by default; tune via env vars without code changes: `AISP_MCP_PREVIEW_LIMIT` (max chars) and `AISP_MCP_PREVIEW_ITEMS` (max list/dict items).

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
    
    def get_optimization_goal(self):
        # Return "speed" (default), "memory", "throughput", or "latency"
        return "speed"

def get_benchmark():
    return OptimizedMyTechnique()

if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness
    BenchmarkHarness().run(get_benchmark())
```

### Multi-Metric Benchmarks

Benchmarks can optimize for different goals:

| Goal | Primary Metric | Example |
|------|----------------|---------|
| `speed` | Speedup (x) | Flash Attention, CUDA Graphs |
| `memory` | Memory savings (%) | Gradient checkpointing, quantization |
| `throughput` | Tokens/sec | Batched inference |
| `latency` | Time to first token | Streaming generation |

```python
def get_optimization_goal(self):
    return "memory"  # This benchmark optimizes for memory reduction
```

See `CONTRIBUTING.md` for full coding standards.

---

## Memory Tracking

All benchmarks automatically track GPU memory usage:

| Metric | Description |
|--------|-------------|
| `peak_mb` | Maximum memory allocated during execution |
| `allocated_mb` | Memory allocated at measurement point |
| `reserved_mb` | Total memory reserved by CUDA allocator |
| `memory_savings_pct` | Reduction vs baseline (for memory optimizations) |

Memory tracking is **always enabled** globally—individual benchmarks cannot disable it. This ensures consistent data for trade-off analysis.

```bash
# View memory data in results
python -m cli.aisp bench analyze

# Memory-focused benchmarks show savings prominently
# Example: Gradient checkpointing shows "💾 57% memory saved"
```

---

## Analysis & Visualization

### Dashboard UI
```bash
python tools/dashboard/server.py --port 8100
# Open http://localhost:8100
```

### Interactive TUI
```bash
python -m cli.aisp bench tui          # Rich curses interface
python -m cli.aisp bench tui --simple # Basic menu
```

### Analysis Commands

| Command | Description |
|---------|-------------|
| `analyze` | Multi-metric analysis with leaderboards and Pareto frontier |
| `whatif` | Find optimizations matching your constraints |
| `stacking` | Which optimizations combine well together |
| `power` | Power efficiency rankings (ops/watt) |
| `cost` | Cost savings analysis with configurable GPU pricing |
| `scaling` | How optimizations scale with workload size |

```bash
# What-If: "What optimizations fit my 24GB VRAM and 100ms latency budget?"
python -m cli.aisp bench whatif --vram 24 --latency 100

# Stacking: "Which optimizations can I combine?"
python -m cli.aisp bench stacking

# Cost: "What's the $/operation impact on different GPUs?"
python -m cli.aisp bench cost --gpu H100
python -m cli.aisp bench cost --gpu A100 --top 10

# Power: "What's most energy efficient?"
python -m cli.aisp bench power

# Scaling: "How do these scale with workload size?"
python -m cli.aisp bench scaling

# Full analysis with Pareto frontier
python -m cli.aisp bench analyze
```

### GPU Pricing

| GPU | Rate | Use Case |
|-----|------|----------|
| B200 | $5.00/hr | Latest Blackwell |
| H100 | $3.50/hr | Production inference |
| A100 | $2.00/hr | Training/inference |
| L40S | $1.50/hr | Inference |
| A10G | $1.00/hr | Cost-optimized |
| T4 | $0.50/hr | Budget inference |

### API Endpoints

The dashboard server exposes REST APIs for programmatic access:

| Endpoint | Description |
|----------|-------------|
| `/api/benchmarks` | All benchmark results with multi-metric data |
| `/api/summary` | Aggregated statistics |
| `/api/analysis/leaderboards` | Speed and memory leaderboards |
| `/api/analysis/pareto` | Pareto-optimal benchmarks |
| `/api/analysis/tradeoffs` | Speed vs memory trade-off data |
| `/api/analysis/recommendations` | Use-case based recommendations |
| `/api/analysis/whatif` | Constraint-based solver |
| `/api/analysis/stacking` | Optimization compatibility matrix |
| `/api/analysis/power` | Power efficiency rankings |
| `/api/analysis/cost` | Cost per operation analysis |
| `/api/analysis/scaling` | Scaling characteristics |

```bash
# Example: Get trade-off data as JSON
curl http://localhost:8100/api/analysis/tradeoffs | jq .
```

---

## CLI Utilities

```bash
# KV cache sizing
python tools/cli/aisp bench utils --tool kv-cache -- \
    --layers 80 --hidden 8192 --tokens 4096 --batch 8 --dtype fp8

# Cost per token
python tools/cli/aisp bench utils --tool cost-per-token -- \
    --avg-power 800 --throughput 1500 --electricity-cost 0.16

# Hardware probe
python tools/cli/aisp bench utils --tool probe-hw
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

## Informational Benchmarks

Some benchmarks are marked "informational"—they demonstrate techniques but may not show speedup due to:
- Multi-GPU requirements (pipeline parallelism, disaggregated inference)
- System topology dependencies (NUMA awareness)
- Experimental APIs (FlexAttention on Blackwell)

These are valuable for learning HOW to implement patterns, even if not faster on single-GPU setups.

```python
# In run_all_benchmarks.py
INFORMATIONAL_BENCHMARKS = {
    "ch3": {"numa_unaware"},        # NUMA topology dependent
    "ch4": {"dataparallel_basic"},  # Requires multi-GPU
    "ch14": {"sliding_window_bench"},  # FlexAttention API
    "ch15": {"disaggregated_inference", "inference_placement"},  # Multi-GPU
    # ...
}
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TORCHINDUCTOR_CACHE_DIR` | Torch Inductor cache location | `.torch_inductor` |
| `CUDA_HOME` | CUDA installation path | `/usr/local/cuda` |
| `RANK` | Process rank for distributed training | `0` |
| `WORLD_SIZE` | Total processes for distributed training | GPU count |
| `LOCAL_RANK` | Local GPU rank | `0` |
| `MASTER_ADDR` | Distributed training coordinator | `localhost` |
| `MASTER_PORT` | Distributed training port | `29500` |

---

## Notes

- `setup.sh` installs system prerequisites (drivers, CUDA, Nsight)
- `python tools/testing/run_all_benchmarks.py --targets ch*` for regression suites
- `artifacts/` holds run outputs; clean via `python cleanup.py`
