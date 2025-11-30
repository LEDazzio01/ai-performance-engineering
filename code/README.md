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
python ch07/optimized_memory_access.py

# Or use the CLI
python -m cli.aisp bench run --targets ch07 --profile minimal
# Point at a different benchmark root (defaults to this repo)
python -m cli.aisp bench run --bench-root /path/to/benchmarks --targets ch07
```

---

## Directory Layout

| Path | Description |
|------|-------------|
| `ch01` - `ch20` | One directory per chapter with baseline/optimized benchmarks |
| `labs/` | Deep-dive labs for matmul, MoE, FlexAttention, distributed training, etc. |

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
### Project root selection
- Default project root: the repo directory you're in (e.g., `ai-performance-engineering/code`).
- Override everywhere with `--bench-root /path/to/project`.
  - CLI: `python -m cli.aisp bench run --bench-root /path/to/project --targets ch07`
  - Full runner: `python -m core.harness.run_all_benchmarks --bench-root /path/to/project`
  - Dashboard backend: `python -m dashboard.api.server serve --bench-root /path/to/project --data /path/to/benchmark_test_results.json`
- Environment variables are ignored for project-root selection; use flags only.
- In the dashboard UI, you can change the project root live (no restart) from the Settings tab’s “Project Root” card; the choice is stored in your browser.

### Direct Execution
```bash
python ch07/optimized_memory_access.py
```

### Using the CLI
```bash
# List available targets
python -m cli.aisp bench list-targets --chapter ch07

# Run with profiling
python -m cli.aisp bench run --targets ch07 --profile minimal

# Compare baseline vs optimized
python -m cli.aisp bench compare ch07.baseline_memory_access ch07.optimized_memory_access

# Quick verification (lightweight smoke test)
python -m cli.aisp bench verify --targets ch07
```

### Using the Harness Directly
```python
from core.harness.benchmark_harness import BenchmarkHarness
from ch07.optimized_memory_access import get_benchmark

harness = BenchmarkHarness()
results = harness.run(get_benchmark())
print(results)
```

---

## Profiling

```bash
# Timeline profile (nsys)
nsys profile -o timeline python ch07/optimized_memory_access.py

# Kernel analysis (ncu)
ncu -o kernel_analysis python ch07/optimized_memory_access.py

# Open in NVIDIA Nsight
nsys-ui timeline.nsys-rep
ncu-ui kernel_analysis.ncu-rep
```

---

## MCP Integration (aisp MCP server)

Start the server with `python -m mcp.mcp_server --serve` (or via `mcp.json`). Responses emit a `text` content entry containing the JSON envelope. Tool descriptions returned by `tools/list` (or `python -m mcp.mcp_server --list`) embed Inputs/Outputs/Expectations so clients can surface parameter guidance and side-effect hints.

**76 MCP tools available** organized by category. See [docs/mcp_tools.md](docs/mcp_tools.md) for complete reference with parameters and examples.

### Benchmarking (17 tools)
| Tool | Description |
|------|-------------|
| `run_benchmarks` | Run benchmarks with optional profiling |
| `verify_benchmarks` | Quick verification smoke tests |
| `available_benchmarks` | List available benchmark targets |
| `benchmark_targets` | List benchmark targets by chapter |
| `list_chapters` | Scan and list all chapters |
| `benchmark_report` | Generate benchmark report |
| `benchmark_export` | Export benchmarks to file |
| `benchmark_compare_runs` | Compare benchmark runs |
| `test_speed` | Run speed tests |
| `test_disk` | Disk I/O benchmark |
| `test_pcie` | PCIe H2D/D2H bandwidth test |
| `test_mem_hierarchy` | Memory hierarchy stride test |
| `test_tensor_core` | Tensor Core throughput test |
| `test_sfu` | Special function unit benchmark |
| `test_network` | Network throughput tests |
| `test_network_loopback` | Loopback TCP throughput test |
| `test_roofline` | Roofline model benchmark |

### GPU (5 tools)
`gpu_info`, `gpu_bandwidth`, `gpu_topology`, `gpu_topology_matrix`, `gpu_power`

### System (13 tools)
`system_software`, `system_dependencies`, `system_context`, `system_capabilities`, `system_parameters`, `container_limits`, `cpu_memory_analysis`, `full_system_analysis`, `status`, `triage`, `context_summary`, `context_full`, `ai_status`

### Profiling (10 tools)
`profile_nsys`, `profile_ncu`, `nsys_summary`, `compare_nsys`, `compare_ncu`, `nsys_ncu_available`, `profile_flame`, `profile_memory`, `profile_kernels`, `profile_roofline`

### Analysis (5 tools)
`analyze_bottlenecks`, `analyze_pareto`, `analyze_scaling`, `analyze_stacking`, `analyze_whatif`

### Optimization (3 tools)
`recommend`, `optimize_roi`, `optimize_techniques`

### Distributed (2 tools)
`distributed_plan`, `distributed_nccl`

### Inference (2 tools)
`inference_vllm`, `inference_quantization`

### AI/LLM (2 tools)
`ask`, `explain`

### HuggingFace (2 tools)
`hf_search`, `hf_trending`

### Cluster & Cost (2 tools)
`cluster_slurm`, `cost_estimate`

### Code Analysis (7 tools)
`warp_divergence`, `bank_conflicts`, `memory_access`, `comm_overlap`, `data_loading`, `energy_analysis`, `predict_scaling`

### Export (3 tools)
`export_csv`, `export_pdf`, `export_html`

### Utility (3 tools)
`help`, `suggest_tools`, `launch_plan`

**Configuration:**
- `isError` mirrors the payload `status` field returned in the JSON envelope
- Response trimming via env vars: `AISP_MCP_PREVIEW_LIMIT` (max chars), `AISP_MCP_PREVIEW_ITEMS` (max list/dict items)

---

## Creating a New Benchmark

```python
#!/usr/bin/env python3
"""Optimized: Description of optimization."""

import torch
from core.harness.benchmark_harness import BaseBenchmark
from core.benchmark.metrics import compute_memory_transfer_metrics

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
    from core.harness.benchmark_harness import BenchmarkHarness
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
python -m dashboard.api.server serve --port 8100 --bench-root /path/to/benchmarks --data /path/to/benchmark_test_results.json
# Open http://localhost:8100 (defaults to repo root if --bench-root is omitted)

# Or start the Next.js frontend plus backend:
#   SKIP_BACKEND=1 ./start-dashboard.sh    # then start backend manually with --bench-root as above
#   ./start-dashboard.sh                   # uses repo root as bench root
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
python cli/aisp.py bench utils --tool kv-cache -- \
    --layers 80 --hidden 8192 --tokens 4096 --batch 8 --dtype fp8

# Cost per token
python cli/aisp.py bench utils --tool cost-per-token -- \
    --avg-power 800 --throughput 1500 --electricity-cost 0.16

# Hardware probe
python cli/aisp.py bench utils --tool probe-hw
```

---

## Validation

```bash
# Run tests
pytest tests/ -v

# Validate benchmark imports
python core/scripts/validate_imports.py

# Check metrics coverage
python core/scripts/update_custom_metrics.py --analyze

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
    "ch03": {"numa_unaware"},        # NUMA topology dependent
    "ch04": {"dataparallel_basic"},  # Requires multi-GPU
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
- `python core/harness/run_all_benchmarks.py --targets ch*` for regression suites
- `artifacts/` holds run outputs; clean via `python cleanup.py`
