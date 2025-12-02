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
  - Full runner: `python -m core.harness.run_benchmarks --bench-root /path/to/project`
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

# Using the Harness Directly
```python
from core.harness.benchmark_harness import BenchmarkHarness
from ch07.optimized_memory_access import get_benchmark

harness = BenchmarkHarness()
results = harness.run(get_benchmark())
print(results)
```

---

## Profiling

Two interfaces available:
- **CLI** (`aisp` command) - for terminal users
- **MCP tools** (`aisp_*`) - for AI assistants via MCP protocol

### Quick Status Check
```bash
aisp info status              # System status overview
aisp info gpu                 # GPU information
aisp hw speed                 # Quick GPU sanity test
```

### Nsight Systems (Timeline)
```bash
aisp profile nsys python train.py --batch 32
aisp profile nsys --preset light python script.py
aisp profile compare ch11     # Compare baseline vs optimized
```

### Nsight Compute (Kernel Deep-Dive)
```bash
aisp profile ncu python gemm_kernel.py
aisp profile ncu --workload-type tensor_core python attention.py
```

### PyTorch Profiler
```bash
aisp profile torch train.py --batch 32
aisp profile torch --mode memory train.py
```

### Bottleneck Analysis
```bash
aisp profile bottleneck       # Identify bottlenecks
aisp profile roofline         # Roofline model analysis
aisp ai ask "Why is my kernel slow?"
```

### Open Results in NVIDIA Nsight
```bash
nsys-ui artifacts/mcp-profiles/mcp_nsys.nsys-rep
ncu-ui artifacts/mcp-profiles/mcp_ncu.ncu-rep
```

---

## GPU Performance Reference

### GPU Hardware Specs

| GPU | FP16 Tensor | FP8 Tensor | Memory | Bandwidth | SMs |
|-----|-------------|------------|--------|-----------|-----|
| **B200** (Blackwell) | 2250 TFLOPS | 4500 TFLOPS | 192 GB HBM3e | 8 TB/s | 148 |
| **H100** SXM (Hopper) | 1979 TFLOPS | 3958 TFLOPS | 80 GB HBM3 | 3.35 TB/s | 132 |
| **A100** SXM (Ampere) | 312 TFLOPS | — | 80 GB HBM2e | 2 TB/s | 108 |

### Key NCU Metrics

**Utilization (Higher = Better)**
| Metric | Good | Warning | Critical | Meaning |
|--------|------|---------|----------|---------|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | >60% | 30-60% | <30% | Compute utilization |
| `gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed` | >70% | 40-70% | <40% | Memory bandwidth |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | >50% | 25-50% | <25% | Occupancy |
| `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` | >50% | 20-50% | <20% | Tensor Core util |

**Stalls (Lower = Better)**
| Metric | Good | Warning | Critical | Meaning |
|--------|------|---------|----------|---------|
| `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` | <10% | 10-30% | >30% | Waiting for L2/DRAM |
| `smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct` | <15% | 15-30% | >30% | Waiting for L1/shared |
| `smsp__warp_issue_stalled_barrier_per_warp_active.pct` | <10% | 10-25% | >25% | Sync overhead |

**Red Flags**
| Metric | Threshold | Issue |
|--------|-----------|-------|
| `l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum` | >0 | Register spills! |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | >1000 | Bank conflicts |

### Bottleneck Decision Tree

```
1. Check utilization:
   ├─ SM >60%, DRAM <40%  → COMPUTE BOUND (good!)
   ├─ DRAM >60%, SM <40%  → MEMORY BOUND (optimize memory)
   └─ Both <40%           → Check occupancy & stalls

2. If both utilizations low:
   ├─ Occupancy <25%      → OCCUPANCY LIMITED
   │   └─ Check: registers, shared mem, block size
   ├─ Top stall >30%      → STALL DOMINATED
   │   └─ Fix based on stall type
   └─ Local mem >0        → REGISTER SPILLS (critical!)

3. Stall fixes:
   ├─ long_scoreboard     → prefetch, async copy, better locality
   ├─ barrier/membar      → reduce syncs, warp-level primitives
   ├─ short_scoreboard    → increase ILP, unroll, reorder
   └─ math_pipe_throttle  → Already compute-bound (good)
```

### Roofline Analysis

```python
# Calculate arithmetic intensity
ai = flops / bytes_transferred  # FLOP/byte

# Ridge point (where compute meets memory)
ridge_point = peak_tflops / peak_bandwidth_tbps

# If AI < ridge_point: memory bound
# If AI > ridge_point: compute bound

# H100 FP16: ridge = 1979 / 3.35 = 591 FLOP/byte
# GEMM M=N=K=4096: AI ≈ 682 FLOP/byte → compute bound ✓
```

### Optimization Patterns

| Bottleneck | Fix |
|------------|-----|
| **Memory Bound** | Tiling, vectorized loads (float4), shared memory, async copy (cp.async) |
| **Low Occupancy** | Reduce registers/shared mem, try block sizes 128/256/512, `__launch_bounds__` |
| **Register Spills** | Reduce array sizes, limit unrolling, split kernels |
| **Bank Conflicts** | Pad shared memory `smem[N][M+1]`, swizzle patterns |
| **Warp Divergence** | Sort data by operation, use predication over branches |

### Kernel Optimization Hints

| Kernel Type | Key Optimization | Why |
|-------------|------------------|-----|
| GEMM | Tensor Cores + Tiling | O(N³) compute, O(N²) data |
| Attention | FlashAttention | Fuses softmax, reduces memory |
| Elementwise | Fusion | Memory bound, reduce launches |
| Reduction | Warp shuffle | Tree reduction in registers |
| Softmax | Online algorithm | Single pass, O(N) memory |
| LayerNorm | Fused kernel | Avoid multiple reads |

### Anti-Patterns to Avoid

1. **Uncoalesced access** — Threads accessing scattered memory
2. **Excessive syncs** — `__syncthreads()` in tight loops
3. **Serial reductions** — Use parallel tree reduction
4. **Register spills** — Watch `l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum`
5. **Small kernels** — Fuse or batch to amortize launch overhead
6. **FP32 on Tensor Cores** — Use FP16/BF16/FP8 for 10x+ speedup
7. **Unaligned matrices** — Pad to multiples of 16/32 for Tensor Cores

---

## MCP Integration (aisp MCP server)

Start the server with `python -m mcp.mcp_server --serve` (or via `mcp.json`). Responses emit a `text` content entry containing the JSON envelope. Tool descriptions returned by `tools/list` (or `python -m mcp.mcp_server --list`) embed Inputs/Outputs/Expectations so clients can surface parameter guidance and side-effect hints.

**73 MCP tools available** organized by category. All tools are prefixed with `aisp_`. See [docs/mcp_tools.md](docs/mcp_tools.md) for complete reference with parameters and examples.

### Quick Start (MCP)
```
aisp_triage()                    # GPU + software health + context (START HERE)
aisp_status()                    # Fast health check only
aisp_recommend(model_size=70, gpus=8, goal='throughput')
aisp_ask(question='Why is my attention kernel slow?')
```

### Benchmarking
| Tool | Description |
|------|-------------|
| `aisp_run_benchmarks` | Run benchmarks with optional profiling |
| `aisp_benchmark_targets` | List benchmark targets by chapter |
| `aisp_list_chapters` | Scan and list all chapters |
| `aisp_benchmark_report` | Generate benchmark report |
| `aisp_benchmark_export` | Export benchmarks to file |
| `aisp_benchmark_compare_runs` | Compare benchmark runs |
| `aisp_benchmark_triage` | Analyze results and get recommendations |
| `aisp_hw_speed` | Run quick GPU speed tests |
| `aisp_hw_disk` | Disk I/O benchmark |
| `aisp_hw_pcie` | PCIe H2D/D2H bandwidth test |
| `aisp_hw_cache` | Memory hierarchy stride test |
| `aisp_hw_tc` | Tensor Core throughput test |
| `aisp_hw_network` | Network throughput tests |
| `aisp_hw_roofline` | Roofline model benchmark |
| `aisp_hw_p2p` | GPU-to-GPU P2P bandwidth test |
| `aisp_hw_nccl` | NCCL collective bandwidth test |
| `aisp_hw_ib` | InfiniBand bandwidth test |

### GPU
`aisp_gpu_info`, `aisp_gpu_bandwidth`, `aisp_gpu_topology`, `aisp_gpu_topology_matrix`, `aisp_gpu_power`

### System
`aisp_system_software`, `aisp_system_dependencies`, `aisp_system_context`, `aisp_system_capabilities`, `aisp_system_full`, `aisp_status`, `aisp_triage`, `aisp_context_summary`, `aisp_context_full`, `aisp_ai_status`

### Profiling
| Tool | Description |
|------|-------------|
| `aisp_profile_nsys` | Run Nsight Systems profiling |
| `aisp_profile_ncu` | Run Nsight Compute profiling |
| `aisp_profile_torch` | Run PyTorch profiler |
| `aisp_profile_hta` | Holistic Trace Analysis |
| `aisp_profile_compare` | Visual flame graph comparison |
| `aisp_nsys_summary` | Quick nsys stats |
| `aisp_compare_nsys` | Compare nsys profiles |
| `aisp_compare_ncu` | Compare ncu profiles |
| `aisp_profile_flame` | Flame graph data |
| `aisp_profile_memory` | VRAM timeline, leak detection |
| `aisp_profile_kernels` | CUDA kernel breakdown |
| `aisp_profile_roofline` | Compute vs memory bound analysis |

### Analysis
`aisp_analyze_bottlenecks`, `aisp_analyze_pareto`, `aisp_analyze_scaling`, `aisp_analyze_stacking`, `aisp_analyze_whatif`, `aisp_analyze_memory_patterns`, `aisp_analyze_dataloader`, `aisp_analyze_comm_overlap`, `aisp_analyze_energy`

### Optimization
`aisp_recommend`, `aisp_optimize_roi`, `aisp_optimize_techniques`

### Distributed
`aisp_distributed_plan`, `aisp_distributed_nccl`, `aisp_launch_plan`, `aisp_cluster_slurm`, `aisp_predict_scaling`

### Inference
`aisp_inference_vllm`, `aisp_inference_quantization`

### AI/LLM
`aisp_ask`, `aisp_explain`, `aisp_suggest_tools`

### HuggingFace
`aisp_hf` (search, trending, download)

### Cost
`aisp_cost_estimate`

### Export
`aisp_export_csv`, `aisp_export_pdf`, `aisp_export_html`

### Job Management
`aisp_job_status` (poll async jobs)

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
# In run_benchmarks.py
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
- `python core/harness/run_benchmarks.py --targets ch*` for regression suites
- `artifacts/` holds run outputs; clean via `python cleanup.py`
