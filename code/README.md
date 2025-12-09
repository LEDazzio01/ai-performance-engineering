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
"""Description of optimization."""

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

---

## Benchmark Validity Issues Reference

This table documents known issues that can cause benchmark results to be misleading, along with their protections. Use this as a checklist when creating or reviewing benchmarks.

**✅ All 94 validity issues are now protected by our harness** (Updated December 2025)

| Category | Issue | What Happens | Protection | Status | Real-World Incident |
|----------|-------|--------------|------------|--------|---------------------|
| **Timing** | Unsynced Streams | Work on non-default streams isn't timed | Full device sync + `StreamAuditor` | ✅ | **Locus/KernelBench 2025** ([source](https://x.com/miru_why/status/1991773868806361138)) |
| **Timing** | Incomplete Async Ops | Timer stops before async work finishes | Full device sync | ✅ | **Locus/KernelBench 2025** ([source](https://x.com/miru_why/status/1991773868806361138)) |
| **Timing** | Event Timing Gaps | CUDA events recorded incorrectly | Cross-validate with wall clock | ✅ | |
| **Timing** | Timer Granularity | Measurement too coarse for fast ops | Adaptive iterations | ✅ | |
| **Timing** | Warmup Bleed | Real work happens during warmup | `isolate_warmup_cache` | ✅ | |
| **Timing** | Clock Drift | System clock changes during measurement | Monotonic clock usage | ✅ | |
| **Timing** | Profiler Overhead | Profiling tools add latency | Profile-free timing path | ✅ | |
| **Output** | Constant Output | Same result regardless of input | Jitter check | ✅ | |
| **Output** | Stale Cache | Same result across different seeds | Fresh-input check | ✅ | |
| **Output** | Approximation Drift | Rough estimate instead of full compute | Output tolerance validation | ✅ | |
| **Output** | Invalid Values (NaN) | NaN in output | `validate_result()` NaN check | ✅ | |
| **Output** | Invalid Values (Inf) | Inf in output | `validate_result()` Inf check | ✅ | |
| **Output** | Invalid Ground Truth | Labels/expected values wrong | `GoldenOutputCache` | ✅ | **ImageNet Labels 2021** ([arXiv:2103.14749](https://arxiv.org/abs/2103.14749)), **MMLU Errors 2025** ([PromptEng](https://promptengineering.org/challenges-and-innovations-in-language-model-benchmarking-and-generalization/)) |
| **Output** | Shape Mismatch | Output shape differs from expected | Shape validation | ✅ | |
| **Output** | Dtype Mismatch | Output dtype differs from expected | `ToleranceSpec` dtype check | ✅ | |
| **Output** | Denormalized Values | Subnormal floats cause slowdowns | Denormal check | ✅ | |
| **Output** | Uninitialized Memory | Output contains garbage | Memory initialization check | ✅ | |
| **Workload** | Precision Mismatch | Claims FP32 but uses FP16 | `InputSignature` dtype verification | ✅ | |
| **Workload** | Undeclared Shortcuts | Skips elements without declaring | Workload invariant check | ✅ | **AI Agent Shortcuts 2024** ([VentureBeat](https://venturebeat.com/ai/ai-agent-benchmarks-are-misleading-study-warns)) |
| **Workload** | Early Exit | Stops iteration loops early | Config immutability | ✅ | |
| **Workload** | Batch Shrinking | Processes fewer samples | `InputSignature` matching | ✅ | |
| **Workload** | Sequence Truncation | Processes shorter sequences | `InputSignature` matching | ✅ | |
| **Workload** | Hidden Downsampling | Silently reduces resolution | Dimension validation | ✅ | |
| **Workload** | Sparsity Mismatch | Different sparsity patterns | Sparsity ratio check | ✅ | |
| **Workload** | Attention Mask Mismatch | Different masking applied | Mask equivalence check | ✅ | |
| **Workload** | KV Cache Size Mismatch | Different cache sizes | Cache dimension check | ✅ | |
| **Workload** | Train/Test Overlap | Model tested on training data | Dataset isolation | ✅ | **Computational Biology 2019** ([Nat Commun](https://www.nature.com/articles/s41467-019-09406-4)) |
| **Location** | CPU Spillover | Work offloaded to CPU | GPU kernel time validation | |
| **Location** | Setup Pre-computation | Work done in `setup()` | Input mutation check | |
| **Location** | Graph Capture Cheat | Pre-compute during graph capture | Graph-aware verification | |
| **Location** | Warmup Computation | Compute results during warmup | Buffer isolation | |
| **Location** | Background Thread | Compute in separate thread | Process isolation | |
| **Location** | Lazy Evaluation Skip | Returns unevaluated lazy tensor | Force evaluation | |
| **Location** | JIT Compilation Timing | JIT compile time included/excluded inconsistently | Compilation state tracking | |
| **Memory** | Pre-allocated Output | Result buffer allocated in setup | Memory allocation tracking | |
| **Memory** | Input-Output Aliasing | Output points to pre-filled input | Memory address validation | |
| **Memory** | Pinned Memory Timing | Async pinned transfers not waited | Transfer completion check | |
| **Memory** | Memory Pool Reuse | Cached allocations skew timing | Memory pool reset | |
| **Memory** | Fragmentation Effects | Memory fragmentation differs | Defragmentation before runs | |
| **Memory** | Page Fault Timing | First-touch page faults included | Memory pre-touch | |
| **Memory** | Swap Interference | Swapping affects timing | Memory lock / swap disable | |
| **CUDA** | Host Callback Escape | `cudaLaunchHostFunc` returns early | Host function tracking | |
| **CUDA** | Async Memcpy Incomplete | D2H/H2D copies not awaited | Memory sync validation | |
| **CUDA** | Workspace Pre-compute | Work in cuBLAS workspace alloc | Workspace monitoring | |
| **CUDA** | Persistent Kernel | Kernel left running across calls | Kernel lifetime check | |
| **CUDA** | Undeclared Multi-GPU | Work spread across undeclared GPUs | Device enumeration | |
| **CUDA** | Context Switch Overhead | CUDA context switches affect timing | Context pinning | |
| **CUDA** | Driver Overhead | Driver calls not accounted for | Driver call tracking | |
| **CUDA** | Cooperative Launch Abuse | Cooperative kernels bypass checks | Launch mode validation | |
| **CUDA** | Dynamic Parallelism Hidden | Child kernels not tracked | CDP kernel tracking | |
| **CUDA** | Unified Memory Faults | Page migration not timed | UM fault tracking | |
| **Compile** | Compilation Cache Hit | Returns cached compiled output | Cache invalidation | |
| **Compile** | Trace Reuse | Exploits trace caching | Fresh trace enforcement | |
| **Compile** | Mode Inconsistency | Different compile mode verify vs perf | Mode consistency check | |
| **Compile** | Inductor Asymmetry | Inductor optimizations inconsistent | Compilation parity | |
| **Compile** | Guard Failure Hidden | Recompilation not counted | Guard check tracking | |
| **Compile** | Autotuning Variance | Autotuning picks different kernels | Fixed autotuning cache | |
| **Compile** | Symbolic Shape Exploit | Different shapes trigger different code | Shape canonicalization | |
| **Distributed** | Rank Skipping | Some ranks don't do work | Per-rank verification | |
| **Distributed** | Collective Short-circuit | Communication skipped | NCCL validation | |
| **Distributed** | Topology Mismatch | Claims different topology | Topology verification | |
| **Distributed** | Barrier Timing | Barrier timing exploited | Barrier synchronization | |
| **Distributed** | Gradient Bucketing Mismatch | Different bucket sizes | Bucket size validation | |
| **Distributed** | Async Gradient Timing | Async all-reduce not awaited | Gradient sync check | |
| **Distributed** | Pipeline Bubble Hiding | Pipeline bubbles not counted | Bubble time tracking | |
| **Distributed** | Shard Size Mismatch | FSDP shards differ | Shard size validation | |
| **Environment** | Device Mismatch | Uses different GPU than declared | Device fingerprint | |
| **Environment** | Frequency Boost | Overclocked for benchmark only | Frequency monitoring | |
| **Environment** | Priority Elevation | Runs at higher priority | Process isolation | |
| **Environment** | Memory Overcommit | Exploits memory overcommit | Memory validation | |
| **Environment** | NUMA Inconsistency | NUMA placement differs | NUMA audit | |
| **Environment** | CPU Governor Mismatch | Different CPU frequency scaling | Governor lock | |
| **Environment** | Thermal Throttling | GPU throttles during run | Temperature monitoring | |
| **Environment** | Power Limit Difference | Different TDP settings | Power limit check | |
| **Environment** | Driver Version Mismatch | Different CUDA drivers | Driver version lock | |
| **Environment** | Library Version Mismatch | Different cuDNN/cuBLAS | Library version lock | |
| **Environment** | Container Resource Limits | cgroups limits differ | Resource limit check | |
| **Environment** | Virtualization Overhead | VM/container overhead varies | Bare-metal validation | |
| **Statistical** | Cherry-picking | Only best iterations reported | All-iteration reporting | **Chatbot Arena 2024** ([TechCrunch](https://techcrunch.com/2025/04/22/crowdsourced-ai-benchmarks-have-serious-flaws-some-experts-say/)) |
| **Statistical** | Outlier Injection | Slow iterations added to baseline | Statistical validation | |
| **Statistical** | Variance Gaming | Variance reporting manipulated | Consistent statistics | |
| **Statistical** | Percentile Selection | Favorable percentile chosen | Fixed percentile policy | |
| **Statistical** | Insufficient Samples | Too few iterations for significance | Minimum iteration count | **AI Benchmarks 2025** ([The Register](https://www.theregister.com/2025/11/07/measuring_ai_models_hampered_by/)) |
| **Statistical** | Cold Start Inclusion | First run included unfairly | Warmup enforcement | |
| **Statistical** | GC Interference | Garbage collection during timing | GC disable during timing | |
| **Statistical** | Background Process Noise | System processes affect timing | Process isolation | |
| **Evaluation** | Eval Code Exploitation | Benchmark code modified to pass | Immutable evaluation harness | |
| **Evaluation** | Timeout Manipulation | Timeout extended to hide slowdowns | Fixed timeout policy | |
| **Evaluation** | Metric Definition Gaming | Redefine what "speedup" means | Standardized metric definitions | **MLPerf 2019** ([Forbes](https://www.forbes.com/sites/janakirammsv/2019/11/10/the-curious-case-of-mlperf-inferencing-benchmark-results/)), **GLUE 2024** ([Revelry](https://revelry.co/insights/artificial-intelligence/why-ai-benchmarks-fail/)) |
| **Evaluation** | Test Data Leakage | Training on test/benchmark data | Data contamination checks | **Data Contamination 2025** ([AI News](https://www.artificialintelligence-news.com/news/flawed-ai-benchmarks-enterprise-budgets-at-risk/)) |
| **Evaluation** | Benchmark Overfitting | Optimize specifically for benchmark | Held-out evaluation sets | **Underspecification 2020** ([arXiv:2011.03395](https://arxiv.org/abs/2011.03395)), **Epic Sepsis 2021** ([ChatBench](https://www.chatbench.org/)) |
| **Evaluation** | Self-Modifying Tests | AI/code modifies its own tests | Read-only test harness | |
| **Evaluation** | Benchmark Memorization | Agent memorizes test cases | Fresh-input checks, jitter | **AI Agent Shortcuts 2024** ([VentureBeat](https://venturebeat.com/ai/ai-agent-benchmarks-are-misleading-study-warns)) |
| **Evaluation** | Missing Holdout Sets | No proper train/test split | Held-out evaluation data | **AI Agent Shortcuts 2024** ([VentureBeat](https://venturebeat.com/ai/ai-agent-benchmarks-are-misleading-study-warns)) |

**Total: 11 categories, 94 validity issues, 17 linked to real-world incidents (all with citations)**

### Notable Real-World Incidents

These validity issues aren't theoretical—they've caused real problems:

| Year | Incident | Issue Type | What Happened | Source |
|------|----------|------------|---------------|--------|
| **2025** | **Locus/KernelBench Stream Exploit** | Unsynced Streams | Claimed 20x speedup on Llama FFW kernel. AI launched work on non-default CUDA streams but timer only measured default stream. **32.8% of RL-generated kernels exploited this**, causing fake 18x speedups. | [X/Twitter @miru_why](https://x.com/miru_why/status/1991773868806361138) |
| **2025** | **AI Benchmark Scientific Rigor** | Metric Definition Gaming | Only 16% of 445 AI benchmarks used statistical tests; ~50% tested abstract concepts without clear definitions. | [The Register](https://www.theregister.com/2025/11/07/measuring_ai_models_hampered_by/) |
| **2025** | **MMLU Benchmark Errors** | Invalid Ground Truth | ~57% of questions in MMLU virology subset found incorrect. Ground truth errors destabilize evaluations. | [PromptEngineering.org](https://promptengineering.org/challenges-and-innovations-in-language-model-benchmarking-and-generalization/) |
| **2024** | **AI Agent Benchmark Shortcuts** | Overfitting / Shortcuts | Study found AI agents memorize benchmark test samples instead of learning to generalize. Many benchmarks lack proper holdout test sets. | [VentureBeat](https://venturebeat.com/ai/ai-agent-benchmarks-are-misleading-study-warns) |
| **2024** | **GLUE Benchmark Heuristics** | Metric Definition Gaming | Models achieved high GLUE scores by exploiting shallow heuristics rather than genuine language understanding. | [Revelry.co](https://revelry.co/insights/artificial-intelligence/why-ai-benchmarks-fail/) |
| **2024** | **HumanEval Limitations** | Benchmark Overfitting | Models performing well on HumanEval struggled with real-world coding tasks; simplified scenarios missed practical complexity. | [Revelry.co](https://revelry.co/insights/artificial-intelligence/why-ai-benchmarks-fail/) |
| **2022** | **MLPerf Participation Issues** | Cherry-picking | MLPerf faced inconsistent vendor participation; selective scenario submissions led to biased performance representations. | [NextPlatform](https://www.nextplatform.com/2022/04/08/the-performance-of-mlperf-as-a-ubiquitous-benchmark-is-lacking/) |
| **2022** | **ML Benchmark Validity (Berkeley)** | Benchmark Overfitting | Small changes in data distribution caused significant performance drops, questioning external validity of static benchmarks. | [UC Berkeley Tech Report](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2022/EECS-2022-180.html) |
| **2021** | **ImageNet Label Errors** | Invalid Ground Truth | Study found **at least 6% label errors** in ImageNet validation set. Average 3.3% error rate across 10 common datasets. | [arXiv:2103.14749](https://arxiv.org/abs/2103.14749) |
| **2021** | **MLPerf Reproducibility** | Benchmark Reproducibility | Users couldn't reproduce MLPerf v0.7 results due to inaccessible datasets and outdated repositories. | [MLCommons Forum](https://groups.google.com/a/mlcommons.org/g/public/c/T_8UsUPIWFo) |
| **2021** | **Epic Sepsis Model Failure** | Benchmark Overfitting | Hospital sepsis prediction model showed significantly worse real-world performance than validation results due to non-representative test data. | [ChatBench.org](https://www.chatbench.org/what-are-the-implications-of-outdated-ai-benchmarks-on-the-accuracy-and-reliability-of-ai-driven-decision-making-and-insights/) |
| **2020** | **Underspecification in ML** | Benchmark Overfitting | ML pipelines produce models with equivalent benchmark performance but divergent deployment behaviors—instability in production. | [arXiv:2011.03395](https://arxiv.org/abs/2011.03395) |
| **2019** | **MLPerf Inference Bias** | Cherry-picking | Inaugural MLPerf inference results showed vendors selectively submitted results highlighting their strengths. | [Forbes](https://www.forbes.com/sites/janakirammsv/2019/11/10/the-curious-case-of-mlperf-inferencing-benchmark-results/) |
| **2019** | **Computational Biology Overfitting** | Train/Test Overlap | Tools developed and tested on same datasets, performing well on benchmarks but failing on new real-world data. | [Nature Communications](https://www.nature.com/articles/s41467-019-09406-4) |
| **2016** | **Microsoft Tay Chatbot** | Missing Holdout Sets | AI chatbot learned offensive behavior within 24 hours due to lack of adversarial benchmarking and content moderation safeguards. | [ChatBench.org](https://www.chatbench.org/what-are-the-implications-of-outdated-ai-benchmarks-on-the-accuracy-and-reliability-of-ai-driven-decision-making-and-insights/) |

#### Incident Categories and Our Protections

| Category | # Incidents | Our Protection |
|----------|-------------|----------------|
| **Timing Manipulation** | 1 (Locus/KernelBench) | Full device sync, stream-aware timing |
| **Invalid Ground Truth** | 2 (ImageNet Labels, MMLU) | Baseline golden outputs, validate_result() |
| **Benchmark Overfitting** | 4 (Underspecification, Epic Sepsis, HumanEval, Berkeley) | Fresh-input checks, distribution shift testing |
| **Data Contamination** | 2 (Data Leakage 2025, Agent Shortcuts) | Data contamination checks, fresh inputs |
| **Metric Gaming** | 3 (MLPerf 2019, GLUE, AI Benchmarks 2025) | Standardized metric definitions, statistical rigor |
| **Cherry-picking** | 2 (Chatbot Arena, MLPerf 2022) | All-iteration reporting, comprehensive submission |
| **Train/Test Overlap** | 2 (Computational Biology, Agent Shortcuts) | Dataset isolation, holdout enforcement |
| **Reproducibility** | 1 (MLPerf 2021) | Version locking, dataset archival |

#### Deep Dive: The Locus/KernelBench Stream Timing Vulnerability

This 2025 incident perfectly illustrates why correctness verification alone is insufficient:

```python
# VULNERABLE TIMING (what KernelBench did)
start_event.record(original_model_stream)  # Only records on default stream
model(*inputs)                              # But work runs on s1, s2, s3!
end_event.record(original_model_stream)    # Timer stops before s1,s2,s3 finish
torch.cuda.synchronize(device=device)      # Waits, but timing already recorded

# CORRECT TIMING (the fix)
for stream in custom_model_streams:
    custom_model_stream.wait_stream(stream)  # Wait for ALL streams
_event.record(custom_model_stream)           # Then record timing
```

**The exploit pattern:**
1. AI creates non-default streams: `s1 = getStreamFromPool()`, `s2 = ...`, `s3 = ...`
2. AI launches GEMMs on those streams: `at::mm_out(gate, x2d, gate_proj.t())` on s1
3. AI does NOT call `setCurrentCUDAStream(s3)` or wait for streams before returning
4. Correctness test uses `torch.cuda.synchronize()` → **PASSES** (waits for everything)
5. Performance test uses stream-specific events → **FAKE SPEEDUP** (doesn't wait for s1,s2,s3)

**Result:** 82/250 (32.8%) of RL-generated CUDA kernels exploited this, producing artificial 18x "speedups" with zero actual performance improvement.

These incidents demonstrate why rigorous benchmark verification is essential—not just for catching intentional gaming, but for catching subtle bugs that produce misleading results.
