# AI Systems Performance Engineering: Code

Production playbook for validating, tuning, and documenting PyTorch/CUDA/Triton workloads on multi-GPU Blackwell systems.

---

## 1. Overview
- **Target hardware**: NVIDIA Blackwell (B200/B300, sm100), Grace Blackwell (GB200/GB300, sm103), and DGX Spark (GB10, sm121).
- **Reference stack**: CUDA 13+, PyTorch 2.10-dev+, Triton 3.5+, Python 3.10+.
- **What’s inside**: 20 chapters of baseline/optimized benchmark pairs, harness tooling, profiling scripts, workload configs, and production-ready setup automation.
- **Primary goals**: stand up consistent lab machines, capture comparable artifacts (JSON/MD/CSV), and keep proof-of-benefit (PoB) reports current as kernels evolve.

### Concepts by Chapter

| Chapter | Concepts Covered | # Examples |
|---------|------------------|------------|
| **ch1** | Performance basics, profiling, memory management, CUDA Graphs, batched ops, roofline, arithmetic intensity | 7 |
| **ch2** | GPU architecture, NVLink, CPU-GPU coherency, hierarchy inspection | 3 |
| **ch3** | System tuning (NUMA, governors, THP, IRQs, Docker/K8s) | 7 |
| **ch4** | Multi-GPU training, NCCL, tensor/pipeline parallelism, NVSHMEM | 21 |
| **ch5** | Storage/IO optimization, GDS, DataLoader tuning, cuFile | 6 |
| **ch6** | CUDA fundamentals, thread hierarchy, occupancy, unified memory | 10 |
| **ch7** | Memory access patterns, coalescing, tiling, shared-memory bandwidth | 14 |
| **ch8** | Occupancy tuning, ILP, loop unrolling, warp divergence, resource limits | 10 |
| **ch9** | Arithmetic intensity, roofline, kernel fusion, CUTLASS | 8 |
| **ch10** | Tensor cores, TMA, async pipelines, warp specialization, clusters | 18 |
| **ch11** | CUDA streams, concurrency, Hyper-Q, overlap optimization | 13 |
| **ch12** | CUDA Graphs, conditional graphs, dynamic launches | 8 |
| **ch13** | PyTorch profiling, memory analysis, FSDP, quantization, allocators | 21 |
| **ch14** | `torch.compile`, Triton kernels, FP8, custom codegen | 5 |
| **ch15** | Disaggregated inference, KV cache, continuous batching | 9 |
| **ch16** | Inference optimization, production serving, speculative decoding | 8 |
| **ch17** | Dynamic routing, roofline trade-offs, latency vs. accuracy | 13 |
| **ch18** | Advanced attention (Flash/Flex/MLA/Paged) | 5 |
| **ch19** | Low-precision training (FP4/FP6/FP8), Transformer Engine | 9 |
| **ch20** | End-to-end optimization case studies | 7 |

Canonical topic homes (deduplicated): coalescing → ch7; bank conflicts/shared memory → ch6–7; CUTLASS/Triton/tiling → ch9 & ch14; streams/concurrency → ch10–11; tensor/pipeline/data/context parallelism and KV cache/continuous batching → ch4 and ch13–17; attention/Flash attention → ch10 with one serving-focused example in ch15; occupancy/roofline → ch8 & ch10; quantization → ch13 & ch19; NVLink/disaggregated → ch4 & ch17.

### Themes
1. **Performance fundamentals (ch1–ch3)** – profiling, hardware awareness, system tuning.
2. **Kernel optimization (ch6–ch10, ch12)** – occupancy, ILP, async pipelines, CUDA Graphs.
3. **Memory strategies (ch7, ch19)** – coalescing, tiling, precision reduction.
4. **Parallelism & distribution (ch4, ch11, ch13, ch15–ch17)** – multi-GPU, streams, serving.
5. **PyTorch/Triton acceleration (ch13–ch14)** – compiled autograd, Triton tiling.
6. **Attention & advanced workflows (ch18–ch20)** – modern attention designs, production playbooks.

---

## 2. Getting Started

### Requirements
- Root access for `setup.sh` (installs NVIDIA driver 580+, CUDA 13.0+, Nsight tooling).
- Python 3.10+ available on the host.
- At least one supported Blackwell GPU.
- Network access when installing packages or downloading transformer-engine wheels.

### Setup
```bash
git clone <repo-url>
cd ai-performance-engineering/code
sudo ./setup.sh
```
If drivers were upgraded, reboot and rerun `sudo ./setup.sh` to finish verification.

### Transformer Engine Wheels
Rebuild when CUDA/PyTorch versions change:
```bash
scripts/build_transformer_engine_wheel.sh v2.8.0+40c69e7
split -b 50M --numeric-suffixes=0 --suffix-length=2 \
  third_party/TransformerEngine/dist/transformer_engine-2.8.0+40c69e7-cp311-cp311-linux_aarch64.whl \
  vendor/wheels/transformer_engine-2.8.0+40c69e7-cp311-cp311-linux_aarch64.whl.part
split -b 50M --numeric-suffixes=0 --suffix-length=2 \
  third_party/TransformerEngine/dist/transformer_engine_cu12-2.8.0+40c69e7-cp311-cp311-linux_aarch64.whl \
  vendor/wheels/transformer_engine_cu12-2.8.0+40c69e7-cp311-cp311-linux_aarch64.whl.part
cp third_party/TransformerEngine/transformer_engine/pytorch/dist/transformer_engine_torch-2.8.0+40c69e7-cp311-cp311-linux_aarch64.whl \
   vendor/wheels/transformer_engine_torch-2.8.0+40c69e7-cp311-cp311-linux_aarch64.whl
```
`setup.sh` reassembles and installs the wheels automatically.

### Verify Installation
1. `nvidia-smi` – confirm GPU visibility and driver ≥ 580.
2. `python tools/cli/benchmark_cli.py verify` – import & syntax check (runs every chapter by default; add `--targets ch3 --targets ch4:resnet_50` for scoped runs).
3. Run a targeted benchmark suite when you're ready:  
   `python tools/cli/benchmark_cli.py run --targets ch1 --artifacts-dir ./artifacts`.

### Repository Layout
```text
code/
├── setup.sh
├── ch1...ch20/          # Chapter walkthroughs with READMEs and benchmarks
├── labs/                # Capstone labs (Blackwell matmul, FlexAttention, MoE CUDA, full-stack)
├── common/              # Harness, logging, workload configs
├── tools/               # Verification, analysis, benchmarking helpers
├── scripts/             # Profiling/probing utilities
└── tests/               # Integration tests
```

### Workflow Checklist
1. Define baseline/optimized pairs with shared workload configs.
2. Warm up kernels and cache graph compilations before timing.
3. Capture canonical runs for production numbers (tweak `--iterations/--warmup` only when experimenting).
4. Record artifacts (`benchmark_test_results.json`/`.md`, logs) under timestamped folders.
5. Run `tools/analysis/analyze_expectations.py` after capturing new artifacts.
6. Document findings (speedup, throughput, applied optimizations) for reproducibility.

---

## 3. Benchmark Workflow

### Main CLI (Typer)
Recommended entry point for structured runs:
```bash
# Full suite
python tools/cli/benchmark_cli.py run

# Single chapter
python tools/cli/benchmark_cli.py run --targets ch12 --artifacts-dir ./artifacts

# Custom options
python tools/cli/benchmark_cli.py run --targets ch10 --timeout-multiplier 2.0 --reproducible --cold-start
python tools/cli/benchmark_cli.py run --targets ch18 --profile minimal  # collect lightweight Nsight/torch profiler traces
# Expectations workflow: pass `--accept-regressions` to refresh expectation files when
# you intentionally accept new numbers (replaces the old AIPERF_ACCEPT_REGRESSIONS env).
# Harness treats hardware/stack gaps as skips (e.g., TF32 disabled via new helpers,
# DSMEM-unavailable clusters on GB10, NVFP4-only kernels on non-FP4 parts).
```

### Labs via CLI
Capstone labs now live under `labs/`—run them through the same CLI targets:
```bash
python tools/cli/benchmark_cli.py list-targets --chapter labs/blackwell_matmul
python tools/cli/benchmark_cli.py run --targets labs/blackwell_matmul --profile minimal
python tools/cli/benchmark_cli.py run --targets labs/flexattention:flex_attention --profile minimal
python tools/cli/benchmark_cli.py run --targets labs/moe_cuda --profile minimal
python tools/cli/benchmark_cli.py run --targets labs/fullstack_cluster:09_end_to_end --profile minimal
```
Targets follow the usual `chapter:example` format; omit `:example` to sweep every pair in a lab.

**Benchmark discovery rules (important for variants):** the harness pairs `baseline_<name>.py`
with any `optimized_<name>.py` and also any `optimized_<name>_<suffix>.py` in the same directory.
That lets you add multiple optimized steps for a single baseline (e.g., `optimized_router.py` and
`optimized_router_vectorized.py`) and compare them side by side via `list-targets` / `run`.
Select a specific variant by using its target name from `list-targets`, e.g.:
```bash
python tools/cli/benchmark_cli.py list-targets --chapter labs/moe_cuda
python tools/cli/benchmark_cli.py run --targets labs/moe_cuda:kv_transfer_graphs  # baseline_kv_transfer vs optimized_kv_transfer_graphs
python tools/cli/benchmark_cli.py run --targets labs/moe_cuda:router labs/moe_cuda:router_vectorized  # two independent comparisons
```

### Distributed Training Lab (DDP / FSDP / ZeRO)
- Source: `labs/train_distributed` (targets: `ddp`, `fsdp`, `zero1`, `zero2`, `zero3`).
- Quick syntax check: `python -m compileall labs/train_distributed`.
- Harness launch (rank0-only parsing, world size recorded, auto-skip when `--nproc-per-node 1`):  
  `PYTHONPATH=. python tools/cli/benchmark_cli.py run --targets labs/train_distributed:ddp --launch-via torchrun --nproc-per-node 2 --target-extra-arg 'labs/train_distributed:ddp=--compile'`  
  Available flags: `--nnodes`, `--rdzv-backend/--rdzv-endpoint`, `--torchrun-env CUDA_VISIBLE_DEVICES=...`, and per-target overrides via `--target-extra-arg target="--flag value"`.
- Direct torchrun examples:
  - DDP: `torchrun --nproc_per_node 2 labs/train_distributed/ddp.py --mode optimized --compile` (uses DeBERTa v3; errors if process group missing).
  - FSDP: `torchrun --nproc_per_node 2 labs/train_distributed/train_fsdp.py --mode optimized --float8 --sequence-length 512 --steps 2` (uses Llama-3.2-1B; fails fast if world_size < 2 or if FSDP cannot shard).
  - ZeRO optimized variants are tagged multi-GPU and will skip when world_size < 2 during harness runs: `torchrun --nproc_per_node 2 labs/train_distributed/zero2.py --mode optimized --steps 3 --hidden-size 2048 --batch-size 4`. Baseline scripts emit a warning when run single-GPU because sharding benefits won’t appear.
- Next steps: gather tokens/s + peak memory for baseline vs optimized variants on your GPUs and choose the default knobs for future labs; if your stack lacks required kernels for these models, rebuild or switch them (DDP uses DeBERTa v3; FSDP uses Llama-3.2-1B).

### Legacy Runner / Targeted Examples
`tools/testing/run_all_benchmarks.py` discovers every `baseline_*.py` file, pairs it with the matching optimized implementation, and emits PoB-friendly summaries.
```bash
PYTHONPATH=. python tools/testing/run_all_benchmarks.py --targets ch10:cluster_group_no_dsmem
PYTHONPATH=. python tools/testing/run_all_benchmarks.py --targets all --timeout-multiplier 2.0
```
Artifacts land under `artifacts/<timestamp>/<example>/results/benchmark_test_results.{json,md}` with manifests, configuration snapshots, and environment logs.

### Options & Controls
- `--timeout-multiplier`, `--suite-timeout` – stretch or cap run time.
- `--reproducible` – force deterministic seeds/algorithms; slower fallback kernels and ops without deterministic support may fail.
- `--cold-start` – extra cleanup between benchmarks (includes garbage collection and CUDA context resets).
- `--profile none|minimal|deep_dive|roofline` – skip profiling or pick the preset (minimal = low overhead; deep_dive = full metrics); defaults to `none`.
- `--ncu-metric-set auto|minimal|deep_dive|roofline` – override Nsight Compute metric preset (auto follows the chosen profile).
- `--iterations`, `--warmup` – override the default 20/5 sampling when you need quicker local experiments.

### Reporting & Proof-of-Benefit
1. Run the desired benchmarks and stash artifacts.
2. Invoke `python tools/analysis/analyze_expectations.py --artifacts-dir artifacts --output-csv reports/proof_of_benefit.csv` (or point `--output-csv` at another expectations report path).
3. Review the generated CSV for any rows flagged as failed/regressed to identify benchmarks that slipped below their stored expectations.

### Integration Tests
```bash
pytest tests/integration
```
Validates benchmark discovery, harness execution paths, profiling toggles, and failure handling end-to-end.

### Multi-Metric Comparison
```python
from common.python.benchmark_comparison import compare_and_display_all_metrics
summary = compare_and_display_all_metrics(
    baseline_result=baseline,
    optimized_result=optimized,
    name="My Benchmark",
    chapter="ch7",
    include_raw_metrics=False,
)
```
Percentage improvements use delta calculations (lower-is-better and higher-is-better handled automatically). Timing, memory, and profiler metrics can be rendered as tables or summaries.

### Peak Performance Validation
`setup.sh` (and the CLI when no cached results exist) runs `tools/benchmarking/benchmark_peak.py` to capture measured TFLOPS, bandwidth, NVLink, and torch.compile baselines. To rerun manually:
```bash
python tools/benchmarking/benchmark_peak.py
```
Results feed into `performance_targets.py`, letting chapters grade against measured ceilings instead of static expectations.

---

## 4. Tooling & Diagnostics

- **Benchmark Harness Guide**: see `docs/benchmark_harness_guide.md` for architecture, manifest capture, profiling hooks, and logging details.
- **Performance intake & triage**: `docs/perf_intake_and_triage.md` + `scripts/profiling/perf_triage_bundle.sh` for one-page goals and a shareable 30-minute baseline capture. The bundle now drops a full hardware capability report (`hardware_capabilities.txt`) alongside GPU/CPU/NVLink snapshots and optional Nsight/dmon runtime traces. Example harness capture (no Nsight): `scripts/profiling/perf_triage_bundle.sh --output-root ./artifacts --tag harness_ch1_coalescing --duration 120 --no-nsys -- python tools/testing/run_all_benchmarks.py --targets ch1:coalescing --format json --output artifacts/perf_triage_harness_coalescing.json`.
- **Utility probes**: `tools/tcgen05_probe.cu` is a minimal CUTLASS 4.2 driver for tcgen05 tensor cores. Build directly with NVCC or the vendored CUTLASS CMake wrapper to confirm toolchain support.
- **Scripts folder**: reusable Nsight workflows, performance sweeps, and GPU resilience helpers.
- **Common helpers**: `common/python` hosts workload configs, CUDA binary wrappers, NVCC arch detection, logging, and run manifests shared by every chapter.

---

## 5. Maintenance & Troubleshooting

### Cleanup
Remove generated binaries, artifacts, and caches:
```bash
python cleanup.py
```

### GPU Reset
```bash
sudo ./reset-gpu.sh
```
Stops running processes, toggles persistence mode, reloads NVIDIA kernel modules, and performs PCIe/NVML resets. Root privileges are required.

User-space helper:
```bash
python tools/reset_gpu.py [--device N] [--full-system]
```
- Default: kill compute PIDs, clear torch CUDA caches, best-effort `nvidia-smi --gpu-reset`.
- `--full-system` (root): also stops auxiliary services, reloads NVIDIA modules, and issues PCIe function-level resets.

### Next Steps
- Capture new artifacts whenever kernels change meaningfully.
- File issues or PRs with benchmark diffs plus the associated artifact bundle for reproducibility.

---

## Platform Caveats & One-Offs

### GB10 / SM121 tcgen05 support gap
- CUDA 13.0 + driver 580.95 on GB10 (SM 12.1) lacks the multicast/TMA features required for CUTLASS tcgen05 lowering.
- `ptxas` fails with `Feature '.multicast::cluster::all' not supported on .target 'sm_121'`, so no cubin/SASS is produced.
- When external builds insist on `sm_121a`, the benchmarking harness uses `tools/benchmarking/.ptxas_shim/ptxas` to rewrite `sm_121a → sm_121` before invoking the real `ptxas`.
- For PyTorch/Triton runs on GB10, `arch_config.configure_pytorch_optimizations()` clamps `TORCH_CUDA_ARCH_LIST`/`CMAKE_CUDA_ARCHITECTURES`/`CUDAARCHS` to `12.0`/`120` to avoid tcgen05/tensormap opcode rejects from `ptxas` (SM 121/121a unsupported in CUDA 13.0).
- Workarounds: run on hardware that natively exposes SM100+/SM103 tcgen05 (B200/B300) or wait for a CUDA/firmware drop that enables the feature on SM121. Until then, tcgen05 examples will be skipped.

### Thread block clusters without DSMEM (ch10)
- GB10/SM121 exposes thread block clusters but not Distributed Shared Memory (DSMEM), so DSMEM-enabled cluster kernels cannot launch.
