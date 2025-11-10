# AI Systems Performance Engineering: Code

Production playbook for validating, tuning, and documenting PyTorch/CUDA/Triton workloads on multi-GPU Blackwell systems.

---

## 1. Overview
- **Target hardware**: NVIDIA Blackwell (B200/B300, sm100), Grace Blackwell (GB200/GB300, sm103), and DGX Spark (GB10, sm121).
- **Reference stack**: CUDA 13+, PyTorch 2.9+, Triton 3.5+, Python 3.10+.
- **What’s inside**: 20 chapters of baseline/optimized benchmark pairs, harness tooling, profiling scripts, workload configs, and production-ready setup automation.
- **Primary goals**: stand up consistent lab machines, capture comparable artifacts (JSON/MD/CSV), and keep proof-of-benefit (PoB) reports current as kernels evolve.

### Concepts by Chapter

| Chapter | Concepts Covered | # Examples |
|---------|------------------|------------|
| **ch1** | Performance basics, profiling, memory management, CUDA Graphs, batched ops, roofline, arithmetic intensity | 21 |
| **ch2** | GPU architecture, NVLink, CPU-GPU coherency, hierarchy inspection | 6 |
| **ch3** | System tuning (NUMA, governors, THP, IRQs, Docker/K8s) | 11 |
| **ch4** | Multi-GPU training, NCCL, tensor/pipeline parallelism, NVSHMEM | 7 |
| **ch5** | Storage/IO optimization, GDS, DataLoader tuning, cuFile | 7 |
| **ch6** | CUDA fundamentals, thread hierarchy, occupancy, unified memory | 14 |
| **ch7** | Memory access patterns, coalescing, tiling, shared-memory bandwidth | 20 |
| **ch8** | Occupancy tuning, ILP, loop unrolling, warp divergence, resource limits | 13 |
| **ch9** | Arithmetic intensity, roofline, kernel fusion, CUTLASS | 17 |
| **ch10** | Tensor cores, TMA, async pipelines, warp specialization, clusters | 17 |
| **ch11** | CUDA streams, concurrency, Hyper-Q, overlap optimization | 11 |
| **ch12** | CUDA Graphs, conditional graphs, dynamic launches | 14 |
| **ch13** | PyTorch profiling, memory analysis, FSDP, quantization, allocators | 23 |
| **ch14** | `torch.compile`, Triton kernels, FP8, custom codegen | 5 |
| **ch15** | Disaggregated inference, KV cache, continuous batching | 10 |
| **ch16** | Inference optimization, production serving, speculative decoding | 8 |
| **ch17** | Dynamic routing, roofline trade-offs, latency vs. accuracy | 13 |
| **ch18** | Advanced attention (Flash/Flex/MLA/Paged) | 18 |
| **ch19** | Low-precision training (FP4/FP6/FP8), Transformer Engine | 15 |
| **ch20** | End-to-end optimization case studies | 14 |

### Learning Themes
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
2. `python3 tools/verification/verify_all_benchmarks.py` – import & syntax check.
3. Optional smoke pass:  
   `python tools/cli/benchmark_cli.py ch1 --artifacts-dir ./artifacts --smoke-test`.

### Repository Layout
```text
code/
├── setup.sh
├── ch1...ch20/          # Chapter walkthroughs with READMEs and benchmarks
├── common/              # Harness, logging, workload configs
├── tools/               # Verification, analysis, benchmarking helpers
├── scripts/             # Profiling/probing utilities
└── tests/               # Integration tests
```

### Workflow Checklist
1. Define baseline/optimized pairs with shared workload configs.
2. Warm up kernels and cache graph compilations before timing.
3. Capture smoke runs for quick validation, then full runs for production numbers.
4. Record artifacts (`benchmark_test_results.json`/`.md`, logs) under timestamped folders.
5. Run `tools/analysis/generate_proof_of_benefit.py` after capturing new artifacts.
6. Document findings (speedup, throughput, applied optimizations) for reproducibility.

---

## 3. Benchmark Workflow

### Main CLI (Typer)
Recommended entry point for structured runs:
```bash
# Full suite
python tools/cli/benchmark_cli.py

# Single chapter
python tools/cli/benchmark_cli.py ch12 --artifacts-dir ./artifacts

# Custom options
python tools/cli/benchmark_cli.py ch10 --timeout-multiplier 2.0 --reproducible --cold-start
python tools/cli/benchmark_cli.py ch18 --profile  # collect Nsight/torch profiler traces
```

### Legacy Runner / Targeted Examples
`tools/testing/run_all_benchmarks.py` discovers every `baseline_*.py` file, pairs it with the matching optimized implementation, and emits PoB-friendly summaries.
```bash
PYTHONPATH=. python tools/testing/run_all_benchmarks.py --chapter ch10 --only-examples cluster_group_no_dsmem
PYTHONPATH=. python tools/testing/run_all_benchmarks.py --chapter all --timeout-multiplier 2.0
```
Artifacts land under `artifacts/<timestamp>/<example>/results/benchmark_test_results.{json,md}` with manifests, configuration snapshots, and environment logs.

### Options & Controls
- `--timeout-multiplier`, `--suite-timeout` – stretch or cap run time.
- `--reproducible` – force deterministic seeds and algorithms.
- `--cold-start` – extra cleanup between benchmarks (includes garbage collection and CUDA context resets).
- `--profile/--no-profile` – opt into Nsight/Torch tracing when needed.
- `BENCHMARK_SMOKE_TEST=1` – shrink workloads via shared configs.

### Reporting & Proof-of-Benefit
1. Run the desired benchmarks (smoke or full) and stash artifacts.
2. Invoke `python tools/analysis/generate_proof_of_benefit.py --artifacts-dir artifacts --output-csv reports/proof_of_benefit.csv`.
3. Review `reports/proof_of_benefit.csv` for entries still below the 1.05× goal.

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

### Next Steps
- Capture new artifacts whenever kernels change meaningfully.
- File issues or PRs with benchmark diffs plus the associated artifact bundle for reproducibility.

---

## Platform Caveats & One-Offs

### GB10 / SM121 tcgen05 support gap
- CUDA 13.0 + driver 580.95 on GB10 (SM 12.1) lacks the multicast/TMA features required for CUTLASS tcgen05 lowering.
- `ptxas` fails with `Feature '.multicast::cluster::all' not supported on .target 'sm_121'`, so no cubin/SASS is produced.
- Workarounds: run on hardware that natively exposes SM100+/SM110 tcgen05 (B200/B300) or wait for a CUDA/firmware drop that enables the feature on SM121. Until then, tcgen05 examples stay in `tools/` as diagnostics rather than chapter benchmarks.

### Thread block clusters without DSMEM (ch10)
- GB10/SM121 exposes thread block clusters but not Distributed Shared Memory (DSMEM), so DSMEM-enabled cluster kernels cannot launch.
- `ch10` now ships DSMEM-free binaries (`baseline_cluster_group_no_dsmem.*`, `optimized_cluster_group_no_dsmem.*`). Build them with:  
  `make -C ch10 baseline_cluster_group_no_dsmem_sm121 optimized_cluster_group_no_dsmem_sm121`
- Run via the harness:  
  `PYTHONPATH=. python tools/testing/run_all_benchmarks.py --chapter ch10 --only-examples cluster_group_no_dsmem`
- `optimized_cluster_group.py` automatically falls back to the DSMEM-free variant whenever `cudaDevAttrClusterLaunch` or DSMEM capability checks fail, keeping PoB runs valid on SM121 hardware.
