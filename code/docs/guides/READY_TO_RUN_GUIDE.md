# Ready-to-Run Guide (8x NVIDIA B200)

This runbook walks through the minimum steps required to bring the repository
online on a fresh 8x B200 system, validate the install, and reproduce the
baseline benchmarks that back the docs in this repo. Follow the sections in
order; every command is self-contained so you can copy/paste directly.

---

## 1. Prerequisites

- **Hardware**: 1x host with 8× NVIDIA B200 (148 SM, 180 GB HBM3e each) and
  NVLink/NVSwitch backplane.
- **OS**: Ubuntu 22.04 LTS or Rocky 9 (other distros work if CUDA 13.0 drivers
  are supported).
- **Permissions**: `sudo` access to install drivers and configure NVSwitch.
- **Storage**: 300 GB free for package caches, Nsight traces, and benchmark
  artifacts.
- **Python**: System Python ≥ 3.10 (the setup script manages virtualenvs).

If you are upgrading the NVIDIA driver, plan for a reboot after the first run
of `setup.sh`.

---

## 2. Clone and Configure

```bash
git clone https://github.com/ai-performance-engineering/code.git
cd code
```

> Tip: use a stable branch (e.g. `main` or a release tag). The examples below
> assume `HEAD` of `main`.

---

## 3. One-Command Setup

```bash
sudo ./setup.sh
```

What this does:

1. Installs the NVIDIA 580 driver (if missing) and CUDA 13.0 toolkit.
2. Sets up a Python virtual environment with PyTorch 2.9 (CUDA 13.0 wheels),
   transformer-engine, and support tools (Nsight CLIs, CUTLASS DSL).
3. Compiles core CUDA samples (`ch2`, `ch6`, `ch7`, `ch8`, `ch9`, `ch10`,
   `ch11`, `ch12`) to verify the toolchain.
4. Runs capability checks (`verify_pytorch.py`, `verify_nvlink.py`,
   `verify_cutlass.py`, `verify_working_optimizations.py`).

If the driver was upgraded, rerun `sudo ./setup.sh` after the reboot so the
CUDA packages and Python environment finish installing.

---

## 4. Quick Health Checks

Run these in order after setup:

```bash
# 1. Confirm GPUs and NVLink topology
nvidia-smi

# 2. Verify framework & CUDA availability
./verify_pytorch.py

# 3. Validate NVLink fabric
./verify_nvlink.py

# 4. CUTLASS support (tensor core pipeline tests)
./verify_cutlass.py

# 5. Run the consolidated smoke tests
./run_all_tests.sh
```

Expected highlights (captured Oct 31, 2025):

- `verify_nvlink.py` reports ~250 GB/s P2P and ~273 GB/s NCCL all-reduce.
- `run_all_tests.sh` exits 0 and stores JSON logs under `test_results/`.
- CUDA sample binaries land in each chapter directory (`ch7/hbm3e_peak_bandwidth`,
  `ch10/tma_2d_pipeline_blackwell`, etc.).

If any test fails, re-run the individual script with `--verbose` to inspect the
exact command invocation.

---

## 5. Reproduce Baseline Benchmarks

### Peak Hardware Capabilities

```bash
python3 benchmark_peak.py
```

Outputs JSON in the repo root (`BENCHMARK_PEAK_RESULTS_*.json`). Healthy B200
nodes reproduce the following ranges:

- **HBM3e bandwidth**: 2.7–2.8 TB/s aggregate
- **FP16 compute**: 1.28–1.30 PFLOPS (65% of theoretical peak)
- **NVLink all-reduce**: 270±5 GB/s

### Large GPT Inference Sweep (39.5B params)

```bash
python3 ch16/test_gpt_large_optimized.py \
  --profile-name large_gpt_benchmark \
  --compile reduce-overhead \
  --benchmark-configs configs/ch16/benchmark_gpt_large.yaml
```

Artifacts show up under `test_results/large_gpt_benchmark_*.json`. You should
match the published numbers within ±3%:

| Batch / Seq | Eager (ms) | Compiled (ms) | Peak Memory (GB) |
|-------------|-----------:|--------------:|-----------------:|
| 4 / 2048    | ~516       | ~519          | 80.6 → 82.5      |
| 2 / 4096    | ~535       | ~573          | 159.6 → 163.7    |
| 1 / 8192    | ~576       | ~708          | 159.6 → 159.2    |

Switch to the `configs/ch16/benchmark_gpt_large_max_autotune.yaml` profile to
replicate the `max-autotune` runs (`speedup ≈ 1.00x` across contexts). For FP8
weight-only experiments, run `configs/ch16/benchmark_gpt_large_fp8.yaml` and
compare against `test_results/large_gpt_fp8_weights_*.json`.

### MoE Throughput Sanity Check

```bash
python3 ch16/moe_performance_benchmark.py
```

Use `synthetic_moe_results.json` to confirm the throughput delta between eager
and compiled paths (expect ~11.6K tok/s eager vs. ~10.1K tok/s compiled using
the stock Triton kernels).

---

## 6. Common Troubleshooting

- **Driver mismatch (`CUDA driver version is insufficient`)**  
  Rerun `sudo ./setup.sh`; the script pins R580. Check `nvidia-smi` afterwards.

- **CUTLASS import warnings (`Failed to import CUTLASS lib`)**  
  Harmless. Inductor cleans up compiled caches aggressively; we disable the
  CUTLASS backend in production runs until NVIDIA ships fixed wheels.

- **`torch.compile` regression on >8K sequence**  
  Use the eager path or switch to `--compile max-autotune`. The model-size
  analysis document explains the measured trade-offs.

- **OOM on 39B benchmark**  
  Ensure swap is disabled (prevents stalls) and no other processes use VRAM.
  If running other jobs, reduce batch size or enable tensor-parallel (`tp=2`).

---

## 7. Next Steps

- Deep dive on memory and latency trends: see `MODEL_SIZE_ANALYSIS.md`.
- Plan migrations from A100/H100 fleets: start with `docs/migration_to_sm100.md`.
- Tune architectures beyond the defaults: consult `docs/architecture_guides.md`
  and `docs/TODO.md` for in-progress optimizations.

All automation scripts are idempotent—rerun them whenever the environment
changes or after pulling new commits.

