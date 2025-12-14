# AI Systems Performance Engineering

## Summary
Reference implementation of high-performance PyTorch, CUDA, and Triton workloads for NVIDIA Blackwell platforms.
The repository packages 20 focused chapters, advanced labs, and the shared benchmarking harness so you can profile baselines, apply optimizations, and capture artifacts that prove performance gains.

## Learning Goals
- Understand how the chapters, labs, and shared tooling fit together.
- Stand up a reproducible environment for PyTorch 2.10-dev + CUDA 13 workloads on Blackwell GPUs.
- Run the benchmark harness directly or through the Typer CLI for automated artifact capture.
- Validate peak hardware characteristics before grading optimizations against stored expectations.

## Directory Layout
| Path | Description |
| --- | --- |
| `ch01` - `ch20` | One directory per chapter with baseline/optimized benchmarks, workload configs, and `compare.py` harness entrypoints. |
| `labs/` | Deep-dive labs for matmul, routing, FlexAttention, MoE, persistent decode, distributed training, and more. |
| `core/benchmark/`, `profiling/`, `core/`, `optimization/`, `analysis/` | Shared harness, logging, workload metadata, profiling, and optimization utilities used by every chapter. |
| `python -m cli.aisp bench` | Typer-based CLI for running and profiling targets with reproducible artifacts. |
| `docs/` + `core/scripts/` | Operational guides, profiling workflows, and setup/reset helpers (`setup.sh`, `cleanup.py`, `reset-gpu.sh`). |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ai-performance-engineering
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements_latest.txt
python -m cli.aisp bench list-targets --chapter ch01
python -m cli.aisp bench run --targets ch01 --profile minimal
```
- `setup.sh` installs system prerequisites (drivers, CUDA, Nsight) and should be rerun after driver upgrades.
- Use `python core/harness/run_benchmarks.py --targets ch*` for automated regression suites.
- `python core/analysis/analyze_expectations.py --artifacts-dir artifacts` compares new runs to stored thresholds.

## Validation Checklist
- `pytest tests/integration` succeeds to confirm harness discovery and CLI plumbing.
- `python core/benchmark/benchmark_peak.py` reports TFLOP/s, bandwidth, and NVLink numbers close to the published ceilings.

## Notes
- `core/scripts/profile_all_workloads.sh` and `ncu_template.ini` capture Nsight traces with consistent metric sets.
- `benchmark_profiles/` and `artifacts/` hold run outputs; clean them via `python cleanup.py` when rotating hardware.
- `docs/perf_intake_and_triage.md` outlines the standard intake bundle for performance investigations.
