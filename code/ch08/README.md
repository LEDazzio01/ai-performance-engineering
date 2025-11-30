# Chapter 8 - Occupancy & Pipeline Tuning

## Summary
Concentrates on resource balancing: adjust block sizes, registers, and shared memory to keep SMs full while hiding TMEM latency via double buffering, loop unrolling, and async pipelines.

## Learning Goals
- Tune occupancy explicitly and observe how register counts limit resident CTAs.
- Apply double buffering and async staging to overlap DRAM fetch with compute.
- Use tiling, loop unrolling, and AI-specific thresholds to control latency vs throughput.
- Measure how pipelined schedules change SM/TMEM utilization using the shared harness.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_occupancy_tuning.py`, `optimized_occupancy_tuning.py`, `optimized_occupancy_tuning_bs64.py`, `optimized_occupancy_tuning_maxrreg32.py`, `occupancy_api_example.cu`, `occupancy_tuning.cu` | Occupancy studies that sweep CTA shapes, register caps, and API-computed limits. |
| `baseline_double_buffering.cu`, `baseline_double_buffering.py`, `optimized_double_buffering_pipelined.cu`, `optimized_double_buffering.py`, `double_buffering_kernels.cu` | Double-buffered kernels and their Python drivers showing how to keep tensor cores busy. |
| `baseline_hbm.cu`, `baseline_hbm.py`, `optimized_hbm.py`, `optimized_hbm_vectorized.cu`, `hbm_kernels.cu` | HBM streaming workloads that compare scalar, vectorized, and asynchronous fetch patterns. |
| `baseline_loop_unrolling.cu`, `baseline_loop_unrolling.py`, `optimized_loop_unrolling.cu`, `optimized_loop_unrolling.py`, `loop_unrolling_kernels.cu` | Loop unrolling case studies targeting various ILP regimes. |
| `baseline_threshold.py`, `baseline_thresholdtma.py`, `optimized_threshold.py`, `optimized_thresholdtma.py`, `threshold_kernels.cu`, `threshold_tma_benchmark_base.py` | Threshold operators implemented with scalar, vectorized, and TMA-backed pipelines. |
| `baseline_tiling.py`, `baseline_tiling_tcgen05.py`, `optimized_tiling.py`, `optimized_tiling_tcgen05.py`, `tiling_kernels.cu`, `tiling_extension_tcgen05.py` | Tile schedulers for tcgen05 matmuls, including safe fallbacks when tcgen05 isn't available. |
| `compare.py`, `requirements.txt`, `expectations_gb10.json`, `ai_optimization_kernels.cu` | Harness entry, dependencies, regression thresholds, and AI-optimization helper kernels. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ch08
python compare.py --profile none
python cli/aisp.py bench list-targets --chapter ch08
python cli/aisp.py bench run --targets ch08 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Expectation baselines live next to each chapter in `expectations_gb10.json`; refresh with `--update-expectations` after validating new hardware.

## Validation Checklist
- Nsight Compute traces for `optimized_double_buffering_pipelined.cu` should show overlapping smem/ldgst transactions with minimal idle cycles.
- `python optimized_occupancy_tuning.py --report` emits CTA/block configurations hitting the highest occupancy without exceeding register limits.
- `python compare.py --examples threshold` confirms the TMA-backed kernels reducing latency vs scalar reference implementations.

## Notes
- `arch_config.py` exposes toggles for enabling/disabling tcgen05 lowering per GPU so the same scripts work on SM100 and SM121.
- `build/` caches CUDA object files per configuration; clean via `python cleanup.py --include-build` when adjusting toolchains.
