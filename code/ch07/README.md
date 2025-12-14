# Chapter 7 - Memory Access Patterns

## Summary
Teaches how memory layout drives performance: coalesced copies, tiled matmuls, async prefetch, TMA transfers, and shared-memory staging for lookup-heavy workloads.

## Learning Goals
- Measure the gap between scalar, coalesced, and vectorized memory moves.
- Use shared-memory tiling, TMA, and async copy to keep tensor cores saturated.
- Analyze lookup-heavy workloads and mitigate cache-thrashing access patterns.
- Quantify transpose and gather/scatter penalties to justify layout changes.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_copy_scalar.cu`, `baseline_copy_uncoalesced.cu`, `baseline_uncoalesced_copy.py`, `optimized_copy_coalesced.cu`, `optimized_copy_vectorized.cu`, `optimized_copy_vectorized_sm121` | Copy kernels highlighting coalescing, vector width, and warp-level efficiency. |
| `baseline_hbm3e_copy.cu`, `baseline_hbm3e_peak.cu`, `optimized_hbm3e_copy.cu`, `optimized_hbm3e_peak.cu`, `baseline_hbm3ecopy.py`, `optimized_hbm3ecopy.py` | HBM3e peak-bandwidth probes with CUDA and Python harnesses. |
| `baseline_async_prefetch.cu`, `optimized_async_prefetch.cu`, `baseline_tma_copy.cu`, `baseline_tma_copy.py`, `optimized_async_prefetch.py` | Async/TMA samples that overlap global-memory fetch with computation. |
| `baseline_matmul.cu`, `baseline_matmul_cuda.py`, `optimized_matmul_cuda.py`, `optimized_matmul_tiled.cu` | Matmul implementations to contrast naive global-memory access with shared-memory tiling and warp-level reuse. |
| `baseline_lookup.cu`, `baseline_lookup.py`, `optimized_lookup.cu`, `lookup_pytorch.py` | Cache-sensitive lookup workloads demonstrating how to reorganize tables for better locality. |
| `baseline_transpose.cu`, `baseline_transpose.py`, `optimized_copy_vectorized.cu`, `optimized_transpose.py` | Transpose and gather/scatter experiments that show how to minimize bank conflicts. |
| `compare.py`, `Makefile`, `expectations_b200.json`, `memory_access_pytorch.py` | Harness entry, build recipes, expectation thresholds, and PyTorch validation scripts. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python ch07/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch07
python -m cli.aisp bench run --targets ch07 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Expectation baselines live next to each chapter in `expectations_b200.json`; refresh with `--update-expectations` after validating new hardware.

## Validation Checklist
- `python baseline_hbm3ecopy.py --bytes 1073741824` reports noticeably lower GB/s than `optimized_hbm3ecopy.py`, proving vectorization plus async copies work.
- `python compare.py --examples async_prefetch` shows optimized_async_prefetch reducing total kernel count while preserving accuracy.
- Nsight Compute captures of `optimized_matmul_tiled.cu` hit >80% shared-memory bandwidth utilization with minimal bank conflicts.

## Notes
- Toggle `TORCH_COMPILE_MODE` when using the Python matmul wrappers to verify fusion benefits alongside the raw CUDA kernels.
- HBM tooling reads real peak numbers from `benchmark_peak_results_*.json` when present, providing realistic reference ceilings.
