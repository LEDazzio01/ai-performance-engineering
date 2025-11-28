# Chapter 19 - Low-Precision Training & Memory Systems

## Summary
Explores NVFP4/FP8 workflows, KV-cache quantization, memory double buffering, and adaptive allocators so low-precision experiments remain numerically safe while squeezing every byte of HBM.

## Learning Goals
- Benchmark FP4/FP6/FP8 training loops with calibration and validation hooks.
- Overlap KV-cache prefetch with compute while respecting precision constraints.
- Implement dynamic quantized caches that switch formats mid-run without drift.
- Design allocator helpers to monitor and rebalance fragmented memory pools.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_nvfp4_training.py`, `optimized_nvfp4_training.py`, `native_fp4_quantization.py`, `native_fp6_quantization.py`, `native_fp8_training.py` | Training and quantization recipes that switch between FP8 and NVFP4 with automatic calibration. |
| `baseline_mxfp8_moe.py`, `optimized_mxfp8_moe.py`, `mxfp8_moe_common.py` | MXFP8 MoE grouped GEMM benchmarks contrasting unfused block quantization with TE-backed tcgen05 pipelines (`--top-k`/`--cuda-graphs` flags supported). |
| `baseline_memory_double_buffering.py`, `optimized_memory_double_buffering.py`, `memory_allocator_with_monitoring.py`, `dynamic_memory_allocator.py`, `_allocator_worker.py` | Memory-management helpers covering double buffering, instrumentation, and adaptive worker pools. |
| `baseline_kv_prefetch_overlap.cu`, `optimized_kv_prefetch_overlap.cu`, `kv_prefetch_overlap_sm121` binaries | CUDA kernels proving that quantized KV prefetch can overlap with compute when using cp.async pipelines. |
| `baseline_dynamic_quantized_cache.py`, `optimized_dynamic_quantized_cache.py`, `dynamic_quantized_cache.py`, `token_precision_switching.py`, `dynamic_precision_switching.py` | Quantized cache management for dynamically switching between precisions based on accuracy budgets. |
| `fp4_hardware_kernel.cu`, `fp8_hardware_kernel.cu`, `custom_allocator_retry.py`, `adaptive_parallelism_strategy.py`, `adaptive_parallelism_worker_pool.py` | Hardware-level kernels and adaptive scheduling helpers for heterogeneous precision fleets. |
| `compare.py`, `arch_config.py`, `expectations_gb10.json` | Harness entry, architecture toggles, and stored expectation data. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ch19
python compare.py --profile none
python tools/cli/aisp bench list-targets --chapter ch19
python tools/cli/aisp bench run --targets ch19 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Expectation baselines live next to each chapter in `expectations_gb10.json`; refresh with `--update-expectations` after validating new hardware.

## Validation Checklist
- `python optimized_nvfp4_training.py --calibrate` warms up with FP8, then switches to NVFP4 and matches the baseline's accuracy thresholds.
- `python optimized_dynamic_quantized_cache.py --trace` logs precision transitions with bounded error, confirming correctness of token-level switching.
- `nvcc -o optimized_kv_prefetch_overlap_sm121 optimized_kv_prefetch_overlap.cu` plus the baseline binary show measurable overlap improvements in Nsight Compute.

## Notes
- `arch_config.py` exposes `ENABLE_NVFP4`/`ENABLE_TF32` toggles per device, making it easy to compare precision recipes.
- `validate_quantization_performance.py` aggregates accuracy vs throughput numbers into CSV form for proof-of-benefit reporting.
