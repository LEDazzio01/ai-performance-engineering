# Chapter 6 - CUDA Programming Fundamentals

## Summary
Moves from Python into CUDA C++: write first kernels, reason about occupancy, control memory layouts, and experiment with ILP, launch bounds, and unified memory on Blackwell devices.

## Learning Goals
- Write and launch custom kernels that mirror the harness workloads.
- Understand how occupancy, launch bounds, and register pressure interact.
- Use ILP and vectorized memory ops to increase throughput per thread.
- Validate unified memory and allocator tuning on Blackwell GPUs.

## Directory Layout
| Path | Description |
| --- | --- |
| `my_first_kernel.cu`, `simple_kernel.cu`, `baseline_add.cu`, `optimized_add.cu`, `baseline_add.py`, `optimized_add.py` | Hello-world kernels plus Python wrappers for verifying CUDA build chains and launch parameters. |
| `baseline_add_tensors.cu`, `optimized_add_tensors.cu`, `baseline_add_tensors.py`, `optimized_add_tensors.py` | Tensor-oriented adds with automatic pinned-memory staging and correctness checks. |
| `baseline_attention_ilp.py`, `baseline_gemm_ilp.py`, `optimized_gemm_ilp.py`, `optimized_ilp_low_occupancy_vec4.cu`, `optimized_ilp_extreme_low_occupancy_vec4.cu` | Instruction-level parallelism studies that manipulate loop unrolling, registers, and vector width. |
| `baseline_bank_conflicts.cu`, `optimized_bank_conflicts.cu`, `baseline_launch_bounds*.{py,cu}`, `optimized_launch_bounds*.{py,cu}` | Bank conflict and launch-bound exercises to highlight shared memory layouts and CTA sizing. |
| `baseline_autotuning.py`, `optimized_autotuning.py`, `memory_pool_tuning.cu`, `stream_ordered_allocator/` | Autotuning harness plus allocator experiments for controlling fragmentation and stream ordering. |
| `unified_memory.cu`, `occupancy_api.cu`, `baseline_quantization_ilp.py`, `optimized_quantization_ilp.py` | Unified memory demo, occupancy calculator sample, and quantization-focused ILP workloads. |
| `compare.py`, `Makefile`, `expectations_b200.json`, `workload_config.py` | Harness entry, build scripts, expectation baselines, and workload settings. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python ch06/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch06
python -m cli.aisp bench run --targets ch06 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Expectation baselines live next to each chapter in `expectations_b200.json`; refresh with `--update-expectations` after validating new hardware.

## Validation Checklist
- `nvcc -o baseline_add_sm121 baseline_add.cu` vs the optimized vectorized version shows a clear bandwidth delta when inspected with Nsight Compute.
- `python optimized_autotuning.py --search` converges to the same schedule as the curated preset and logs the score table under `artifacts/`.
- `python compare.py --examples ilp` confirms optimized ILP kernels achieving higher instructions-per-byte with identical outputs.

## Notes
- `arch_config.py` forces SM-specific compile flags (e.g., disabling pipelines on unsupported GPUs) so targets fail gracefully on older hardware.
- CUDA extensions in `cuda_extensions/` can be imported directly into notebooks for interactive prototyping.
