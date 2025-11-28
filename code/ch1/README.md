# Chapter 1 - Performance Fundamentals

## Summary
Establishes the baseline benchmarking discipline: measure goodput, remove Python-side stalls, and apply foundational CUDA optimizations such as pinned memory, batched transfers, and CUDA Graphs.

## Learning Goals
- Profile PyTorch training loops with the shared harness to identify memory-transfer and launch bottlenecks.
- Apply pinned memory, tensor preallocation, and larger batch sizes to raise GPU utilization.
- Capture and replay CUDA Graphs to trim kernel launch overhead for steady-state workloads.
- Compare hand-written GEMM kernels in batched vs. strided forms to understand arithmetic intensity.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_performance.py`, `optimized_performance.py` | Goodput-focused training loop pair that toggles pinned memory, CUDA Graphs, and tensor preallocation. |
| `baseline_guided_decoding.py`, `optimized_guided_decoding.py`, `baseline_guided_decoding_math.py`, `optimized_guided_decoding_math.py` | Guided decoding microbenchmark plus math-only validator for deterministic correctness checks. |
| `baseline_ilp_basic.py`, `optimized_ilp_basic.py` | Instruction-level parallelism exercises that expose loop unrolling and register reuse benefits. |
| `baseline_warp_specialization.py`, `optimized_warp_specialization.py` | Warp-specialized producer/consumer example that demonstrates overlapped memory movement. |
| `baseline_gemm.cu`, `optimized_gemm_batched.cu`, `optimized_gemm_strided.cu` | CUDA GEMM variants (single, batched, strided) used to illustrate launch amortization and memory coalescing. |
| `compare.py`, `workload_config.py`, `arch_config.py`, `expectations_gb10.json` | Harness entrypoint, workload shapes, architecture overrides, and stored expectation thresholds. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ch1
python compare.py --profile none
python tools/cli/aisp bench list-targets --chapter ch1
python tools/cli/aisp bench run --targets ch1 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Expectation baselines live next to each chapter in `expectations_gb10.json`; refresh with `--update-expectations` after validating new hardware.

## Validation Checklist
- `python compare.py` reports optimized_performance achieving >=2x tokens/sec vs the baseline on default microbatch sizes.
- Running `make && ./baseline_gemm_sm121` vs `optimized_gemm_batched_sm121` shows a substantial drop in launch count and total runtime.
- `python baseline_guided_decoding_math.py --validate` and the optimized twin agree numerically for every request.

## Notes
- `requirements.txt` pins lightweight extras (Typer, tabulate) used by helper scripts.
- `Makefile` builds the CUDA GEMM binaries with SM-specific suffixes for quick diffing.
