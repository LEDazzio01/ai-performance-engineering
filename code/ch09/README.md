# Chapter 9 - Arithmetic Intensity & Kernel Fusion

## Summary
Explores how to move workloads along the roofline: raise arithmetic intensity with tiling, fuse memory-bound kernels, and deploy CUTLASS/Triton/inline-PTX paths built for Blackwell tensor cores.

## Learning Goals
- Separate compute-bound vs memory-bound behaviors and adjust kernels accordingly.
- Design micro-tiling schedules that balance register pressure with data reuse.
- Leverage CUTLASS and Triton for rapid iteration while keeping custom CUDA fallbacks.
- Fuse reduction-heavy kernels (e.g., norm + activation) to eliminate redundant memory trips.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_compute_bound.py`, `optimized_compute_bound.py`, `baseline_memory_bound.py`, `optimized_memory_bound.py` | Reference kernels that isolate compute vs bandwidth ceilings and demonstrate tuning strategies. |
| `baseline_micro_tiling_matmul.cu`, `baseline_micro_tiling_matmul.py`, `optimized_micro_tiling_matmul.cu`, `optimized_micro_tiling_matmul.py` | Micro-tiling matmuls with explicit register blocking and cp.async prefetch. |
| `baseline_cutlass_gemm.cu`, `baseline_cutlass_gemm.py`, `optimized_cutlass_gemm.cu`, `optimized_cutlass_gemm.py`, `warp_specialized_cuda.cu` | CUTLASS-driven matmuls and warp-specialized kernels showcasing tcgen05 lowering and occupancy tuning. |
| `baseline_fused_l2norm.cu`, `baseline_fused_l2norm.py`, `optimized_fused_l2norm.cu`, `optimized_fused_l2norm.py`, `fusedL2Norm/` | Fusion examples that merge L2 norm + scaling while staying numerically stable. |
| `baseline_triton.py`, `optimized_triton.py`, `warp_specialized_triton.py` | Triton counterparts for quick prototyping and verifying compiler-generated PTX on Blackwell. |
| `baseline_warp_specialization_producer_consumer.py`, `optimized_warp_specialization_producer_consumer.py`, `two_stage_pipeline.cu` | Warp cooperation samples emphasizing producer/consumer pipelines and inline PTX hooks. |
| `compare.py`, `requirements.txt`, `expectations_gb10.json` | Harness hooks plus regression thresholds for every example. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ch09
python compare.py --profile none
python cli/aisp.py bench list-targets --chapter ch09
python cli/aisp.py bench run --targets ch09 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Expectation baselines live next to each chapter in `expectations_gb10.json`; refresh with `--update-expectations` after validating new hardware.

## Validation Checklist
- `python baseline_compute_bound.py --summaries` reports much higher arithmetic intensity than `baseline_memory_bound.py`, matching the roofline plots.
- `python optimized_cutlass_gemm.py --sizes 4096 4096 8192` hits TFLOP/s numbers close to `benchmark_peak_results.json` for the same device.
- `python compare.py --examples fused_l2norm` confirms numerically identical outputs before and after fusion.

## Notes
- `inline_ptx_example.cu` demonstrates how to wrap tcgen05 intrinsics safely with architecture guards.
- `requirements.txt` includes Triton nightly pinning so the kernels track PyTorch 2.10-dev features.
