# B200 Large-Model Benchmark Status

## Executive Summary
- Ran the 39.5B parameter GPT benchmark with both `reduce-overhead` and `max-autotune`; the latter yields only ~0.6-1.0% gains even with six timed iterations.
- Added tensor-parallel execution across two B200s; per-GPU memory shrank to ~41 GB, but latency increased because of NVLink transfers.
- Prototyped FP8 weight-only linear layers (per-output scaling): model memory fell to ~55 GB, though per-iteration dequantization cut throughput by ~10-15%.
- Installed CUTLASS sources plus NVIDIA Python bindings, yet TorchInductor still warns that `cutlass._mlir` is missing, so CUTLASS kernels remain disabled.

Raw measurements and metadata:
- `test_results/large_gpt_benchmark_20251028_135159.json` (reduce-overhead, warmup 1 / iters 2)
- `test_results/large_gpt_benchmark_max_autotune_20251028_140501.json` (max-autotune without CUTLASS, warmup 3 / iters 6)
- `test_results/large_gpt_benchmark_max_autotune_cutlass_20251028_150143.json` (max-autotune with CUTLASS packages installed - still falls back to Triton/ATen)
- `test_results/large_gpt_tp2_20251028_143806.json` (tensor-parallel eager prototype)
- `test_results/large_gpt_fp8_weights_20251028_151819.json` (FP8 weight-only eager prototype)

## Benchmark Configuration
- Architecture: 48 layers, `d_model=8192`, `n_heads=64`, `d_ff=32768`
- Dtype: FP16 (weights occupy ~79 GB before quantization)
- Workloads: forward-only inference
  - `(batch=4, seq=2048)`
  - `(batch=2, seq=4096)`
  - `(batch=1, seq=8192)`
- Warmup / iters: noted per experiment below
- torch.compile modes: `reduce-overhead` and `max-autotune`

## Measured Results
### torch.compile mode = `reduce-overhead` (warmup 1, iters 2)
| Configuration        | Eager (ms) | Compiled (ms) | Speedup | Eager Throughput (tok/s) | Compiled Throughput (tok/s) | Peak Memory (GB) |
|---------------------|-----------:|--------------:|--------:|-------------------------:|-----------------------------:|-----------------:|
| Batch=4, Seq=2048   | 516.31     | 518.89        | 0.99x   | 15,866                   | 15,788                       | 80.6 -> 82.5     |
| Batch=2, Seq=4096   | 534.85     | 573.33        | 0.93x   | 15,317                   | 14,288                       | 159.6 -> 163.7   |
| Batch=1, Seq=8192   | 576.05     | 708.02        | 0.81x   | 14,221                   | 11,570                       | 159.6 -> 159.2   |

Notes:
- Activation tensors stay small (~0.13 GB), but transformer state reaches ~160 GB.
- Compiled runs shift slightly more memory, and speedups are negative without autotune.

### torch.compile mode = `max-autotune` (warmup 3, iters 6)
| Configuration        | Eager (ms) | Compiled (ms) | Speedup | Eager Throughput (tok/s) | Compiled Throughput (tok/s) | Peak Memory (GB) |
|---------------------|-----------:|--------------:|--------:|-------------------------:|-----------------------------:|-----------------:|
| Batch=4, Seq=2048   | 512.83     | 512.29        | 1.00x   | 15,974                   | 15,991                       | 80.6 -> 82.5     |
| Batch=2, Seq=4096   | 536.11     | 532.57        | 1.01x   | 15,280                   | 15,382                       | 159.6 -> 163.7   |
| Batch=1, Seq=8192   | 1,307.05   | 1,304.82      | 1.00x   | 6,268                    | 6,278                        | 80.6 -> 80.5     |

Notes:
- Max-autotune removes regressions but gains stay within measurement noise (~0.6% at 4K tokens).
- Even with CUTLASS sources/bindings and `TORCHINDUCTOR_CUTLASS_DIR` set, Inductor logs `ModuleNotFoundError: cutlass._mlir` and skips CUTLASS backends.

### FP8 weight-only prototype (no torch.compile, warmup 1, iters 3)
| Configuration        | Latency (ms) | Throughput (tok/s) | Peak Memory (GB) |
|---------------------|-------------:|--------------------:|-----------------:|
| Batch=4, Seq=2048   | 613.12       | 13,361              | 54.9             |
| Batch=2, Seq=4096   | 635.99       | 12,881              | 54.9             |
| Batch=1, Seq=8192   | 677.96       | 12,083              | 54.9             |

Notes:
- MLP feed-forward layers and the output head now store weights as FP8 with per-output scaling buffers; activations remain FP16.
- Peak memory drops ~31% compared with FP16 weights (80.6 GB -> 54.9 GB).
- Throughput falls 10-15% because dequantization currently happens outside the matmul; fused FP8 kernels are still required for speedups.

### Tensor-parallel eager prototype (2 GPUs, warmup 0, iters 1)
| Configuration        | Latency (ms) | Throughput (tok/s) | Peak Memory / GPU (GB) |
|---------------------|-------------:|--------------------:|-----------------------:|
| Batch=4, Seq=2048   | 1,458.68     | 5,616               | 41.1                   |
| Batch=2, Seq=4096   | 559.40       | 14,644              | 41.1                   |
| Batch=1, Seq=8192   | 597.33       | 13,714              | 41.1                   |

Notes:
- Even partitioning layers across `cuda:0`/`cuda:1` halves per-GPU memory, but NVLink transfers stall medium-length sequences.
- Communication overlap and pipelining are needed before tensor-parallel shows net throughput gains.

## Interpretation
- Long contexts keep the workload memory-bound; `torch.compile` fusion alone does not overcome the HBM ceiling.
- Max-autotune slightly improves runtime but sits within ~1% of eager; meaningful gains likely require CUTLASS kernels or deeper fusion.
- Two-way tensor parallelism doubles available memory headroom, yet added NVLink latency negates speedups without overlap strategies.
- FP8 weight-only storage cuts ~25 GB of HBM at the cost of lower throughput; integrating quantization into the matmul is the next logical step.

## Recommendations & Next Experiments
1. Resolve the missing `cutlass._mlir` dependency so TorchInductor can actually emit CUTLASS kernels; re-run max-autotune afterward.
2. Fuse the FP8 weight-only path with CUTLASS or Triton kernels so dequantization happens inside the GEMM rather than materializing FP16 weights.
3. Extend the tensor-parallel prototype with scatter/gather overlap or pipeline parallelism to hide NVLink delays.
4. Profile with Nsight Compute or PyTorch Profiler to pinpoint the dominant memory-bound kernels (FlashAttention vs. MLP) before writing custom kernels.
