# B200 Large-Model Benchmark Status

## Executive Summary
- After installing the NVIDIA CUTLASS DSL wheel (`nvidia-cutlass-dsl==4.2.1`), max-autotune with `torch.compile` now delivers up to **1.03x** on shorter batches, but still hovers ~1.00x on 8K+ tokens.
- Tensor-parallel execution across two B200s still halves per-GPU memory (~41 GB) at the cost of additional NVLink latency.
- Fused FP8 weight-only linears (using `torch._scaled_mm`) hold throughput within 1-2% of FP16 while dropping peak memory to ~42 GB.
- TorchInductor continues to log CUTLASS import failures (`cuda.bindings` during cleanup), so Triton kernels remain the fallback despite the new package.

Raw measurements and metadata:
- `test_results/large_gpt_benchmark_20251028_135159.json` (reduce-overhead, warmup 1 / iters 2)
- `test_results/large_gpt_benchmark_max_autotune_20251028_140501.json` (max-autotune without CUTLASS, warmup 3 / iters 6)
- `test_results/large_gpt_benchmark_max_autotune_cutlass_20251028_161008.json` (max-autotune with CUTLASS DSL 4.2.1, warmup 3 / iters 6, `attention=sdpa`)
- `test_results/large_gpt_tp2_20251028_143806.json` (tensor-parallel eager prototype)
- `test_results/large_gpt_fp8_weights_20251028_151819.json` (legacy FP8 weight-only eager prototype)
- `test_results/large_gpt_fp8_fused_20251028_171215.json` (FP8 weight-only fused via `_scaled_mm`, eager)

> **Latest hardware check (2025‑10‑31):** benchmark automation ran on a single
> NVIDIA GB10 (SM 12.1). Those artifacts do **not** represent B200 performance,
> so the tables below continue to reference the validated 2025‑10‑28 B200 runs.
> Re-run the large-model suite on Blackwell hardware before replacing the
> numbers.

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

### torch.compile mode = `max-autotune` (CUTLASS DSL 4.2.1, warmup 3, iters 6, attention=sdpa)
| Configuration        | Eager (ms) | Compiled (ms) | Speedup | Eager Throughput (tok/s) | Compiled Throughput (tok/s) | Peak Memory (GB) |
|---------------------|-----------:|--------------:|--------:|-------------------------:|-----------------------------:|-----------------:|
| Batch=8, Seq=1024   | 513.36     | 496.57        | 1.03x   | 15,958                   | 16,497                      | 80.6 -> 80.2     |
| Batch=4, Seq=2048   | 507.20     | 510.93        | 0.99x   | 16,151                   | 16,034                      | 80.6 -> 80.5     |
| Batch=2, Seq=4096   | 725.54     | 743.31        | 0.98x   | 11,291                   | 11,021                      | 80.6 -> 80.3     |
| Batch=2, Seq=8192   | 2,352.52   | 2,338.95      | 1.01x   | 6,964                    | 7,005                       | 82.2 -> 80.9     |
| Batch=1, Seq=12K    | 1,841.68   | 1,824.08      | 1.01x   | 6,672                    | 6,737                       | 81.4 -> 81.7     |
| Batch=1, Seq=16K    | 2,536.88   | 2,521.58      | 1.01x   | 6,458                    | 6,498                       | 82.2 -> 81.3     |

Notes:
- CUTLASS kernels still emit repeated `Failed to import CUTLASS lib` warnings (cleanup path cannot load `cuda.bindings`), but the shorter workloads now show a measurable 3% gain.
- Longer sequences remain memory-bound; the compile path tracks eager within ~1%.

### FP8 weight-only (fused via `_scaled_mm`, no torch.compile, warmup 1, iters 3)
| Configuration        | Latency (ms) | Throughput (tok/s) | Peak Memory (GB) |
|---------------------|-------------:|--------------------:|-----------------:|
| Batch=8, Seq=1024   | 516.13       | 15,872              | 41.8             |
| Batch=4, Seq=2048   | 519.41       | 15,772              | 41.8             |
| Batch=2, Seq=4096   | 530.51       | 15,442              | 41.8             |
| Batch=2, Seq=8192   | 1,100.84     | 14,883              | 43.7             |
| Batch=1, Seq=12K    | 856.39       | 14,349              | 42.7             |
| Batch=1, Seq=16K    | 1,184.14     | 13,836              | 43.7             |

Notes:
- Linear weights are stored in FP8 with per-output scaling; activations are quantised row-wise to FP8 on the fly and multiplied via `torch._scaled_mm` to avoid re-materialising FP16 weights.
- Peak memory drops by ~35% relative to FP16 weights (80.6 GB -> 41.8 GB for 2-4K token runs) while staying within ~1-2% of FP16 throughput up to 4K tokens.
- Longer contexts still trend memory-bound; further fusion (FlashAttention, pipeline overlap) will be needed to maintain the gap beyond 8K tokens.

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
- Fused FP8 weight-only storage frees >35% of HBM while nearly matching FP16 throughput on 2-4K contexts; remaining work is to extend the benefit to 8K+ tokens.

## Recommendations & Next Experiments
1. Debug the remaining CUTLASS import failure (`cuda.bindings` during `move_cutlass_compiled_cache`) so Inductor can retain CUTLASS kernels without warnings.
2. Push the fused FP8 weight-only path to 8K+ tokens (larger tiles, fused activation quantization) and benchmark under `torch.compile`.
3. Extend the tensor-parallel prototype with scatter/gather overlap or pipeline parallelism to hide NVLink delays.
4. Profile with Nsight Compute or PyTorch Profiler to pinpoint the dominant memory-bound kernels (FlashAttention vs. MLP) before writing custom kernels.
