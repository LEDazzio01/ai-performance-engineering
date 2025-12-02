# Lab - Persistent Decode & TMA Prefill

## Summary
Demonstrates Blackwell-friendly persistent decode kernels and TMA-powered prefill paths, all validated via Python harnesses plus CUDA/Triton implementations.

## Learning Goals
- Contrast naive decode loops against persistent kernels that pin CTAs per sequence.
- Adopt TMA-based prefill to stream activations into shared memory with minimal latency.
- Benchmark CUDA vs Triton implementations with unified validation utilities.
- Mix CUDA Graphs into the decode path to remove residual launch overhead.
- Bias decode versus prefill work onto high- and low-priority streams to visualize SM scheduling and tail-latency impact.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_persistent_decode.py`, `optimized_persistent_decode_cuda.py`, `optimized_persistent_decode_graphs.py`, `optimized_persistent_decode_triton.py` | Persistent decode variants spanning CUDA, graphs, and Triton. |
| `baseline_tma_prefill_decode.py`, `optimized_tma_prefill_decode.py`, `baseline_native_tma_prefill_decode.py`, `optimized_native_tma_prefill_decode.py` | Prefill workloads illustrating cp.async vs native TMA scheduling. |
| `right_sized_decode.py` | Flag-driven wrapper to compare small/medium/large decode tiers and quantization modes. |
| `kv_locality_microbench.py` | H2D copy microbench comparing pageable vs pinned (NUMA-local/remote) host slabs against HBM-resident tensors. |
| `baseline_nvlink_offload.py`, `optimized_nvlink_offload.py`, `nvlink_offload_common.py` | KV-cache offload microbench contrasting pageable/blocking copies against pinned/async swaps over NVLink-C2C. |
| `baseline_paged_kv_offload.py`, `optimized_paged_kv_offload.py`, `paged_kv_offload_common.py` | Paged/NVMe-style KV-cache microbench that gates FP8 KV on fused FlashAttention availability and contrasts pageable CPU storage against pinned + memmap-backed prefetch. |
| `persistent_decode_common.py`, `tma_extension.py`, `expectations_gb10.json` | Shared helpers, CUDA extension wrappers, and expectation thresholds. |

## Running the Benchmarks
Use the benchmark harness for comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ai-performance-engineering
python -m cli.aisp bench list-targets --chapter labs/persistent_decode
python -m cli.aisp bench run --targets labs/persistent_decode --profile minimal
```
- Targets follow the `labs/persistent_decode:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/persistent_decode:<workload>="--flag value"` to sweep schedule knobs.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/persistent_decode --profile minimal` compares all persistent/TMA variants in one sweep.
- `python labs/persistent_decode/kv_locality_microbench.py --rows 256 --cols 4096 --iters 200` contrasts pageable vs pinned (NUMA-local/remote) host slabs against HBM copies; flags override defaults (no env vars).
- `python -m cli.aisp bench run --targets labs/persistent_decode:kv_locality_microbench --target-extra-arg labs/persistent_decode:kv_locality_microbench="--rows 256 --cols 4096 --iters 200"` drives the harness version with the same flags.
- `python labs/persistent_decode/optimized_persistent_decode_graphs.py --iterations 50` shows lower launch overhead than `baseline_persistent_decode.py`.
- `python labs/persistent_decode/optimized_native_tma_prefill_decode.py --validate` matches the math reference while reporting achieved memory throughput.
- `python -m cli.aisp bench run --targets labs/persistent_decode:paged_kv_offload_baseline labs/persistent_decode:paged_kv_offload_optimized --profile minimal` compares naive FP8 KV with no fusion guard versus the fused-gated, memmap-backed path.
- `optimized_tma_prefill_decode.py` SKIPs automatically when `tma_ready` is false (e.g., GB10 toolchains that cannot emit tensormap instructions yet); expectations include `hw_skip_reason` to mark this as not applicable instead of a regression.
- `python -m cli.aisp bench run --targets labs/persistent_decode:right_sized_decode --target-extra-arg labs/persistent_decode:right_sized_decode="--tier small --quantization int4 --backend triton"` compares right-sized decode tiers; swap `--tier large` or `--backend graphs` to see the impact of bigger slices or graph capture.

## Notes
- The paged KV microbench assumes PyTorch 2.10+ and CUDA 13 with B200/GB200-class hardware for fused FP8 attention; it falls back to FP16 automatically when fusion is unavailable.
- Set `TORCH_COMPILE_MODE` or `TMA_TILE_SIZE` via env vars before invoking the harness to sweep tile sizes.
- Right-sized decode flags (`--tier`, `--quantization`, `--block-k`, `--num-programs`) replace the old env-var driven knobs; pass them via `--target-extra-arg` when using the harness.
- `tma_extension.py` caches builds under `~/.cache/torch_extensions`; clean the cache when switching CUDA versions.
- FlashAttention-3 ≥ 3.1.0 and Transformer Engine ≥ 2.8.0 are required for the FP8/TMA fast paths; the harness aborts on missing dependencies instead of silently degrading.
