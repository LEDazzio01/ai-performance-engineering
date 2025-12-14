# Lab - Persistent Decode & TMA Prefill

## Summary
Demonstrates Blackwell-friendly persistent decode kernels and TMA-powered prefill paths, all validated via Python harnesses plus CUDA/Triton implementations.

## Learning Goals
- Contrast naive decode loops against persistent kernels that pin CTAs per sequence.
- Adopt TMA-based prefill to stream activations into shared memory with minimal latency.
- Benchmark CUDA vs Triton implementations with unified validation utilities.
- Mix CUDA Graphs into the decode path to remove residual launch overhead.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_persistent_decode.py`, `optimized_persistent_decode_cuda.py`, `optimized_persistent_decode_graphs.py`, `optimized_persistent_decode_triton.py` | Persistent decode variants spanning CUDA, graphs, and Triton. |
| `baseline_tma_prefill_decode.py`, `optimized_tma_prefill_decode.py`, `baseline_native_tma_prefill_decode.py`, `optimized_native_tma_prefill_decode.py` | Prefill workloads illustrating cp.async vs native TMA scheduling. |
| `kv_locality_microbench.py` | Pinned/pageable/NUMA host slab copy microbench (HBM vs local/remote pinned vs pageable). |
| `persistent_decode_common.py`, `tma_extension.py`, `expectations_b200.json` | Shared helpers, CUDA extension wrappers, and expectation thresholds. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/persistent_decode
python -m cli.aisp bench run --targets labs/persistent_decode --profile minimal
```
- Targets follow the `labs/persistent_decode:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/persistent_decode:<workload>="--flag value"` to sweep schedule knobs.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/persistent_decode --profile minimal` compares all persistent/TMA variants in one sweep.
- `python labs/persistent_decode/optimized_persistent_decode_graphs.py --iterations 50` shows lower launch overhead than `baseline_persistent_decode.py`.
- `python labs/persistent_decode/optimized_native_tma_prefill_decode.py --validate` matches the math reference while reporting achieved memory throughput.
- `python labs/persistent_decode/kv_locality_microbench.py` surfaces H2D copy time deltas for pageable vs pinned slabs; add `QUICK=1` for a short run.

## Notes
- Set `TORCH_COMPILE_MODE` or `TMA_TILE_SIZE` via env vars before invoking the harness to sweep tile sizes.
- `tma_extension.py` caches builds under `~/.cache/torch_extensions`; clean the cache when switching CUDA versions.
