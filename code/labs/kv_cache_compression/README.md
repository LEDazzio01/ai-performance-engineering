# Lab - KV-Cache Compression (FP8 → NVFP4)

## Summary
Benchmarks a KV-cache heavy attention block using Transformer Engine 2.10 (CUDA 13) recipes: MXFP8 block scaling as the baseline and an NVFP4 path that calibrates in FP8 before switching to FP4 tensor cores on Blackwell.

## Learning Goals
- Compare MXFP8 vs NVFP4 block scaling for KV-cache heavy attention.
- Verify FP4 tensor-core speedups while maintaining accuracy through FP8 calibration.
- Exercise graceful fallbacks when FP4 kernels are unavailable.
- Capture reproducible metrics via the benchmark harness.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_kv_cache.py` | MXFP8 baseline benchmark. |
| `optimized_kv_cache_nvfp4.py` | NVFP4 path with FP8 calibration and automatic fallback. |
| `kv_cache_common.py` | Shared shapes/utilities for both paths. |
| `tma_prefetch_extension.py`, `tma_prefetch_ext.cu` | TMA async prefetch demo for KV cache tiles. |
| `expectations_b200.json`, `__init__.py` | Regression thresholds and harness target exports. |

## Running the Benchmarks
Use the benchmark harness to pick targets explicitly.
```bash
python -m cli.aisp bench list-targets --chapter labs/kv_cache_compression
python -m cli.aisp bench run --targets labs/kv_cache_compression:kv_cache --profile minimal
python -m cli.aisp bench run --targets labs/kv_cache_compression:kv_cache_nvfp4 --profile minimal
```
- Pass `--target-extra-arg labs/kv_cache_compression:kv_cache_nvfp4="--flag value"` to sweep shapes or calibration options.

## Validation Checklist
- Baseline MXFP8 target runs without requiring FP4 support.
- NVFP4 target either reports FP4 tensor-core usage or cleanly falls back to MXFP8 with a log message.
- Harness artifacts align with `expectations_b200.json`; mismatches flag regression risk.

## Notes
- FP4 path requires Blackwell-class NVFP4 support; tensor dimensions stay multiples of 16 for FP4/FP8 GEMM kernels.
- Everything is self-contained in `labs/kv_cache_compression/`—no cross-chapter imports.
