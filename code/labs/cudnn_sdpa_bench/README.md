# Lab - cuDNN SDPA Bench

## Summary
Benchmarks cuDNN fused scaled-dot-product attention against Flash and math backends, using identical workloads with CLI-controlled backend selection.

## Learning Goals
- Compare cuDNN SDPA throughput/latency to Flash and math implementations on the same shapes.
- Exercise the benchmark CLI with per-target backend overrides instead of environment variables.
- Capture Nsight traces and harness artifacts for regression tracking.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_flash_sdp.py`, `optimized_flash_sdp.py` | Attention microbenchmarks; backend is chosen via `--backend {auto,cudnn,flash,math}`. |
| `expectations_gb10.json` | Regression thresholds captured on GB10. |
| `__init__.py` | Exports harness targets for the CLI. |

## Running the Benchmarks
Use the benchmark harness to sweep backends.
```bash
cd ai-performance-engineering
python -m cli.aisp bench list-targets --chapter labs/cudnn_sdpa_bench
python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --profile minimal --target-extra-arg labs/cudnn_sdpa_bench:flash_sdp="--backend cudnn"
python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --profile minimal --target-extra-arg labs/cudnn_sdpa_bench:flash_sdp="--backend flash"
python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --target-extra-arg labs/cudnn_sdpa_bench:flash_sdp="--backend math"
```
- `--backend` is CLI-only; env vars are ignored by design.

## Validation Checklist
- Harness runs succeed for `cudnn`, `flash`, and `math` backends without code changes.
- Nsight traces land under `benchmark_profiles/labs/cudnn_sdpa_bench/<run_id>` and artifacts under `artifacts/<run_id>/...`.
- Optimized path meets or exceeds expectations in `expectations_gb10.json`; failures flag regressions.

## Notes
- Profiles are lightweight; use `--profile none` when you only need correctness checks.
- Hopper-class expectations differ; refresh `expectations_gb10.json` after validating new hardware.
