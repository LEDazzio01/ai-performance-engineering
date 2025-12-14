# Lab - cuDNN SDPA Bench

## Summary
Microbenchmarks cuDNN fused scaled-dot-product attention against Flash and math backends with explicit CLI backend selection.

## Learning Goals
- Compare cuDNN fused SDPA to Flash and math backends on identical shapes.
- Capture Nsight traces per backend to inspect kernel fusion and launch counts.
- Keep regression thresholds per architecture in `expectations_b200.json`.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_flash_sdp.py`, `optimized_flash_sdp.py` | Shared attention microbenchmarks; backend chosen via `--backend {auto,cudnn,flash,math}` passed with `--target-extra-arg`. |
| `expectations_b200.json` | Current golden timings on B200 for regression checking. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/cudnn_sdpa_bench
python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench --profile minimal
```
- Targets follow the `labs/cudnn_sdpa_bench:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/cudnn_sdpa_bench:<workload>="--flag value"` to sweep schedule knobs.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --profile minimal --target-extra-arg labs/cudnn_sdpa_bench:flash_sdp="--backend cudnn"` captures cuDNN with Nsight traces.
- `python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --target-extra-arg labs/cudnn_sdpa_bench:flash_sdp="--backend flash"` compares the Flash path against cuDNN.
- `python -m cli.aisp bench run --targets labs/cudnn_sdpa_bench:flash_sdp --target-extra-arg labs/cudnn_sdpa_bench:flash_sdp="--backend math"` sanity-checks the math backend where fused kernels are unsupported.

## Notes
- Backend selection is CLI-only; environment variables are intentionally ignored.
- Profiling outputs are stored under `benchmark_profiles/labs/cudnn_sdpa_bench/<run_id>` with harness artifacts in `artifacts/<run_id>/`.
