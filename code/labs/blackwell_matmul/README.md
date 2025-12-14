# Lab - Blackwell Matmul Suite

## Summary
Ports the four-part Blackwell matmul deep dive into the harness: start with a naive CUDA kernel, then layer pipeline loads, real TMA, and cluster DSMEM broadcasts until you surpass the baseline roofline.

## Learning Goals
- Reproduce the reference matmul trajectory (baseline -> pipelined -> TMA -> cluster).
- Compare PyTorch harness timings against the CUDA extensions while reusing the same shapes.
- Validate kernels on SM100/103 targets and gracefully skip DSMEM-only paths on SM121.
- Capture dual roofline metadata (SM vs TMEM) for every variant.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_blackwell_matmul.py`, `optimized_blackwell_matmul_pipeline.py`, `optimized_blackwell_matmul_tma.py`, `optimized_blackwell_matmul_cluster.py` | Python entrypoints for each stage of the matmul tutorial. |
| `blackwell_benchmarks.py`, `run_blackwell_matmul.py` | Harness adapters and standalone runner for quick sweeps and metadata capture. |
| `grace_blackwell_extension.py`, `grace_blackwell_kernels.cu` | PyTorch extension and CUDA kernels implementing the baseline and optimized kernels. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/blackwell_matmul
python -m cli.aisp bench run --targets labs/blackwell_matmul --profile minimal
```
- Targets follow the `labs/blackwell_matmul:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/blackwell_matmul:<workload>="--flag value"` to sweep schedule knobs.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/blackwell_matmul:blackwell_matmul_cluster --profile minimal` delivers higher TFLOP/s than the baseline and emits artifacts under `artifacts/labs_blackwell_matmul*`.
- `python labs/blackwell_matmul/run_blackwell_matmul.py --variant pipeline --size 4096 --roofline-meta artifacts/matmul_meta.csv` saves roofline metadata alongside timings.
- DSM-aware variants error out early on GPUs that lack cluster DSMEM support, preventing misleading results.

## Notes
- `run_blackwell_matmul.py` accepts `--variant baseline|pipeline|tma|cluster` plus `--size` to mirror the blog walkthrough.
- TMA kernels require CUDA 13.0+ and SM100/103 hardware; on GB10 they log a warning and skip execution.
