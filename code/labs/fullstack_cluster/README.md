# Lab - Full-Stack Blackwell Cluster

## Summary
Replays the entire performance-engineering arc as scenarios: from system prep to streaming inference, plus the original cluster GEMM CUDA kernels wired into the harness.

## Learning Goals
- Run scenario benchmarks that stitch together chapters into end-to-end workflows.
- Inspect cluster GEMM kernels (baseline and DSMEM/TMA optimized) via the CUDA extension.
- Track GPU requirements, expected shapes, and automation scripts in one place.
- Collect artifact bundles that summarize every phase of the scenario.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_01_system_foundations.py` ... `baseline_09_end_to_end.py`, `optimized_01_system_foundations.py` ... `optimized_09_end_to_end.py`, `scenario_benchmark.py` | Scenario scripts that orchestrate system, kernel, compiler, memory, serving, and end-to-end phases. |
| `baseline_cluster_gemm.py`, `optimized_cluster_gemm.py`, `baseline_cluster_gemm_tcgen05.py`, `optimized_cluster_gemm_tcgen05.py` | Python entrypoints for the cluster GEMM kernels with tcgen05 fallbacks. |
| `capstone_extension.py`, `capstone_kernels.cu`, `capstone_kernels_tcgen05.cu`, `capstone_benchmarks.py` | PyTorch extension, CUDA kernels, and harness hooks for the GEMM showcase. |
| `run_lab_fullstack_cluster.py`, `gpu_requirements.py`, `expectations_b200.json` | Standalone runner, hardware requirement helper, and expectation file. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/fullstack_cluster
python -m cli.aisp bench run --targets labs/fullstack_cluster --profile minimal
```
- Targets follow the `labs/fullstack_cluster:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/fullstack_cluster:<workload>="--flag value"` to sweep schedule knobs.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/fullstack_cluster --profile minimal` records per-phase metrics for the entire scenario suite.
- `python labs/fullstack_cluster/run_lab_fullstack_cluster.py --size 2048` builds the extension on first run and prints baseline vs optimized TFLOP/s.
- KF-specific kernels skip gracefully on hardware lacking tcgen05 or DSMEM, ensuring CI signal stays meaningful.

## Notes
- `gpu_requirements.py` reports the minimum GPU count, memory, and features for each scenario; consult it before scheduling runs.
- `capstone_extension.py` caches builds under `~/.cache/torch_extensions`; run `python cleanup.py --include-extensions` when switching CUDA versions.
