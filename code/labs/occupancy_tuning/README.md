# Lab - Triton Occupancy & Schedule Sweep

## Summary
Sweeps Triton matmul schedules for ProtonNet-style workloads on Blackwell, comparing the baseline schedule against optimized block/warp dimensions and reporting how each choice affects occupancy and FLOP/s.

## Learning Goals
- Measure how Triton block sizes map to achieved occupancy on SM100/121.
- Autogenerate schedule sweeps and record best-performing parameter sets.
- Compare baseline schedules to curated optimized variants packaged with the lab.
- Integrate selected schedules into harness targets for regression tracking.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_proton_matmul.py`, `optimized_proton_matmul_bm128_bn128_bk32_nw8.py`, `optimized_proton_matmul_bm64_bn64_bk32_nw2.py`, `optimized_proton_matmul_bm64_bn256_bk32.py`, `optimized_proton_matmul_bm128_bn256_bk64.py` | Baseline and optimized Triton schedules covering multiple block/warp configurations. |
| `triton_matmul.py`, `triton_matmul_schedules.py` | Core Triton kernel and schedule definitions used by the harness. |
| `sweep_schedules.py` | Utility for enumerating candidate schedules and logging throughput/occupancy to `artifacts/`. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/occupancy_tuning
python -m cli.aisp bench run --targets labs/occupancy_tuning --profile minimal
```
- Targets follow the `labs/occupancy_tuning:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/occupancy_tuning:<workload>="--flag value"` to sweep schedule knobs.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/occupancy_tuning --profile minimal` executes every schedule defined in the lab.
- `python labs/occupancy_tuning/sweep_schedules.py --output artifacts/occupancy_tuning.csv` enumerates schedules and highlights the top performer.
- `python labs/occupancy_tuning/optimized_proton_matmul_bm128_bn128_bk32_nw8.py --validate` compares outputs against the baseline to ensure correctness.

## Notes
- Add new schedules to `triton_matmul_schedules.py` and regenerate the harness targets by rerunning the sweep script.
- `expectations_b200.json` records FLOP/s per schedule so improvements show up in CI.
