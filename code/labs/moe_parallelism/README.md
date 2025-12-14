# Lab - MoE Parallelism Planner

## Summary
Provides scenario planning for mixture-of-experts clusters: memory budgeting, network affinity, parallelism breakdown, and pipeline schedules expressed as baseline/optimized harness targets.

## Learning Goals
- Quantify memory budgets for experts, routers, and KV caches before deploying models.
- Explore different grouping strategies (hashing, topology-aware) and their throughput impact.
- Model network affinity to decide where experts should live in an NVLink/NVSwitch fabric.
- Simulate pipeline schedules to identify bottlenecks before touching production systems.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_memory_budget.py`, `optimized_memory_budget.py` | Memory planners that prove how optimized layouts free additional HBM for experts. |
| `baseline_moe_grouping.py`, `optimized_moe_grouping.py`, `plan.py` | Grouping strategies spanning naive, locality-aware, and latency-balanced heuristics. |
| `baseline_network_affinity.py`, `optimized_network_affinity.py` | Network-affinity calculators comparing NVLink, NVSwitch, and PCIe hops. |
| `baseline_parallelism_breakdown.py`, `optimized_parallelism_breakdown.py`, `baseline_pipeline_schedule.py`, `optimized_pipeline_schedule.py` | Parallelism and scheduling studies for sharding experts across GPUs. |
| `benchmarking.py`, `run_lab.py`, `__init__.py` | Lab driver, Typer CLI, and exports used by the harness. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/moe_parallelism
python -m cli.aisp bench run --targets labs/moe_parallelism --profile minimal
```
- Targets follow the `labs/moe_parallelism:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/moe_parallelism:<workload>="--flag value"` to sweep schedule knobs.

## Validation Checklist
- `python -m cli.aisp bench run --targets labs/moe_parallelism --profile minimal` runs every planner pair and drops JSON/Markdown summaries.
- `python labs/moe_parallelism/run_lab.py --scenario grouped` prints an actionable plan (experts/GPU, bandwidth needs) for the chosen scenario.
- `python labs/moe_parallelism/optimized_memory_budget.py --validate` ensures optimized allocations meet the same correctness checks as the baseline.

## Notes
- `plan.py` centralizes scenario definitions so you only update one file when adding a new MoE topology.
- `benchmarking.py` can emit Markdown tables for documentation by passing `--format markdown`.
