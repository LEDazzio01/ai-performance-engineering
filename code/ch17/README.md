# Chapter 17 - Dynamic Routing & Hybrid Serving

## Summary
Blends router design, disaggregated inference, and profiling discipline so Blackwell clusters can route queries between prefill/decode pools, MoE experts, and pipeline stages without sacrificing utilization.

## Learning Goals
- Implement dynamic routers that react to TTFT, TPOT, and KV-locality metrics.
- Profile complete inference stacks (prefill + decode) under realistic synthetic loads.
- Blend pipeline parallelism with routing logic for long-context workloads.
- Document profiling steps (roofline, Nsight) specific to the routing lab.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_dynamic_routing.py`, `optimized_dynamic_routing.py`, `dynamic_routing.py`, `early_rejection.py` | Routing controllers that evolve from static heuristics to telemetry-driven admission and rejection policies. |
| `baseline_inference_full.py`, `optimized_inference_full.py`, `baseline_prefill_decode_disagg.py`, `optimized_prefill_decode_disagg.py`, `baseline_prefill_decode_disagg_multigpu.py`, `optimized_prefill_decode_disagg_multigpu.py` | End-to-end inference flows modeling separate prefill and decode pools, both single-node and multi-GPU. |
| `baseline_pipeline_parallelism.py`, `optimized_pipeline_parallelism.py` | Pipeline parallel workloads combining compute and KV-transfer scheduling. |
| `baseline_moe_router_uniform.py`, `optimized_moe_router_uniform_topology.py` | Comparable MoE router benchmark pair contrasting uniform vs topology-aware routing while keeping outputs invariant via shared expert weights. |
| `moe_router_uniform_demo.py`, `moe_router_topology_demo.py` | MoE routing demos (non-benchmark) contrasting uniform vs topology-aware expert selection. |
| `baseline_routing_static.py`, `optimized_routing_static.py` | Router variants for static/dynamic sharding decisions (comparable benchmarks). |
| `baseline_memory.py`, `optimized_memory.py`, `blackwell_profiling_guide.py` | Memory-bound case studies plus profiling guides tailored to routing workloads (use `aisp tools roofline` for roofline analysis). |
| `compare.py`, `Makefile`, `expectations_b200.json`, `dynamo_config.yaml` | Harness entry, build rules, expectation baselines, and Dynamo config knobs. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python ch17/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch17
python -m cli.aisp bench run --targets ch17 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Expectation baselines live next to each chapter in `expectations_b200.json`; refresh with `--update-expectations` after validating new hardware.

## Validation Checklist
- `python optimized_dynamic_routing.py --trace` logs TTFT/TPOT trends that settle faster than the baseline's oscillations.
- `python optimized_pipeline_parallelism.py --profile minimal` shows overlapping prefill/decode segments with fewer idle bubbles.
- `python -m cli.aisp tools roofline` reproduces the documented roofline points using your latest captures.

## Demos (Non-Benchmark)
Run the MoE router demos via `aisp demos`:
```bash
python -m cli.aisp demos ch17-moe-router-uniform
python -m cli.aisp demos ch17-moe-router-topology
```

## Notes
- `blackwell_profiling_guide.py` walks through Nsight Systems/Compute captures and interpreting roofline vs occupancy bottlenecks for routing-heavy workloads.
- `baseline_prefill_decode_disagg_multigpu.py` can run in simulation-only mode by passing `--simulate-fabric`, avoiding the need for multi-node hardware while iterating on routing logic.
