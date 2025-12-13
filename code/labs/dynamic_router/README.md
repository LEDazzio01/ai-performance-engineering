# Lab - Dynamic Prefill/Decode Router

## Summary
Simulates and benchmarks dynamic routing policies for large-scale inference: split GPUs into prefill/decode pools, monitor TTFT/TPOT, honor KV locality, and migrate traffic only when the score gap warrants it.

## Learning Goals
- Compare naive round-robin routing with telemetry-driven policies that stabilize TTFT.
- Prototype migration budgets, KV-locality boosts, and per-pool thresholds.
- Drive the router against synthetic workloads or real vLLM engines.
- Export detailed metrics (TTFT, TPOT, queue depth) for visualization.

## Directory Layout
| Path | Description |
| --- | --- |
| `router_round_robin.py`, `router_policy.py`, `driver.py` | Core router logic plus a synthetic simulator for deterministic comparisons (tool; not a benchmark target). |
| `baseline_dynamic_router_vllm.py`, `optimized_dynamic_router_vllm.py`, `vllm_runner.py` | Integrations for running the routing policy against vLLM instances. |
| `baseline_dual_pool_vllm.py`, `optimized_dual_pool_vllm.py` | Shared-pool vs dual-pool TTFT benchmarks that reuse `vllm_runner.py`. |
| `cheap_eval.py`, `eval_stack.py` | Cheap quality + latency + MoE telemetry stack that emits the six “thin checks” scorecard (tool; not a benchmark target). |
| `scorecard.py` | CLI to render a ship/no-ship table from emitted JSONLs (text + optional matplotlib PNG). |
| `topology.py`, `topology_probe.py` | NUMA/GPU topology helpers plus a tool that writes `artifacts/topology/topology.json` for routing hints. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ai-performance-engineering
python -m cli.aisp bench list-targets --chapter labs/dynamic_router
python -m cli.aisp bench run --targets labs/dynamic_router --profile minimal
```
- Targets follow the `labs/dynamic_router:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/dynamic_router:<workload>="--flag value"` to sweep schedule knobs.
- Run the cheap eval stack tools directly:
  - `python labs/dynamic_router/cheap_eval.py --no-vllm --request-count 48`
- Render a CLI table (and optional plot) from past runs: `python labs/dynamic_router/scorecard.py artifacts/dynamic_router/cheap_eval/full_* --plot artifacts/dynamic_router/cheap_eval/scorecard.png`.

## Validation Checklist
- `python labs/dynamic_router/driver.py --mode baseline` vs `--mode optimized` shows lower TTFT variance and higher TPOT for the optimized policy.
- `python labs/dynamic_router/cheap_eval.py` emits the six-check scorecard (quality, TTFT/decode latency, MoE health, throughput/goodput) under `artifacts/dynamic_router/cheap_eval/full_<timestamp>/`.
- `python -m cli.aisp bench run --targets labs/dynamic_router:dynamic_router_vllm --target-extra-arg labs/dynamic_router:dynamic_router_vllm="--model /path/to/model --decode-gpus 0,1"` succeeds on systems with ≥2 GPUs and local model weights.
- `python -m cli.aisp bench run --targets labs/dynamic_router:dual_pool_vllm --target-extra-arg labs/dynamic_router:dual_pool_vllm="--model /path/to/model --prefill-gpus 0 --decode-gpus 1"` contrasts shared vs dual pools and emits per-pool TTFT/queue depth.
- `python labs/dynamic_router/topology_probe.py` persists GPU↔NUMA mappings to `artifacts/topology/topology.json` for downstream hints.

## Notes
- Cheap eval stack: runs mini quality slice (MMLU-mini/GSM8K-lite/TruthfulQA-lite/domain), TTFT p50/p95 (warm/cold), decode p50/p95 for 128/512/2048 tokens, MoE drop %, expert imbalance (CV), router entropy/margin, plus throughput/goodput. Artifacts: `quality.jsonl`, `latency.jsonl`, `moe_router.jsonl`, `moe_traffic.jsonl`, optional `tps_goodput.json`, and `scorecard.json`. Replay real telemetry via `--metrics-dir` and `--baseline-scorecard`; add `--no-vllm` to force synthetic quality.
- vLLM dual-pool TTFT lab: baseline shared pool vs optimized dual pool. Requires `vllm` installed, a local model path, and ≥2 GPUs. Flags include `--prefill-gpus`, `--decode-gpus`, `--long-prompt-tokens`, `--short-prompt-tokens`, `--prefill-burst`, `--decode-requests`, `--continue-requests`, `--max-tokens`, `--prefill-ctx-thresh`.
- Router knobs: `driver.py` accepts `--scenario`, `--arrival-rate`, `--burst-factor`, `--decode-cost-penalty`, `--log-json`; scoring factors KV locality, queue depth urgency, and decode cost.
- `eval_stack.py` is mock-friendly—swap synthetic generators for real engine hooks (TTFT, decode latency, router entropy, expert histogram, drop rate) without changing run folder layout.
