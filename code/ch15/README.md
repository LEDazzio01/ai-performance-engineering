# Chapter 15 - Disaggregated Inference & KV Management

## Summary
Addresses large-scale inference concerns: disaggregated compute/storage, KV-cache pooling over NVLink, continuous batching, and mixture-of-experts serving patterns.

## Learning Goals
- Benchmark monolithic vs disaggregated inference paths and quantify fabric costs.
- Design KV-cache managers that gracefully span local and remote HBM pools.
- Implement continuous batching and queueing so decode throughput stays high.
- Serve MoE models efficiently by pairing routing with optimized communication.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_inference_monolithic.py`, `optimized_inference_monolithic.py`, `disaggregated_inference.py` | Single-box inference loops that establish the baseline before disaggregation. |
| `baseline_disaggregated_inference.py`, `optimized_disaggregated_inference.py`, `baseline_prefill_decode_disagg.py`, `optimized_prefill_decode_disagg.py` | Disaggregated pipelines modeling remote prefills, decode overlap, and NVLink pooling. |
| `baseline_inference_placement.py`, `optimized_inference_placement.py`, `placement_sim.py` | NVLink-local placement simulator comparing cross-node TP/EP (baseline) against heuristic-aligned sticky decode + local MoE experts (optimized). |
| `baseline_kv_cache_management.py`, `optimized_kv_cache_management.py`, `kv_cache_management_math.py`, `optimized_kv_cache_nvlink_pool.py`, `baseline_kv_cache_local_only.py` | KV-cache orchestration utilities with local-only, math-only, and NVLink-pooled variants. |
| `baseline_continuous_batching.py`, `optimized_continuous_batching.py` | Continuous batching scheduler demonstrating TTFT-aware queueing. |
| `moe_validation/moe_validation.py` | Routing guardrails tool: overflow/Gini/entropy sweeps and token-drop detection alongside loss/throughput. (Tool; not a benchmark target.) |
| `baseline_moe_inference.py`, `optimized_moe_inference.py` | Inference-specific MoE workloads that pair router load with communication control. |
| `compare.py`, `requirements.txt`, `expectations_gb10.json`, `Makefile` | Harness entry and dependencies for inference-focused validation. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ch15
python compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch15
python -m cli.aisp bench run --targets ch15 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Expectation baselines live next to each chapter in `expectations_gb10.json`; refresh with `--update-expectations` after validating new hardware.

## Validation Checklist
- `python optimized_disaggregated_inference.py --profile minimal` shows reduced fabric stalls compared to the baseline while maintaining accuracy parity.
- `python optimized_kv_cache_management.py --validate` confirms eviction + promotion policies keep decode latency within the budget.
- `python compare.py --examples continuous_batching` proves optimized scheduling increases tokens/sec vs naive queue draining.
- `python -m cli.aisp tools moe-validation -- --out artifacts/moe_validation.json` sweeps top-k and capacity-factor guardrails while logging overflow rate, Gini, router entropy, and loss/throughput.
- `python -m cli.aisp bench run --targets ch15:inference_placement --profile none` contrasts naive cross-node TP/EP versus NVLink-local prefill/decode with sticky sessions and local-first MoE routing.

## Notes
- `disaggregated_inference.py` can run purely in simulation mode; set `--simulate-network` when hardware isn't wired for NVLink pooling.
- `Makefile` wraps the MPI/UCX targets needed for the multi-node decode experiments.
- All new placement sims assume PyTorch >= 2.10 with CUDA 13 so that Blackwell (B200/B300, GB200/GB300) NVLink bandwidths and bf16 defaults model realistically; the harness degrades gracefully on earlier stacks.
