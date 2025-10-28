# Architecture-Specific Tuning Guides

This document captures concrete, reproducible tuning recipes for common model
families on Blackwell hardware. Each section links to runnable scripts in this
repository so results can be reproduced and extended.

## Dense Autoregressive Transformers

- **Benchmark**: `ch16/test_gpt_large_optimized.py` (now with FlexAttention and
  transformer_engine FP8 support)
- **Long-context workloads**: Use the built-in `--fp8-mode auto --attention-backend flex`
  flags to cover 8K–16K token sequences. The script now tests `Batch=1, Seq=16K`
  and records precision/mode in JSON output.
- **Tensor-parallel validation**: Run `python ch16/multi_gpu_validation.py --tensor-parallel-gpus 4`
  to confirm numerical consistency across partitions after changing layer
  placement or attention backend.

Recommended settings:

| Scenario           | Notes                                                                                 |
| ------------------ | -------------------------------------------------------------------------------------- |
| Latency-critical   | Prefer `--compile-mode reduce-overhead --attention-backend auto` (Flex when available) |
| Throughput focused | Enable `--fp8-mode transformer-engine` once TE is installed and validated             |

## Mixture-of-Experts (MoE)

- **Benchmark**: `python ch16/moe_performance_benchmark.py --num-experts 16 --top-k 2`
  reports latency and tokens/sec for configurable expert/top-k combinations.
- **Routing diagnostics**: Extend the script with `--compile` to measure the
  benefits of `torch.compile` on expert MLPs.
- **Integration**: The MoE benchmark shares the same embedding/projection code
  as the dense GPT harness, making it easy to compare apples-to-apples.

## Inference Serving (8x B200)

- **Functional load test**: `torchrun --nproc_per_node=8 ch16/inference_server_load_test.py --duration 120`
  drives the production server with synthetic traffic, recording throughput and
  latency percentiles (per-rank results aggregated on rank 0).
- **Profiling**: Wrap the server (or any workload) with `python tools/memory_profiler.py ch16/inference_serving_8xb200.py`
  to capture CUDA memory hot spots.
- **Power envelope**: `python tools/power_monitor.py -- torchrun --nproc_per_node=8 ch16/inference_server_load_test.py ...`
  samples GPU power draw while the load test command runs, exporting JSON
  metrics for dashboards.

## Accuracy & Quality Evaluation

- `python ch16/perplexity_eval.py data/tokens.txt --seq-len 1024 --stride 256`
  computes cross-entropy/perplexity on pre-tokenized datasets, ensuring that
  optimizations (e.g. FP8) do not regress accuracy.
- Integrate this script into the continuous benchmark harness to capture both
  performance and quality metrics per build.

## Continuous Benchmarking

- Create a JSON config (see `docs/examples/continuous_benchmark.json`) listing
  critical benchmarks.
- Run `python tools/continuous_benchmark.py docs/examples/continuous_benchmark.json`
  nightly to produce timestamped summaries under `benchmark_runs/`.
- Feed resulting JSON into dashboards (Grafana, Superset) for regression alerts.

These guides will be expanded as we validate additional architectures (vision
transformers, diffusion, recommendation) and as upstream libraries expose new
Blackwell-optimized kernels.
