# Chapter 18 - Advanced Attention & Decoding

## Summary
Collects modern decoder techniques-FlexAttention, FlexDecoding, speculative and paged attention workflows-implemented in both PyTorch and CUDA/Triton so you can iterate quickly while validating kernels on real hardware.

## Learning Goals
- Prototype FlexAttention/FlexDecoding workloads with custom masks, score mods, and KV-cache integration.
- Evaluate speculative decoding pipelines that trade extra compute for lower latency.
- Test tensor-core optimized attention kernels tailored for Blackwell tmem limits.
- Validate integration points with serving frameworks (vLLM) using the provided runners.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_flexdecoding.py`, `optimized_flexdecoding.py`, `optimized_flexdecoding_math.py`, `baseline_speculative_decoding.py`, `optimized_speculative_decoding.py`, `optimized_speculative_decoding_math.py` | Decoding workloads comparing eager vs math-only vs optimized pipeline variants. |
| `baseline_tensor_cores.py`, `optimized_tensor_cores.py`, `flashmla_kernel.cu`, `warp_specialized_triton.py` | Tensor-core attention kernels plus Triton equivalents for rapid validation. |
| `flex_attention_native.py`, `flex_attention_enhanced.py`, `flex_attention_large_model.py`, `kv_cache_integration_example.py` | FlexAttention examples ranging from toy sizes to large models with KV-cache reuse. |
| `baseline_v1_engine_loop.py`, `optimized_v1_engine_loop.py` | V1 EngineCore/CoreClient polling loops showing the executed_flag quirk and prompt KV reclamation via `report_finished_ids`. |
| `v1_bucketed_decode_loop.py` | V1 polling loop that reuses bucketed workspaces/masks and can export graph/allocator counters for Prometheus scraping. |
| `baseline_vllm_decode_graphs.py`, `optimized_vllm_decode_graphs.py` | Decode loop harness contrasting ragged CUDA-graph recaptures against bucketed/padded shapes, preallocated workspaces, and lazy KV compaction. |
| `baseline_cudagraph_bucketing.py`, `optimized_cudagraph_bucketing.py`, `cudagraph_bucketing_common.py` | Dynamic-shape decode demos that contrast graph churn vs bucketed/pre-warmed CUDA Graph Tree captures. |
| `vllm_monitoring.py`, `monitoring_bundle.py`, `configs/vllm_monitoring.yaml` | Emit Prometheus/Grafana bundles for vLLM v1 metrics (TTFT, prefill/decode split, KV cache, queue churn, CUDA graph mode) with overrideable metric names and alert thresholds. (Tool; not a benchmark target.) |
| `run_vllm_decoder.py`, `configs/`, `paged_attn_common.py`, `workload_config.py` | Serving integrations and config presets for pushing workloads through vLLM or custom harnesses. |
| `compare.py`, `expectations_gb10.json`, `test_flex_attention.py` | Harness entry, regression thresholds, and pytest coverage for FlexAttention APIs. |

### KV cache sizing helper (standalone utility)
- `core/scripts/utilities/kv_cache_calc.py` — CLI for quick KV-cache sizing from (L, H, T, N, dtype), including overhead and optional GPU budget/reserve. Not a benchmark target; run directly:
  ```bash
  python core/scripts/utilities/kv_cache_calc.py \
    --layers 80 --hidden 8192 --tokens 4096 --batch 8 \
    --dtype fp8 --gpu-mem-gb 192 --kv-overhead-frac 0.10 \
    --reserve-activations-gb 40
  ```

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ch18
python compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch18
python -m cli.aisp bench run --targets ch18 --profile minimal
# Run the CUDA-graph bucketing simulators via the CLI (per-target flags only):
python -m cli.aisp bench run \
  --targets ch18:cudagraph_bucketing,ch18:cudagraph_bucketing_optimized \
  --target-extra-arg ch18:cudagraph_bucketing="--vllm-model gpt-oss-20b" \
  --target-extra-arg ch18:cudagraph_bucketing_optimized="--vllm-model gpt-oss-20b"
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Expectation baselines live next to each chapter in `expectations_gb10.json`; refresh with `--update-expectations` after validating new hardware.

## Validation Checklist
- `python optimized_flexdecoding.py --profiling` reports significantly fewer kernels and lower latency than the baseline while matching decoded tokens.
- `python run_vllm_decoder.py --config configs/flex_attention.yaml` completes with accuracy parity vs the native FlexAttention path.
- `python test_flex_attention.py` passes locally, confirming mask/score-mod helpers are wired correctly.

## Notes
- `flex_attention` scripts accept env vars like `BLOCK_SIZE`, `DOC_SPAN`, and `SEQ_LEN` so you can sweep shapes without editing code.
- `flashmla_kernel.cu` includes the Blackwell-specific tensor memory guard to keep compilation healthy on SM121 hardware.
- CUDA-graph bucketing demos default to `--vllm-model gpt-oss-20b`; override with `--vllm-model <name>` or disable capture bins via `--no-vllm-bins`.
- `baseline_vllm_decode_graphs.py`, `optimized_vllm_decode_graphs.py`, and `v1_bucketed_decode_loop.py` accept `--prom-port` to expose `vllm:decode_graph_recaptures_total`, `vllm:decode_allocator_bytes`, and `vllm:decode_kv_compactions_total` for quick scraping alongside the main vLLM metrics endpoint.
- To validate on live traffic: (1) run `python ch18/v1_bucketed_decode_loop.py --use-vllm --model <id> --prom-port 9300` next to your vLLM server and scrape both `/metrics` endpoints to watch graph recaptures/allocator bytes; (2) point Prometheus at the decode demo’s port and confirm `vllm:decode_graph_recaptures_rate` and `vllm:decode_allocator_mb` populate the new recording rules from `ch18/vllm_monitoring.py`.

### Live validation quickstart
Run the bucketed loop beside a live vLLM server and scrape both endpoints:
```bash
# 1) start vLLM (example)
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000 --api-key token-abc123

# 2) start the bucketed decode loop with metrics exposed
python ch18/v1_bucketed_decode_loop.py --use-vllm --model meta-llama/Llama-3.1-8B-Instruct --prom-port 9300
```
Point Prometheus at `http://localhost:8000/metrics` (vLLM) and `http://localhost:9300/metrics` (demo). You should see `vllm:decode_graph_recaptures_rate` and `vllm:decode_allocator_mb` populate via the recording rules defined in `ch18/vllm_monitoring.py`; also export `vllm:decode_kv_compactions_total` if you want compaction visibility.

## Monitoring bundle quickstart
Export the vLLM v1 monitoring assets used throughout the chapter:
```bash
python -m cli.aisp tools vllm-monitoring -- --outdir artifacts/vllm_monitoring
```
- Override metric names and alert thresholds via `--config /path/to/your_metrics.yaml`, e.g. `python -m cli.aisp tools vllm-monitoring -- --config /path/custom.yaml`.
