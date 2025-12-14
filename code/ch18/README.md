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
| `baseline_flexdecoding.py`, `optimized_flexdecoding.py`, `optimized_flexdecoding_graphs.py`, `baseline_v1_engine_loop.py`, `optimized_v1_engine_loop.py`, `v1_engine_loop_common.py` | Decode-loop workloads comparing eager vs graph-backed execution while surfacing capture/recapture behavior. |
| `baseline_tensor_cores.py`, `optimized_tensor_cores.py`, `flashmla_kernel.cu`, `warp_specialized_triton.py` | Tensor-core attention kernels plus Triton equivalents for rapid validation. |
| `flex_attention_native.py`, `flex_attention_enhanced.py`, `flex_attention_large_model.py`, `kv_cache_integration_example.py` | FlexAttention examples ranging from toy sizes to large models with KV-cache reuse. |
| `baseline_vllm_v1_integration.py`, `optimized_vllm_v1_integration.py`, `baseline_vllm_decode_graphs.py`, `optimized_vllm_decode_graphs.py`, `configs/`, `workload_config.py` | Serving integrations and config presets for pushing workloads through vLLM or custom harnesses. |
| `speculative_decode/spec_config_sweep.py` | Tooling to sweep speculative-decoding configs and summarize latency/throughput tradeoffs. |
| `compare.py`, `expectations_b200.json`, `test_flex_attention.py` | Harness entry, regression thresholds, and pytest coverage for FlexAttention APIs. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python ch18/compare.py --profile none
python -m cli.aisp bench list-targets --chapter ch18
python -m cli.aisp bench run --targets ch18 --profile minimal
```
- Override `--profile` or `--iterations` per workload when capturing Nsight traces.
- Expectation baselines live next to each chapter in `expectations_b200.json`; refresh with `--update-expectations` after validating new hardware.

## Validation Checklist
- `python optimized_flexdecoding.py --profiling` reports significantly fewer kernels and lower latency than the baseline while matching decoded tokens.
- `python run_vllm_decoder.py --config configs/flex_attention.yaml` completes with accuracy parity vs the native FlexAttention path.
- `python test_flex_attention.py` passes locally, confirming mask/score-mod helpers are wired correctly.

## Notes
- `flex_attention` scripts accept env vars like `BLOCK_SIZE`, `DOC_SPAN`, and `SEQ_LEN` so you can sweep shapes without editing code.
- `flashmla_kernel.cu` includes the Blackwell-specific tensor memory guard to keep compilation healthy on SM121 hardware.
