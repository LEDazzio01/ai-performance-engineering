# Lab - Fast NanoChat

## Summary
Decode-focused microbenchmarks inspired by nanochat that show how common serving optimizations (pinned memory, streams, compile/graphs, FP8/FP4, warp specialization, multi-GPU) change TTFT, TPOT, and throughput.

## Learning Goals
- Contrast eager vs pinned/streamed vs compiled/graph decode paths on the same workload.
- Measure FP8/FP4 tensor-core benefits relative to FP16/BF16 baselines.
- Validate Triton warp-specialized decode kernels against Python math and harness expectations.
- Observe NVLink-C2C behavior by scaling the decode loop to 8×B200.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_fast_nanochat.py` | Eager decode baseline with pageable prompts on a single stream. |
| `optimized_fast_nanochat_pinned.py`, `optimized_fast_nanochat_streams.py` | Pinned-host and dual-stream variants that remove host bottlenecks. |
| `optimized_fast_nanochat_compile.py`, `optimized_fast_nanochat_graph.py`, `optimized_fast_nanochat_graph_full.py` | `torch.compile` and CUDA Graph variants (decode-only and full prefill+decode). |
| `optimized_fast_nanochat_fp8.py`, `optimized_fast_nanochat_fp4.py` | Transformer Engine FP8/FP4 decode paths (FP4 falls back gracefully when unsupported). |
| `optimized_fast_nanochat_warp_specialized.py`, `triton_fused_decode.py` | Triton fused decode MLP with warp specialization/TMA-style pointers. |
| `optimized_fast_nanochat_8xgpu.py` | Multi-GPU launcher to stress NVLink-C2C bandwidth on 8×B200. |
| `optimized_fast_nanochat_double_buffer_tma_cuda.py` | CUDA double-buffered decode using TMA-style loads. |
| `nanochat_common.py`, `expectations_8x_b200.json`, `__init__.py` | Shared helpers, regression thresholds, and harness target exports. |

## Running the Benchmarks
Use the benchmark harness for repeatable runs and artifact capture.
```bash
cd ai-performance-engineering
python -m cli.aisp bench list-targets --chapter labs/fast_nanochat
python -m cli.aisp bench run --targets labs/fast_nanochat --profile none
```
- Target labels: `baseline_fast_nanochat`, `optimized_fast_nanochat_pinned`, `..._streams`, `..._compile`, `..._graph`, `..._graph_full`, `..._fp8`, `..._fp4`, `..._warp_specialized`, `..._8xgpu`.
- Pass extra flags via `--target-extra-arg labs/fast_nanochat:<target>="--flag value"`.

## Validation Checklist
- Baseline vs pinned/streams shows improved TTFT and TPOT with lower host wait time.
- Compile/graph variants emit fewer kernels and higher tokens/sec than the baseline in harness output.
- FP8/FP4 runs fall back cleanly if FP4 kernels are unavailable; expectations remain within tolerance.
- Warp-specialized Triton kernel matches math outputs and reports higher throughput; expectation file stays green.
- 8×GPU run exercises NVLink-C2C without graph capture failures when launched via `torchrun`.

## Notes
- All targets emit TTFT, TPOT mean, decode time, total time, and tokens/sec in `custom_metrics` for easy diffing.
- FP4 path requires NVFP4 (Blackwell); otherwise it reports the fallback in logs.
- Multi-GPU target expects eight visible GPUs; reduce `--world-size` if you customize the script for smaller boxes.
