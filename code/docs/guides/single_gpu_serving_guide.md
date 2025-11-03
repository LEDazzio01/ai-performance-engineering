# Single-GPU Production Serving Guide

This guide captures the proven configuration for running the GPT-Large
production workload on a single Blackwell GPU. Use it when capacity planning
requires per-node isolation, or when deploying to edge sites where NVLink mesh
hardware is unavailable.

## Prerequisites
- CUDA 12.6+ with Nsight tools installed (`nsys`, `ncu`).
- PyTorch 2.5+ with FlexAttention and TorchDynamo enabled (see
  `requirements_latest.txt`).
- Transformer Engine 1.9.0a or newer (for FP8 GEMM support).
- Repository synchronized to the tag that shipped the FlexAttention vmap fix.

## 1. Environment Configuration
Export the minimal variable set so leftover NVLink tuning cannot leak into the
single-GPU tier.

```bash
export CUDA_VISIBLE_DEVICES=0
export TORCH_USE_CUDA_DSA=1               # catch illegal memory access early
export PYTHONPATH=$PWD
unset NCCL_P2P_DISABLE NCCL_P2P_LEVEL
```

> **Tip:** Keep these exports in the process supervisor unit file so restarts
> always pick up the safe defaults.

## 2. Warmup & Verification
Run the optimized benchmark once to compile kernels, verify tensor-parallel is
disabled, and produce a baseline JSON artifact.

```bash
python ch16/test_gpt_large_optimized.py \
    --tensor-parallel-gpus 1 \
    --iters 6 \
    --warmup 2 \
    --fp8-mode transformer-engine \
    --attention-backend flex \
    --output-json artifacts/single_gpu_serving/warmup.json
```

- Confirms FlexAttention + FP8 path is healthy on a lone GPU.
- Emits `tokens_per_joule`, latency statistics, and max absolute diff against
  the reference path.

## 3. Launching the Inference Server
Package `compare_inference_methods()` from `ch16/inference_optimizations_blackwell.py`
into the production entrypoint so both eager and optimized graph handles are
available.

```python
from ch16.inference_optimizations_blackwell import compare_inference_methods

def build_model():
    handles = compare_inference_methods(
        tensor_parallel_gpus=1,
        attention_backend="flex",
        fp8_mode="transformer-engine",
        compile_mode="reduce-overhead"
    )
    return handles["optimized_model"]
```

For CLI-driven serving (e.g. Triton or FastAPI managed by systemd):

```bash
python ch16/inference_optimizations_blackwell.py \
    --tensor-parallel-gpus 1 \
    --attention-backend flex \
    --fp8-mode transformer-engine \
    --compile-mode reduce-overhead \
    --max-new-tokens 512
```

## 4. Latency & Throughput Targets
- **Expected throughput**: 1,850–2,050 tokens/sec after warmup (prompt length
  1K, generation length 256).
- **P99 latency budget**: 2.3–2.6 s for 4-way batching; keep max batch size ≤16
  to avoid memory spikes beyond 48 GB.
- **Power draw**: 500–525 W with FP8 kernels engaged (≈3.6 tokens/joule).

## 5. Production Observability
- Scrape `artifacts/single_gpu_serving/warmup.json` for latency and accuracy
  metrics at deploy time.
- Enable the built-in Prometheus exporter (`--export-prometheus 9090`) inside
  `inference_optimizations_blackwell.py` to expose live throughput gauges.
- Attach `tools/power_monitor.py` in canaries to compute rolling
  tokens-per-joule and detect thermal throttling.

## 6. Release Checklist
1. Warmup benchmark succeeds with `max_abs_diff == 0.0`.
2. Load test passes:
   ```bash
   torchrun --nproc_per_node=1 \
     ch16/inference_server_load_test.py \
     --duration 120 \
     --target-qps 4 \
     --prompt-len-min 128 \
     --prompt-len-max 512 \
     --max-new-tokens 256 \
     --allow-smaller-world \
     --output-json artifacts/single_gpu_serving/load_test.json
   ```
3. Nsight profile captured (see `docs/nsight_fp8_flexattention.md`).
4. Alert thresholds updated with new throughput and latency envelopes.
5. Rollback plan validated (`docs/single_gpu_serving_fallback.md`).

## 7. Common Pitfalls
- **Leftover NCCL env**: Forgetting to unset `NCCL_P2P_DISABLE` keeps NVLink
  disabled when you later scale back to multi-GPU. Always use scoped service
  units.
- **Torch.compile regressions**: For 40B+ checkpoints, add
  `--compile-mode skip` if warmup exceeds 3 minutes or the server logs Dynamo
  graph breaks.
- **Long-context spikes**: Force the request router to batch 8 K+ token
  prompts separately. Monitor `cudaMallocAsync` fallbacks in logs; persistent
  hits mean FP8 activations are pushing memory beyond the safe 48 GB budget.

This playbook keeps the single-GPU tier aligned with the canonical production
configuration while preserving the knobs needed for emergency rollbacks.
