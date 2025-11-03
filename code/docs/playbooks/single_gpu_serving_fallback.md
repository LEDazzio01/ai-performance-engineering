# Single-GPU Serving Fallback Playbook

When NVLink capacity or multi-GPU orchestration is unavailable, fall back to a
single-GPU configuration to keep traffic flowing. This playbook captures the
exact steps we exercise during rollback drills.

## When to Trigger the Fallback
- NVLink mesh degraded (`verify_nvlink.py` reports less than 18 active links).
- NCCL collectives repeatedly hang or tear down the tensor-parallel server.
- Production is pinned to PCIe-only hardware and sustained QPS drops by >30%.
- Capacity planning needs a “fits anywhere” deployment (edge nodes, cold
  standby racks).

## Golden Path Checklist
1. **Freeze tensor-parallel workers**: drain requests, stop the 8×GPU server,
   keep the load balancer pointed at the single-GPU pool only after warmup.
2. **Export safe NCCL defaults** (prevents leftover env from disabling CUDA P2P):
   ```bash
   unset NCCL_P2P_DISABLE NCCL_P2P_LEVEL
   export CUDA_VISIBLE_DEVICES=0
   ```
3. **Warm the model** with the optimized single-GPU benchmark:
   ```bash
   python ch16/test_gpt_large_optimized.py \
       --iters 3 \
       --warmup 1 \
       --tensor-parallel-gpus 1 \
       --output-json fallback_benchmark.json
   ```
   This primes TorchDynamo graphs, confirms memory headroom, and records the
   baseline latency envelope we pin alerts to during fallback mode.
4. **Launch the lightweight inference loop**:
   ```bash
   python ch16/inference_optimizations_blackwell.py
   ```
   - Default execution runs on a single GPU and emits throughput/latency
     comparisons for eager vs. compiled kernels.
   - Wrap this script inside your serving harness (Gunicorn, Triton, FastAPI)
     by importing `compare_inference_methods()` and reusing the compiled model
     handle.
5. **Enable back-pressure controls**:
   - Cap max batch size at 8–16 requests (`--max-new-tokens` 128–256).
   - Enable request queue timeouts at 5 s to avoid tail-latency explosions.
   - Route long-context users to batch windows ≥4 s so GPU memory stays below
     80 %.

### Optional: Fallback Load Test
The primary load-test harness now supports a single GPU when you add
`--allow-smaller-world` and reduce the generation length. Example:

```bash
mkdir -p artifacts/single_gpu_fallback
PYTHONPATH=$PWD \
torchrun --nproc_per_node=1 \
  ch16/inference_server_load_test.py \
  --duration 20 \
  --target-qps 3 \
  --max-new-tokens 4 \
  --prompt-len-min 32 \
  --prompt-len-max 64 \
  --allow-smaller-world \
  --output-json artifacts/single_gpu_fallback/load_test.json
```

This path automatically disables tensor-parallel optimisations (FlexAttention,
torch.compile, CUDA graphs) that rely on NVLink. Expect ~1.8K tokens/sec, P99
latency ≈2.5 s, and use the JSON output to feed your alerting thresholds during
rollback.

For trace capture during drills:

```bash
nsys profile --force-overwrite true \
  --output=artifacts/single_gpu_fallback/nsys/serving_trace \
  --duration=15 --delay=5 --sample=none \
  PYTHONPATH=$PWD torchrun --nproc_per_node=1 \
    ch16/inference_server_load_test.py \
    --duration 20 \
    --target-qps 3 \
    --max-new-tokens 4 \
    --prompt-len-min 32 \
    --prompt-len-max 64 \
    --allow-smaller-world
```

## Capacity Expectations
- GPT-40B class models sustain ~1,850 tokens/sec after warmup (TorchDynamo,
  FlexAttention enabled).
- Peak memory footprint during 4K-token prompts: ~42 GB; leave 6–8 GB free for
  CUDA graphs and NCCL buffers.
- Power draw ~510 W; expect 3.6 tokens/joule (52 % of the multi-GPU system).

## Operational Tips
- **Prometheus hooks**: scrape the JSON artifact emitted by
  `test_gpt_large_optimized.py` for latency/tokens-per-joule gauges.
- **Graceful recovery**: keep `CUDA_VISIBLE_DEVICES` scoped to the process or
  service unit; once NVLink recovers you can restart the 8×GPU server in a
  parallel unit and shift traffic over.
- **Model refresh cadence**: reuse the same checkpoint that powers the
  multi-GPU tier. The fallback scripts accept standard `state_dict` loads—set
  `MODEL_PATH=/models/llama40b.pt` and add a short loader wrapper before
  calling `compare_inference_methods()`.
- **CI guardrail**: add a nightly job that runs the warmup benchmark with
  `CUDA_VISIBLE_DEVICES=0` to detect regressions in single-GPU kernel paths.
