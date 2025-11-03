# Incident Response Runbook – Multi-GPU Serving

This runbook gives on-call engineers a single place to recover production
serving when the 8×B200 tensor-parallel tier degrades. Keep it alongside the
pager rotation notes and update the timestamps after every drill.

## 0. Rapid Triage
- **Detect**: Alerts from load-test mirroring, NVLink topology monitors, or QPS
  regression dashboards fire.
- **Decide**: If symptoms involve NVLink loss, NCCL hangs, or PCIe-only
  failover, jump to the fallback workflow below while the multi-GPU tier is
  investigated offline.

## 1. Execute the Fallback Playbook
Follow the detailed checklist in `docs/single_gpu_serving_fallback.md`. The key
steps are summarised here for quick reference:

1. Drain tensor-parallel workers and freeze the 8×GPU deployment.
2. Reset NCCL state and scope serving to a single device:
   ```bash
   unset NCCL_P2P_DISABLE NCCL_P2P_LEVEL
   export CUDA_VISIBLE_DEVICES=0
   ```
3. Warm the model to prime TorchDynamo graphs and capture the baseline JSON:
   ```bash
   python ch16/test_gpt_large_optimized.py \
       --iters 3 \
       --warmup 1 \
       --tensor-parallel-gpus 1 \
       --output-json fallback_benchmark.json
   ```
4. Launch the single-GPU serving path:
   ```bash
   python ch16/inference_optimizations_blackwell.py
   ```
5. Apply back-pressure controls (batch ≤16, queue timeout 5 s, long contexts in
   ≥4 s windows). See the fallback playbook for full guidance.
6. Validate the fallback load test so on-call monitoring dashboards have fresh
   baselines:
   ```bash
   mkdir -p artifacts/single_gpu_fallback
   PYTHONPATH=$PWD torchrun --nproc_per_node=1 \
     ch16/inference_server_load_test.py \
     --duration 20 \
     --target-qps 3 \
     --max-new-tokens 4 \
     --prompt-len-min 32 \
     --prompt-len-max 64 \
     --allow-smaller-world \
     --output-json artifacts/single_gpu_fallback/load_test.json
   ```

> **Bookmark**: For rationale, capacity expectations, and operational notes,
> read `docs/single_gpu_serving_fallback.md`.

## 2. Preserve Forensics
- Capture the latest Nsight trace and load-test artifacts (per the load-testing
  guide) so the root cause can be analysed post-incident.
- Recommended fallback Nsight capture (matches the command above):
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
- Archive `artifacts/single_gpu_fallback/`, `load_test_latest/`, `fallback_benchmark.json`, and system logs to the
  incident ticket.

## 3. Restore the Multi-GPU Tier
- Once NVLink/NCCL issues are resolved, warm a fresh load test:
  ```bash
  ./tools/orchestrate_8xb200_load_test.sh 900 180 postmortem_validation
  ```
- If throughput and latency match the published baselines, gradually shift
  traffic back from the fallback pool.

## 4. Post-Incident Checklist
- Fill in the incident doc with timelines, commands run, and artifacts captured.
- Update this runbook (and the fallback playbook) with any deviations that
  improved response time.

**Last updated:** November 2, 2025
