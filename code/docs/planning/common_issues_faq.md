# Common Issues FAQ (LLM Inference & Training)

This FAQ captures the recurring issues we see while operating large language
models on 8×B200 systems. Each entry includes the symptom, root cause, and the
fix or mitigation that worked in production.

## 1. Warmup Hangs When Using `torch.compile`
- **Symptom**: Benchmark never exits warmup; CPU pegged at 100%; no GPU activity.
- **Root Cause**: Inductor compiler stalls on models >40B parameters.
- **Fix**: Re-run with `--compile-mode smart` (uses `safe_compile()` timeout) or
  add `--skip-compile`. Capture a repro bundle via
  `python tools/torch_compile_repro.py --output-dir artifacts/torch_compile_repro`
  before filing upstream bugs. See `docs/torch_compile_troubleshooting.md`.

## 2. NVLink Bandwidth Looks Like PCIe
- **Symptom**: `nvidia-smi topo -m` reports `PHB`, throughput <30 GB/s per link.
- **Root Cause**: `NCCL_P2P_DISABLE=1` leftover env disables NVLink peer access.
- **Fix**: Export the validated config:
  ```bash
  export NCCL_P2P_LEVEL=NVL
  export NCCL_P2P_DISABLE=0
  export NCCL_IB_DISABLE=1
  export NCCL_SHM_DISABLE=0
  export NCCL_NET_GDR_LEVEL=5
  ```
  Then rerun `python scripts/capture_fp8_flexattention_traces.sh`. Refer to
  `docs/nvlink_pcie_playbook.md` for topology diagnostics.

## 3. Throughput Drops After Rollback to Single GPU
- **Symptom**: Tokens/sec falls below 1.5K after moving traffic to fallback.
- **Root Cause**: Leftover tensor-parallel env vars keep kernels in multi-GPU
  mode and prevent CUDA graphs re-use.
- **Fix**: Follow the rollback checklist in
  `docs/single_gpu_serving_guide.md` (unset `NCCL_*` vars, pin
  `CUDA_VISIBLE_DEVICES=0`). Warm the model with
  `python ch16/test_gpt_large_optimized.py --tensor-parallel-gpus 1 --skip-compile`.

## 4. Memory Exhaustion on 32K+ Token Runs
- **Symptom**: OOM at ~70% progress when testing ultra-long contexts.
- **Root Cause**: KV cache + activation tensors exceed 48 GB per GPU.
- **Fix**: Run the automated sweep:
  ```bash
  python tools/long_context_validation.py \
    --sequence-lengths 32768 65536 \
    --tensor-parallel-gpus 8 \
    --output-json artifacts/long_context/validation.json
  ```
  Use the report to adjust batch size and enable activation checkpointing in
  the decode stack. Document the final memory headroom in
  `docs/long_context_playbook.md` (template pending data).

## 5. Power Efficiency Regressions After Enabling FP8
- **Symptom**: Tokens/joule drop below 6 despite FP8 kernels being active.
- **Root Cause**: Mixed-precision fallback triggered (activations running in
  BF16) due to invalid calibration.
- **Fix**: Regenerate scaling factors with
  `python ch16/fp8_transformer_engine.py --recalibrate --output-json artifacts/fp8/recalibration.json`
  and re-run the power sweep using
  `python tools/precision_power_sweep.py --modes fp16 bf16 fp8_te --skip-compile`.
  Compare outputs with `python tools/power_efficiency_analyzer.py`.

If you encounter a new issue, add an entry following the same structure and
link to any new scripts or guides that were created during the fix.
