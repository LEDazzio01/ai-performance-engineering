# Migrating Workloads to NVIDIA Blackwell (SM_100 / B200)

This playbook captures the steps we use to migrate production inference
deployments from Ampere (A100) or Hopper (H100) to Blackwell-class GPUs
(B200/GB200, SM_100). Every recommendation references scripts or measurements
already tracked in this repository to keep the process auditable.

---

## 1. Executive Summary

- **Throughput expectation**: 15–16K tokens/sec on a 39.5B GPT at batch 4 /
  seq 2048 (FP16 eager). Larger contexts remain memory-bound even on B200.
- **Key enablers**: FlexAttention kernels, transformer-engine FP8 modes, and
  properly tuned NVLink/NVSwitch fabrics. All are validated in `run_all_tests.sh`.
- **Risks**: CUTLASS integration (known import warnings), `torch.compile`
  regressions on ≥8K tokens, and NVSwitch firmware mismatches. Mitigate with
  nightly benchmarks and the ready-to-run scripts provided here.

---

## 2. Pre-Migration Checklist (Source Hardware)

1. **Capture Baselines**
   - `python3 ch16/test_gpt_large_optimized.py --profile baseline`  
     Stores latency/throughput under `test_results/`.
   - `python3 tools/memory_profiler.py --config configs/ch16/memory_profile.yaml`
     Logs peak allocator state.
   - `python3 tools/power_monitor.py --duration 300`  
     Measure wattage for cost-per-token comparisons.
   - `python3 ch16/perplexity_eval.py --checkpoint <path>`  
     Record quality metrics.

2. **Archive Environment**
   - Export driver/CUDA versions (`nvidia-smi`, `nvcc --version`).
   - Snapshot Python environment (`pip freeze > baseline_requirements.txt`).

3. **Collect Nsight Traces (Optional)**
   - `./scripts/profile/enabled_nsight_systems.sh` for representative loads.

Store everything alongside the JSON outputs so you can diff against the B200
results later (see `tools/continuous_benchmark.py --archive`).

---

## 3. Target Environment Preparation (B200 / SM_100)

1. **Provision Hardware**
   - 8× B200 with NVSwitch backplane.
   - 2× 96-core CPUs and ≥1 TB system RAM if you plan to saturate all GPUs.

2. **Install Software**
   - `sudo ./setup.sh` (installs R580 driver + CUDA 13.0 + PyTorch 2.9 stack).
   - Verify with `./verify_pytorch.py`, `./verify_nvlink.py`,
     `./verify_cutlass.py`.

3. **Confirm Fabric Health**
   - Expect ~250 GB/s P2P and ~273 GB/s NCCL all-reduce (see `verify_nvlink`).
   - Run `python3 benchmark_peak.py` to record HBM (2.7–2.8 TB/s) and compute
     (1.28–1.30 PFLOPS) ceilings.

Refer to `docs/READY_TO_RUN_GUIDE.md` if you need a full walkthrough.

---

## 4. Porting & Feature Enablement

1. **FlexAttention / Flash Decoding**
   - Ensure `--attention-backend auto` for all benchmark scripts.
   - Verify the kernels actives via `python3 ch16/test_gpt_large_optimized.py --print-config`.

2. **Precision Strategy**
   - Start with FP16 to validate functionality.
   - Switch to transformer-engine FP8:  
     `python3 ch16/test_gpt_large_optimized.py --fp8-mode te-fused`.
   - For weight-only compression, reuse configs from
     `configs/ch16/benchmark_gpt_large_fp8.yaml`.

3. **Compiler Strategy**
   - Default to eager for 8K+ tokens; use `--compile max-autotune` for 2–4K
     sequences where autotune breaks even.
   - Disable CUTLASS via `TORCHINDUCTOR_DISABLE_CUTLASS=1` until NVIDIA fixes
     the `cuda.bindings` cleanup bug.

4. **Parallelism & Sharding**
   - Validate tensor-parallel splits:  
     `python3 ch16/multi_gpu_validation.py --tp-size 2`.
   - Measure NVLink impact using `test_results/large_gpt_tp2_*.json`; expect
     41 GB per GPU but lower throughput without overlap.

---

## 5. Validation on B200

| Check | Command | Expected Output |
|-------|---------|-----------------|
| Functional | `./run_all_tests.sh` | Exit code 0, new JSON under `test_results/` |
| Large GPT inference | `python3 ch16/test_gpt_large_optimized.py --profile large_gpt_benchmark` | Matches `MODEL_SIZE_ANALYSIS.md` tables (≤±3%) |
| FP8 sanity | `python3 ch16/test_gpt_large_optimized.py --profile large_gpt_fp8` | Peak memory ~55 GB, throughput within 12% of FP16 |
| MoE throughput | `python3 ch16/moe_performance_benchmark.py` | ~11.6K tok/s eager, see `synthetic_moe_results.json` |
| Power envelope | `python3 tools/power_monitor.py --duration 300` | ~2.3 kW under peak GPT load |

Run `tools/continuous_benchmark.py --input test_results/*.json --label b200_migration`
to package the comparison bundle. This keeps source vs. target deltas traceable.

---

## 6. Rollout Checklist

- ✅ **Performance parity or better** (latency, throughput within ±5% of target).
- ✅ **Quality unchanged** (perplexity or accuracy within statistical noise).
- ✅ **Power budget met** (≤2.4 kW sustained per 8× node).
- ✅ **Observability ready** (Nsight traces captured, alerts updated).
- ✅ **Fallback plan documented** (scripted toggle back to FP16/BF16 paths).

Only flip production traffic once all boxes are checked and the benchmark
artifacts are archived.

---

## 7. Known Issues & Workarounds

- **torch.compile regressions (≥8K tokens)**  
  Stick to eager or reduce-overhead mode and monitor `MODEL_SIZE_ANALYSIS.md`
  for future updates.

- **CUTLASS cleanup warnings**  
  Benign but noisy. Disable CUTLASS until NVIDIA resolves
  `ModuleNotFoundError: cutlass._mlir`.

- **NVSwitch firmware drift**  
  If `verify_nvlink.py` reports <200 GB/s, update NVSwitch firmware using
  NVIDIA's out-of-band tools before debugging PyTorch.

- **FP8 numerical drift**  
  Validate with `ch16/perplexity_eval.py` on your real checkpoints before
  shipping. Keep the eager FP16 path as a fallback.

---

## 8. Appendix: Files to Update When Migration Completes

- Add the new JSON artifacts under `test_results/`.
- Update `MODEL_SIZE_ANALYSIS.md` and `MODEL_SIZE_RECOMMENDATIONS.md` with your
  measurements.
- Move the corresponding items in `docs/TODO.md` from “TODO” to “Completed”.

Following this checklist keeps the repo self-consistent and auditable for future
migrations (Grace-Blackwell, Rubin, etc.).
