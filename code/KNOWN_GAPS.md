# Known Gaps and Limitations

**Last Updated**: October 29, 2025  
**Purpose**: Honest documentation of what's NOT implemented, tested, or working

This document exists because we found fabricated performance claims in the codebase. Moving forward, we document both what works AND what doesn't.

---

## ✅ Latest Validation

- `./run_all_tests.sh` completed successfully on NVIDIA B200 hardware (2025-10-28 14:23 UTC)  
  Results: `test_results_20251028_142311/`
- Remaining gaps below are still open until the underlying implementation/testing work is finished.

---

## 🚨 Critical Gaps

### 1. FP8 Quantization - AVAILABLE
**Status**: ✅ transformer_engine integration and validation hooks

**What changed:**
- `ch16/test_gpt_large_optimized.py` now supports `--fp8-mode transformer-engine`
  (auto-falls back to weight-only when TE is missing)
- helper module `ch16/fp8_transformer_engine.py` converts linear layers to
  TE kernels and manages `fp8_autocast`
- JSON outputs report the active precision mode for each workload

**Validation:**
- `python ch16/test_gpt_large_optimized.py --fp8-mode auto`
- Accuracy guard: `python ch16/perplexity_eval.py ...` against FP32 baseline

---

### 2. Performance Claims - FABRICATED

**Status**: ❌ Documentation corrected

**What was claimed:**
- 1.65x torch.compile speedup
- 14,050 tokens/sec throughput

**Reality (from `gpt_oss_120b_results.json`):**
- 0.98x "speedup" (actually slower)
- 5,636 tokens/sec throughput

**Gap**: 68% speedup error, 149% throughput error

**Fix**: All docs now updated with actual numbers

---

### 3. MODEL_SIZE_ANALYSIS.md - MISSING (Now Fixed)

**Status**: ✅ Created with honest results

**Was claimed**: Referenced in multiple places but didn't exist  
**Now**: Comprehensive analysis based on actual benchmarks

---

## ⚠️ Partially Implemented

### 4. Large Model Testing (30B+)
**Status**: ⚠️ Expanded coverage, still gathering hardware baselines

**What works now:**
- ✅ Multiple batch/sequence regimes including 12K & 16K tokens (`test_gpt_large_optimized.py`)
- ✅ FlexAttention, transformer_engine FP8, and tensor-parallel validation hooks
- ✅ JSON output enriched with precision/attention metadata for CI dashboards

**Still missing:**
- ❌ Cross-architecture sweeps (vision, diffusion, recommenders)
- ❌ Hardware-derived bottleneck analysis (Nsight traces)
- ❌ Verified results on full 8x B200 systems (current runs on smaller rigs)

---

### 5. torch.compile Benefits - UNRELIABLE
**Status**: ⚠️ Works sometimes, fails sometimes

**Measured results:**
- Small models (1B): 1.02x speedup
- Medium models (8B): 1.00x speedup  
- Large models (40B): 0.98x speedup (regression)

**Why?**
- Blackwell baseline is already fast
- Many workloads are memory-bound
- Compilation overhead not always amortized
- Need architecture-specific tuning

**Recommendation**: Profile before assuming torch.compile helps

---

### 6. Multi-GPU Support
**Status**: ⚠️ Validation scripts added; awaiting full-scale hardware run

**Progress:**
- `ch16/multi_gpu_validation.py` checks tensor-parallel correctness locally
- `ch16/inference_server_load_test.py` exercises the 8x B200 server under load

**Outstanding:**
- Run the load test + validation on actual 8x B200 clusters and archive logs
- Capture NVLink bandwidth/latency metrics during stress runs

---

## 📝 Missing Features

### 7. FlexAttention Integration
**Status**: ✅ Implemented in GPT benchmark & inference server

- `test_gpt_large_optimized.py` selects FlexAttention via `--attention-backend flex|auto`
- Inference server routes through FlexAttention and exposes sliding-window masks
- Load tester exercises FlexAttention paths across tensor-parallel ranks

---

### 8. Long Sequence Testing (8K+ tokens)
**Status**: ✅ 12K/16K coverage baked into benchmark suite

- Default workloads now include 12K & 16K token scenarios
- JSON results capture per-workload memory footprint for regression tracking
- Remaining TODO: extend to 32K once silicon with enough memory is available

---

### 9. MoE (Mixture of Experts) Models
**Status**: ✅ Dedicated benchmark + TE support

- `ch16/moe_performance_benchmark.py` measures throughput/latency across expert counts
- Shares embedding & output stack with dense GPT benchmark for fair comparisons
- Continuous benchmark config includes the MoE run for regression tracking

---

### 10. Production Inference Server
**Status**: ⚠️ Load harness landed; need real cluster telemetry

- `ch16/inference_server_load_test.py` (torchrun) generates traffic and aggregates latency percentiles
- Completion callbacks expose per-request latencies for custom dashboards
- Next step: capture Nsight/NVML traces on production hardware and document results

---

## 🔬 Testing Gaps

### 11. Memory Profiling
**Status**: ⚠️ Tooling added; integrate into CI pipeline

- `python tools/memory_profiler.py <script>` captures operator-level CUDA
  memory usage and exports Chrome traces
- Nsight deep dives still required for kernel-level bottlenecks

---

### 12. Accuracy/Quality Testing
**Status**: ⚠️ Perplexity harness available; expand coverage

- `ch16/perplexity_eval.py` computes cross-entropy/perplexity on tokenized corpora
- Need curated evaluation sets + scripted FP32 vs FP8 comparisons

---

### 13. Power/Energy Measurements
**Status**: ⚠️ Sampling harness in place

- `tools/power_monitor.py` samples NVML power while running arbitrary commands
- Combine with continuous benchmark runs to compute cost/token; capture real hardware baselines

---

## 🛠️ Tool Limitations

### 14. Profiling Integration
**Status**: ⚠️ Foundational tooling in place; automate Nsight capture next

- `tools/memory_profiler.py` wraps workloads with torch.profiler memory stats
- Continuous benchmark harness can hook profiling commands, but Nsight CI exports remain TODO

---

### 15. Continuous Benchmarking
**Status**: ✅ Configurable automation shipped

- `tools/continuous_benchmark.py` runs suite from JSON config and archives results
- Example config: `docs/examples/continuous_benchmark.json`
- Next step: wire into CI/cron on production hardware

---

## 📚 Documentation Gaps

### 16. Architecture-Specific Guides
**Status**: ✅ Initial guide published (`docs/architecture_guides.md`)
- Covers dense GPT, MoE, inference serving, and evaluation hooks
- Add vision/diffusion sections once validated

---

### 17. Migration Guides
**Status**: ✅ `docs/migration_to_b200.md` documents step-by-step workflow
- Includes environment prep, validation checklists, and performance parity checks

---

## ✅ What Actually Works

To balance the gaps, here's what's validated:

1. ✅ **Basic hardware access**: B200 detected, CUDA works
2. ✅ **HBM3e bandwidth**: Measured at 2.73 TB/s (35% of theoretical)
3. ✅ **FP16 compute**: 1291 TFLOPS achieved
4. ✅ **Small model examples**: Run successfully (even if not optimal)
5. ✅ **Test infrastructure**: pytest suite works
6. ✅ **CUDA kernel examples**: Compile and run
7. ✅ **Documentation**: Now honest about limitations

---

## 🎯 Priority Fixes

### Immediate (This Week)
1. ⚠️ Run load tester + power monitor on production 8x B200 hardware and archive results
2. ⚠️ Capture Nsight Systems traces for FP8 + FlexAttention workloads

### Short-term (This Month)
3. 🔄 Expand perplexity harness with curated evaluation sets (FP16 vs FP8 deltas)
4. 🔄 Add automated alerts on continuous benchmark regressions

### Long-term (This Quarter)
5. 📝 Extend architecture guide with vision/diffusion best practices
6. 📝 Document end-to-end MoE deployment (routing telemetry, autoscaling)
7. 📝 Publish power-efficiency baselines (tokens/J) for major workloads

---

## 🤝 Contributing

If you implement any of these missing pieces:

1. **Update this document** (remove from "gaps")
2. **Add actual test results** (no fabricated numbers)
3. **Document limitations** (be honest about what doesn't work)
4. **Include reproduction steps** (make it verifiable)

---

## 📖 Related Documentation

- `MODEL_SIZE_ANALYSIS.md` - Honest performance results
- `MODEL_SIZE_RECOMMENDATIONS.md` - Updated with realistic expectations
- `docs/performance_baseline.md` - Validated baseline metrics
- `docs/architecture_guides.md` - Architecture-specific tuning recipes
- `docs/migration_to_b200.md` - Migration checklist from A100/H100
- `ch16/test_fp8_quantization_real.py` - Honest FP8 implementation

---

## Disclaimer

This document exists because we found and fixed fabricated claims. We're committed to:

✅ **Honesty** over hype  
✅ **Measured results** over projections  
✅ **Clear limitations** over vague promises  
✅ **Reproducible benchmarks** over aspirational claims  

If you find more gaps or false claims, please document them here.

**Remember**: It's better to have honest gaps than dishonest claims.
