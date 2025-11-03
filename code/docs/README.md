# Documentation Index

## TMA (Tensor Memory Accelerator) Issues

### Current Status (Oct 31, 2025)

**TL;DR**: TMA descriptors are broken on B200/GB10 with CUDA 13.0. Use fallback paths (default).

### Quick Reference

| Document | Purpose | When to Read |
|----------|---------|-------------|
| **[TMA_STATUS_SUMMARY.md](TMA_STATUS_SUMMARY.md)** | 📊 Executive summary, test results, next steps | **START HERE** |
| **[nvidia_tma_bug_report.md](nvidia_tma_bug_report.md)** | 🐛 Formal bug report for NVIDIA support | Sharing with NVIDIA |
| **[ROOT_CAUSE_ANALYSIS.md](ROOT_CAUSE_ANALYSIS.md)** | 🔍 PyTorch/Triton root cause analysis | Understanding PyTorch issues |

### TMA Documentation

#### [TMA_STATUS_SUMMARY.md](TMA_STATUS_SUMMARY.md)
**Executive Summary** - Read this first

- ✅ What works (fallback paths)
- ❌ What's broken (TMA descriptors)
- 📋 Test results from B200 and GB10
- 🔧 Workarounds and recommendations
- 📞 What to share with NVIDIA

#### [nvidia_tma_bug_report.md](nvidia_tma_bug_report.md)
**Formal Bug Report** - For NVIDIA support tickets

- Detailed reproduction steps (B200 and GB10)
- Environment configurations
- Error messages and logs
- Parameter dumps for debugging
- Ready to share with NVIDIA support

## PyTorch/Triton Issues

### [ROOT_CAUSE_ANALYSIS.md](ROOT_CAUSE_ANALYSIS.md)
**Root Cause Analysis** - PyTorch and experimental features

Topics covered:
1. **torch.compile reliability issues**
   - Why large models (>40B) hang
   - Partial compilation mitigation (implemented)
   - Smart strategy selection (implemented)
   
2. **Experimental features**
   - Symmetric memory shim (documented, configurable)
   - Triton SM architecture patch (documented, configurable)

3. **Upstream dependencies**
   - What requires PyTorch fixes
   - What requires Triton fixes
   - Timeline for resolution

## Other Documentation

- [ch14/GB10_TRITON_SETUP_GUIDE.md](../ch14/GB10_TRITON_SETUP_GUIDE.md) - Complete GB10 (SM 12.1) Triton setup
- [ch14/GB10_TRITON_TMA_STATUS.md](../ch14/GB10_TRITON_TMA_STATUS.md) - Detailed GB10 TMA analysis
- [common/README.md](../common/README.md) - torch_compile_safe utility documentation
- [docs/READY_TO_RUN_GUIDE.md](READY_TO_RUN_GUIDE.md) - Full install + validation runbook
- [docs/incident_response.md](incident_response.md) - On-call playbook with single-GPU fallback
- [MODEL_SIZE_ANALYSIS.md](../MODEL_SIZE_ANALYSIS.md) - Performance vs. sequence/precision
- [docs/migration_to_sm100.md](migration_to_sm100.md) - A100/H100 → B200 migration checklist

## Quick Navigation

### I'm new here, what should I read?
1. [TMA_STATUS_SUMMARY.md](TMA_STATUS_SUMMARY.md) - Current TMA status
2. [ROOT_CAUSE_ANALYSIS.md](ROOT_CAUSE_ANALYSIS.md) - PyTorch/Triton issues
3. [READY_TO_RUN_GUIDE.md](READY_TO_RUN_GUIDE.md) - Install + reproduce baselines

### I need to report TMA issues to NVIDIA
- [nvidia_tma_bug_report.md](nvidia_tma_bug_report.md) - Ready-to-share bug report

### I'm debugging torch.compile hangs
- [ROOT_CAUSE_ANALYSIS.md](ROOT_CAUSE_ANALYSIS.md) - Root cause and mitigations
- [common/README.md](../common/README.md) - Safe compilation utilities

### I'm setting up GB10 (SM 12.1)
- [ch14/GB10_TRITON_SETUP_GUIDE.md](../ch14/GB10_TRITON_SETUP_GUIDE.md) - Complete setup guide
- [ch14/README.md](../ch14/README.md) - GB10 documentation index

### I'm migrating from A100/H100 to Blackwell
- [migration_to_sm100.md](migration_to_sm100.md) - End-to-end migration playbook
- [../MODEL_SIZE_ANALYSIS.md](../MODEL_SIZE_ANALYSIS.md) - Expected memory/latency deltas

---

**Last Updated**: October 31, 2025
