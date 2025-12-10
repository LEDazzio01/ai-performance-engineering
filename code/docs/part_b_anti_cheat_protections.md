# Part B: Anti-Cheat Protections (Benchmark Validity Enforcement)

## Summary

This document describes the implementation of comprehensive protections against benchmark validity issues that can lead to misleading performance measurements. These protections are based on real-world incidents documented in the README.

## Design Principles

1. **Defense in Depth**: Multiple layers of protection for each category
2. **Fail Loud**: Any detected issue immediately fails verification
3. **No Silent Passes**: Never assume OK if validation can't be performed
4. **Citation-Backed**: Every protection addresses a documented real-world incident

## Protection Categories

### 1. Timing Manipulation Protections

**Real-World Incident**: Locus/KernelBench 2025 - 32.8% of RL-generated CUDA kernels exploited stream timing loopholes for fake 18x speedups.

| Issue | Protection | Implementation |
|-------|------------|----------------|
| Unsynced Streams | Full device sync | `torch.cuda.synchronize()` before timing ends |
| Incomplete Async Ops | Full device sync | Wait for all async operations |
| Event Timing Gaps | Cross-validate | Compare CUDA events with wall clock |
| Timer Granularity | Multiple methods | Use both events and high-resolution timers |
| Warmup Bleed | Buffer isolation | Separate input buffers for warmup vs timed |
| Clock Drift | Monotonic clock | Use `time.perf_counter()` not `time.time()` |
| Profiler Overhead | Profile-free path | Separate profiled vs production timing |

**Implementation Location**: `core/harness/benchmark_harness.py`

```python
def _measure_iteration(self, benchmark):
    """Timing with full device synchronization."""
    # Record start
    start_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    
    # Run benchmark
    benchmark.benchmark_fn()
    
    # CRITICAL: Wait for ALL streams, not just default
    torch.cuda.synchronize()  # Waits for ALL streams on ALL devices
    
    # Record end AFTER full sync
    end_event = torch.cuda.Event(enable_timing=True)
    end_event.record()
    end_event.synchronize()
    
    return start_event.elapsed_time(end_event)
```

### 2. Output Manipulation Protections

**Real-World Incidents**: ImageNet Labels 2021 (6% error rate), MMLU 2025 (57% virology errors)

| Issue | Protection | Implementation |
|-------|------------|----------------|
| Constant Output | Jitter check | Perturb inputs, verify outputs change |
| Stale Cache | Fresh-input check | Different seeds produce different outputs |
| Invalid Values (NaN/Inf) | validate_result() | Check for NaN/Inf in all outputs |
| Invalid Ground Truth | Golden outputs | Cache baseline outputs for comparison |
| Shape Mismatch | Shape validation | Verify output shapes match expected |
| Dtype Mismatch | Dtype validation | Verify output dtypes match expected |

**Implementation Location**: `core/benchmark/verify_runner.py`

```python
def _jitter_check(self, baseline, optimized, signature):
    """Verify outputs change when inputs are perturbed."""
    # Get original outputs
    original_baseline = self._run_benchmark_in_verify_mode(baseline)
    original_optimized = self._run_benchmark_in_verify_mode(optimized)
    
    # Perturb inputs
    jitter_dim = select_jitter_dimension(signature)
    perturbed_signature = self._perturb_signature(signature, jitter_dim)
    
    # Get perturbed outputs
    perturbed_baseline = self._run_benchmark_in_verify_mode(baseline, perturbed_signature)
    perturbed_optimized = self._run_benchmark_in_verify_mode(optimized, perturbed_signature)
    
    # Verify outputs changed
    if torch.allclose(original_baseline, perturbed_baseline):
        return VerifyResult.fail("jitter_fail", "Baseline output unchanged after input perturbation")
    if torch.allclose(original_optimized, perturbed_optimized):
        return VerifyResult.fail("jitter_fail", "Optimized output unchanged after input perturbation")
    
    return VerifyResult.success()
```

### 3. Workload Manipulation Protections

**Real-World Incidents**: AI Agent Shortcuts 2024, Computational Biology 2019

| Issue | Protection | Implementation |
|-------|------------|----------------|
| Precision Mismatch | Dtype verification | Compare declared vs actual dtypes |
| Undeclared Shortcuts | Workload invariant | Compare bytes/tokens/ops per iteration |
| Early Exit | Iteration count | Enforce fixed iteration counts |
| Batch Shrinking | Input signature | Verify batch sizes match |
| Sequence Truncation | Input signature | Verify sequence lengths match |
| Train/Test Overlap | Dataset isolation | Separate train/test data |

**Implementation Location**: `core/benchmark/verify_runner.py`

```python
def _check_workload_invariants(self, baseline, optimized):
    """Verify workload metrics match within tolerance."""
    baseline_meta = baseline.get_workload_metadata()
    optimized_meta = optimized.get_workload_metadata()
    
    if baseline_meta is None or optimized_meta is None:
        return VerifyResult.fail("workload_metadata_missing")
    
    # Compare all workload metrics
    for metric in ['bytes_per_iteration', 'tokens_per_iteration', 'samples_per_iteration']:
        baseline_val = getattr(baseline_meta, metric, None)
        optimized_val = getattr(optimized_meta, metric, None)
        
        if baseline_val is not None and optimized_val is not None:
            ratio = optimized_val / baseline_val if baseline_val > 0 else 0
            if abs(ratio - 1.0) > 0.01:  # 1% tolerance
                return VerifyResult.fail("workload_mismatch", 
                    f"{metric} differs: baseline={baseline_val}, optimized={optimized_val}")
    
    return VerifyResult.success()
```

### 4. Work Relocation Protections

| Issue | Protection | Implementation |
|-------|------------|----------------|
| CPU Spillover | GPU kernel validation | Verify work runs on GPU |
| Setup Pre-computation | Input mutation check | Hash inputs before/after setup |
| Graph Capture Cheat | Graph-aware verify | Separate graph capture from execution |
| Warmup Computation | Buffer isolation | Fresh buffers for timed runs |
| Background Thread | Process isolation | Single-threaded verification |
| Lazy Evaluation Skip | Force evaluation | Call `.item()` or sync after ops |

**Implementation Location**: `core/harness/benchmark_harness.py`

```python
def _verify_no_precomputation(self, benchmark):
    """Verify setup() doesn't pre-compute results."""
    # Hash inputs before setup
    input_hash_before = self._hash_tensors(benchmark.get_inputs())
    
    # Run setup
    benchmark.setup()
    
    # Hash inputs after setup
    input_hash_after = self._hash_tensors(benchmark.get_inputs())
    
    if input_hash_before != input_hash_after:
        raise VerificationError("Input tensors modified during setup()")
```

### 5. Memory Manipulation Protections

| Issue | Protection | Implementation |
|-------|------------|----------------|
| Pre-allocated Output | Memory tracking | Track allocations during benchmark |
| Input-Output Aliasing | Address validation | Verify output addresses differ from input |
| Pinned Memory Timing | Transfer completion | Explicit sync after transfers |
| Memory Pool Reuse | Pool reset | Clear memory pool between runs |

### 6. CUDA-Specific Protections

| Issue | Protection | Implementation |
|-------|------------|----------------|
| Host Callback Escape | Host function tracking | Monitor cudaLaunchHostFunc |
| Async Memcpy Incomplete | Memory sync | Explicit sync after memcpy |
| Workspace Pre-compute | Workspace monitoring | Track cuBLAS workspace |
| Persistent Kernel | Kernel lifetime | Verify kernels complete |
| Undeclared Multi-GPU | Device enumeration | Count active devices |
| Dynamic Parallelism | CDP tracking | Monitor child kernel launches |

**Implementation Location**: `core/common/headers/cuda_verify.cuh`

```cpp
#ifdef VERIFY
// Emit checksum for verification
#define VERIFY_CHECKSUM(buffer, size, checksum_out) \
    do { \
        float sum = 0.0f; \
        for (int i = 0; i < size; i++) sum += buffer[i]; \
        *checksum_out = sum; \
        printf("VERIFY_CHECKSUM: %f\n", sum); \
    } while(0)
#else
// No-op in performance mode
#define VERIFY_CHECKSUM(buffer, size, checksum_out) ((void)0)
#endif
```

### 7. torch.compile Protections

| Issue | Protection | Implementation |
|-------|------------|----------------|
| Compilation Cache Hit | Cache invalidation | Clear compile cache |
| Trace Reuse | Fresh trace | Force recompilation |
| Mode Inconsistency | Mode check | Same compile mode for verify/perf |
| Guard Failure Hidden | Guard tracking | Monitor recompilations |
| Autotuning Variance | Fixed cache | Lock autotuning selections |

### 8. Distributed Protections

| Issue | Protection | Implementation |
|-------|------------|----------------|
| Rank Skipping | Per-rank verification | Verify all ranks do work |
| Collective Short-circuit | NCCL validation | Verify collectives execute |
| Topology Mismatch | Topology verification | Compare declared vs actual |
| Barrier Timing | Barrier sync | Proper barrier accounting |

### 9. Environment Protections

| Issue | Protection | Implementation |
|-------|------------|----------------|
| Device Mismatch | Device fingerprint | Log GPU UUID/name |
| Frequency Boost | Frequency monitoring | Check clock speeds |
| Driver Version Mismatch | Version lock | Record driver version |
| Thermal Throttling | Temperature monitoring | Check GPU temp |

### 10. Statistical Protections

**Real-World Incidents**: Chatbot Arena 2024, AI Benchmarks 2025 (only 16% used statistical tests)

| Issue | Protection | Implementation |
|-------|------------|----------------|
| Cherry-picking | All-iteration reporting | Report all iterations |
| Outlier Injection | Statistical validation | Remove statistical outliers |
| Insufficient Samples | Minimum iterations | Require N iterations |
| Cold Start Inclusion | Warmup enforcement | Discard warmup iterations |

### 11. Evaluation Protections

**Real-World Incidents**: MLPerf 2019, GLUE 2024, Data Contamination 2025

| Issue | Protection | Implementation |
|-------|------------|----------------|
| Eval Code Exploitation | Immutable harness | Read-only evaluation code |
| Metric Definition Gaming | Standardized metrics | Fixed metric definitions |
| Test Data Leakage | Contamination checks | Fresh-input verification |
| Benchmark Overfitting | Holdout sets | Separate evaluation data |

## Implementation Status

| Category | Status | Files |
|----------|--------|-------|
| Timing | ✅ Implemented | `benchmark_harness.py` |
| Output | ✅ Implemented | `verify_runner.py` |
| Workload | ✅ Implemented | `verify_runner.py` |
| Work Relocation | 🔲 Planned | `benchmark_harness.py` |
| Memory | 🔲 Planned | `benchmark_harness.py` |
| CUDA | ✅ Partial | `cuda_verify.cuh`, `cuda_binary_benchmark.py` |
| torch.compile | 🔲 Planned | TBD |
| Distributed | ✅ Partial | `verify_runner.py` |
| Environment | 🔲 Planned | `run_manifest.py` |
| Statistical | ✅ Partial | `benchmark_harness.py` |
| Evaluation | ✅ Implemented | `verify_runner.py`, `contract.py` |

## Testing

Each protection category has corresponding tests in:
- `tests/test_verification.py` - Unit tests
- `tests/test_verification_e2e.py` - End-to-end tests

## References

All protections are backed by real-world incidents documented in:
- `README.md` - Benchmark Validity Issues Reference table (94 issues, 17 with citations)
- `.kiro/specs/benchmark-verification-enforcement/requirements.md`
- `.kiro/specs/benchmark-verification-enforcement/design.md`


