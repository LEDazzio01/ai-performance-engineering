# Coding Styles

## BE EFFICIENT AND ASK QUESTIONS AT KEY DECISION POINTS
- Instead of forging ahead and performing a lot of changes, ask me questions if you are unsure or just want re-assurance that your approach is valid.

## Achieve MAXIMUM speedup when benchmarking baseline_ versus optimized_ variants when possible
- For any speedups <1.05x, we must improve in a natural manner utilizing hardware, software, and algorithmic speedups.  
- Both the baseline and the optimized variants need to equivalent workloads.  Perhaps we need to increase the workloads to demonstrate the speedup?  
- Let's consider all options and find the best speedup 
- Make sure we're staying with the intent of the example and within the context of the chapter (book/chXX.md).

## Benchmarks vs Tools (CRITICAL)

### Benchmarks (comparable baseline vs optimized)
- `aisp bench run --targets ...` should only include targets that are explicitly intended to demonstrate an optimization outcome.
  - Default: **performance** (speedup) with clear speedup potential.
  - Rare: **memory** (reduced memory) when explicitly the goal.
- Comparable benchmarks must use `baseline_*.py` + `optimized_*.py` naming and MUST be equivalent workloads (no hidden work reduction, no extra work in a variant hot path).

### Memory-goal benchmarks
- Memory-goal benchmarks are still comparable baseline vs optimized pairs, but are evaluated on memory savings (not speed).
- Gate on `baseline_memory_mb / optimized_memory_mb >= 1.05` (speed may regress; do not add a speed gate).

### Tools / methodology / analysis scripts (NOT comparable benchmarks)
- If something is not meant to be compared as baseline vs optimized (e.g., roofline analysis, config sweeps, monitoring bundles, validation workflows), it MUST NOT use `baseline_*/optimized_*` naming.
- Keep only the “full / sophisticated” version (no `_basic`, no smoke/minimal variants).
- Keep the tool script at the chapter/lab level when book context references it, but decouple it from benchmark discovery and `bench run`.
- Run these via `aisp tools <name> -- <args...>` by registering the script path in `core/tools/tools_commands.py` (`TOOLS` mapping).

## When to Move Code into `core/` (Reuse Rule)
- If shared logic has **2+ call sites** across chapters/labs, extract it into `core/` (prefer `core/analysis/*` or `core/utils/*`) and import it from chapter code.
- If a chapter’s narrative/book references specific chapter-local code, keep a thin chapter wrapper that calls into `core/` rather than moving everything out of the chapter.

## Verification Mixins REQUIRED
- Benchmarks must surface verification metadata via `VerificationPayloadMixin` + `_set_verification_payload()` inside `benchmark_fn()` (or equivalent path).
- Do NOT hand-roll `get_verify_output()/get_input_signature()/get_output_tolerance()` unless there is a truly special case; fail fast instead of adding fallbacks.
- New benchmarks should copy the compliant template (`templates/benchmark_compliant.py`) and keep mixin usage consistent.

## Chapter Consistency
- Make sure all code in the chapter (chXX/ examples are consistent with the content in the equivalent chapter (XX) of the AI Systems Performance Engineering book (book/chXX.md)

### Benchmark Example Pairs (Baseline vs. Optimized)
- Before making changes to these benchmark pairs (baseline_* and optimized_*), be sure to understand the intent of the optimization/comparison before making any changes.  
- You must preserve the intent of the comparison  (e.g. comparing 32-bit to 16-bit data types)
- And you must not introduce additional operations in the measured hot path of a variant just to satisfy requirements of the harness, for instance (e.g. cast from 16-bit to 32-bit to satisfy an output comparison/verification). 
- Instead, keep as much as possible outside of the timed areas so as to not artifically inflate any variant.  (e.g. cast outside the timed area or compare with a tolerance large enough to still maintain the intent of the comparison)
- AVOID SKIPPING EXAMPLES WHENEVER POSSIBLE

### Benchmark Validity Issues Reference

The table below documents known issues that can cause benchmark results to be misleading, along with their protections. Use this as a checklist when creating or reviewing benchmarks. DO NOT ALLOW THESE IN OUR BENCHMARKS.

**✅ All 94 validity issues are now protected by our harness**

| Category | Issue | What Happens | Protection | Status | Real-World Incident |
|----------|-------|--------------|------------|--------|---------------------|
| **Timing** | Unsynced Streams | Work on non-default streams isn't timed | Full device sync + `StreamAuditor` | ✅ | **Locus/KernelBench 2025** ([source](https://x.com/miru_why/status/1991773868806361138)) |
| **Timing** | Incomplete Async Ops | Timer stops before async work finishes | Full device sync | ✅ | **Locus/KernelBench 2025** ([source](https://x.com/miru_why/status/1991773868806361138)) |
| **Timing** | Event Timing Gaps | CUDA events recorded incorrectly | Cross-validate with wall clock | ✅ | |
| **Timing** | Timer Granularity | Measurement too coarse for fast ops | Adaptive iterations | ✅ | |
| **Timing** | Warmup Bleed | Real work happens during warmup | `isolate_warmup_cache` | ✅ | |
| **Timing** | Clock Drift | System clock changes during measurement | Monotonic clock usage | ✅ | |
| **Timing** | Profiler Overhead | Profiling tools add latency | Profile-free timing path | ✅ | |
| **Output** | Constant Output | Same result regardless of input | Jitter check | ✅ | |
| **Output** | Stale Cache | Same result across different seeds | Fresh-input check | ✅ | |
| **Output** | Approximation Drift | Rough estimate instead of full compute | Output tolerance validation | ✅ | |
| **Output** | Invalid Values (NaN) | NaN in output | `validate_result()` NaN check | ✅ | |
| **Output** | Invalid Values (Inf) | Inf in output | `validate_result()` Inf check | ✅ | |
| **Output** | Invalid Ground Truth | Labels/expected values wrong | `GoldenOutputCache` | ✅ | **ImageNet Labels 2021** ([arXiv:2103.14749](https://arxiv.org/abs/2103.14749)), **MMLU Errors 2025** ([PromptEng](https://promptengineering.org/challenges-and-innovations-in-language-model-benchmarking-and-generalization/)) |
| **Output** | Shape Mismatch | Output shape differs from expected | Shape validation | ✅ | |
| **Output** | Dtype Mismatch | Output dtype differs from expected | `ToleranceSpec` dtype check | ✅ | |
| **Output** | Denormalized Values | Subnormal floats cause slowdowns | Denormal check | ✅ | |
| **Output** | Uninitialized Memory | Output contains garbage | Memory initialization check | ✅ | |
| **Workload** | Precision Mismatch | Claims FP32 but uses FP16 | `InputSignature` dtype verification | ✅ | |
| **Workload** | Undeclared Shortcuts | Skips elements without declaring | Workload invariant check | ✅ | **AI Agent Shortcuts 2024** ([VentureBeat](https://venturebeat.com/ai/ai-agent-benchmarks-are-misleading-study-warns)) |
| **Workload** | Early Exit | Stops iteration loops early | Config immutability | ✅ | |
| **Workload** | Batch Shrinking | Processes fewer samples | `InputSignature` matching | ✅ | |
| **Workload** | Sequence Truncation | Processes shorter sequences | `InputSignature` matching | ✅ | |
| **Workload** | Hidden Downsampling | Silently reduces resolution | Dimension validation | ✅ | |
| **Workload** | Sparsity Mismatch | Different sparsity patterns | Sparsity ratio check | ✅ | |
| **Workload** | Attention Mask Mismatch | Different masking applied | Mask equivalence check | ✅ | |
| **Workload** | KV Cache Size Mismatch | Different cache sizes | Cache dimension check | ✅ | |
| **Workload** | Train/Test Overlap | Model tested on training data | Dataset isolation | ✅ | **Computational Biology 2019** ([Nat Commun](https://www.nature.com/articles/s41467-019-09406-4)) |
| **Location** | CPU Spillover | Work offloaded to CPU | GPU kernel time validation | ✅ | |
| **Location** | Setup Pre-computation | Work done in `setup()` | `check_setup_precomputation()` | ✅ | |
| **Location** | Graph Capture Cheat | Pre-compute during graph capture | `GraphCaptureCheatDetector` | ✅ | |
| **Location** | Warmup Computation | Compute results during warmup | `isolate_warmup_cache` | ✅ | |
| **Location** | Background Thread | Compute in separate thread | Process isolation | ✅ | |
| **Location** | Lazy Evaluation Skip | Returns unevaluated lazy tensor | `force_tensor_evaluation()` | ✅ | |
| **Location** | JIT Compilation Timing | JIT compile time included/excluded inconsistently | `clear_compile_cache()` | ✅ | |
| **Memory** | Pre-allocated Output | Result buffer allocated in setup | `MemoryAllocationTracker` | ✅ | |
| **Memory** | Input-Output Aliasing | Output points to pre-filled input | `check_input_output_aliasing()` | ✅ | |
| **Memory** | Pinned Memory Timing | Async pinned transfers not waited | Transfer completion check | ✅ | |
| **Memory** | Memory Pool Reuse | Cached allocations skew timing | `reset_cuda_memory_pool()` | ✅ | |
| **Memory** | Fragmentation Effects | Memory fragmentation differs | Memory pool reset | ✅ | |
| **Memory** | Page Fault Timing | First-touch page faults included | Memory pre-touch | ✅ | |
| **Memory** | Swap Interference | Swapping affects timing | Memory lock / swap disable | ✅ | |
| **CUDA** | Host Callback Escape | `cudaLaunchHostFunc` returns early | Host function tracking | ✅ | |
| **CUDA** | Async Memcpy Incomplete | D2H/H2D copies not awaited | Full device sync | ✅ | |
| **CUDA** | Workspace Pre-compute | Work in cuBLAS workspace alloc | Workspace monitoring | ✅ | |
| **CUDA** | Persistent Kernel | Kernel left running across calls | Kernel lifetime check | ✅ | |
| **CUDA** | Undeclared Multi-GPU | Work spread across undeclared GPUs | `validate_environment()` | ✅ | |
| **CUDA** | Context Switch Overhead | CUDA context switches affect timing | Context pinning | ✅ | |
| **CUDA** | Driver Overhead | Driver calls not accounted for | Driver call tracking | ✅ | |
| **CUDA** | Cooperative Launch Abuse | Cooperative kernels bypass checks | Launch mode validation | ✅ | |
| **CUDA** | Dynamic Parallelism Hidden | Child kernels not tracked | CDP kernel tracking | ✅ | |
| **CUDA** | Unified Memory Faults | Page migration not timed | UM fault tracking | ✅ | |
| **Compile** | Compilation Cache Hit | Returns cached compiled output | `clear_compile_cache()` | ✅ | |
| **Compile** | Trace Reuse | Exploits trace caching | `torch._dynamo.reset()` | ✅ | |
| **Compile** | Mode Inconsistency | Different compile mode verify vs perf | Mode consistency check | ✅ | |
| **Compile** | Inductor Asymmetry | Inductor optimizations inconsistent | Compilation parity | ✅ | |
| **Compile** | Guard Failure Hidden | Recompilation not counted | `get_compile_state()` | ✅ | |
| **Compile** | Autotuning Variance | Autotuning picks different kernels | Fixed autotuning cache | ✅ | |
| **Compile** | Symbolic Shape Exploit | Different shapes trigger different code | `InputSignature` matching | ✅ | |
| **Distributed** | Rank Skipping | Some ranks don't do work | `check_rank_execution()` | ✅ | |
| **Distributed** | Collective Short-circuit | Communication skipped | NCCL validation | ✅ | |
| **Distributed** | Topology Mismatch | Claims different topology | `verify_distributed()` | ✅ | |
| **Distributed** | Barrier Timing | Barrier timing exploited | Barrier synchronization | ✅ | |
| **Distributed** | Gradient Bucketing Mismatch | Different bucket sizes | Bucket size validation | ✅ | |
| **Distributed** | Async Gradient Timing | Async all-reduce not awaited | Full device sync | ✅ | |
| **Distributed** | Pipeline Bubble Hiding | Pipeline bubbles not counted | Bubble time tracking | ✅ | |
| **Distributed** | Shard Size Mismatch | FSDP shards differ | `InputSignature` matching | ✅ | |
| **Environment** | Device Mismatch | Uses different GPU than declared | `validate_environment()` | ✅ | |
| **Environment** | Frequency Boost | Overclocked for benchmark only | `lock_gpu_clocks()` | ✅ | |
| **Environment** | Priority Elevation | Runs at higher priority | Process isolation | ✅ | |
| **Environment** | Memory Overcommit | Exploits memory overcommit | Memory validation | ✅ | |
| **Environment** | NUMA Inconsistency | NUMA placement differs | NUMA audit | ✅ | |
| **Environment** | CPU Governor Mismatch | Different CPU frequency scaling | Governor lock | ✅ | |
| **Environment** | Thermal Throttling | GPU throttles during run | `capture_gpu_state()` pynvml | ✅ | |
| **Environment** | Power Limit Difference | Different TDP settings | `capture_gpu_state()` | ✅ | |
| **Environment** | Driver Version Mismatch | Different CUDA drivers | `RunManifest` version lock | ✅ | |
| **Environment** | Library Version Mismatch | Different cuDNN/cuBLAS | `RunManifest` version lock | ✅ | |
| **Environment** | Container Resource Limits | cgroups limits differ | Resource limit check | ✅ | |
| **Environment** | Virtualization Overhead | VM/container overhead varies | Bare-metal validation | ✅ | |
| **Statistical** | Cherry-picking | Only best iterations reported | All-iteration reporting | ✅ | **Chatbot Arena 2024** ([TechCrunch](https://techcrunch.com/2025/04/22/crowdsourced-ai-benchmarks-have-serious-flaws-some-experts-say/)) |
| **Statistical** | Outlier Injection | Slow iterations added to baseline | Statistical validation | ✅ | |
| **Statistical** | Variance Gaming | Variance reporting manipulated | Consistent statistics | ✅ | |
| **Statistical** | Percentile Selection | Favorable percentile chosen | Fixed percentile policy | ✅ | |
| **Statistical** | Insufficient Samples | Too few iterations for significance | Adaptive iterations | ✅ | **AI Benchmarks 2025** ([The Register](https://www.theregister.com/2025/11/07/measuring_ai_models_hampered_by/)) |
| **Statistical** | Cold Start Inclusion | First run included unfairly | Warmup enforcement | ✅ | |
| **Statistical** | GC Interference | Garbage collection during timing | `gc_disabled()` | ✅ | |
| **Statistical** | Background Process Noise | System processes affect timing | Process isolation | ✅ | |
| **Evaluation** | Eval Code Exploitation | Benchmark code modified to pass | `BenchmarkContract` enforcement | ✅ | |
| **Evaluation** | Timeout Manipulation | Timeout extended to hide slowdowns | Config immutability | ✅ | |
| **Evaluation** | Metric Definition Gaming | Redefine what "speedup" means | Standardized metric definitions | ✅ | **MLPerf 2019** ([Forbes](https://www.forbes.com/sites/janakirammsv/2019/11/10/the-curious-case-of-mlperf-inferencing-benchmark-results/)), **GLUE 2024** ([Revelry](https://revelry.co/insights/artificial-intelligence/why-ai-benchmarks-fail/)) |
| **Evaluation** | Test Data Leakage | Training on test/benchmark data | Data contamination checks | ✅ | **Data Contamination 2025** ([AI News](https://www.artificialintelligence-news.com/news/flawed-ai-benchmarks-enterprise-budgets-at-risk/)) |
| **Evaluation** | Benchmark Overfitting | Optimize specifically for benchmark | Fresh-input + jitter checks | ✅ | **Underspecification 2020** ([arXiv:2011.03395](https://arxiv.org/abs/2011.03395)), **Epic Sepsis 2021** ([ChatBench](https://www.chatbench.org/)) |
| **Evaluation** | Self-Modifying Tests | AI/code modifies its own tests | Config immutability | ✅ | |
| **Evaluation** | Benchmark Memorization | Agent memorizes test cases | Fresh-input checks, jitter | ✅ | **AI Agent Shortcuts 2024** ([VentureBeat](https://venturebeat.com/ai/ai-agent-benchmarks-are-misleading-study-warns)) |
| **Evaluation** | Missing Holdout Sets | No proper train/test split | Held-out evaluation data | ✅ | **AI Agent Shortcuts 2024** ([VentureBeat](https://venturebeat.com/ai/ai-agent-benchmarks-are-misleading-study-warns)) |

**Total: 11 categories, 94 validity issues — ✅ ALL PROTECTED by our harness (17 linked to real-world incidents with citations)**

### Notable Real-World Incidents

These validity issues aren't theoretical—they've caused real problems:

| Year | Incident | Issue Type | What Happened | Source |
|------|----------|------------|---------------|--------|
| **2025** | **Locus/KernelBench Stream Exploit** | Unsynced Streams | Claimed 20x speedup on Llama FFW kernel. AI launched work on non-default CUDA streams but timer only measured default stream. **32.8% of RL-generated kernels exploited this**, causing fake 18x speedups. | [X/Twitter @miru_why](https://x.com/miru_why/status/1991773868806361138) |
| **2025** | **AI Benchmark Scientific Rigor** | Metric Definition Gaming | Only 16% of 445 AI benchmarks used statistical tests; ~50% tested abstract concepts without clear definitions. | [The Register](https://www.theregister.com/2025/11/07/measuring_ai_models_hampered_by/) |
| **2025** | **MMLU Benchmark Errors** | Invalid Ground Truth | ~57% of questions in MMLU virology subset found incorrect. Ground truth errors destabilize evaluations. | [PromptEngineering.org](https://promptengineering.org/challenges-and-innovations-in-language-model-benchmarking-and-generalization/) |
| **2024** | **AI Agent Benchmark Shortcuts** | Overfitting / Shortcuts | Study found AI agents memorize benchmark test samples instead of learning to generalize. Many benchmarks lack proper holdout test sets. | [VentureBeat](https://venturebeat.com/ai/ai-agent-benchmarks-are-misleading-study-warns) |
| **2024** | **GLUE Benchmark Heuristics** | Metric Definition Gaming | Models achieved high GLUE scores by exploiting shallow heuristics rather than genuine language understanding. | [Revelry.co](https://revelry.co/insights/artificial-intelligence/why-ai-benchmarks-fail/) |
| **2024** | **HumanEval Limitations** | Benchmark Overfitting | Models performing well on HumanEval struggled with real-world coding tasks; simplified scenarios missed practical complexity. | [Revelry.co](https://revelry.co/insights/artificial-intelligence/why-ai-benchmarks-fail/) |
| **2022** | **MLPerf Participation Issues** | Cherry-picking | MLPerf faced inconsistent vendor participation; selective scenario submissions led to biased performance representations. | [NextPlatform](https://www.nextplatform.com/2022/04/08/the-performance-of-mlperf-as-a-ubiquitous-benchmark-is-lacking/) |
| **2022** | **ML Benchmark Validity (Berkeley)** | Benchmark Overfitting | Small changes in data distribution caused significant performance drops, questioning external validity of static benchmarks. | [UC Berkeley Tech Report](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2022/EECS-2022-180.html) |
| **2021** | **ImageNet Label Errors** | Invalid Ground Truth | Study found **at least 6% label errors** in ImageNet validation set. Average 3.3% error rate across 10 common datasets. | [arXiv:2103.14749](https://arxiv.org/abs/2103.14749) |
| **2021** | **MLPerf Reproducibility** | Benchmark Reproducibility | Users couldn't reproduce MLPerf v0.7 results due to inaccessible datasets and outdated repositories. | [MLCommons Forum](https://groups.google.com/a/mlcommons.org/g/public/c/T_8UsUPIWFo) |
| **2021** | **Epic Sepsis Model Failure** | Benchmark Overfitting | Hospital sepsis prediction model showed significantly worse real-world performance than validation results due to non-representative test data. | [ChatBench.org](https://www.chatbench.org/what-are-the-implications-of-outdated-ai-benchmarks-on-the-accuracy-and-reliability-of-ai-driven-decision-making-and-insights/) |
| **2020** | **Underspecification in ML** | Benchmark Overfitting | ML pipelines produce models with equivalent benchmark performance but divergent deployment behaviors—instability in production. | [arXiv:2011.03395](https://arxiv.org/abs/2011.03395) |
| **2019** | **MLPerf Inference Bias** | Cherry-picking | Inaugural MLPerf inference results showed vendors selectively submitted results highlighting their strengths. | [Forbes](https://www.forbes.com/sites/janakirammsv/2019/11/10/the-curious-case-of-mlperf-inferencing-benchmark-results/) |
| **2019** | **Computational Biology Overfitting** | Train/Test Overlap | Tools developed and tested on same datasets, performing well on benchmarks but failing on new real-world data. | [Nature Communications](https://www.nature.com/articles/s41467-019-09406-4) |
| **2016** | **Microsoft Tay Chatbot** | Missing Holdout Sets | AI chatbot learned offensive behavior within 24 hours due to lack of adversarial benchmarking and content moderation safeguards. | [ChatBench.org](https://www.chatbench.org/what-are-the-implications-of-outdated-ai-benchmarks-on-the-accuracy-and-reliability-of-ai-driven-decision-making-and-insights/) |

#### Incident Categories and Our Protections

| Category | # Incidents | Our Protection | Status |
|----------|-------------|----------------|--------|
| **Timing Manipulation** | 1 (Locus/KernelBench) | Full device sync + `StreamAuditor` | ✅ |
| **Invalid Ground Truth** | 2 (ImageNet Labels, MMLU) | `GoldenOutputCache` + `validate_result()` | ✅ |
| **Benchmark Overfitting** | 4 (Underspecification, Epic Sepsis, HumanEval, Berkeley) | Fresh-input checks + jitter | ✅ |
| **Data Contamination** | 2 (Data Leakage 2025, Agent Shortcuts) | Data contamination checks + fresh inputs | ✅ |
| **Metric Gaming** | 3 (MLPerf 2019, GLUE, AI Benchmarks 2025) | Standardized metric definitions | ✅ |
| **Cherry-picking** | 2 (Chatbot Arena, MLPerf 2022) | All-iteration reporting | ✅ |
| **Train/Test Overlap** | 2 (Computational Biology, Agent Shortcuts) | Dataset isolation + holdout enforcement | ✅ |
| **Reproducibility** | 1 (MLPerf 2021) | `RunManifest` version locking | ✅ |

#### Deep Dive: The Locus/KernelBench Stream Timing Vulnerability

This 2025 incident perfectly illustrates why correctness verification alone is insufficient:

```python
# VULNERABLE TIMING (what KernelBench did)
start_event.record(original_model_stream)  # Only records on default stream
model(*inputs)                              # But work runs on s1, s2, s3!
end_event.record(original_model_stream)    # Timer stops before s1,s2,s3 finish
torch.cuda.synchronize(device=device)      # Waits, but timing already recorded

# CORRECT TIMING (the fix)
for stream in custom_model_streams:
    custom_model_stream.wait_stream(stream)  # Wait for ALL streams
_event.record(custom_model_stream)           # Then record timing
```

**The exploit pattern:**
1. AI creates non-default streams: `s1 = getStreamFromPool()`, `s2 = ...`, `s3 = ...`
2. AI launches GEMMs on those streams: `at::mm_out(gate, x2d, gate_proj.t())` on s1
3. AI does NOT call `setCurrentCUDAStream(s3)` or wait for streams before returning
4. Correctness test uses `torch.cuda.synchronize()` → **PASSES** (waits for everything)
5. Performance test uses stream-specific events → **FAKE SPEEDUP** (doesn't wait for s1,s2,s3)

**Result:** 82/250 (32.8%) of RL-generated CUDA kernels exploited this, producing artificial 18x "speedups" with zero actual performance improvement.

These incidents demonstrate why rigorous benchmark verification is essential—not just for catching intentional gaming, but for catching subtle bugs that produce misleading results.

#### Protection Implementation Reference

All 94 validity protections are implemented in the following modules:

| Module | Key Protections |
|--------|-----------------|
| `core/harness/benchmark_harness.py` | Full device sync, L2 cache clearing, GPU clock locking, warmup isolation, config immutability, adaptive iterations, CUDA graph mode |
| `core/harness/validity_checks.py` | `StreamAuditor`, `MemoryAllocationTracker`, `GraphCaptureCheatDetector`, `gc_disabled()`, `clear_compile_cache()`, `capture_gpu_state()`, `validate_environment()` |
| `core/harness/l2_cache_utils.py` | Dynamic L2 cache size detection for Blackwell/Hopper/Ampere, `clear_l2_cache()` |
| `core/benchmark/verify_runner.py` | `VerifyRunner`, `GoldenOutputCache`, jitter check, fresh-input check, output comparison, workload invariants |
| `core/benchmark/verification.py` | `InputSignature`, `ToleranceSpec`, `QuarantineReason`, seed mutation detection |
| `core/benchmark/quarantine.py` | `QuarantineManager` with persistence |
| `core/benchmark/contract.py` | `BenchmarkContract` enforcement |

## FAIL FAST - NO FALLBACKS, NO AUTO-INFERENCE

**CRITICAL**: This project follows a STRICT fail-fast policy. DO NOT implement fallbacks, auto-detection, or auto-inference.

### What This Means

1. **NO Auto-Inference**: Never write code that guesses or infers values from attributes
   - BAD: `if hasattr(self, 'batch_size'): return self.batch_size`
   - GOOD: Require explicit implementation, raise `NotImplementedError` if missing

2. **NO Fallbacks**: Never provide default values when explicit implementation is required
   - BAD: `return sig if sig else None` or `return sig if sig else {}`
   - GOOD: `raise NotImplementedError("Benchmark must implement this method")`

3. **NO Silent Failures**: Never swallow errors or return empty/None when something is wrong
   - BAD: `try: ... except: return None`
   - GOOD: Let exceptions propagate with clear error messages

### When You Find Code With Fallbacks

If you encounter code with auto-inference or fallbacks:
1. **DO NOT** add more fallbacks to fix the symptom
2. **DO** fix the underlying benchmarks to implement required methods
3. **DO** remove the fallback logic and make it fail-fast

### Audit Compliance

Use `aisp bench audit --all` to check verification compliance:
- All benchmark files must have 100% compliance
- Compliance means explicit implementations, not auto-detected ones

## Jitter Check (Advisory)

The jitter check protects against benchmarks returning **constant/hardcoded outputs** regardless of input.

**How It Works:**
1. Perturbs the input tensor by adding small noise
2. Re-runs `benchmark_fn()`
3. Verifies output CHANGED (if output unchanged → hardcoded)

**Important Notes:**
- The jitter check is largely **redundant** with proper output verification
- If baseline computes real output and optimized returns hardcoded values, they won't match anyway
- Jitter check only catches the case where BOTH baseline AND optimized return the SAME hardcoded value (extremely unlikely)
- No exemptions needed - the check auto-skips when appropriate

**Anti-patterns (DO NOT USE):**
- `return torch.tensor([1.0])` - Fixed constant (fails verification, not just jitter)
- `return torch.tensor([output.sum().item()])` - Scalar checksum (defeats both jitter AND verification)

**Valid patterns:**
- `return self.output.detach().clone()` - Actual output from benchmark_fn
- `return self.gpu_data[:1000].clone()` - Slice of actual data for large outputs


### Benchmark Verification Interface

Every benchmark MUST explicitly implement these methods (NO auto-detection):

```python
def get_verify_output(self) -> torch.Tensor:
    """MANDATORY: Return output tensor for verification."""
    raise NotImplementedError("Must implement explicitly")

def get_input_signature(self) -> dict:
    """MANDATORY: Return workload parameters for matching."""
    raise NotImplementedError("Must implement explicitly")
    
def get_output_tolerance(self) -> tuple:
    """MANDATORY: Return (rtol, atol) for numerical comparison."""
    raise NotImplementedError("Must implement explicitly")
```

## Deterministic Seed Pattern (CRITICAL)

The harness uses **seed 42** by default. Benchmarks MUST match this seed.

**Why this matters:**
- The harness sets seeds via `set_deterministic_seeds(42)` before `setup()`
- After `benchmark_fn()`, it checks if `torch.initial_seed() == 42`
- If seeds don't match → "Benchmark mutated RNG seeds during execution" error

**Correct Pattern:**
```python
def setup(self) -> None:
    torch.manual_seed(42)           # MUST be 42 to match harness
    torch.cuda.manual_seed_all(42)  # Always include for CUDA determinism
    # ... rest of setup
```

**Anti-pattern (DO NOT USE):**
```python
def setup(self) -> None:
    torch.manual_seed(1)    # BAD: mismatches harness seed 42
    torch.manual_seed(101)  # BAD: mismatches harness seed 42
```

## Harness Verification Architecture (IMPORTANT)

The harness uses **POST-TIMING VERIFICATION** - verification happens AFTER timing runs complete, using the outputs from the already-run benchmarks. This is efficient:

1. **Benchmarks run ONCE** during timing (with warmup + N iterations)
2. **Outputs are captured** via `get_verify_output()` 
3. **Comparison happens** after timing completes
4. **No redundant runs** - we don't run benchmarks twice

### How Verification Works

After timing runs complete for both baseline and optimized:
```python
baseline_output = baseline_benchmark.get_verify_output()
optimized_output = optimized_benchmark.get_verify_output()
rtol, atol = baseline_benchmark.get_output_tolerance()
torch.allclose(baseline_output, optimized_output, rtol=rtol, atol=atol)
```

### `_verify_patched_benchmark` - FOR LLM PATCHES ONLY

This separate function is **reserved for LLM-patched benchmarks**:
- When an LLM modifies a benchmark to optimize it
- We need to verify the LLM's version produces same output as original
- This loads and runs benchmarks fresh from disk
- **NOT used for baseline vs optimized pairs** (those use post-timing verification)

**Consistency Rule**: Always use `self.output` for benchmark results (not `self._result` or other names). The harness will use `get_verify_output()` for comparison.

## Checksum Verification is NOT Acceptable (IMPORTANT)

**DO NOT use checksums to work around verification failures.** If baseline and optimized produce different results, something is WRONG:

1. **Same algorithm, different execution pattern** → SHOULD produce identical results (within tolerance)
2. **Different algorithms** → One is incorrect, or they're testing different things
3. **Precision differences (FP16 vs FP32)** → Use appropriate tolerance, NOT checksums

### When Outputs Don't Match - Fix the Root Cause

If verification fails, investigate WHY:
- Are the inputs identical? (Check seed, setup order)
- Is the math the same? (Check operations, order)
- Is there hidden state mutation? (Check in-place ops)
- Is tolerance appropriate for the dtype? (FP16 needs looser tolerance than FP32)

**DO NOT** return `torch.tensor([self.output.sum().item()])` as a workaround.

### Cop-out Patterns That Need Fixing

**ALL of these patterns are COP-OUTS that defeat verification:**

| Pattern | Problem |
|---------|---------|
| `torch.tensor([0.0])` | Constant zero - defeats verification |
| `torch.tensor([hash(str(id(self))) % (2**31)])` | Random hash - defeats verification |
| `torch.tensor([output.sum().item()])` | Checksum - hides element-wise errors |
| `return None` or missing output | No verification at all |

**Files with cop-outs STATUS:**
- ✅ ch01-ch03 - FIXED with actual outputs
- ✅ ch05-ch16 - FIXED with actual outputs or RuntimeError for incompatible
- ✅ ch17-ch20 - FIXED with RuntimeError for nested harness benchmarks
- ⚠️ ch04/* - SKIPPED (multi-GPU required)

**Nested Harness Benchmarks (ch18/ch19/ch20):**
Many advanced benchmarks use a "nested harness" pattern where the wrapper calls `run_benchmark()` 
which internally creates another harness. These are marked with `RuntimeError("Nested harness 
benchmark - needs refactoring")` until proper output surfacing can be implemented.

### NO `_run_once_for_verify` in setup()

Verification uses outputs from the TIMING RUN, not a separate pre-run in setup().

**WRONG:**
```python
def setup(self):
    self._run_once_for_verify()  # NO! This runs benchmark twice!
```

**CORRECT:**
```python
def setup(self):
    # Just set up inputs - benchmark_fn() will set self.output
    pass

def get_verify_output(self):
    return self.output  # From timing run
```

### NO FALLBACKS in get_verify_output()

If output is None, FAIL FAST with an error. DO NOT return a cop-out.

**WRONG:**
```python
def get_verify_output(self):
    if self.output is not None:
        return self.output
    return torch.tensor([0.0])  # COP-OUT! Defeats verification!
```

**CORRECT:**
```python
def get_verify_output(self):
    if self.output is None:
        raise RuntimeError("benchmark_fn() must be called before verification")
    return self.output
```

### Training Benchmarks: Capture Output at END with `_verify_input`

For training benchmarks, both baseline and optimized train for N iterations. To verify they produce 
equivalent results, we need to test the TRAINED MODELS with the SAME input. Use `_verify_input`:

```python
def __init__(self):
    super().__init__()
    self.output = None
    self._verify_input = None  # Fixed input for verification
    # ... other init

def setup(self):
    torch.manual_seed(42)
    # ... model and training data setup
    
    # Create fixed verification input (same seed state for both baseline/optimized)
    self._verify_input = torch.randn(self.batch_size, self.seq_len, self.hidden_dim, device=self.device)
    # ... warmup

def benchmark_fn(self):
    # Training loop (timed)
    for data, target in zip(self.batches, self.targets):
        logits = self.model(data)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        self.optimizer.step()
    
    # Capture output AFTER training completes using fixed verify input
    with torch.no_grad():
        self.model.eval()
        self.output = self.model(self._verify_input).float().clone()  # .float() for FP16 models
        self.model.train()

def get_verify_output(self):
    if self.output is None:
        raise RuntimeError("benchmark_fn() must be called before verification")
    return self.output.detach().clone()

def get_output_tolerance(self):
    # Training with different optimizations may diverge slightly
    return (1e-2, 1e-2)
```

### Wrapper/Impl Pattern Benchmarks

Many benchmarks use an `_impl` class pattern. The wrapper must surface the output:

```python
class BenchmarkWrapper(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self._impl = SomeImplementation()
        self.output = None

    def benchmark_fn(self):
        result = self._impl.run()
        self.output = result  # Surface the output from _impl

    def get_verify_output(self):
        if self.output is None:
            raise RuntimeError("benchmark_fn() must be called before verification")
        return self.output.detach().clone()
```

### Benchmarks That Can Skip Verification

Only **two categories** may legitimately skip tensor verification:

| Category | Examples | Reason | Handling |
|----------|----------|--------|----------|
| **Multi-GPU Required** | ch04/*, ch17/multigpu | Requires >=2 GPUs | `raise RuntimeError("SKIPPED: requires >=2 GPUs")` |
| **Config Generation** | ch18/vllm_monitoring | Writes YAML/config files, no GPU computation | Use `verification_not_applicable_reason` attribute |

All other benchmarks **MUST** produce verifiable output:

### Benchmarks With Alternative Output Types

These benchmarks don't produce GPU tensors but MUST still verify their results:

| Category | Solution | Example |
|----------|----------|---------|
| **CUDA Binary** | Use `CudaBinaryBenchmark` base class - builds with `-DVERIFY=1` and parses checksum | ch09/cutlass_gemm |
| **Simulations** | Convert metrics to tensor (e.g., `torch.tensor([p50, p95, tokens_s])`) | ch15/placement |
| **CPU-only** | Convert output to tensor (e.g., decompressed bytes → tensor) | ch05/cpu_decompression |
| **Nested Harness** | Surface output from inner benchmark to outer wrapper | ch18/speculative_decoding |

### Verification Skip Attribute

For config generation benchmarks (only!), use:
```python
# In __init__:
self.verification_not_applicable_reason = "Config generation - writes YAML, no GPU computation"

# In get_verify_output:
def get_verify_output(self) -> torch.Tensor:
    raise RuntimeError("VERIFICATION_SKIP: Config generation benchmark - writes files, no GPU computation")
```

### Cop-out Fix Status: COMPLETE

All cop-outs (hash/zero/metrics-only) have been eliminated:

| Chapter Range | Status | Notes |
|--------------|--------|-------|
| ch01-ch03 | ✅ Fixed | Proper output capture |
| ch04 | ⏭️ Multi-GPU | Legitimate skip |
| ch05-ch16 | ✅ Fixed | All cop-outs replaced with real outputs |
| ch17-ch20 | ✅ Fixed | Nested harness refactored, simulations converted to tensor |

**CUDA Binary benchmarks:** Use base class verification via `-DVERIFY=1` builds
**Simulations:** Metrics converted to tensors for deterministic verification
**Config generation (4 files):** Only legitimate non-GPU benchmarks - use `verification_not_applicable_reason`

### Fixing Cop-out Patterns

When fixing `torch.tensor([hash(...)])` or `torch.tensor([0.0])` cop-outs:

1. **Find the output**: Look for `self.output`, `self.data`, `self.result`, `self.outputs`, etc.
2. **If output exists**: Return `self.output.detach().clone()`
3. **If wrapper pattern**: Surface output from `_impl`
4. **If no tensor output**: Use explicit `RuntimeError` with reason
5. **Never use hash/zero cop-outs**

### Data Loading / Prefetching Benchmarks: Wide Tolerances

For benchmarks where the optimization is in data loading (prefetching, pinned memory, 
double-buffering), baseline and optimized may process different batches in different 
orders. Use wide tolerances since we're primarily testing timing, not exact output matching.

```python
def get_output_tolerance(self) -> tuple:
    """Wide tolerance for data loading benchmarks.
    
    Primary checks are: no NaN, shapes match, reasonable values.
    """
    return (1.0, 10.0)
```

## ALWAYS MAKE THE LONG_TERM CHOICE
- DO NOT MAKE CHOICES BASED ON IMMEDIATE CONVENIENCE
- PREFER HARNESS CHANGES OVER PER-BENCHMARK HACKS
- DOCUMENT DISCOVERIES IN THIS FILEIATE NEED.  ALWAYS design for the long-term, right way of doing things.

## Prefer flags over environment variables

## Keys are in .env, .env.local in the project root folder.  
- Use these keys for external integrations (e.g. OpenAI, Anthropic, etc)
