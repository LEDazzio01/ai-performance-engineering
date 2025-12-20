# AI Systems Performance Engineering - Code and Benchmark Verifications

[![O'Reilly Book](../img/ai_sys_perf_engg_cover_cheetah_sm.png)](https://www.amazon.com/Systems-Performance-Engineering-Optimizing-Algorithms/dp/B0F47689K8/)

## Summary
Reference implementation of high-performance PyTorch, CUDA, and Triton workloads for NVIDIA Blackwell platforms.
The repository packages 20 focused chapters, advanced labs, and the shared benchmarking harness so you can profile baselines, apply optimizations, and capture artifacts that prove performance gains.

## Benchmark Verification & Validity Protections

This section describes the current benchmark verification system (correctness + workload equivalence) and the harness validity/anti-cheat protections.

This content used to live in `docs/implementation_status.md` and is now maintained in `README.md`.

### TODO

- No open items right now.

### Coverage

| Area | Coverage |
| --- | --- |
| Interface standardization & migration | 11/11 (100%) |
| Anti-cheat protections | 32/32 (100%) |
| Harness & CLI integration | 15/15 (100%) |
| Triton benchmarking best practices | 17/17 (100%) |
| **Total** | **75/75 (100%)** |

### Verification Model

#### Post-timing verification
Verification is **post-timing**: benchmarks run once for timing (warmup + measured iterations), then the harness compares outputs from those same runs.

#### Required benchmark interface
Benchmarks are required to provide explicit verification metadata:
- `get_input_signature()` (workload equivalence)
- `get_verify_output()` (output tensor(s) to compare)
- `get_output_tolerance()` (rtol/atol policy)
- `validate_result()` (sanity checks like NaN/Inf)

The harness is **fail-fast**: missing methods are treated as errors, not auto-inferred.

#### Verification payload mixin
`core/benchmark/verification_mixin.py` provides `VerificationPayloadMixin` and `_set_verification_payload()` so benchmarks can register verification inputs/outputs without hand-rolling boilerplate. The recommended starting point is `templates/benchmark_compliant.py`.

#### Output & overfitting checks
`core/benchmark/verify_runner.py` compares baseline vs optimized outputs and enforces:
- dtype-aware tolerances (via `core/benchmark/verification.py`)
- golden output caching (`GoldenOutputCache`)
- fresh-input and jitter checks (to detect cached/constant outputs)
- workload invariants + signature matching

#### Enforcement phases
Verification can run in phases (`DETECT`, `QUARANTINE`, `GATE`) using `core/benchmark/verification.py` + `core/benchmark/quarantine.py` and the CLI `--verify-phase` option.

### CUDA-L1 Reward-Hacking Protections

Reward-hacking cases identified in the CUDA-L1 paper are covered by the harness:

| Case | Protection |
| --- | --- |
| Improper timing measurement | Full device sync + `StreamAuditor` |
| Lazy evaluation | `force_tensor_evaluation()` |
| Hyperparameter manipulation | `InputSignature` + signature matching |
| Result caching | Fresh-input check |
| Mathematical short-circuit | Workload invariant check |
| Pre-allocated tensors | `MemoryAllocationTracker` |
| Direct shape matching | Signature validation |
| Pre-computed parameters | `check_setup_precomputation()` |

### Validity / Anti-Cheat Protections (Implementation Map)

#### Timing & measurement correctness

| Protection | Primary implementation |
| --- | --- |
| Full device sync | `core/harness/benchmark_harness.py` (`full_device_sync=True`) |
| Adaptive iterations | `core/harness/benchmark_harness.py` (`adaptive_iterations=True`) |
| Event timing cross-validation | `core/harness/benchmark_harness.py` (`cross_validate_timing=True`) |
| Warmup buffer isolation | `core/harness/benchmark_harness.py` (`isolate_warmup_cache=True`) |
| L2 cache clearing | `core/harness/benchmark_harness.py` (`clear_l2_cache`) |
| GPU clock locking | `core/harness/benchmark_harness.py` (`lock_gpu_clocks()`) |
| GC disabled during timing | `core/harness/validity_checks.py` (`gc_disabled()`) |
| Config immutability | `core/harness/benchmark_harness.py` (`enforce_config_immutability=True`) |
| Memory pool reset | `core/harness/benchmark_harness.py` (`reset_memory_pool=True`) |

#### Streams, graphs, and CUDA-specific protections

| Protection | Primary implementation |
| --- | --- |
| Stream auditing + sync completeness | `core/harness/validity_checks.py` (`StreamAuditor`, `audit_streams()`, `check_stream_sync_completeness()`) |
| Graph capture cheat detection | `core/harness/validity_checks.py` (`GraphCaptureCheatDetector`, `check_graph_capture_integrity()`) |
| Setup precomputation detection | `core/harness/validity_checks.py` (`check_setup_precomputation()`) |
| Force tensor evaluation | `core/harness/validity_checks.py` (`force_tensor_evaluation()`) |
| CUDA verify header | `core/common/headers/cuda_verify.cuh` (`VERIFY_CHECKSUM`) |
| CUDA binary symbol inspection | `core/benchmark/cuda_binary_benchmark.py` (`check_perf_binary_clean()`) |

#### Workload & output protections

| Protection | Primary implementation |
| --- | --- |
| Signature matching | `core/benchmark/verify_runner.py` (`_verify_signatures_match()`) |
| Workload invariant checks | `core/benchmark/verify_runner.py` (`_check_workload_invariants()`) |
| Fresh-input check | `core/benchmark/verify_runner.py` (`_run_fresh_input_check()`) |
| Jitter check | `core/benchmark/verify_runner.py` (`_run_jitter_check()`) |
| Golden output caching | `core/benchmark/verify_runner.py` (`GoldenOutputCache`) |
| Seed mutation detection | `core/benchmark/verification.py` (`detect_seed_mutation()`) |
| Input-output aliasing check | `core/benchmark/verify_runner.py` (`_check_input_output_aliasing()`) |
| Skip-flag detection | `core/benchmark/quarantine.py` (`detect_skip_flags()`) |

#### Environment & distributed protections

| Protection | Primary implementation |
| --- | --- |
| Environment validation | `core/harness/validity_checks.py` (`validate_environment()`) |
| GPU state capture (power/thermals) | `core/harness/validity_checks.py` (`capture_gpu_state()`) |
| Compile cache clearing | `core/harness/validity_checks.py` (`clear_compile_cache()`) |
| Distributed topology verification | `core/benchmark/verify_runner.py` + `core/harness/validity_checks.py` (`verify_distributed()`, `gather_rank_outputs()`, `verify_distributed_outputs()`) |

Note: `validate_environment()` treats virtualization (hypervisor present) as invalid; benchmarks are supported only on bare metal.

### CLI / CI Entry Points

| Task | Command |
| --- | --- |
| Audit verification compliance | `python -m cli.aisp bench audit --all` |
| Verify baseline/optimized correctness | `python -m cli.aisp bench verify -t ch12:graph_bandwidth` |
| Generate verification report | `python -m cli.aisp bench verify-report --gpu H100` |
| Generate quarantine report | `python -m cli.aisp bench quarantine-report --format markdown` |
| Print theoretical peaks | `python -m cli.aisp bench theoretical-peak --gpu H100` |

### Files Reference

| Category | File |
| --- | --- |
| Core data models | `core/benchmark/verification.py` |
| Verification mixin | `core/benchmark/verification_mixin.py` |
| Verify runner | `core/benchmark/verify_runner.py` |
| Quarantine manager | `core/benchmark/quarantine.py` |
| Benchmark contract | `core/benchmark/contract.py` |
| Benchmark harness | `core/harness/benchmark_harness.py` |
| Validity checks | `core/harness/validity_checks.py` |
| L2 cache utils | `core/harness/l2_cache_utils.py` |
| Verification reports | `core/analysis/reporting/verification_report.py` |
| CLI commands | `core/benchmark/bench_commands.py` |
| Audit script | `core/scripts/audit_verification_compliance.py` |
| Migration script | `core/scripts/migrate_verification_methods.py` |
| Pair validation | `core/scripts/validate_benchmark_pairs.py` |
| CI compliance check | `core/scripts/ci/check_verification_compliance.py` |

## Benchmark Validity Issues Reference

This table documents known issues that can cause benchmark results to be misleading, along with their protections. Use this as a checklist when creating or reviewing benchmarks.

**✅ All 94 validity issues are now protected by our harness** (Updated December 2025)

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
| **Workload** | Undeclared Shortcuts | Skips elements without declaring | Workload invariant check | ✅ | **AI Agent Benchmark Shortcuts 2024** ([arXiv:2407.01502](https://arxiv.org/abs/2407.01502)) |
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
| **Evaluation** | Metric Definition Gaming | Redefine what "speedup" means | Standardized metric definitions | ✅ | **MLPerf 2019** ([Forbes (archived)](https://web.archive.org/web/20191112035148/https://www.forbes.com/sites/janakirammsv/2019/11/10/the-curious-case-of-mlperf-inferencing-benchmark-results/)), **GLUE 2024** ([Revelry](https://revelry.co/insights/artificial-intelligence/why-ai-benchmarks-fail/)) |
| **Evaluation** | Test Data Leakage | Training on test/benchmark data | Data contamination checks | ✅ | **Data Contamination 2025** ([AI News](https://www.artificialintelligence-news.com/news/flawed-ai-benchmarks-enterprise-budgets-at-risk/)) |
| **Evaluation** | Benchmark Overfitting | Optimize specifically for benchmark | Fresh-input + jitter checks | ✅ | **Underspecification 2020** ([arXiv:2011.03395](https://arxiv.org/abs/2011.03395)), **Epic Sepsis 2021** ([ChatBench](https://www.chatbench.org/)) |
| **Evaluation** | Self-Modifying Tests | AI/code modifies its own tests | Config immutability | ✅ | |
| **Evaluation** | Benchmark Memorization | Agent memorizes test cases | Fresh-input checks, jitter | ✅ | **AI Agent Benchmark Shortcuts 2024** ([arXiv:2407.01502](https://arxiv.org/abs/2407.01502)) |
| **Evaluation** | Missing Holdout Sets | No proper train/test split | Held-out evaluation data | ✅ | **AI Agent Benchmark Shortcuts 2024** ([arXiv:2407.01502](https://arxiv.org/abs/2407.01502)) |

**Total: 11 categories, 94 validity issues — ✅ ALL PROTECTED by our harness (17 linked to real-world incidents with citations)**

### Notable Real-World Incidents

These validity issues aren't theoretical—they've caused real problems:

| Year | Incident | Issue Type | What Happened | Source |
|------|----------|------------|---------------|--------|
| **2025** | **Locus/KernelBench Stream Exploit** | Unsynced Streams | Claimed 20x speedup on Llama FFW kernel. AI launched work on non-default CUDA streams but timer only measured default stream. **32.8% of RL-generated kernels exploited this**, causing fake 18x speedups. | [X/Twitter @miru_why](https://x.com/miru_why/status/1991773868806361138) |
| **2025** | **AI Benchmark Scientific Rigor** | Metric Definition Gaming | Only 16% of 445 AI benchmarks used statistical tests; ~50% tested abstract concepts without clear definitions. | [The Register](https://www.theregister.com/2025/11/07/measuring_ai_models_hampered_by/) |
| **2025** | **MMLU Benchmark Errors** | Invalid Ground Truth | ~57% of questions in MMLU virology subset found incorrect. Ground truth errors destabilize evaluations. | [PromptEngineering.org](https://promptengineering.org/challenges-and-innovations-in-language-model-benchmarking-and-generalization/) |
| **2024** | **AI Agent Benchmark Shortcuts** | Overfitting / Shortcuts | Analysis found many agent benchmarks lack proper holdout sets, leading to shortcutting and overfitting instead of robust generalization. | [arXiv:2407.01502](https://arxiv.org/abs/2407.01502) |
| **2024** | **GLUE Benchmark Heuristics** | Metric Definition Gaming | Models achieved high GLUE scores by exploiting shallow heuristics rather than genuine language understanding. | [Revelry.co](https://revelry.co/insights/artificial-intelligence/why-ai-benchmarks-fail/) |
| **2024** | **HumanEval Limitations** | Benchmark Overfitting | Models performing well on HumanEval struggled with real-world coding tasks; simplified scenarios missed practical complexity. | [Revelry.co](https://revelry.co/insights/artificial-intelligence/why-ai-benchmarks-fail/) |
| **2022** | **MLPerf Participation Issues** | Cherry-picking | MLPerf faced inconsistent vendor participation; selective scenario submissions led to biased performance representations. | [NextPlatform](https://www.nextplatform.com/2022/04/08/the-performance-of-mlperf-as-a-ubiquitous-benchmark-is-lacking/) |
| **2022** | **ML Benchmark Validity (Berkeley)** | Benchmark Overfitting | Small changes in data distribution caused significant performance drops, questioning external validity of static benchmarks. | [UC Berkeley Tech Report](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2022/EECS-2022-180.html) |
| **2021** | **ImageNet Label Errors** | Invalid Ground Truth | Study found **at least 6% label errors** in ImageNet validation set. Average 3.3% error rate across 10 common datasets. | [arXiv:2103.14749](https://arxiv.org/abs/2103.14749) |
| **2021** | **MLPerf Reproducibility** | Benchmark Reproducibility | Users couldn't reproduce MLPerf v0.7 results due to inaccessible datasets and outdated repositories. | [MLCommons Forum](https://groups.google.com/a/mlcommons.org/g/public/c/T_8UsUPIWFo) |
| **2021** | **Epic Sepsis Model Failure** | Benchmark Overfitting | Hospital sepsis prediction model showed significantly worse real-world performance than validation results due to non-representative test data. | [ChatBench.org](https://www.chatbench.org/what-are-the-implications-of-outdated-ai-benchmarks-on-the-accuracy-and-reliability-of-ai-driven-decision-making-and-insights/) |
| **2020** | **Underspecification in ML** | Benchmark Overfitting | ML pipelines produce models with equivalent benchmark performance but divergent deployment behaviors—instability in production. | [arXiv:2011.03395](https://arxiv.org/abs/2011.03395) |
| **2019** | **MLPerf Inference Bias** | Cherry-picking | Inaugural MLPerf inference results showed vendors selectively submitted results highlighting their strengths. | [Forbes (archived)](https://web.archive.org/web/20191112035148/https://www.forbes.com/sites/janakirammsv/2019/11/10/the-curious-case-of-mlperf-inferencing-benchmark-results/) |
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

#### Additional Real-World Incidents (supplemental)

These additions were not included in the incident counts above.

| Year | Incident | Issue Type | What Happened | Source |
|------|----------|------------|---------------|--------|
| **2025** | **Sakana AI Scientist Evaluation** | Evaluation Integrity | Independent evaluation found frequent experiment failures and hallucinated numerical results, challenging reliability claims for AI-generated research outputs. | [arXiv:2502.14297](https://arxiv.org/abs/2502.14297) |
| **2023** | **NLP Evaluation Data Contamination** | Test Data Leakage | Position paper warns that LLMs trained on benchmark test splits can inflate reported scores and mask real generalization. | [arXiv:2310.18018](https://arxiv.org/abs/2310.18018) |
| **2024** | **Benchmark Data Contamination Survey** | Data Contamination | Survey catalogs contamination pathways across LLM benchmarks and highlights mitigation gaps. | [arXiv:2406.04244](https://arxiv.org/abs/2406.04244) |

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

**Verification Commands:**
```bash
aisp bench verify                              # Execute verification on any benchmark pair
aisp bench verify-report --gpu H100            # Generate detailed verification report
aisp bench theoretical-peak --gpu H100         # Show theoretical peak performance
aisp bench quarantine-report --format markdown # View quarantined benchmarks
```

## Learning Goals
- Understand how the chapters, labs, and shared tooling fit together.
- Stand up a reproducible environment for PyTorch 2.10-dev + CUDA 13 workloads on Blackwell GPUs.
- Run the benchmark harness directly or through the Typer CLI for automated artifact capture.
- Validate peak hardware characteristics before grading optimizations against stored expectations.

## Directory Layout
| Path | Description |
| --- | --- |
| `ch01` - `ch20` | One directory per chapter with baseline/optimized benchmarks, workload configs, and `compare.py` harness entrypoints. |
| `labs/` | Deep-dive labs for matmul, routing, FlexAttention, MoE, persistent decode, distributed training, and more. |
| `core/benchmark/`, `profiling/`, `core/`, `optimization/`, `analysis/` | Shared harness, logging, workload metadata, profiling, and optimization utilities used by every chapter. |
| `python -m cli.aisp bench` | Typer-based CLI for running and profiling targets with reproducible artifacts. |
| `docs/` + `core/scripts/` | Operational guides, profiling workflows, and setup/reset helpers (`setup.sh`, `cleanup.py`, `reset-gpu.sh`). |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
cd ai-performance-engineering
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements_latest.txt
python -m cli.aisp bench list-targets --chapter ch01
python -m cli.aisp bench run --targets ch01 --profile minimal
```
- `setup.sh` installs system prerequisites (drivers, CUDA, Nsight) and should be rerun after driver upgrades.
- Use `python core/harness/run_benchmarks.py --targets ch*` for automated regression suites.
- `python core/analysis/analyze_expectations.py --artifacts-dir artifacts` compares new runs to stored thresholds.

## Validation Checklist
- `pytest tests/integration` succeeds to confirm harness discovery and CLI plumbing.
- `python core/benchmark/benchmark_peak.py` reports TFLOP/s, bandwidth, and NVLink numbers close to the published ceilings.
- `python -m cli.aisp bench verify -t ch12:graph_bandwidth` validates baseline/optimized correctness.
- `python -m cli.aisp bench audit --all` checks verification compliance (signatures, outputs, workload metadata).

## Docs
- `docs/api-reference.md` (CLI/MCP/Dashboard/Python API overview)
- `docs/benchmark_harness_guide.md` (harness architecture and run modes)
- `docs/perf_intake_and_triage.md` (standard intake bundle for investigations)

## Notes
- `core/scripts/profile_all_workloads.sh` and `ncu_template.ini` capture Nsight traces with consistent metric sets.
- `benchmark_profiles/` and `artifacts/` hold run outputs; clean them via `python cleanup.py` when rotating hardware.
