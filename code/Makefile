# AI Performance Engineering - Development Makefile
#
# Common development tasks for the codebase.
# Run `make help` to see available targets.

.PHONY: help test lint validate coverage check all clean

# Default target
help:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "       AI Performance Engineering - Complete CLI Reference"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "SYSTEM & DIAGNOSTICS"
	@echo "  system-info            Full system information"
	@echo "  system-info-quick      Quick system summary"
	@echo "  system-test-all        Run all tests (disk, GPU, network)"
	@echo "  verify-deps            Check third-party dependencies"
	@echo "  verify-cutlass         Verify CUTLASS setup for Blackwell"
	@echo "  check-updates          Check for upstream CUTLASS/TE updates"
	@echo "  alert-updates          Alert on updates (Slack/email)"
	@echo ""
	@echo "GPU CONTROL"
	@echo "  gpu-status             Show GPU status"
	@echo "  gpu-pin                Pin GPU clocks to maximum"
	@echo "  gpu-unpin              Unpin GPU clocks"
	@echo ""
	@echo "BATCH OPTIMIZATION (MODEL=xxx GPUS=n MEMORY=n)"
	@echo "  models-fit             Models that fit in memory"
	@echo "  quantization-compare   Compare quantization options"
	@echo "  multi-gpu-scaling      Multi-GPU scaling analysis"
	@echo "  cloud-cost             Cloud cost estimation"
	@echo "  deploy-config          Generate deployment config"
	@echo "  finetune-estimate      Fine-tuning time estimate"
	@echo "  compound-optimizations Show compound optimization effects"
	@echo ""
	@echo "PROFILING"
	@echo "  profile-flame          Generate flame graph"
	@echo "  profile-memory         Memory timeline analysis"
	@echo "  profile-kernels        Kernel breakdown"
	@echo "  profile-roofline       Roofline analysis"
	@echo ""
	@echo "ANALYSIS"
	@echo "  analyze-bottlenecks    Find performance bottlenecks"
	@echo "  analyze-recommendations Get optimization recommendations"
	@echo "  analyze-pareto         Pareto frontier analysis"
	@echo "  analyze-scaling        Scaling analysis"
	@echo "  analyze-cost           Cost analysis"
	@echo ""
	@echo "ADVANCED SYSTEM ANALYSIS (NEW!)"
	@echo "  analyze-cpu-mem        CPU/memory hierarchy (caches, NUMA, TLB)"
	@echo "  analyze-sysparams      Kernel/system parameters"
	@echo "  analyze-container      Container/cgroups limits"
	@echo "  analyze-divergence     Warp divergence analysis"
	@echo "  analyze-bank-conflicts Shared memory bank conflicts"
	@echo "  analyze-memory-access  Memory access coalescing"
	@echo "  analyze-full           Complete system analysis"
	@echo "  tune-matmul            Auto-tune matmul kernel"
	@echo "  tune-attention         Auto-tune attention kernel"
	@echo "  optimization-list      List all optimization techniques"
	@echo "  optimization-playbooks Show pre-defined playbooks"
	@echo "  optimization-optimal   Find optimal stack (TARGET=10 DIFFICULTY=medium)"
	@echo ""
	@echo "PARALLELISM PLANNING (MODEL=xxx)"
	@echo "  parallelism-topology   Show GPU topology"
	@echo "  parallelism-recommend  Get parallelism recommendations"
	@echo "  parallelism-sharding   ZeRO/FSDP/HSDP sharding recommendations"
	@echo "  parallelism-launch     Generate launch commands"
	@echo "  parallelism-bottleneck Analyze bottlenecks"
	@echo "  parallelism-scaling    Scaling efficiency analysis"
	@echo "  parallelism-whatif     What-if configuration analysis"
	@echo "  parallelism-batch-size Find max batch size"
	@echo "  parallelism-auto-tune  Auto-tune configuration"
	@echo "  parallelism-inference  Inference optimization"
	@echo "  parallelism-validate   Validate configuration"
	@echo "  parallelism-dry-run    Quick dry-run test"
	@echo "  parallelism-pareto     Cost/throughput Pareto analysis"
	@echo "  parallelism-slurm      Generate SLURM script"
	@echo ""
	@echo "DISTRIBUTED TRAINING & ADVANCED (NEW!)"
	@echo "  nccl-tune              NCCL tuning recommendations"
	@echo "  nccl-diagnose          Diagnose NCCL issues"
	@echo "  rlhf-memory            RLHF/DPO memory calculator"
	@echo "  rlhf-compare           Compare RLHF algorithms"
	@echo "  moe-optimize           MoE parallelism optimizer"
	@echo "  long-context           Long context optimization"
	@echo "  vllm-config            vLLM configuration generator"
	@echo "  vllm-compare-engines   Compare inference engines"
	@echo "  comm-overlap           Communication overlap analyzer"
	@echo ""
	@echo "LLM-POWERED OPTIMIZATION (NEW!)"
	@echo "  llm-advisor            LLM-powered optimization advisor"
	@echo "  llm-advisor-inference  LLM advisor for inference"
	@echo ""
	@echo "🧠 INTELLIGENT OPTIMIZER (GOAL=xxx MODEL=xxx)"
	@echo "  smart-optimize         Auto-profile + LLM analysis"
	@echo "  smart-vllm             Optimize live vLLM server"
	@echo "  smart-tgi              Optimize live TGI server"
	@echo "  smart-discover         Discover compound optimizations"
	@echo "  smart-monitor          Real-time training monitor"
	@echo "  smart-cluster          Multi-cluster job orchestrator"
	@echo "  smart-status           Check LLM provider status"
	@echo ""
	@echo "UNIFIED PERF CLI (LLM-Powered)"
	@echo "  perf-status            System status overview"
	@echo "  perf-ask               LLM Q&A (QUESTION='why slow?')"
	@echo "  perf-explain           Book + LLM explanation (TOPIC=xxx)"
	@echo "  perf-troubleshoot      Error diagnosis (ERROR='CUDA OOM')"
	@echo "  perf-optimize          Auto-optimization with LLM"
	@echo "  perf-recommend         Parallelism recommendations"
	@echo "  perf-benchmark         Run benchmark (TARGET=ch08:matmul)"
	@echo "  perf-monitor           Real-time GPU monitoring"
	@echo "  troubleshoot-oom       Quick: CUDA OOM troubleshoot"
	@echo "  troubleshoot-nccl      Quick: NCCL timeout troubleshoot"
	@echo ""
	@echo "UI & TUI"
	@echo "  dashboard              Start web dashboard (localhost:8765)"
	@echo "  tui                    Start terminal UI"
	@echo ""
	@echo "ADVANCED OPTIMIZATION"
	@echo "  playbooks              List optimization playbooks"
	@echo "  playbook-apply         Apply playbook (PLAYBOOK=inference-speed)"
	@echo "  whatif                 What-if performance simulation"
	@echo "  auto-optimize          Auto-discover optimizations"
	@echo "  leaderboard            Performance leaderboard"
	@echo "  kernel-efficiency      Kernel efficiency analysis"
	@echo "  stacking               Compound optimization stacking"
	@echo "  tradeoffs              Multi-objective tradeoff analysis"
	@echo "  ask                    Ask AI assistant (Q='your question')"
	@echo "  generate-patch         Generate optimization code patches"
	@echo "  history                Historical performance trends"
	@echo "  roi                    ROI calculator (COST=10000)"
	@echo "  report                 Generate full performance report"
	@echo ""
	@echo "COMPARISON & MONITORING"
	@echo "  compare                A/B comparison (A=baseline B=optimized)"
	@echo "  monitor                Real-time GPU monitoring"
	@echo "  regression             Detect performance regressions"
	@echo ""
	@echo "MODEL ZOO"
	@echo "  model-list             List pre-optimized model configs"
	@echo "  model-config           Get optimal config (MODEL=llama-70b)"
	@echo "  model-recommend        Hardware recommendations"
	@echo ""
	@echo "LLM-POWERED ANALYSIS (DYNAMIC!)"
	@echo "  llm-ask                Ask LLM about performance (Q='your question')"
	@echo "  llm-analyze            LLM-powered profile analysis"
	@echo "  llm-status             Check LLM backend status"
	@echo ""
	@echo "DISTRIBUTED TRAINING (MULTI-NODE)"
	@echo "  cluster-topology       Discover cluster topology"
	@echo "  cluster-scaling        Multi-node scaling analysis"
	@echo "  cluster-recommend      Parallelism recommendation"
	@echo ""
	@echo "INFERENCE OPTIMIZATION (vLLM)"
	@echo "  inference-config       Generate vLLM config (MODEL=llama-70b)"
	@echo "  inference-benchmark    Benchmark endpoint (ENDPOINT=http://...)"
	@echo "  rlhf-config            RLHF training config (ALGO=ppo MODEL_PARAMS=70)"
	@echo ""
	@echo "UNIFIED CLI"
	@echo "  perf                   Unified CLI (run: perf help)"
	@echo "  status                 Quick system status"
	@echo ""
	@echo "TESTING & VALIDATION"
	@echo "  test                   Run all pytest tests"
	@echo "  test-fast              Run tests without slow markers"
	@echo "  validate               Validate benchmark imports"
	@echo "  lint                   Run linters"
	@echo "  audit                  Audit for silent fallbacks"
	@echo "  check                  Run all validation checks"
	@echo ""
	@echo "REPORTS"
	@echo "  coverage               Generate coverage report"
	@echo "  coverage-md            Coverage as Markdown"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "Examples:"
	@echo "  make compound-optimizations MODEL=llama-70b"
	@echo "  make deploy-config MODEL=mixtral-8x7b GPUS=8"
	@echo "  make multi-gpu-scaling MODEL=llama-70b"
	@echo "  make dashboard"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# =============================================================================
# Testing
# =============================================================================

test:
	python -m pytest tests/ -v --tb=short

test-fast:
	python -m pytest tests/ -v --tb=short -m "not slow"

test-cov:
	python -m pytest tests/ -v --tb=short --cov=common/python --cov-report=term-missing --cov-report=html

# =============================================================================
# Validation
# =============================================================================

validate:
	python scripts/validate_imports.py

validate-ch%:
	python scripts/validate_imports.py --chapter $*

# CUTLASS setup verification (critical for Blackwell/SM100a builds)
verify-cutlass:
	@echo "Verifying CUTLASS setup for Blackwell..."
	python tools/verification/verify_cutlass_setup.py
	@echo ""
	@echo "Verifying CUTLASS runtime (torch.compile backend)..."
	python tools/verification/verify_cutlass.py

# Quick pre-build check for third-party dependencies
verify-deps:
	@echo "Checking third-party dependency setup..."
	@echo ""
	@echo "1. CUTLASS Setup:"
	@python tools/verification/verify_cutlass_setup.py
	@echo ""
	@echo "2. TransformerEngine:"
	@python -c "import transformer_engine.pytorch as te; print('   ✓ TransformerEngine imported successfully')" 2>/dev/null || echo "   ✗ TransformerEngine not available"
	@echo ""
	@echo "3. PyTorch CUDA:"
	@python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'   ✓ PyTorch {torch.__version__} with CUDA {torch.version.cuda}')"

# Check for upstream updates (CUTLASS, TransformerEngine)
check-updates:
	@python scripts/check_upstream_versions.py --check-te-cutlass

check-updates-json:
	@python scripts/check_upstream_versions.py --json

# Alert on updates (supports Slack, email, file output)
# Usage: make alert-updates SLACK_WEBHOOK_URL=https://hooks.slack.com/...
alert-updates:
	@python scripts/alert_dependency_updates.py

alert-updates-slack:
	@python scripts/alert_dependency_updates.py --slack-webhook $(SLACK_WEBHOOK_URL)

alert-updates-quiet:
	@python scripts/alert_dependency_updates.py --quiet

# =============================================================================
# Performance Analysis (CLI parity with Dashboard UI)
# =============================================================================

# System diagnostics
system-info:
	@python -m tools.cli.diagnostics_cli info

system-info-quick:
	@python -m tools.cli.diagnostics_cli info --quick

system-test-all:
	@python -m tools.cli.diagnostics_cli test all

# GPU control
gpu-status:
	@python -m tools.cli.perf_cli gpu status

gpu-pin:
	@python -m tools.cli.perf_cli gpu pin --max

gpu-unpin:
	@python -m tools.cli.perf_cli gpu unpin

# Batch optimization analysis
models-fit:
	@python -m tools.cli.batch_cli models-fit --memory $(or $(MEMORY),80)

quantization-compare:
	@python -m tools.cli.batch_cli quantization --model $(or $(MODEL),llama-70b)

multi-gpu-scaling:
	@python -m tools.cli.batch_cli multi-gpu --model $(or $(MODEL),llama-70b)

cloud-cost:
	@python -m tools.cli.batch_cli cloud-cost --model $(or $(MODEL),llama-70b)

deploy-config:
	@python -m tools.cli.batch_cli deploy-config --model $(or $(MODEL),llama-70b) --gpus $(or $(GPUS),4)

finetune-estimate:
	@python -m tools.cli.batch_cli finetune --model $(or $(MODEL),llama-70b)

compound-optimizations:
	@python -m tools.cli.batch_cli compound --model $(or $(MODEL),llama-70b)

# Profiling
profile-flame:
	@python -m tools.profiling flame

profile-memory:
	@python -m tools.profiling memory

profile-kernels:
	@python -m tools.profiling kernels

profile-roofline:
	@python -m tools.profiling roofline

# Analysis
analyze-bottlenecks:
	@python -m tools.cli.perf_cli bottlenecks

analyze-recommendations:
	@python -m tools.cli.perf_cli recommend

analyze-pareto:
	@python -m tools.cli.perf_cli pareto

analyze-scaling:
	@python -m tools.cli.perf_cli scaling

analyze-cost:
	@python -m tools.cli.perf_cli cost

# Advanced Analysis (NEW!)
analyze-cpu-mem:
	@python -m tools.advanced_analysis cpu-mem

analyze-sysparams:
	@python -m tools.advanced_analysis sysparams

analyze-container:
	@python -m tools.advanced_analysis container

analyze-divergence:
	@python -m tools.advanced_analysis divergence

analyze-bank-conflicts:
	@python -m tools.advanced_analysis bank-conflicts

analyze-memory-access:
	@python -m tools.advanced_analysis memory-access

analyze-full:
	@python -m tools.advanced_analysis full

# Auto-tuning
tune-matmul:
	@python -m tools.advanced_analysis tune --kernel matmul

tune-attention:
	@python -m tools.advanced_analysis tune --kernel attention

# Optimization Stack
optimization-list:
	@python -m tools.advanced_analysis list

optimization-playbooks:
	@python -m tools.advanced_analysis playbook

optimization-optimal:
	@python -m tools.advanced_analysis optimal --target $(or $(TARGET),10) --difficulty $(or $(DIFFICULTY),medium)

# Parallelism planning
parallelism-topology:
	@python -m tools.parallelism_planner topology

parallelism-recommend:
	@python -m tools.parallelism_planner recommend $(or $(MODEL),llama-3.1-70b)

parallelism-launch:
	@python -m tools.parallelism_planner launch

parallelism-sharding:
	@python -m tools.parallelism_planner sharding $(or $(MODEL),llama-3.1-70b)

parallelism-bottleneck:
	@python -m tools.parallelism_planner bottleneck $(or $(MODEL),llama-3.1-70b)

parallelism-scaling:
	@python -m tools.parallelism_planner scaling $(or $(MODEL),llama-3.1-70b)

parallelism-whatif:
	@python -m tools.parallelism_planner whatif $(or $(MODEL),llama-3.1-70b)

parallelism-batch-size:
	@python -m tools.parallelism_planner batch-size $(or $(MODEL),llama-3.1-70b)

parallelism-auto-tune:
	@python -m tools.parallelism_planner auto-tune $(or $(MODEL),llama-3.1-70b)

parallelism-inference:
	@python -m tools.parallelism_planner inference $(or $(MODEL),llama-3.1-70b)

parallelism-validate:
	@python -m tools.parallelism_planner validate $(or $(MODEL),llama-3.1-70b)

parallelism-dry-run:
	@python -m tools.parallelism_planner dry-run $(or $(MODEL),llama-3.1-70b)

parallelism-pareto:
	@python -m tools.parallelism_planner pareto $(or $(MODEL),llama-3.1-70b)

parallelism-slurm:
	@python -m tools.parallelism_planner slurm

# DISTRIBUTED TRAINING & ADVANCED (NEW!)
nccl-tune:
	@python -m tools.parallelism_planner nccl --nodes=$(or $(NODES),1) --gpus=$(or $(GPUS),8)

nccl-diagnose:
	@python -m tools.parallelism_planner nccl --diagnose

rlhf-memory:
	@python -m tools.parallelism_planner rlhf $(or $(MODEL),llama-3.1-70b)

rlhf-compare:
	@python -m tools.parallelism_planner rlhf $(or $(MODEL),llama-3.1-70b) --compare

moe-optimize:
	@python -m tools.parallelism_planner moe $(or $(MODEL),mixtral-8x7b)

long-context:
	@python -m tools.parallelism_planner long-context $(or $(MODEL),llama-3.1-70b) --seq-length=$(or $(SEQ),128000)

vllm-config:
	@python -m tools.parallelism_planner vllm $(or $(MODEL),llama-3.1-70b) --target=$(or $(TARGET),throughput)

vllm-compare-engines:
	@python -m tools.parallelism_planner vllm $(or $(MODEL),llama-3.1-70b) --compare-engines

comm-overlap:
	@python -m tools.parallelism_planner comm-overlap $(or $(MODEL),llama-3.1-70b)

# LLM-POWERED OPTIMIZATION (NEW!)
llm-advisor:
	@python -m tools.parallelism_planner llm-advisor $(or $(MODEL),llama-3.1-70b) --goal=$(or $(GOAL),throughput)

# CLUSTER RESILIENCE (Fault Tolerance, Spot, Elastic)
fault-tolerance:
	@curl -s "http://localhost:8765/api/cluster/fault-tolerance?params=$(or $(PARAMS),70)&nodes=$(or $(NODES),1)&gpus=$(or $(GPUS),8)&hours=$(or $(HOURS),24)&spot=$(or $(SPOT),false)&cloud=$(or $(CLOUD),aws)" | python3 -m json.tool

spot-config:
	@curl -s "http://localhost:8765/api/cluster/spot-config?params=$(or $(PARAMS),70)&cloud=$(or $(CLOUD),aws)" | python3 -m json.tool

elastic-scaling:
	@curl -s "http://localhost:8765/api/cluster/elastic-scaling?params=$(or $(PARAMS),70)&nodes=$(or $(NODES),4)&traffic=$(or $(TRAFFIC),variable)" | python3 -m json.tool

cluster-diagnose:
	@curl -s "http://localhost:8765/api/cluster/diagnose?error=$(or $(ERROR),NCCL%20timeout)" | python3 -m json.tool

llm-advisor-inference:
	@python -m tools.parallelism_planner llm-advisor $(or $(MODEL),llama-3.1-70b) --goal=latency --inference

# Intelligent Optimizer - THE BRAIN of the optimization system
smart-optimize:
	@python -m tools.intelligent_optimizer optimize --goal=$(or $(GOAL),throughput)

smart-vllm:
	@python -m tools.intelligent_optimizer vllm --url=$(or $(URL),http://localhost:8000)

smart-tgi:
	@python -m tools.intelligent_optimizer tgi --url=$(or $(URL),http://localhost:8080)

smart-discover:
	@python -m tools.intelligent_optimizer discover --workload=$(or $(WORKLOAD),training)

smart-monitor:
	@python -m tools.intelligent_optimizer monitor --interval=$(or $(INTERVAL),30)

smart-cluster:
	@python -m tools.intelligent_optimizer cluster --model=$(or $(MODEL),llama-3.1-70b) --gpus=$(or $(GPUS),8) --nodes=$(or $(NODES),1) --scheduler=$(or $(SCHEDULER),slurm)

smart-status:
	@python -m tools.intelligent_optimizer status

# Dashboard
dashboard:
	@echo "Starting dashboard at http://localhost:8765"
	@python -m tools.dashboard.server

# TUI (Terminal UI)
tui:
	@python -m tools.cli.tui

# =============================================================================
# UNIFIED PERF CLI (LLM-Powered)
# =============================================================================

# Quick status overview
perf-status:
	@python -m tools.cli.perf status

# LLM-powered Q&A (QUESTION="Why is my kernel slow?")
perf-ask:
	@python -m tools.cli.perf ask $(QUESTION)

# Explain with book + LLM (TOPIC=flash-attention)
perf-explain:
	@python -m tools.cli.perf explain $(or $(TOPIC),flash-attention)

# Troubleshoot errors (ERROR="CUDA OOM")
perf-troubleshoot:
	@python -m tools.cli.perf troubleshoot $(ERROR)

# Auto-optimization with LLM
perf-optimize:
	@python -m tools.cli.perf optimize --auto $(if $(MODEL),--model $(MODEL))

# Parallelism recommendations
perf-recommend:
	@python -m tools.cli.perf recommend $(or $(MODEL),llama-3.1-70b)

# Run benchmark (TARGET=ch08:matmul)
perf-benchmark:
	@python -m tools.cli.perf benchmark $(TARGET)

# Real-time monitoring
perf-monitor:
	@python -m tools.cli.perf monitor

# Troubleshooting shortcuts
troubleshoot-oom:
	@python -m tools.cli.perf troubleshoot "CUDA out of memory"

troubleshoot-nccl:
	@python -m tools.cli.perf troubleshoot "NCCL timeout"

troubleshoot-slow:
	@python -m tools.cli.perf troubleshoot "training is slow"

troubleshoot-nan:
	@python -m tools.cli.perf troubleshoot "loss is NaN"

# =============================================================================
# Advanced Optimization Features (Complete UI Parity)
# =============================================================================

# Optimization playbooks
playbooks:
	@python -m tools.cli.advanced_cli playbooks

playbook-apply:
	@python -m tools.cli.advanced_cli playbooks --apply $(or $(PLAYBOOK),inference-speed)

# What-if simulation
whatif:
	@python -m tools.cli.advanced_cli whatif

whatif-fp8:
	@python -m tools.cli.advanced_cli whatif --fp8 --compile --flash-attn

# Auto-optimization
auto-optimize:
	@python -m tools.cli.advanced_cli auto-optimize --dry-run

auto-optimize-apply:
	@python -m tools.cli.advanced_cli auto-optimize --apply

# Leaderboard
leaderboard:
	@python -m tools.cli.advanced_cli leaderboard

# Kernel efficiency
kernel-efficiency:
	@python -m tools.cli.advanced_cli kernel-efficiency

# Stacking analysis (compound optimizations)
stacking:
	@python -m tools.cli.advanced_cli stacking

# Tradeoffs analysis
tradeoffs:
	@python -m tools.cli.advanced_cli tradeoffs

# LLM assistant
ask:
	@python -m tools.cli.advanced_cli ask "$(or $(Q),How can I improve performance?)"

# Generate optimization patches
generate-patch:
	@python -m tools.cli.advanced_cli generate-patch

# Historical trends
history:
	@python -m tools.cli.advanced_cli history

# ROI calculator
roi:
	@python -m tools.cli.advanced_cli roi --current-cost $(or $(COST),10000)

# Full report generation
report:
	@python -m tools.cli.advanced_cli report --output performance_report.html

# =============================================================================
# Comparison & Monitoring
# =============================================================================

# A/B comparison
compare:
	@python -m tools.cli.perf_engineer compare $(or $(A),baseline) $(or $(B),optimized)

# Real-time monitoring
monitor:
	@python -m tools.cli.perf_engineer monitor --duration $(or $(DURATION),60)

# Regression detection
regression:
	@python -m tools.cli.perf_engineer regression

# =============================================================================
# Model Zoo
# =============================================================================

# List pre-optimized configs
model-list:
	@python -m tools.cli.perf_engineer model list

# Get optimal config for a model
model-config:
	@python -m tools.cli.perf_engineer model config $(or $(MODEL),llama-70b)

# Hardware recommendations
model-recommend:
	@python -m tools.cli.perf_engineer model recommend

# =============================================================================
# Unified CLI Entry Point
# =============================================================================

# Unified perf CLI
perf:
	@python -m tools.cli.perf_engineer $(ARGS)

# Quick status
status:
	@python -m tools.cli.perf_engineer status

# =============================================================================
# LLM-Powered Analysis (DYNAMIC - NOT HARD-CODED!)
# =============================================================================

# Ask LLM about performance
llm-ask:
	@python -m tools.llm_engine ask $(or $(Q),"How can I improve my model's inference performance?")

# LLM-powered profile analysis
llm-analyze:
	@python -m tools.llm_engine analyze $(if $(PROFILE),--profile $(PROFILE))

# Check LLM backend status
llm-status:
	@python -m tools.llm_engine status

# =============================================================================
# Distributed Training (Multi-Node Clusters)
# =============================================================================

# Discover cluster topology
cluster-topology:
	@python -m tools.distributed_analysis topology

# Multi-node scaling analysis
cluster-scaling:
	@python -m tools.distributed_analysis scaling --model-params $(or $(MODEL_PARAMS),70)

# Parallelism recommendation
cluster-recommend:
	@python -m tools.distributed_analysis recommend --model-params $(or $(MODEL_PARAMS),70) $(if $(NODES),--nodes $(NODES))

# =============================================================================
# Inference Optimization (vLLM Integration)
# =============================================================================

# Generate vLLM config
inference-config:
	@python -m tools.inference_optimizer config --model $(or $(MODEL),llama-70b) --target $(or $(TARGET),balanced)

# Benchmark inference endpoint
inference-benchmark:
	@python -m tools.inference_optimizer benchmark --endpoint $(or $(ENDPOINT),http://localhost:8000)

# RLHF training configuration
rlhf-config:
	@python -m tools.inference_optimizer rlhf --model-params $(or $(MODEL_PARAMS),70) --algorithm $(or $(ALGO),ppo)

lint:
	@echo "Running flake8..."
	-flake8 common/python/ scripts/ --max-line-length=120 --ignore=E501,W503
	@echo ""
	@echo "Running mypy..."
	-mypy common/python/benchmark_metrics.py common/python/profiler_config.py --ignore-missing-imports

metrics:
	python scripts/update_custom_metrics.py --analyze

metrics-validate:
	python scripts/update_custom_metrics.py --validate

metrics-apply:
	python scripts/update_custom_metrics.py --apply

# =============================================================================
# Reports
# =============================================================================

coverage:
	python scripts/benchmark_coverage.py

coverage-md:
	python scripts/benchmark_coverage.py --markdown

coverage-json:
	python scripts/benchmark_coverage.py --json

# =============================================================================
# Maintenance
# =============================================================================

audit:
	python scripts/audit_silent_fallbacks.py

audit-fix:
	python scripts/audit_silent_fallbacks.py --fix

# Benchmark warmup validation - CRITICAL for accurate measurements
# Low warmup causes JIT/compile overhead to be included in timing results.
audit-warmup:
	@echo "Auditing benchmark warmup settings..."
	python scripts/audit_warmup_settings.py

audit-warmup-strict:
	@echo "Auditing benchmark warmup settings (including recommended levels)..."
	python scripts/audit_warmup_settings.py --check-recommended --verbose

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage 2>/dev/null || true

# =============================================================================
# Composite Targets
# =============================================================================

check: validate metrics lint audit-warmup
	@echo ""
	@echo "✅ All checks passed!"

all: test validate metrics coverage
	@echo ""
	@echo "✅ All tasks completed!"



