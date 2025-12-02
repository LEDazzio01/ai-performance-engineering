# AI Systems Performance - Unified API Reference

This document describes the unified 10-domain API that powers all interfaces (CLI, MCP, Dashboard, Python API).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AI Systems Performance                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User Interfaces                                                           │
│   ┌─────────┐  ┌─────────┐  ┌─────────────┐  ┌─────────────┐               │
│   │   CLI   │  │   MCP   │  │  Dashboard  │  │  Python API │               │
│   │  aisp   │  │  Tools  │  │   Web UI    │  │   Direct    │               │
│   └────┬────┘  └────┬────┘  └──────┬──────┘  └──────┬──────┘               │
│        │            │              │                │                       │
│        └────────────┴──────────────┴────────────────┘                       │
│                              │                                              │
│   ┌──────────────────────────▼──────────────────────────┐                   │
│   │           PerformanceEngine (core/engine.py)         │                   │
│   │                                                      │                   │
│   │   10 Domains:                                        │                   │
│   │   ┌─────┐ ┌──────┐ ┌───────┐ ┌───────┐ ┌────────┐   │                   │
│   │   │ gpu │ │system│ │profile│ │analyze│ │optimize│   │                   │
│   │   └─────┘ └──────┘ └───────┘ └───────┘ └────────┘   │                   │
│   │   ┌───────────┐ ┌─────────┐ ┌─────────┐ ┌──┐ ┌────┐ │                   │
│   │   │distributed│ │inference│ │benchmark│ │ai│ │exp │ │                   │
│   │   └───────────┘ └─────────┘ └─────────┘ └──┘ └────┘ │                   │
│   └──────────────────────────────────────────────────────┘                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Python API

```python
from core.engine import get_engine

engine = get_engine()

# GPU info
engine.gpu.info()
engine.gpu.bandwidth()

# Analysis
engine.analyze.bottlenecks()
engine.analyze.pareto()

# Optimization
engine.optimize.recommend(model_size=70, gpus=8)

# AI-powered
engine.ai.ask("Why is my attention kernel slow?")
```

### CLI

```bash
# System info
aisp info status
aisp info gpu

# Optimization recommendations
aisp optimize recommend --model-size 70 --gpus 8

# AI questions
aisp ai ask "Why is my kernel slow?"

# Profiling
aisp profile nsys python train.py
```

### MCP Tools (for AI assistants)

```
aisp_gpu_info          - Get GPU hardware info
aisp_analyze_bottlenecks - Identify performance issues
aisp_recommend         - Get optimization recommendations
aisp_ask              - Ask performance questions
```

---

## Domain Reference

### 1. GPU Domain

Hardware information, topology, power management, and bandwidth testing.

| Operation | Description | CLI | MCP Tool |
|-----------|-------------|-----|----------|
| `info()` | GPU name, memory, temperature, utilization | `aisp info gpu` | `aisp_gpu_info` |
| `topology()` | Multi-GPU topology, NVLink, P2P matrix | `aisp info topo` | `aisp_gpu_topology` |
| `power()` | Power draw, limits, thermal status | - | `aisp_gpu_power` |
| `bandwidth()` | Memory bandwidth test (HBM) | `aisp hw memory` | `aisp_gpu_bandwidth` |
| `nvlink()` | NVLink status and bandwidth | - | `aisp_gpu_topology` |
| `control()` | Clock settings, persistence mode | - | - |

**Python API:**
```python
engine.gpu.info()           # GPU info dict
engine.gpu.topology()       # Topology matrix
engine.gpu.bandwidth()      # Bandwidth test results
engine.gpu.power()          # Power/thermal info
```

---

### 2. System Domain

Software stack, dependencies, and environment information.

| Operation | Description | CLI | MCP Tool |
|-----------|-------------|-----|----------|
| `software()` | PyTorch, CUDA, Python versions | `aisp info status` | `aisp_system_software` |
| `dependencies()` | ML/AI dependency health | `aisp info deps` | `aisp_system_dependencies` |
| `capabilities()` | Hardware features (TMA, FP8, tensor cores) | `aisp info features` | `aisp_system_capabilities` |
| `context()` | Full system context for AI analysis | - | `aisp_system_context` |
| `parameters()` | Kernel parameters affecting performance | - | `aisp_system_parameters` |
| `container()` | Container/cgroup limits | - | `aisp_container_limits` |

**Python API:**
```python
engine.system.software()      # Software versions
engine.system.dependencies()  # Dependency health
engine.system.capabilities()  # Hardware capabilities
engine.system.context()       # Full context for AI
```

---

### 3. Profile Domain

Profiling with Nsight Systems, Nsight Compute, and torch.profiler.

| Operation | Description | CLI | MCP Tool |
|-----------|-------------|-----|----------|
| `flame_graph()` | Flame graph visualization data | `aisp profile flame` | `aisp_profile_flame` |
| `kernels()` | Kernel execution breakdown | - | `aisp_profile_kernels` |
| `memory_timeline()` | Memory allocation timeline | - | `aisp_profile_memory` |
| `hta()` | Holistic Trace Analysis | - | `aisp_profile_hta` |
| `torch()` | torch.profiler capture summary | - | `aisp_profile_torch` |
| `roofline()` | Roofline model data | - | `aisp_profile_roofline` |
| `compare(chapter)` | Compare baseline vs optimized | `aisp profile compare` | `aisp_profile_compare` |
| `list_profiles()` | List available profile pairs | - | - |

**Python API:**
```python
engine.profile.flame_graph()      # Flame graph data
engine.profile.kernels()          # Kernel breakdown
engine.profile.memory_timeline()  # Memory allocations
engine.profile.hta()              # HTA analysis
engine.profile.compare("ch11")    # Compare profiles
```

---

### 4. Analyze Domain

Performance analysis, bottleneck detection, and what-if scenarios.

| Operation | Description | CLI | MCP Tool |
|-----------|-------------|-----|----------|
| `bottlenecks(mode)` | Identify performance bottlenecks | `aisp analyze bottleneck` | `aisp_analyze_bottlenecks` |
| `pareto()` | Pareto frontier (throughput vs latency vs memory) | - | `aisp_analyze_pareto` |
| `scaling()` | Scaling analysis with GPU count | - | `aisp_analyze_scaling` |
| `whatif(...)` | What-if constraint analysis | - | `aisp_analyze_whatif` |
| `stacking()` | Optimization stacking compatibility | - | `aisp_analyze_stacking` |
| `power()` | Power efficiency analysis | - | `aisp_gpu_power` |
| `memory(...)` | Memory access pattern analysis | - | `aisp_memory_access` |
| `warp_divergence()` | Warp divergence analysis | - | `aisp_warp_divergence` |
| `bank_conflicts()` | Shared memory bank conflicts | - | `aisp_bank_conflicts` |
| `leaderboards()` | Performance leaderboards | - | - |

**Python API:**
```python
engine.analyze.bottlenecks()                    # Bottleneck analysis
engine.analyze.bottlenecks(mode="llm")          # AI-powered analysis
engine.analyze.pareto()                         # Pareto frontier
engine.analyze.scaling()                        # Scaling analysis
engine.analyze.whatif(max_vram_gb=24)           # What-if scenarios
engine.analyze.stacking()                       # Technique compatibility
```

---

### 5. Optimize Domain

Optimization recommendations and technique analysis.

| Operation | Description | CLI | MCP Tool |
|-----------|-------------|-----|----------|
| `recommend(model_size, gpus, goal)` | Get recommendations | `aisp optimize recommend` | `aisp_recommend` |
| `techniques()` | List all optimization techniques | - | `aisp_optimize_techniques` |
| `roi()` | Calculate optimization ROI | - | `aisp_optimize_roi` |
| `compound(techniques)` | Analyze compound effects | - | - |
| `playbooks()` | Optimization playbooks | - | - |
| `details(technique)` | Technique details | - | - |

**Python API:**
```python
engine.optimize.recommend(model_size=70, gpus=8)
engine.optimize.recommend(model_size=7, goal="memory")
engine.optimize.techniques()                     # All techniques
engine.optimize.roi()                            # ROI analysis
engine.optimize.compound(["flash-attention", "fsdp"])
```

---

### 6. Distributed Domain

Distributed training: parallelism planning, NCCL tuning, FSDP configuration.

| Operation | Description | CLI | MCP Tool |
|-----------|-------------|-----|----------|
| `plan(model_size, gpus, nodes)` | Plan parallelism strategy | `aisp distributed plan` | `aisp_distributed_plan` |
| `nccl(nodes, gpus)` | NCCL tuning recommendations | `aisp distributed nccl` | `aisp_distributed_nccl` |
| `fsdp(model)` | FSDP configuration | - | - |
| `tensor_parallel(model)` | Tensor parallelism config | - | - |
| `pipeline(model)` | Pipeline parallelism config | - | - |
| `slurm(...)` | Generate SLURM script | `aisp cluster slurm` | `aisp_cluster_slurm` |
| `cost_estimate(...)` | Cloud cost estimation | - | `aisp_cost_estimate` |

**Python API:**
```python
engine.distributed.plan(model_size=70, gpus=16, nodes=2)
engine.distributed.nccl(nodes=2, gpus=8)
engine.distributed.fsdp(model="7b")
engine.distributed.slurm(model="70b", nodes=4, gpus=8)
engine.distributed.cost_estimate(model_size=70, provider="aws")
```

---

### 7. Inference Domain

Inference optimization: vLLM configuration, quantization, deployment.

| Operation | Description | CLI | MCP Tool |
|-----------|-------------|-----|----------|
| `vllm_config(model, target)` | Generate vLLM configuration | `aisp inference vllm` | `aisp_inference_vllm` |
| `quantization(model_size)` | Quantization recommendations | `aisp inference quantize` | `aisp_inference_quantization` |
| `deploy(params)` | Deployment configuration | - | - |
| `estimate(params)` | Inference performance estimate | - | - |

**Python API:**
```python
engine.inference.vllm_config(model="llama-70b", target="throughput")
engine.inference.quantization(model_size=70)    # FP8, INT8, INT4 options
engine.inference.deploy({"model": "70b", "replicas": 4})
```

---

### 8. Benchmark Domain

Benchmark execution, history tracking, and result comparison.

| Operation | Description | CLI | MCP Tool |
|-----------|-------------|-----|----------|
| `run(targets, profile)` | Run benchmarks | `aisp bench run` | `aisp_run_benchmarks` |
| `targets()` | List benchmark targets | `aisp bench targets` | `aisp_benchmark_targets` |
| `history()` | Historical benchmark runs | - | - |
| `data()` | Load benchmark results | - | - |
| `available()` | Available benchmarks | - | `aisp_available_benchmarks` |
| `speed_test()` | Quick speed tests | `aisp hw speed` | `aisp_hw_speed` |

**Python API:**
```python
engine.benchmark.run(targets=["ch07", "ch11"], profile="standard")
engine.benchmark.targets()                      # Available targets
engine.benchmark.history()                      # Historical runs
engine.benchmark.data()                         # Current results
engine.benchmark.speed_test()                   # Quick GEMM/attention test
```

---

### 9. AI Domain

LLM-powered analysis, questions, and explanations.

| Operation | Description | CLI | MCP Tool |
|-----------|-------------|-----|----------|
| `ask(question)` | Ask a performance question | `aisp ai ask` | `aisp_ask` |
| `explain(concept)` | Explain a concept | `aisp ai explain` | `aisp_explain` |
| `analyze_kernel(code)` | AI kernel analysis | - | - |
| `suggest_tools(query)` | Suggest tools for a task | - | `aisp_suggest_tools` |
| `status()` | AI/LLM availability | - | `aisp_ai_status` |

**Python API:**
```python
engine.ai.ask("Why is my attention kernel slow?")
engine.ai.ask("How do I fix CUDA OOM?", include_citations=True)
engine.ai.explain("flash-attention")
engine.ai.suggest_tools("I keep OOMing on 24GB VRAM")
engine.ai.status()                              # Check LLM availability
```

---

### 10. Export Domain

Export reports in various formats.

| Operation | Description | CLI | MCP Tool |
|-----------|-------------|-----|----------|
| `csv(detailed)` | Export to CSV | `aisp report export --format csv` | `aisp_export_csv` |
| `pdf()` | Generate PDF report | `aisp report export --format pdf` | `aisp_export_pdf` |
| `html()` | Generate HTML report | `aisp report export --format html` | `aisp_export_html` |

**Python API:**
```python
csv_content = engine.export.csv()               # Basic CSV
csv_detailed = engine.export.csv(detailed=True) # Detailed CSV
pdf_bytes = engine.export.pdf()                 # PDF report
html_content = engine.export.html()             # HTML report
```

---

## Convenience Methods

The engine also provides top-level convenience methods:

```python
engine.status()           # Quick system status (GPU + software + AI)
engine.triage()           # Status + context + next steps
engine.ask(question)      # Shortcut to engine.ai.ask()
engine.recommend(...)     # Shortcut to engine.optimize.recommend()
engine.list_domains()     # List all domains and operations
```

---

## Interface Mapping Summary

| Engine Domain | CLI Command Group | MCP Tool Prefix | Dashboard API |
|---------------|-------------------|-----------------|---------------|
| `gpu` | `aisp info` | `aisp_gpu_*` | `/api/gpu/*` |
| `system` | `aisp info` | `aisp_system_*` | `/api/software`, `/api/deps` |
| `profile` | `aisp profile` | `aisp_profile_*` | `/api/profiler/*` |
| `analyze` | `aisp analyze` | `aisp_analyze_*` | `/api/analysis/*` |
| `optimize` | `aisp optimize` | `aisp_optimize_*`, `aisp_recommend` | `/api/optimize/*` |
| `distributed` | `aisp distributed` | `aisp_distributed_*` | `/api/parallelism/*` |
| `inference` | `aisp inference` | `aisp_inference_*` | `/api/inference/*` |
| `benchmark` | `aisp bench` | `aisp_benchmark_*`, `aisp_run_*` | `/api/benchmarks/*` |
| `ai` | `aisp ai` | `aisp_ask`, `aisp_explain` | `/api/ai/*` |
| `export` | `aisp report` | `aisp_export_*` | `/api/export/*` |

---

## Error Handling

All operations return dictionaries with consistent structure:

```python
# Success
{
    "success": True,
    "data": {...},
    "timestamp": "2024-01-01T12:00:00Z"
}

# Error
{
    "success": False,
    "error": "Error message",
    "error_type": "AttributeError"
}
```

---

## Best Practices

### 1. Start with Triage

```python
# Get full context before diving in
context = engine.triage()
```

### 2. Use Domain-Specific Methods

```python
# Good: Clear domain separation
engine.analyze.bottlenecks()
engine.optimize.recommend(model_size=70)

# Avoid: Using internal methods directly
engine._handler.detect_bottlenecks()  # Don't do this
```

### 3. Combine AI with Data

```python
# Get hard data first
bottlenecks = engine.analyze.bottlenecks(mode="profile")

# Then ask AI for interpretation
explanation = engine.ai.ask(f"Explain these bottlenecks: {bottlenecks}")
```

### 4. Use Consistent Parameters

```python
# Parameters are consistent across interfaces
# CLI:       aisp optimize recommend --model-size 70 --gpus 8
# Python:    engine.optimize.recommend(model_size=70, gpus=8)
# MCP:       aisp_recommend with {"model_size": 70, "gpus": 8}
```





