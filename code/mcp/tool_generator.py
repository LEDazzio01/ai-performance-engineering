#!/usr/bin/env python3
"""
Auto-generate MCP tool schemas from PerformanceEngine.

This module introspects the PerformanceEngine to generate consistent
MCP tool definitions, reducing manual maintenance and ensuring
all interfaces stay in sync.
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, get_type_hints

# Tool metadata for enhanced descriptions
TOOL_METADATA = {
    # GPU Domain
    "gpu.info": {
        "tags": "gpu, info, snapshot, health-check, inventory, nvidia-smi",
        "speed": "⚡ FAST (~1s)",
        "use_when": "Starting any performance investigation, verifying hardware before profiling",
        "not_for": "Feature detection (system.capabilities), topology (gpu.topology), power throttling (gpu.power)",
        "examples": ["Show GPU names and memory", "What GPUs do I have?", "Check VRAM before loading model"],
    },
    "gpu.bandwidth": {
        "tags": "gpu, memory, bandwidth, throughput, benchmark",
        "speed": "🕐 MEDIUM (~10s)",
        "use_when": "Diagnosing memory-bound kernels, comparing GPU generations",
        "not_for": "Quick health checks (gpu.info), compute benchmarks (hw.speed)",
        "examples": ["Test GPU memory speed", "Is my VRAM bandwidth bottlenecked?"],
    },
    "gpu.topology": {
        "tags": "gpu, multi-gpu, nvlink, pcie, numa, p2p, topology",
        "speed": "⚡ FAST (~2s)",
        "use_when": "Planning multi-GPU training, diagnosing communication bottlenecks",
        "not_for": "Single GPU info (gpu.info), NCCL tuning (distributed.nccl)",
        "examples": ["Show NVLink connections", "Which GPUs can do P2P?", "NUMA affinity"],
    },
    "gpu.power": {
        "tags": "gpu, power, thermal, throttling, temperature, watts",
        "speed": "⚡ FAST (~1s)",
        "use_when": "Diagnosing thermal throttling, checking power limits",
        "not_for": "Memory info (gpu.info), energy optimization (analyze.energy)",
        "examples": ["Are my GPUs throttling?", "Check GPU temperatures", "Power draw vs limit"],
    },
    # System Domain
    "system.software": {
        "tags": "system, cuda, pytorch, python, versions, stack",
        "speed": "⚡ FAST (~1s)",
        "use_when": "Checking software versions, diagnosing compatibility issues",
        "not_for": "Hardware info (gpu.info), feature detection (system.capabilities)",
        "examples": ["What CUDA version?", "Check PyTorch version", "Software stack"],
    },
    "system.dependencies": {
        "tags": "system, dependencies, packages, pip, conda",
        "speed": "⚡ FAST (~2s)",
        "use_when": "Checking installed packages, version conflicts",
        "not_for": "Core software versions (system.software)",
        "examples": ["What packages are installed?", "Check transformers version"],
    },
    "system.capabilities": {
        "tags": "system, features, capabilities, bf16, fp16, flash-attention",
        "speed": "⚡ FAST (~1s)",
        "use_when": "Checking hardware/software feature support",
        "not_for": "Version info (system.software), GPU specs (gpu.info)",
        "examples": ["Does my GPU support BF16?", "Is Flash Attention available?"],
    },
    # Profile Domain
    "profile.nsys": {
        "tags": "profile, nsys, nsight, trace, timeline, cuda",
        "speed": "🕐 SLOW (varies)",
        "use_when": "Deep-diving into CUDA kernel timing, CPU-GPU overlap",
        "not_for": "Quick checks (status), memory profiling (profile.ncu)",
        "examples": ["Profile this training script", "Get kernel timeline"],
    },
    "profile.ncu": {
        "tags": "profile, ncu, nsight-compute, kernel, occupancy, memory",
        "speed": "🕐 SLOW (varies)",
        "use_when": "Analyzing specific kernel performance, occupancy, memory access",
        "not_for": "Full trace (profile.nsys), quick checks (status)",
        "examples": ["Profile kernel occupancy", "Analyze memory coalescing"],
    },
    "profile.torch": {
        "tags": "profile, pytorch, torch, profiler, tensorboard",
        "speed": "🕐 MEDIUM (~30s)",
        "use_when": "PyTorch-specific profiling, operator breakdown",
        "not_for": "CUDA-level profiling (profile.nsys/ncu)",
        "examples": ["Profile PyTorch model", "Get operator breakdown"],
    },
    "profile.compare": {
        "tags": "profile, compare, diff, regression, before-after",
        "speed": "⚡ FAST (~2s)",
        "use_when": "Comparing two profile runs, finding regressions",
        "not_for": "Single profile analysis",
        "examples": ["Compare before/after optimization", "Find performance regression"],
    },
    # Analyze Domain
    "analyze.bottlenecks": {
        "tags": "analyze, bottleneck, diagnosis, slow, performance",
        "speed": "⚡ FAST (~3s)",
        "use_when": "First step in performance investigation",
        "not_for": "Specific profiling (profile.*), optimization (optimize.*)",
        "examples": ["Why is training slow?", "Find bottlenecks", "What's limiting performance?"],
    },
    "analyze.pareto": {
        "tags": "analyze, pareto, tradeoff, optimization, cost-benefit",
        "speed": "⚡ FAST (~2s)",
        "use_when": "Prioritizing optimizations by impact vs effort",
        "not_for": "Finding bottlenecks (analyze.bottlenecks)",
        "examples": ["Best optimizations for my time?", "What gives biggest speedup?"],
    },
    "analyze.scaling": {
        "tags": "analyze, scaling, multi-gpu, efficiency, parallelism",
        "speed": "⚡ FAST (~2s)",
        "use_when": "Planning multi-GPU scaling, predicting speedup",
        "not_for": "Single GPU optimization",
        "examples": ["How will 8 GPUs scale?", "Predict multi-GPU speedup"],
    },
    "analyze.whatif": {
        "tags": "analyze, whatif, simulation, prediction, upgrade",
        "speed": "⚡ FAST (~1s)",
        "use_when": "Predicting impact of hardware/config changes",
        "not_for": "Current performance (analyze.bottlenecks)",
        "examples": ["What if I upgrade to H100?", "Impact of doubling batch size?"],
    },
    # Optimize Domain
    "optimize.recommend": {
        "tags": "optimize, recommend, suggestions, improve, speedup",
        "speed": "⚡ FAST (~2s)",
        "use_when": "Getting optimization recommendations",
        "not_for": "Analysis (analyze.*), profiling (profile.*)",
        "examples": ["How to speed up training?", "Optimization suggestions"],
    },
    "optimize.roi": {
        "tags": "optimize, roi, cost, benefit, priority",
        "speed": "⚡ FAST (~1s)",
        "use_when": "Calculating optimization ROI",
        "not_for": "Getting recommendations (optimize.recommend)",
        "examples": ["Which optimization is most valuable?", "Cost-benefit of Flash Attention"],
    },
    "optimize.techniques": {
        "tags": "optimize, techniques, methods, catalog, options",
        "speed": "⚡ FAST (~1s)",
        "use_when": "Browsing available optimization techniques",
        "not_for": "Specific recommendations (optimize.recommend)",
        "examples": ["What optimizations are available?", "List optimization techniques"],
    },
    # Distributed Domain
    "distributed.plan": {
        "tags": "distributed, multi-gpu, parallelism, strategy, ddp, fsdp",
        "speed": "⚡ FAST (~2s)",
        "use_when": "Planning distributed training strategy",
        "not_for": "Single GPU training",
        "examples": ["How to distribute 70B model?", "DDP vs FSDP?"],
    },
    "distributed.nccl": {
        "tags": "distributed, nccl, allreduce, communication, tuning",
        "speed": "⚡ FAST (~1s)",
        "use_when": "Tuning NCCL for multi-GPU communication",
        "not_for": "Strategy planning (distributed.plan)",
        "examples": ["NCCL environment variables", "Optimize allreduce"],
    },
    # Inference Domain
    "inference.vllm": {
        "tags": "inference, vllm, serving, deployment, throughput",
        "speed": "⚡ FAST (~2s)",
        "use_when": "Configuring vLLM for inference serving",
        "not_for": "Training optimization",
        "examples": ["vLLM config for Llama 70B", "Optimize inference throughput"],
    },
    "inference.quantization": {
        "tags": "inference, quantization, int8, int4, awq, gptq",
        "speed": "⚡ FAST (~1s)",
        "use_when": "Choosing quantization strategy",
        "not_for": "Training (use mixed precision instead)",
        "examples": ["AWQ vs GPTQ?", "Can I run 70B on 24GB?"],
    },
    # Benchmark Domain
    "benchmark.run": {
        "tags": "benchmark, test, measure, performance, baseline",
        "speed": "🕐 SLOW (varies)",
        "use_when": "Running performance benchmarks",
        "not_for": "Quick checks (status)",
        "examples": ["Run memory bandwidth test", "Benchmark my GPU"],
    },
    "benchmark.targets": {
        "tags": "benchmark, targets, goals, expected, comparison",
        "speed": "⚡ FAST (~1s)",
        "use_when": "Getting expected performance targets",
        "not_for": "Running benchmarks (benchmark.run)",
        "examples": ["What should H100 achieve?", "Expected bandwidth for A100"],
    },
    # AI Domain
    "ai.ask": {
        "tags": "ai, llm, question, help, explain",
        "speed": "🕐 MEDIUM (~5s)",
        "use_when": "Asking performance questions in natural language",
        "not_for": "Specific analysis (use domain tools)",
        "examples": ["Why is my GPU at 50% utilization?", "Explain tensor parallelism"],
    },
    "ai.explain": {
        "tags": "ai, llm, explain, interpret, results",
        "speed": "🕐 MEDIUM (~5s)",
        "use_when": "Getting AI explanation of results",
        "not_for": "Getting raw data (use domain tools)",
        "examples": ["Explain these bottlenecks", "What do these metrics mean?"],
    },
    # Export Domain
    "export.csv": {
        "tags": "export, csv, data, spreadsheet",
        "speed": "⚡ FAST (~1s)",
        "use_when": "Exporting data to CSV",
        "not_for": "Reports (export.html/pdf)",
        "examples": ["Export benchmark results to CSV"],
    },
    "export.html": {
        "tags": "export, html, report, web",
        "speed": "⚡ FAST (~2s)",
        "use_when": "Creating HTML reports",
        "not_for": "Raw data (export.csv)",
        "examples": ["Generate HTML performance report"],
    },
    "export.pdf": {
        "tags": "export, pdf, report, document",
        "speed": "🕐 MEDIUM (~5s)",
        "use_when": "Creating PDF reports",
        "not_for": "Quick sharing (export.html)",
        "examples": ["Generate PDF performance report"],
    },
}


def get_tool_description(domain: str, method: str, docstring: str) -> str:
    """Generate enhanced tool description with metadata."""
    key = f"{domain}.{method}"
    meta = TOOL_METADATA.get(key, {})
    
    parts = []
    
    # Tags
    if meta.get("tags"):
        parts.append(f"Tags: {meta['tags']}.")
    
    # Main description from docstring
    if docstring:
        parts.append(docstring.strip().split("\n")[0])
    
    # Speed indicator
    if meta.get("speed"):
        parts.append(meta["speed"])
    
    # Use when
    if meta.get("use_when"):
        parts.append(f"USE WHEN: {meta['use_when']}.")
    
    # Examples
    if meta.get("examples"):
        examples = " | ".join(f'"{e}"' for e in meta["examples"][:2])
        parts.append(f"Examples: {examples}.")
    
    # Not for
    if meta.get("not_for"):
        parts.append(f"NOT FOR: {meta['not_for']}.")
    
    return " ".join(parts)


def generate_tool_schema(method: Callable) -> Dict[str, Any]:
    """Generate JSON schema from method signature."""
    sig = inspect.signature(method)
    hints = {}
    try:
        hints = get_type_hints(method)
    except Exception:
        pass
    
    properties = {}
    required = []
    
    for name, param in sig.parameters.items():
        if name in ("self", "kwargs"):
            continue
        
        prop: Dict[str, Any] = {}
        hint = hints.get(name)
        
        # Type mapping
        if hint == str or (hasattr(hint, "__origin__") and hint.__origin__ == str):
            prop["type"] = "string"
        elif hint == int:
            prop["type"] = "integer"
        elif hint == float:
            prop["type"] = "number"
        elif hint == bool:
            prop["type"] = "boolean"
        elif hint == list or (hasattr(hint, "__origin__") and hint.__origin__ == list):
            prop["type"] = "array"
        elif hint == dict or (hasattr(hint, "__origin__") and hint.__origin__ == dict):
            prop["type"] = "object"
        else:
            prop["type"] = "string"  # Default to string
        
        # Default value
        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(name)
        
        properties[name] = prop
    
    schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    
    return schema


def generate_tools_from_engine() -> List[Tuple[str, str, Dict[str, Any], str, str]]:
    """
    Generate tool definitions from PerformanceEngine.
    
    Returns list of (tool_name, description, schema, domain, method) tuples.
    """
    from core.engine import get_engine, DOMAINS
    
    engine = get_engine()
    tools = []
    
    for domain_name in DOMAINS:
        domain = getattr(engine, domain_name, None)
        if domain is None:
            continue
        
        # Get all public methods
        for method_name in dir(domain):
            if method_name.startswith("_"):
                continue
            
            method = getattr(domain, method_name)
            if not callable(method):
                continue
            
            tool_name = f"aisp_{domain_name}_{method_name}"
            docstring = method.__doc__ or ""
            description = get_tool_description(domain_name, method_name, docstring)
            schema = generate_tool_schema(method)
            
            tools.append((tool_name, description, schema, domain_name, method_name))
    
    return tools


# Consolidated tool definitions - manually curated for best UX
CONSOLIDATED_TOOLS = {
    # === Core Status Tools (4) ===
    "aisp_status": {
        "description": "Tags: status, health, overview, quick-check. "
            "Get quick system status: GPU availability, software versions, AI ready state. "
            "⚡ FAST (<1s). USE FIRST for any investigation. "
            "Returns: {gpus_available, cuda_version, pytorch_version, ai_enabled}. "
            "WORKFLOW: aisp_status → aisp_triage → domain-specific tools.",
        "schema": {"type": "object", "properties": {}},
        "handler": "engine.status",
    },
    "aisp_triage": {
        "description": "Tags: triage, diagnosis, full-check, snapshot. "
            "Comprehensive system triage: hardware, software, bottlenecks, recommendations. "
            "⚡ FAST (~3s). USE WHEN: Starting performance investigation. "
            "Returns: {hardware, software, bottlenecks, top_recommendations}. "
            "WORKFLOW: aisp_status → aisp_triage → aisp_recommend → specific fixes.",
        "schema": {"type": "object", "properties": {}},
        "handler": "engine.triage",
    },
    "aisp_job_status": {
        "description": "Tags: async, job, status, poll. "
            "Check status of async profiling job. "
            "⚡ FAST (<1s). USE WHEN: Waiting for profile to complete. "
            "Returns: {status: 'running'|'completed'|'error', progress, result}.",
        "schema": {
            "type": "object",
            "properties": {"job_id": {"type": "string", "description": "Job ID from async operation"}},
            "required": ["job_id"],
        },
        "handler": "job_status",
    },
    "aisp_suggest_tools": {
        "description": "Tags: help, tools, discover, guide. "
            "Get tool suggestions for your task. "
            "⚡ FAST (<1s). USE WHEN: Not sure which tool to use. "
            "Examples: \"I want to profile\" → suggests profile.* tools.",
        "schema": {
            "type": "object",
            "properties": {"task": {"type": "string", "description": "What you want to do"}},
        },
        "handler": "suggest_tools",
    },
    
    # === GPU Domain (4) ===
    "aisp_gpu_info": {
        "description": "Tags: gpu, info, nvidia-smi, memory, temperature. "
            "Get GPU hardware info: name, VRAM, temp, power, utilization. "
            "⚡ FAST (~1s). USE FIRST when investigating GPU issues. "
            "Examples: \"What GPUs do I have?\" | \"Check VRAM usage\". "
            "NOT FOR: topology (aisp_gpu_topology), power limits (aisp_gpu_power).",
        "schema": {"type": "object", "properties": {}},
        "handler": "engine.gpu.info",
    },
    "aisp_gpu_bandwidth": {
        "description": "Tags: gpu, bandwidth, memory, throughput, benchmark. "
            "Run GPU memory bandwidth test. "
            "🕐 MEDIUM (~10s). USE WHEN: Diagnosing memory-bound kernels. "
            "Examples: \"Test GPU memory speed\" | \"Is bandwidth limiting me?\". "
            "NOT FOR: quick checks (aisp_gpu_info).",
        "schema": {"type": "object", "properties": {}},
        "handler": "engine.gpu.bandwidth",
    },
    "aisp_gpu_topology": {
        "description": "Tags: gpu, topology, nvlink, pcie, p2p, multi-gpu. "
            "Get multi-GPU topology: NVLink, PCIe, NUMA, P2P matrix. "
            "⚡ FAST (~2s). USE WHEN: Planning multi-GPU training. "
            "Params: raw=true for detailed matrix. "
            "Examples: \"NVLink connections\" | \"Can GPUs do P2P?\".",
        "schema": {
            "type": "object",
            "properties": {"raw": {"type": "boolean", "default": False, "description": "Return raw P2P matrix"}},
        },
        "handler": "engine.gpu.topology",
    },
    "aisp_gpu_power": {
        "description": "Tags: gpu, power, thermal, throttling, watts, temperature. "
            "Get GPU power/thermal status: draw, limit, temp, throttling. "
            "⚡ FAST (~1s). USE WHEN: Diagnosing thermal throttling. "
            "Examples: \"Is GPU throttling?\" | \"Check power limit\".",
        "schema": {"type": "object", "properties": {}},
        "handler": "engine.gpu.power",
    },
    
    # === System Domain (3) ===
    "aisp_system_software": {
        "description": "Tags: system, software, cuda, pytorch, versions. "
            "Get software stack versions: Python, CUDA, PyTorch, etc. "
            "⚡ FAST (~1s). USE WHEN: Checking compatibility. "
            "Examples: \"What CUDA version?\" | \"Check PyTorch\".",
        "schema": {"type": "object", "properties": {}},
        "handler": "engine.system.software",
    },
    "aisp_system_dependencies": {
        "description": "Tags: system, dependencies, packages, pip. "
            "Check installed packages and versions. "
            "⚡ FAST (~2s). USE WHEN: Checking package versions.",
        "schema": {"type": "object", "properties": {}},
        "handler": "engine.system.dependencies",
    },
    "aisp_system_context": {
        "description": "Tags: system, context, environment, full-info. "
            "Get comprehensive system context. "
            "Params: level='summary'|'full' for detail control. "
            "⚡ FAST (~2s). Includes CPU, memory, container limits, params.",
        "schema": {
            "type": "object",
            "properties": {
                "level": {"type": "string", "enum": ["summary", "full"], "default": "summary"},
            },
        },
        "handler": "engine.system.context",
    },
    
    # === Profile Domain (5) ===
    "aisp_profile_nsys": {
        "description": "Tags: profile, nsys, nsight-systems, trace, timeline. "
            "Run nsys profiling on a script. Returns async job_id. "
            "🕐 SLOW (varies). USE WHEN: Need CUDA timeline analysis. "
            "WORKFLOW: aisp_profile_nsys → aisp_job_status → analyze results. "
            "Params: script, args, duration, summary=true for quick stats.",
        "schema": {
            "type": "object",
            "properties": {
                "script": {"type": "string", "description": "Script to profile"},
                "args": {"type": "string", "description": "Script arguments"},
                "duration": {"type": "integer", "default": 30, "description": "Max duration (seconds)"},
                "summary": {"type": "boolean", "default": False, "description": "Return summary stats only"},
            },
            "required": ["script"],
        },
        "handler": "engine.profile.nsys",
    },
    "aisp_profile_ncu": {
        "description": "Tags: profile, ncu, nsight-compute, kernel, occupancy. "
            "Run ncu kernel profiling. Returns async job_id. "
            "🕐 SLOW (varies). USE WHEN: Need kernel-level analysis. "
            "Params: script, kernel_filter, metrics.",
        "schema": {
            "type": "object",
            "properties": {
                "script": {"type": "string", "description": "Script to profile"},
                "kernel_filter": {"type": "string", "description": "Filter specific kernels"},
                "metrics": {"type": "array", "description": "Specific metrics to collect"},
            },
            "required": ["script"],
        },
        "handler": "engine.profile.ncu",
    },
    "aisp_profile_torch": {
        "description": "Tags: profile, pytorch, torch-profiler, tensorboard. "
            "Run PyTorch profiler. Returns async job_id. "
            "🕐 MEDIUM (~30s). USE WHEN: Need PyTorch operator breakdown.",
        "schema": {
            "type": "object",
            "properties": {
                "script": {"type": "string", "description": "Script to profile"},
                "wait": {"type": "integer", "default": 1},
                "warmup": {"type": "integer", "default": 1},
                "active": {"type": "integer", "default": 3},
            },
            "required": ["script"],
        },
        "handler": "engine.profile.torch",
    },
    "aisp_profile_flame": {
        "description": "Tags: profile, flame, flamegraph, visualization. "
            "Generate flame graph from profile data. "
            "⚡ FAST (~2s). USE WHEN: Visualizing hotspots.",
        "schema": {
            "type": "object",
            "properties": {
                "profile_path": {"type": "string", "description": "Path to profile file"},
            },
        },
        "handler": "engine.profile.flame",
    },
    "aisp_profile_compare": {
        "description": "Tags: profile, compare, diff, regression. "
            "Compare two profile runs. "
            "⚡ FAST (~2s). USE WHEN: Finding regressions, before/after analysis. "
            "Params: type='nsys'|'ncu' for profile type.",
        "schema": {
            "type": "object",
            "properties": {
                "profile_a": {"type": "string", "description": "First profile path"},
                "profile_b": {"type": "string", "description": "Second profile path"},
                "type": {"type": "string", "enum": ["nsys", "ncu"], "default": "nsys"},
            },
            "required": ["profile_a", "profile_b"],
        },
        "handler": "engine.profile.compare",
    },
    
    # === Analyze Domain (5) ===
    "aisp_analyze_bottlenecks": {
        "description": "Tags: analyze, bottleneck, diagnosis, slow. "
            "Identify performance bottlenecks. "
            "⚡ FAST (~3s). USE FIRST for performance issues. "
            "Examples: \"Why is training slow?\" | \"Find bottlenecks\".",
        "schema": {
            "type": "object",
            "properties": {
                "profile_path": {"type": "string", "description": "Profile to analyze (optional)"},
            },
        },
        "handler": "engine.analyze.bottlenecks",
    },
    "aisp_analyze_pareto": {
        "description": "Tags: analyze, pareto, priority, impact. "
            "Pareto analysis: highest-impact optimizations. "
            "⚡ FAST (~2s). USE WHEN: Prioritizing optimization work.",
        "schema": {"type": "object", "properties": {}},
        "handler": "engine.analyze.pareto",
    },
    "aisp_analyze_scaling": {
        "description": "Tags: analyze, scaling, multi-gpu, efficiency. "
            "Predict multi-GPU scaling efficiency. "
            "⚡ FAST (~2s). USE WHEN: Planning distributed training. "
            "Params: num_gpus, model_size for predictions.",
        "schema": {
            "type": "object",
            "properties": {
                "num_gpus": {"type": "integer", "description": "Target GPU count"},
                "model_size": {"type": "number", "description": "Model size in billions"},
            },
        },
        "handler": "engine.analyze.scaling",
    },
    "aisp_analyze_whatif": {
        "description": "Tags: analyze, whatif, simulation, prediction. "
            "What-if analysis: predict impact of changes. "
            "⚡ FAST (~1s). USE WHEN: Planning upgrades/changes. "
            "Examples: \"What if H100?\" | \"Double batch size?\".",
        "schema": {
            "type": "object",
            "properties": {
                "scenario": {"type": "string", "description": "Scenario to analyze"},
            },
        },
        "handler": "engine.analyze.whatif",
    },
    "aisp_analyze_memory": {
        "description": "Tags: analyze, memory, warp, bank, coalescing. "
            "Memory access pattern analysis: warp divergence, bank conflicts, coalescing. "
            "⚡ FAST (~2s). USE WHEN: Debugging memory-bound kernels. "
            "Params: type='warp'|'bank'|'access' for specific analysis.",
        "schema": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["warp", "bank", "access", "all"], "default": "all"},
                "profile_path": {"type": "string", "description": "Profile to analyze"},
            },
        },
        "handler": "engine.analyze.memory",
    },
    
    # === Optimize Domain (3) ===
    "aisp_recommend": {
        "description": "Tags: optimize, recommend, suggestions, improve. "
            "Get optimization recommendations for your workload. "
            "⚡ FAST (~2s). USE AFTER: aisp_triage or aisp_analyze_bottlenecks. "
            "Examples: \"How to speed up training?\" | \"Optimization suggestions\".",
        "schema": {
            "type": "object",
            "properties": {
                "model_size": {"type": "number", "description": "Model size in billions"},
                "gpu_count": {"type": "integer", "description": "Number of GPUs"},
            },
        },
        "handler": "engine.optimize.recommend",
    },
    "aisp_optimize_roi": {
        "description": "Tags: optimize, roi, cost-benefit, value. "
            "Calculate ROI of optimization techniques. "
            "⚡ FAST (~1s). USE WHEN: Prioritizing optimization work.",
        "schema": {
            "type": "object",
            "properties": {
                "technique": {"type": "string", "description": "Optimization technique to evaluate"},
            },
        },
        "handler": "engine.optimize.roi",
    },
    "aisp_optimize_techniques": {
        "description": "Tags: optimize, techniques, catalog, options. "
            "List available optimization techniques. "
            "⚡ FAST (~1s). USE WHEN: Exploring optimization options.",
        "schema": {"type": "object", "properties": {}},
        "handler": "engine.optimize.techniques",
    },
    
    # === Distributed Domain (3) ===
    "aisp_distributed_plan": {
        "description": "Tags: distributed, plan, strategy, ddp, fsdp, parallelism. "
            "Plan distributed training strategy with cost estimates. "
            "⚡ FAST (~2s). USE WHEN: Planning multi-GPU/node training. "
            "Includes: parallelism strategy, launch config, cost estimate.",
        "schema": {
            "type": "object",
            "properties": {
                "model_size": {"type": "number", "description": "Model size in billions"},
                "gpu_count": {"type": "integer", "description": "Number of GPUs"},
                "gpu_memory": {"type": "number", "description": "GPU memory in GB"},
            },
        },
        "handler": "engine.distributed.plan",
    },
    "aisp_distributed_nccl": {
        "description": "Tags: distributed, nccl, allreduce, communication. "
            "NCCL tuning recommendations. "
            "⚡ FAST (~1s). USE WHEN: Optimizing multi-GPU communication.",
        "schema": {"type": "object", "properties": {}},
        "handler": "engine.distributed.nccl",
    },
    "aisp_cluster_slurm": {
        "description": "Tags: cluster, slurm, hpc, job, submit. "
            "Generate SLURM job scripts. "
            "⚡ FAST (~1s). USE WHEN: Submitting to HPC cluster.",
        "schema": {
            "type": "object",
            "properties": {
                "nodes": {"type": "integer", "description": "Number of nodes"},
                "gpus_per_node": {"type": "integer", "description": "GPUs per node"},
                "script": {"type": "string", "description": "Script to run"},
            },
        },
        "handler": "engine.distributed.slurm",
    },
    
    # === Inference Domain (2) ===
    "aisp_inference_vllm": {
        "description": "Tags: inference, vllm, serving, throughput. "
            "vLLM configuration recommendations. "
            "⚡ FAST (~2s). USE WHEN: Setting up LLM inference serving.",
        "schema": {
            "type": "object",
            "properties": {
                "model": {"type": "string", "description": "Model name/path"},
                "gpu_memory": {"type": "number", "description": "GPU memory in GB"},
            },
        },
        "handler": "engine.inference.vllm",
    },
    "aisp_inference_quantization": {
        "description": "Tags: inference, quantization, int8, int4, awq, gptq. "
            "Quantization strategy recommendations. "
            "⚡ FAST (~1s). USE WHEN: Reducing model memory footprint. "
            "Examples: \"AWQ vs GPTQ?\" | \"Fit 70B on 24GB?\".",
        "schema": {
            "type": "object",
            "properties": {
                "model_size": {"type": "number", "description": "Model size in billions"},
                "target_memory": {"type": "number", "description": "Target memory in GB"},
            },
        },
        "handler": "engine.inference.quantization",
    },
    
    # === Benchmark Domain (4) ===
    "aisp_benchmark_run": {
        "description": "Tags: benchmark, run, test, measure. "
            "Run performance benchmarks. Returns async job_id for long tests. "
            "🕐 VARIES. USE WHEN: Measuring baseline performance.",
        "schema": {
            "type": "object",
            "properties": {
                "benchmark": {"type": "string", "description": "Benchmark name"},
                "params": {"type": "object", "description": "Benchmark parameters"},
            },
        },
        "handler": "engine.benchmark.run",
    },
    "aisp_benchmark_targets": {
        "description": "Tags: benchmark, targets, expected, comparison. "
            "Get expected performance targets for hardware. "
            "⚡ FAST (~1s). USE WHEN: Comparing actual vs expected.",
        "schema": {
            "type": "object",
            "properties": {
                "gpu": {"type": "string", "description": "GPU model name"},
            },
        },
        "handler": "engine.benchmark.targets",
    },
    "aisp_benchmark_report": {
        "description": "Tags: benchmark, report, summary, results. "
            "Generate benchmark report. "
            "⚡ FAST (~2s). USE AFTER: Running benchmarks.",
        "schema": {"type": "object", "properties": {}},
        "handler": "engine.benchmark.report",
    },
    "aisp_benchmark_history": {
        "description": "Tags: benchmark, history, compare, regression. "
            "View and compare benchmark history. "
            "⚡ FAST (~1s). USE WHEN: Tracking performance over time.",
        "schema": {
            "type": "object",
            "properties": {
                "compare": {"type": "boolean", "default": False, "description": "Compare recent runs"},
            },
        },
        "handler": "engine.benchmark.history",
    },
    
    # === AI Domain (3) ===
    "aisp_ask": {
        "description": "Tags: ai, llm, question, natural-language. "
            "Ask any performance question in natural language. "
            "🕐 MEDIUM (~5s). USE WHEN: Need human-friendly explanation. "
            "Examples: \"Why is GPU at 50%?\" | \"Explain tensor parallelism\".",
        "schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Your question"},
            },
            "required": ["question"],
        },
        "handler": "engine.ai.ask",
    },
    "aisp_explain": {
        "description": "Tags: ai, llm, explain, interpret. "
            "Get AI explanation of performance data. "
            "🕐 MEDIUM (~5s). USE WHEN: Need help interpreting results.",
        "schema": {
            "type": "object",
            "properties": {
                "data": {"type": "object", "description": "Data to explain"},
                "context": {"type": "string", "description": "Additional context"},
            },
        },
        "handler": "engine.ai.explain",
    },
    "aisp_ai_status": {
        "description": "Tags: ai, status, llm, available. "
            "Check AI/LLM availability and status. "
            "⚡ FAST (<1s). USE WHEN: Checking if AI features work.",
        "schema": {"type": "object", "properties": {}},
        "handler": "engine.ai.status",
    },
    
    # === Export Domain (3) ===
    "aisp_export_csv": {
        "description": "Tags: export, csv, data, spreadsheet. "
            "Export data to CSV format. "
            "⚡ FAST (~1s).",
        "schema": {
            "type": "object",
            "properties": {
                "data": {"type": "object", "description": "Data to export"},
                "path": {"type": "string", "description": "Output path"},
            },
        },
        "handler": "engine.export.csv",
    },
    "aisp_export_html": {
        "description": "Tags: export, html, report, web. "
            "Export data to HTML report. "
            "⚡ FAST (~2s).",
        "schema": {
            "type": "object",
            "properties": {
                "data": {"type": "object", "description": "Data to export"},
                "path": {"type": "string", "description": "Output path"},
            },
        },
        "handler": "engine.export.html",
    },
    "aisp_export_pdf": {
        "description": "Tags: export, pdf, report, document. "
            "Export data to PDF report. "
            "🕐 MEDIUM (~5s).",
        "schema": {
            "type": "object",
            "properties": {
                "data": {"type": "object", "description": "Data to export"},
                "path": {"type": "string", "description": "Output path"},
            },
        },
        "handler": "engine.export.pdf",
    },
    
    # === Hardware Microbenchmarks (10) ===
    "aisp_hw_speed": {
        "description": "Tags: hw, benchmark, compute, tflops. "
            "GPU compute speed benchmark. "
            "🕐 MEDIUM (~15s). Returns TFLOPS for various dtypes.",
        "schema": {"type": "object", "properties": {}},
        "handler": "hw.speed",
    },
    "aisp_hw_roofline": {
        "description": "Tags: hw, roofline, compute-bound, memory-bound. "
            "Generate roofline model for GPU. "
            "🕐 MEDIUM (~20s). Shows compute vs memory bounds.",
        "schema": {"type": "object", "properties": {}},
        "handler": "hw.roofline",
    },
    "aisp_hw_disk": {
        "description": "Tags: hw, disk, io, storage. "
            "Disk I/O benchmark. "
            "🕐 MEDIUM (~10s). Tests read/write throughput.",
        "schema": {"type": "object", "properties": {}},
        "handler": "hw.disk",
    },
    "aisp_hw_pcie": {
        "description": "Tags: hw, pcie, bandwidth, host-device. "
            "PCIe bandwidth benchmark. "
            "🕐 MEDIUM (~10s). Tests CPU-GPU transfer speed.",
        "schema": {"type": "object", "properties": {}},
        "handler": "hw.pcie",
    },
    "aisp_hw_cache": {
        "description": "Tags: hw, cache, l2, memory-hierarchy. "
            "GPU cache benchmark. "
            "🕐 MEDIUM (~15s). Tests L2 cache performance.",
        "schema": {"type": "object", "properties": {}},
        "handler": "hw.cache",
    },
    "aisp_hw_tc": {
        "description": "Tags: hw, tensor-core, tc, matmul. "
            "Tensor Core benchmark. "
            "🕐 MEDIUM (~15s). Tests TC throughput.",
        "schema": {"type": "object", "properties": {}},
        "handler": "hw.tc",
    },
    "aisp_hw_nccl": {
        "description": "Tags: hw, nccl, collective, allreduce. "
            "NCCL collective benchmark. "
            "🕐 MEDIUM (~30s). Tests multi-GPU communication.",
        "schema": {"type": "object", "properties": {}},
        "handler": "hw.nccl",
    },
    "aisp_hw_ib": {
        "description": "Tags: hw, infiniband, rdma, network. "
            "InfiniBand benchmark. "
            "🕐 MEDIUM (~20s). Tests IB throughput/latency.",
        "schema": {"type": "object", "properties": {}},
        "handler": "hw.ib",
    },
    "aisp_hw_network": {
        "description": "Tags: hw, network, tcp, bandwidth. "
            "Network benchmark (TCP/IB). "
            "🕐 MEDIUM (~15s). Tests network throughput.",
        "schema": {
            "type": "object",
            "properties": {
                "protocol": {"type": "string", "enum": ["tcp", "ib"], "default": "tcp"},
            },
        },
        "handler": "hw.network",
    },
    "aisp_hw_p2p": {
        "description": "Tags: hw, p2p, peer-to-peer, gpu-gpu. "
            "GPU peer-to-peer bandwidth benchmark. "
            "🕐 MEDIUM (~20s). Tests direct GPU-GPU transfer.",
        "schema": {"type": "object", "properties": {}},
        "handler": "hw.p2p",
    },
    
    # === HuggingFace (1 consolidated) ===
    "aisp_hf": {
        "description": "Tags: huggingface, models, download, search. "
            "HuggingFace Hub operations: search, trending, download. "
            "⚡ FAST (~2s). Params: action='search'|'trending'|'download'.",
        "schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["search", "trending", "download"], "default": "search"},
                "query": {"type": "string", "description": "Search query or model name"},
                "limit": {"type": "integer", "default": 10},
            },
        },
        "handler": "hf.action",
    },
}

# Total: 50 tools (down from 88)
# Core: 4, GPU: 4, System: 3, Profile: 5, Analyze: 5, Optimize: 3
# Distributed: 3, Inference: 2, Benchmark: 4, AI: 3, Export: 3, HW: 10, HF: 1



