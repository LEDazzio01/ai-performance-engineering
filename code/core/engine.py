"""
🚀 PerformanceEngine - The Unified Core of AI Systems Performance

This is THE single source of truth for all performance analysis functionality.
All interfaces (CLI, MCP, Web UI, Python API) should use this engine.

Architecture:
    PerformanceEngine (this file)
        ↓
    ┌───────┬───────┬───────┬────────┐
    │  CLI  │  MCP  │  Web  │ Python │
    │ aisp  │ Tools │  UI   │  API   │
    └───────┴───────┴───────┴────────┘

UNIFIED DOMAIN MODEL: 10 clean domains with consistent naming.

Domains:
    1. gpu        - Hardware info, topology, power, bandwidth
    2. system     - Software stack, dependencies, capabilities
    3. profile    - nsys/ncu profiling, flame graphs, HTA
    4. analyze    - Bottlenecks, pareto, scaling, memory patterns
    5. optimize   - Recommendations, ROI, techniques, compound effects
    6. distributed- Parallelism planning, NCCL, FSDP, tensor parallel
    7. inference  - vLLM config, quantization, deployment
    8. benchmark  - Run & track benchmarks, history, targets
    9. ai         - Ask questions, explain concepts, LLM analysis
   10. export     - CSV, PDF, HTML reports

Usage:
    from core.engine import get_engine
    
    engine = get_engine()
    
    # GPU info
    engine.gpu.info()
    engine.gpu.bandwidth()
    engine.gpu.topology()
    
    # System info
    engine.system.software()
    engine.system.dependencies()
    
    # Profiling
    engine.profile.flame_graph()
    engine.profile.compare(chapter="ch11")
    
    # Analysis
    engine.analyze.bottlenecks()
    engine.analyze.pareto()
    engine.analyze.whatif(max_vram_gb=24)
    
    # Optimization
    engine.optimize.recommend(model_size=70, gpus=8)
    engine.optimize.techniques()
    engine.optimize.roi()
    
    # Distributed training
    engine.distributed.plan(model_size=70, gpus=16, nodes=2)
    engine.distributed.nccl()
    
    # Inference
    engine.inference.vllm_config(model="70b")
    engine.inference.quantization()
    
    # Benchmarks
    engine.benchmark.run(targets=["ch07"])
    engine.benchmark.history()
    engine.benchmark.targets()
    
    # AI-powered
    engine.ai.ask("Why is my attention kernel slow?")
    engine.ai.explain("flash-attention")
    
    # Export
    engine.export.csv()
    engine.export.html()
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Union


# Find code root
CODE_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# LAZY LOADING - Don't import heavy modules until needed
# =============================================================================

_handler_instance = None
_analyzer_instance = None


def _get_handler():
    """Lazy load the shared PerfCore facade to avoid import overhead."""
    global _handler_instance
    if _handler_instance is None:
        from core.perf_core import get_core
        _handler_instance = get_core()
    return _handler_instance


def _get_analyzer():
    """Shared analyzer for CLI/UI without HTTP handler."""
    global _analyzer_instance
    if _analyzer_instance is None:
        from core.analysis.performance_analyzer import (
            PerformanceAnalyzer,
            load_benchmark_data,
        )
        _analyzer_instance = PerformanceAnalyzer(load_benchmark_data)
    return _analyzer_instance


def _safe_call(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """Safely call a function, returning error dict on failure."""
    try:
        return func(*args, **kwargs)
    except AttributeError as e:
        return {"error": f"Method not available: {e}", "success": False}
    except Exception as e:
        return {"error": str(e), "success": False}


# =============================================================================
# DOMAIN MODEL SPECIFICATION
# =============================================================================

DOMAINS = {
    "gpu": {
        "description": "GPU hardware info, topology, power, bandwidth",
        "operations": ["info", "topology", "power", "bandwidth", "nvlink", "control"]
    },
    "system": {
        "description": "Software stack, dependencies, environment",
        "operations": ["software", "dependencies", "capabilities", "context", "parameters"]
    },
    "profile": {
        "description": "Profiling with nsys/ncu/torch, flame graphs, HTA",
        "operations": ["nsys", "ncu", "torch", "hta", "flame_graph", "compare", "kernels"]
    },
    "analyze": {
        "description": "Performance analysis and bottleneck detection",
        "operations": ["bottlenecks", "pareto", "scaling", "whatif", "stacking", "power", "memory"]
    },
    "optimize": {
        "description": "Optimization recommendations and techniques",
        "operations": ["recommend", "techniques", "roi", "compound", "playbooks"]
    },
    "distributed": {
        "description": "Distributed training: parallelism, NCCL, FSDP",
        "operations": ["plan", "nccl", "fsdp", "tensor_parallel", "pipeline", "slurm"]
    },
    "inference": {
        "description": "Inference optimization: vLLM, quantization, deployment",
        "operations": ["vllm_config", "quantization", "deploy", "estimate"]
    },
    "benchmark": {
        "description": "Run benchmarks, track history, list targets",
        "operations": ["run", "targets", "history", "data", "compare_runs"]
    },
    "ai": {
        "description": "LLM-powered analysis, questions, explanations",
        "operations": ["ask", "explain", "analyze_kernel", "suggest_tools", "status"]
    },
    "export": {
        "description": "Export reports in CSV, PDF, HTML formats",
        "operations": ["csv", "pdf", "html", "report"]
    },
}


# =============================================================================
# DOMAIN 1: GPU
# =============================================================================

class GPUDomain:
    """
    GPU hardware operations.
    
    Operations:
        info()          - Get GPU name, memory, temperature, power, utilization
        topology()      - Multi-GPU topology: NVLink, PCIe, P2P matrix
        power()         - Power draw, limits, thermal status, throttling
        bandwidth()     - Memory bandwidth test (actual vs theoretical)
        nvlink()        - NVLink status and bandwidth
        control()       - GPU control state (clocks, persistence)
    """
    
    def __init__(self, parent: 'PerformanceEngine'):
        self._parent = parent
    
    def info(self) -> Dict[str, Any]:
        """Get GPU information: name, memory, temperature, power, utilization."""
        return _get_handler().get_gpu_info()
    
    def topology(self) -> Dict[str, Any]:
        """Get GPU topology: NVLink/PCIe connections, P2P matrix, NUMA affinity."""
        return _get_handler().get_gpu_topology()
    
    def power(self) -> Dict[str, Any]:
        """Get power status: draw, limits, temperature, throttling state."""
        return _safe_call(_get_analyzer().get_power_efficiency)
    
    def bandwidth(self) -> Dict[str, Any]:
        """Run GPU memory bandwidth test (actual vs theoretical HBM bandwidth)."""
        return _safe_call(_get_handler().run_gpu_bandwidth_test)
    
    def nvlink(self) -> Dict[str, Any]:
        """Get NVLink status: links per GPU, total bandwidth."""
        return _get_handler().get_nvlink_status()
    
    def control(self) -> Dict[str, Any]:
        """Get GPU control state: clock settings, persistence mode."""
        return _safe_call(_get_handler().get_gpu_control_state)
    
    def topology_matrix(self) -> Dict[str, Any]:
        """Get raw nvidia-smi topo -m output."""
        try:
            proc = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=5)
            return {"stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}
        except Exception as e:
            return {"error": str(e), "success": False}


# =============================================================================
# DOMAIN 2: SYSTEM
# =============================================================================

class SystemDomain:
    """
    System and software stack operations.
    
    Operations:
        software()      - PyTorch, CUDA, Python versions
        dependencies()  - Check ML/AI dependency health
        capabilities()  - Hardware capabilities (TMA, FP8, tensor cores)
        context()       - Full system context for AI analysis
        parameters()    - Kernel parameters affecting performance
        container()     - Container/cgroup limits
    """
    
    def __init__(self, parent: 'PerformanceEngine'):
        self._parent = parent
    
    def software(self) -> Dict[str, Any]:
        """Get software stack: PyTorch, CUDA, cuDNN, Python versions."""
        return _get_handler().get_software_info()
    
    def dependencies(self) -> Dict[str, Any]:
        """Check ML/AI dependency health: torch, triton, flash-attn, etc."""
        return _get_handler().get_dependency_health()
    
    def capabilities(self) -> Dict[str, Any]:
        """Get hardware capabilities: TMA, TMEM, tensor cores, FP8 support."""
        return _get_handler().get_hardware_capabilities()
    
    def context(self) -> Dict[str, Any]:
        """Get full system context for AI analysis."""
        return _get_handler().get_full_system_context()
    
    def parameters(self) -> Dict[str, Any]:
        """Get kernel parameters affecting performance (swappiness, etc.)."""
        return _get_handler().get_system_parameters()
    
    def container(self) -> Dict[str, Any]:
        """Get container/cgroup limits."""
        return _get_handler().get_container_limits()
    
    def cpu_memory(self) -> Dict[str, Any]:
        """Analyze CPU/memory hierarchy: NUMA, caches, TLB."""
        return _get_handler().get_cpu_memory_analysis()


# =============================================================================
# DOMAIN 3: PROFILE
# =============================================================================

class ProfileDomain:
    """
    Profiling operations with nsys, ncu, torch.profiler.
    
    Operations:
        nsys(command)       - Run Nsight Systems profiling
        ncu(command)        - Run Nsight Compute profiling
        torch(script)       - Run torch.profiler capture
        hta(command)        - Run HTA-friendly nsys capture + analysis
        flame_graph()       - Get flame graph visualization data
        compare(chapter)    - Compare baseline vs optimized profiles
        kernels()           - Get kernel execution breakdown
        list_profiles()     - List available profile pairs
        memory_timeline()   - Memory allocation timeline
        roofline()          - Roofline model data
    """
    
    def __init__(self, parent: 'PerformanceEngine'):
        self._parent = parent
    
    def flame_graph(self) -> Dict[str, Any]:
        """Get flame graph visualization data."""
        return _get_handler().get_flame_graph_data()
    
    def kernels(self) -> Dict[str, Any]:
        """Get kernel execution breakdown."""
        return _get_handler().get_kernel_breakdown()
    
    def memory_timeline(self) -> Dict[str, Any]:
        """Get memory allocation timeline."""
        return _get_handler().get_memory_timeline()
    
    def hta(self) -> Dict[str, Any]:
        """Get Holistic Trace Analysis data."""
        return _get_handler().get_hta_analysis()
    
    def torch(self) -> Dict[str, Any]:
        """Get latest torch.profiler capture summary."""
        return _get_handler().get_torch_profiler()
    
    def compile_analysis(self) -> Dict[str, Any]:
        """Get torch.compile analysis."""
        return _get_handler().get_compile_analysis()
    
    def roofline(self) -> Dict[str, Any]:
        """Get roofline model data."""
        return _get_handler().get_roofline_data()
    
    def compare(self, chapter: str) -> Dict[str, Any]:
        """Compare baseline vs optimized profiles for a chapter."""
        return _get_handler().compare_profiles(chapter)
    
    def list_profiles(self) -> Dict[str, Any]:
        """List available profile pairs for comparison."""
        return _get_handler().list_deep_profile_pairs()
    
    def recommendations(self) -> Dict[str, Any]:
        """Get profiling recommendations."""
        return _get_handler().get_profile_recommendations()
    
    def nsys_summary(self, report_path: str) -> Dict[str, Any]:
        """Summarize an existing nsys report."""
        return _safe_call(_get_handler().summarize_nsys_report, report_path)


# =============================================================================
# DOMAIN 4: ANALYZE
# =============================================================================

class AnalyzeDomain:
    """
    Performance analysis and bottleneck detection.
    
    Operations:
        bottlenecks()       - Identify performance bottlenecks
        pareto()            - Pareto frontier analysis
        scaling()           - Scaling analysis (GPU count)
        whatif(params)      - What-if constraint analysis
        stacking()          - Optimization stacking compatibility
        power()             - Power efficiency analysis
        memory()            - Memory access pattern analysis
        warp_divergence()   - Warp divergence analysis
        bank_conflicts()    - Shared memory bank conflict analysis
        leaderboards()      - Categorized performance leaderboards
    """
    
    def __init__(self, parent: 'PerformanceEngine'):
        self._parent = parent
    
    def bottlenecks(self, mode: str = "both") -> Dict[str, Any]:
        """
        Identify performance bottlenecks.
        
        Args:
            mode: "profile" for static analysis, "llm" for AI, "both" for combined
        """
        profile_result = {}
        llm_result = {}
        
        if mode in ["profile", "both"]:
            try:
                profile_result = _safe_call(_get_handler().detect_bottlenecks)
            except Exception as e:
                profile_result = {"error": str(e)}
        
        if mode in ["llm", "both"]:
            try:
                llm_result = _safe_call(_get_handler().run_ai_analysis, "bottleneck")
            except Exception as e:
                llm_result = {"error": str(e)}
        
        return {
            "profile": profile_result if mode != "llm" else None,
            "llm": llm_result if mode != "profile" else None,
            "source": mode,
        }
    
    def pareto(self) -> Dict[str, Any]:
        """Get Pareto frontier analysis (throughput vs latency vs memory)."""
        return _get_analyzer().get_pareto_frontier()
    
    def scaling(self) -> Dict[str, Any]:
        """Analyze scaling behavior with GPU count."""
        return _get_analyzer().get_scaling_analysis()
    
    def whatif(
        self, 
        max_vram_gb: Optional[float] = None,
        max_latency_ms: Optional[float] = None,
        min_throughput: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        What-if analysis: find optimizations meeting constraints.
        
        Args:
            max_vram_gb: Maximum VRAM budget in GB
            max_latency_ms: Maximum latency constraint in ms
            min_throughput: Minimum throughput requirement
        """
        params = {}
        if max_vram_gb is not None:
            params["vram"] = [str(max_vram_gb)]
        if max_latency_ms is not None:
            params["latency"] = [str(max_latency_ms)]
        if min_throughput is not None:
            params["throughput"] = [str(min_throughput)]
        return _get_analyzer().get_whatif_recommendations(params)
    
    def stacking(self) -> Dict[str, Any]:
        """Analyze optimization stacking compatibility."""
        return _get_analyzer().get_optimization_stacking()
    
    def power(self) -> Dict[str, Any]:
        """Power efficiency analysis."""
        return _get_analyzer().get_power_efficiency()
    
    def tradeoffs(self) -> Dict[str, Any]:
        """Trade-off analysis between metrics."""
        return _get_analyzer().get_tradeoff_analysis()
    
    def memory(self, stride: int = 1, element_size: int = 4) -> Dict[str, Any]:
        """Analyze memory access patterns for coalescing."""
        return _get_handler().get_memory_access_patterns(stride, element_size)
    
    def warp_divergence(self, code: str = "") -> Dict[str, Any]:
        """Analyze warp divergence patterns in kernel code."""
        return _get_handler().get_warp_divergence(code)
    
    def bank_conflicts(self, stride: int = 1, element_size: int = 4) -> Dict[str, Any]:
        """Analyze shared memory bank conflicts."""
        return _get_handler().get_bank_conflicts(stride, element_size)
    
    def leaderboards(self) -> Dict[str, Any]:
        """Get categorized performance leaderboards."""
        return _get_analyzer().get_categorized_leaderboards()
    
    def cost(self) -> Dict[str, Any]:
        """Cost analysis for benchmarks."""
        return _get_analyzer().get_cost_analysis()
    
    def predict_scaling(self, model_size: float, gpus: int) -> Dict[str, Any]:
        """Predict scaling behavior for model size and GPU count."""
        return _get_handler().predict_scaling({"model_size": model_size, "gpus": gpus})
    
    def comm_overlap(self, model: str = "llama-3.1-70b") -> Dict[str, Any]:
        """Analyze communication overlap opportunities."""
        return _get_handler().get_comm_overlap_analysis(model)
    
    def data_loading(self) -> Dict[str, Any]:
        """Analyze data loading pipeline."""
        return _get_handler().get_data_loading_analysis()
    
    def energy(self) -> Dict[str, Any]:
        """Energy efficiency analysis."""
        return _get_handler().get_energy_analysis()


# =============================================================================
# DOMAIN 5: OPTIMIZE
# =============================================================================

class OptimizeDomain:
    """
    Optimization recommendations and techniques.
    
    Operations:
        recommend(model_size, gpus, goal)  - Get recommendations
        techniques()                        - List all optimization techniques
        roi()                              - Calculate optimization ROI
        compound(techniques)               - Analyze compound effects
        playbooks()                        - Get optimization playbooks
        details(technique)                 - Get technique details
    """
    
    def __init__(self, parent: 'PerformanceEngine'):
        self._parent = parent
    
    def recommend(
        self, 
        model_size: float = 7, 
        gpus: int = 1, 
        goal: str = "throughput"
    ) -> Dict[str, Any]:
        """
        Get optimization recommendations for a model configuration.
        
        Args:
            model_size: Model size in billions of parameters
            gpus: Number of GPUs available
            goal: Optimization goal: "throughput", "latency", or "memory"
        """
        result = _safe_call(_get_analyzer().get_constraint_recommendations)
        
        # Add model-specific recommendations
        if model_size >= 70:
            techniques = ["Tensor Parallelism", "Pipeline Parallelism", "FP8 Training", 
                         "Flash Attention", "Gradient Checkpointing"]
            steps = [
                f"Use TP={min(8, gpus)} for 70B+ models",
                "Enable FP8 with Transformer Engine",
                "Use Flash Attention 2 for memory efficiency",
                "Enable gradient checkpointing for large batches",
            ]
        elif model_size >= 13:
            techniques = ["FSDP", "Flash Attention", "Mixed Precision", "Gradient Accumulation"]
            steps = [
                "Use FSDP with FULL_SHARD for memory efficiency",
                "Enable Flash Attention",
                "Use BF16 mixed precision",
            ]
        else:
            techniques = ["torch.compile", "Flash Attention", "Mixed Precision"]
            steps = [
                "Enable torch.compile for 2x speedup",
                "Use Flash Attention",
                "Enable BF16 precision",
            ]
        
        return {
            "model_size": model_size,
            "gpus": gpus,
            "goal": goal,
            "techniques": techniques,
            "estimated_speedup": (1.5, 3.0),
            "estimated_memory_reduction": 40 if model_size >= 70 else 20,
            "confidence": 0.8,
            "rationale": f"Recommendations for {model_size}B model on {gpus} GPUs optimizing for {goal}",
            "implementation_steps": steps,
            "success": True,
            **(result if isinstance(result, dict) else {})
        }
    
    def techniques(self) -> Dict[str, Any]:
        """Get list of all available optimization techniques."""
        return _get_handler().get_all_optimizations()
    
    def roi(self) -> Dict[str, Any]:
        """Calculate ROI for optimization techniques."""
        return _get_handler().get_optimization_roi()
    
    def compound(self, techniques: List[str]) -> Dict[str, Any]:
        """Analyze compound effects of multiple techniques together."""
        return _safe_call(_get_handler().get_compound_effect, {"techniques": techniques})
    
    def playbooks(self) -> Dict[str, Any]:
        """Get optimization playbooks for common scenarios."""
        return _safe_call(_get_handler().get_optimization_playbooks)
    
    def details(self, technique: str) -> Dict[str, Any]:
        """Get detailed information about a specific technique."""
        return _safe_call(_get_handler().get_optimization_details, technique)


# =============================================================================
# DOMAIN 6: DISTRIBUTED
# =============================================================================

class DistributedDomain:
    """
    Distributed training: parallelism planning, NCCL, FSDP.
    
    Operations:
        plan(model_size, gpus, nodes)  - Plan parallelism strategy
        nccl(nodes, gpus)              - NCCL tuning recommendations
        fsdp(model)                    - FSDP configuration
        tensor_parallel(model)         - Tensor parallelism config
        pipeline(model)                - Pipeline parallelism config
        slurm(model, nodes, gpus)      - Generate SLURM script
        cost_estimate(params)          - Estimate cloud costs
    """
    
    def __init__(self, parent: 'PerformanceEngine'):
        self._parent = parent
    
    def plan(
        self, 
        model_size: float = 7, 
        gpus: int = 8, 
        nodes: int = 1
    ) -> Dict[str, Any]:
        """
        Plan parallelism strategy for distributed training.
        
        Args:
            model_size: Model size in billions of parameters
            gpus: Total number of GPUs
            nodes: Number of nodes
        """
        try:
            from core.optimization.parallelism_planner.cli import get_parallelism_recommendations
            return get_parallelism_recommendations(
                model_name=f"model-{model_size}b",
                num_gpus=gpus,
                num_nodes=nodes
            )
        except Exception:
            return _safe_call(_get_handler().get_parallelism_recommendations)
    
    def nccl(self, nodes: int = 1, gpus: int = 8, diagnose: bool = False) -> Dict[str, Any]:
        """
        Get NCCL tuning recommendations.
        
        Args:
            nodes: Number of nodes
            gpus: GPUs per node
            diagnose: If True, run diagnostics
        """
        return _get_handler().get_nccl_recommendations(nodes, gpus, diagnose)
    
    def fsdp(self, model: str = "7b") -> Dict[str, Any]:
        """Get FSDP configuration for a model."""
        return _safe_call(_get_handler().get_fsdp_config, model)
    
    def tensor_parallel(self, model: str = "70b") -> Dict[str, Any]:
        """Get tensor parallelism configuration."""
        return _safe_call(_get_handler().get_tensor_parallel_config, model)
    
    def pipeline(self, model: str = "70b") -> Dict[str, Any]:
        """Get pipeline parallelism configuration."""
        return _safe_call(_get_handler().get_pipeline_parallel_config, model)
    
    def slurm(
        self, 
        model: str = "7b", 
        nodes: int = 1, 
        gpus: int = 8, 
        framework: str = "pytorch"
    ) -> Dict[str, Any]:
        """Generate SLURM job script for distributed training."""
        return _get_handler().generate_slurm_script(model, nodes, gpus, framework)
    
    def cost_estimate(
        self,
        model_size: Optional[float] = None,
        training_tokens: Optional[float] = None,
        provider: str = "aws"
    ) -> Dict[str, Any]:
        """
        Estimate cloud costs for training.
        
        Args:
            model_size: Model size in billions
            training_tokens: Training tokens in billions
            provider: Cloud provider (aws, gcp, azure)
        """
        params = {"provider": provider}
        if model_size is not None:
            params["model_size"] = model_size
        if training_tokens is not None:
            params["training_tokens"] = training_tokens
        return _get_handler().get_cloud_cost_estimate(params)
    
    def topology(self) -> Dict[str, Any]:
        """Get parallelism topology info."""
        return _safe_call(_get_handler().get_parallelism_topology)


# =============================================================================
# DOMAIN 7: INFERENCE
# =============================================================================

class InferenceDomain:
    """
    Inference optimization: vLLM, quantization, deployment.
    
    Operations:
        vllm_config(model, target)   - Generate vLLM configuration
        quantization(model_size)     - Quantization recommendations (FP8, INT8, INT4)
        deploy(params)               - Generate deployment configuration
        estimate(params)             - Estimate inference performance
    """
    
    def __init__(self, parent: 'PerformanceEngine'):
        self._parent = parent
    
    def vllm_config(
        self, 
        model: str = "7b", 
        target: str = "throughput",
        compare: bool = False
    ) -> Dict[str, Any]:
        """
        Generate vLLM configuration.
        
        Args:
            model: Model name or size (e.g., "llama-7b", "70b")
            target: Optimization target: "throughput" or "latency"
            compare: If True, compare inference engines
        """
        return _get_handler().get_vllm_config(model, target, compare)
    
    def quantization(self, model_size: Optional[float] = None) -> Dict[str, Any]:
        """
        Get quantization recommendations (FP8, INT8, INT4).
        
        Args:
            model_size: Model size in billions of parameters
        """
        params = {}
        if model_size is not None:
            params["params"] = model_size * 1e9
        return _get_handler().get_quantization_comparison(params)
    
    def deploy(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate deployment configuration."""
        return _safe_call(_get_handler().generate_deploy_config, params)
    
    def estimate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate inference performance."""
        return _safe_call(_get_handler().get_inference_estimate, params)


# =============================================================================
# DOMAIN 8: BENCHMARK
# =============================================================================

class BenchmarkDomain:
    """
    Benchmark execution and history tracking.
    
    Operations:
        run(targets)        - Run benchmarks on specified targets
        targets()           - List available benchmark targets
        history()           - Get historical benchmark runs
        data()              - Load benchmark results data
        compare_runs(a, b)  - Compare two benchmark runs
        available()         - Get available benchmarks with details
    """
    
    def __init__(self, parent: 'PerformanceEngine'):
        self._parent = parent
    
    def run(
        self, 
        targets: List[str],
        profile: str = "minimal",
        llm_analysis: bool = True,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Run benchmarks on specified targets.
        
        Args:
            targets: List of targets (e.g., ["ch07", "ch11:streams"])
            profile: Profile level: "minimal", "standard", "full"
            llm_analysis: Whether to include LLM analysis
            dry_run: If True, just show what would run
        """
        if dry_run:
            return {
                "dry_run": True,
                "targets": targets,
                "profile": profile,
                "llm_analysis": llm_analysis,
                "command": f"aisp bench run -t {' -t '.join(targets)} --profile {profile}",
            }
        
        # Actual execution would be handled by CLI or subprocess
        return {
            "targets": targets,
            "profile": profile,
            "note": "Use CLI for actual benchmark execution: aisp bench run -t <target>",
        }
    
    def targets(self, chapter: Optional[str] = None) -> Dict[str, Any]:
        """
        List available benchmark targets.
        
        Args:
            chapter: Optional chapter filter
        """
        return _get_handler().list_benchmark_targets()
    
    def history(self) -> Dict[str, Any]:
        """Get historical benchmark runs with trends."""
        return _get_handler().get_history_runs()
    
    def trends(self) -> Dict[str, Any]:
        """Get performance trends over time."""
        return _get_handler().get_performance_trends()
    
    def data(self) -> Dict[str, Any]:
        """Load current benchmark results data."""
        return _get_handler().load_benchmark_data()
    
    def available(self) -> Dict[str, Any]:
        """Get available benchmarks with detailed info."""
        return _get_handler().get_available_benchmarks()
    
    def speed_test(self) -> Dict[str, Any]:
        """Run quick speed tests (GEMM, memory, attention)."""
        return _safe_call(_get_handler().run_speed_tests)
    
    def network_test(self) -> Dict[str, Any]:
        """Run network throughput tests."""
        return _safe_call(_get_handler().run_network_tests)


# =============================================================================
# DOMAIN 9: AI
# =============================================================================

class AIDomain:
    """
    AI/LLM-powered analysis and questions.
    
    Operations:
        ask(question)           - Ask a performance question
        explain(concept)        - Explain a concept with citations
        analyze_kernel(code)    - Analyze CUDA kernel code
        suggest_tools(query)    - Suggest tools for a task
        status()                - Check AI/LLM availability
    """
    
    def __init__(self, parent: 'PerformanceEngine'):
        self._parent = parent
    
    def ask(self, question: str, include_citations: bool = True) -> Dict[str, Any]:
        """
        Ask a performance question with optional book citations.
        
        Args:
            question: Your performance question
            include_citations: Whether to include book citations
        """
        result = {"question": question, "success": False}
        
        # Get book citations
        if include_citations:
            try:
                from core.book import get_book_citations
                citations = get_book_citations(question, max_citations=3)
                result["citations"] = citations
            except Exception:
                result["citations"] = []
        
        # Get LLM response
        try:
            from core.llm import llm_call, is_available, PERF_EXPERT_SYSTEM
            
            if not is_available():
                result["error"] = "LLM not configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY."
                return result
            
            # Build context
            context = _get_handler().get_full_system_context()
            gpu_name = context.get("gpu_info", {}).get("name", "Unknown GPU")
            context_str = f"System: {gpu_name}"
            
            prompt = f"Context: {context_str}\n\nQuestion: {question}"
            response = llm_call(prompt, system=PERF_EXPERT_SYSTEM)
            result["answer"] = response
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def explain(self, concept: str) -> Dict[str, Any]:
        """
        Explain a GPU/AI performance concept with book citations.
        
        Args:
            concept: The concept to explain (e.g., "flash-attention", "tensor parallelism")
        """
        return _safe_call(_get_handler().get_llm_explanation, concept)
    
    def analyze_kernel(self, code: str) -> Dict[str, Any]:
        """
        Analyze CUDA kernel code with AI.
        
        Args:
            code: CUDA kernel code to analyze
        """
        return _safe_call(_get_handler().analyze_kernel_with_llm, {"code": code})
    
    def suggest_tools(self, query: str) -> Dict[str, Any]:
        """
        Suggest relevant tools based on user intent.
        
        Args:
            query: User intent (e.g., "I keep OOMing on 24GB VRAM")
        """
        # Return tool suggestions based on query keywords
        suggestions = []
        query_lower = query.lower()
        
        if "oom" in query_lower or "memory" in query_lower or "vram" in query_lower:
            suggestions.extend([
                {"tool": "gpu.info", "reason": "Check current VRAM usage"},
                {"tool": "analyze.memory", "reason": "Analyze memory access patterns"},
                {"tool": "inference.quantization", "reason": "Reduce memory with quantization"},
                {"tool": "optimize.recommend", "reason": "Get memory optimization tips"},
            ])
        
        if "slow" in query_lower or "latency" in query_lower:
            suggestions.extend([
                {"tool": "analyze.bottlenecks", "reason": "Identify bottlenecks"},
                {"tool": "profile.flame_graph", "reason": "Visualize time breakdown"},
                {"tool": "optimize.recommend", "reason": "Get optimization recommendations"},
            ])
        
        if "distributed" in query_lower or "multi-gpu" in query_lower or "parallel" in query_lower:
            suggestions.extend([
                {"tool": "gpu.topology", "reason": "Check GPU interconnects"},
                {"tool": "distributed.plan", "reason": "Plan parallelism strategy"},
                {"tool": "distributed.nccl", "reason": "NCCL tuning recommendations"},
            ])
        
        if not suggestions:
            suggestions = [
                {"tool": "system.context", "reason": "Get full system overview"},
                {"tool": "analyze.bottlenecks", "reason": "Start with bottleneck analysis"},
                {"tool": "ai.ask", "reason": "Ask your specific question"},
            ]
        
        return {
            "query": query,
            "suggestions": suggestions[:5],
            "note": "Call suggested tools in order for best results",
        }
    
    def status(self) -> Dict[str, Any]:
        """Check AI/LLM backend availability."""
        try:
            from core.llm import get_llm_status
            status = get_llm_status()
            return {
                "llm_available": status.get("available", False),
                "provider": status.get("provider"),
                "model": status.get("model"),
                "book_available": True,
            }
        except Exception as e:
            return {
                "llm_available": False,
                "provider": None,
                "model": None,
                "book_available": True,
                "error": str(e),
            }


# =============================================================================
# DOMAIN 10: EXPORT
# =============================================================================

class ExportDomain:
    """
    Export reports in various formats.
    
    Operations:
        csv()           - Export benchmarks to CSV
        csv_detailed()  - Export detailed CSV with all metrics
        pdf()           - Generate PDF report
        html()          - Generate HTML report
        report(params)  - Generate custom report
    """
    
    def __init__(self, parent: 'PerformanceEngine'):
        self._parent = parent
    
    def csv(self, detailed: bool = False) -> str:
        """
        Export benchmarks to CSV.
        
        Args:
            detailed: If True, include all metrics
        """
        if detailed:
            return _get_handler().export_detailed_csv()
        return _get_handler().export_benchmarks_csv()
    
    def pdf(self) -> bytes:
        """Generate PDF report."""
        try:
            return _get_handler().generate_pdf_report()
        except AttributeError:
            return b""
    
    def html(self) -> str:
        """Generate HTML report."""
        try:
            return _get_handler().generate_html_report()
        except AttributeError:
            return "<html><body>Report not available</body></html>"
    
    def report(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate custom report with specified parameters."""
        return _safe_call(_get_handler().generate_report, params)


# =============================================================================
# MAIN ENGINE
# =============================================================================

class PerformanceEngine:
    """
    🚀 The Unified Core of AI Systems Performance
    
    This is THE single source of truth for all performance analysis.
    All interfaces (CLI, MCP, Web UI) should use this engine.
    
    UNIFIED DOMAIN MODEL: 10 clean domains with consistent naming.
    
    Domains:
        engine.gpu          GPU hardware info, topology, power, bandwidth
        engine.system       Software stack, dependencies, capabilities
        engine.profile      nsys/ncu profiling, flame graphs, HTA
        engine.analyze      Bottlenecks, pareto, scaling, memory patterns
        engine.optimize     Recommendations, ROI, techniques, compound
        engine.distributed  Parallelism planning, NCCL, FSDP
        engine.inference    vLLM config, quantization, deployment
        engine.benchmark    Run & track benchmarks, history, targets
        engine.ai           Ask questions, explain concepts, LLM analysis
        engine.export       CSV, PDF, HTML reports
    
    Usage:
        engine = get_engine()
        
        # Quick status
        engine.status()
        
        # GPU info
        engine.gpu.info()
        engine.gpu.bandwidth()
        
        # Analysis
        engine.analyze.bottlenecks()
        engine.analyze.pareto()
        
        # Optimization
        engine.optimize.recommend(model_size=70, gpus=8)
        
        # AI-powered
        engine.ai.ask("Why is my kernel slow?")
        engine.ai.explain("flash-attention")
    """
    
    # Class-level domain specification for introspection
    DOMAINS = DOMAINS
    
    def __init__(self):
        # Lazy-initialized domain instances
        self._gpu = None
        self._system = None
        self._profile = None
        self._analyze = None
        self._optimize = None
        self._distributed = None
        self._inference = None
        self._benchmark = None
        self._ai = None
        self._export = None
    
    # -------------------------------------------------------------------------
    # Domain Properties (10 clean domains)
    # -------------------------------------------------------------------------
    
    @property
    def gpu(self) -> GPUDomain:
        """GPU hardware info, topology, power, bandwidth."""
        if self._gpu is None:
            self._gpu = GPUDomain(self)
        return self._gpu
    
    @property
    def system(self) -> SystemDomain:
        """Software stack, dependencies, capabilities."""
        if self._system is None:
            self._system = SystemDomain(self)
        return self._system
    
    @property
    def profile(self) -> ProfileDomain:
        """nsys/ncu profiling, flame graphs, HTA."""
        if self._profile is None:
            self._profile = ProfileDomain(self)
        return self._profile
    
    @property
    def analyze(self) -> AnalyzeDomain:
        """Performance analysis and bottleneck detection."""
        if self._analyze is None:
            self._analyze = AnalyzeDomain(self)
        return self._analyze
    
    @property
    def optimize(self) -> OptimizeDomain:
        """Optimization recommendations and techniques."""
        if self._optimize is None:
            self._optimize = OptimizeDomain(self)
        return self._optimize
    
    @property
    def distributed(self) -> DistributedDomain:
        """Distributed training: parallelism, NCCL, FSDP."""
        if self._distributed is None:
            self._distributed = DistributedDomain(self)
        return self._distributed
    
    @property
    def inference(self) -> InferenceDomain:
        """Inference optimization: vLLM, quantization."""
        if self._inference is None:
            self._inference = InferenceDomain(self)
        return self._inference
    
    @property
    def benchmark(self) -> BenchmarkDomain:
        """Run & track benchmarks, history, targets."""
        if self._benchmark is None:
            self._benchmark = BenchmarkDomain(self)
        return self._benchmark
    
    @property
    def ai(self) -> AIDomain:
        """AI/LLM-powered analysis and questions."""
        if self._ai is None:
            self._ai = AIDomain(self)
        return self._ai
    
    @property
    def export(self) -> ExportDomain:
        """Export reports in CSV, PDF, HTML formats."""
        if self._export is None:
            self._export = ExportDomain(self)
        return self._export
    
    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------
    
    def status(self) -> Dict[str, Any]:
        """
        Quick system status check.
        
        Returns GPU info, software versions, and AI availability.
        """
        return {
            "gpu": self.gpu.info(),
            "software": self.system.software(),
            "ai": self.ai.status(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    
    def triage(self) -> Dict[str, Any]:
        """
        Quick triage: status + context summary.
        
        Use as a first call before other tools.
        """
        return {
            "status": self.status(),
            "context": self.system.context(),
            "next_steps": [
                "Use engine.analyze.bottlenecks() to identify issues",
                "Use engine.optimize.recommend() for optimization tips",
                "Use engine.ai.ask() for specific questions",
            ],
        }
    
    def ask(self, question: str) -> Dict[str, Any]:
        """Quick AI question (shortcut to engine.ai.ask)."""
        return self.ai.ask(question)
    
    def recommend(self, model_size: float = 7, gpus: int = 1) -> Dict[str, Any]:
        """Quick recommendations (shortcut to engine.optimize.recommend)."""
        return self.optimize.recommend(model_size, gpus)
    
    def list_domains(self) -> Dict[str, Any]:
        """List all available domains and their operations."""
        return {
            "domains": DOMAINS,
            "count": len(DOMAINS),
            "usage": "engine.<domain>.<operation>()",
        }


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_engine_instance: Optional[PerformanceEngine] = None


def get_engine() -> PerformanceEngine:
    """Get the singleton PerformanceEngine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = PerformanceEngine()
    return _engine_instance


def reset_engine() -> None:
    """Reset the singleton engine (useful for testing)."""
    global _engine_instance, _handler_instance, _analyzer_instance
    _engine_instance = None
    _handler_instance = None
    _analyzer_instance = None


# =============================================================================
# BACKWARDS COMPATIBILITY - Deprecated aliases
# =============================================================================

# These are kept for backwards compatibility but will be removed in future versions
GPUEngine = GPUDomain
SystemEngine = SystemDomain
ProfileEngine = ProfileDomain
AnalyzeEngine = AnalyzeDomain
OptimizeEngine = OptimizeDomain
DistributedEngine = DistributedDomain
InferenceEngine = InferenceDomain
TestEngine = BenchmarkDomain  # Renamed
AIEngine = AIDomain
ExportEngine = ExportDomain
