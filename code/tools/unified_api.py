#!/usr/bin/env python3
"""
🌐 Unified API - Single Interface for All Optimization Capabilities

This module provides a SINGLE, CONSISTENT interface for ALL optimization
capabilities in the system. Every feature is accessible through:
1. Python API (direct function calls)
2. CLI (command line interface)
3. REST API (for UI dashboard)

Feature Parity Guarantee:
- Every function exposed here is available through ALL interfaces
- Same parameters, same outputs
- No hidden functionality

Architecture:
    Python API:  unified_api.optimize_distributed(...)
    CLI:         python -m tools.unified_api distributed --model llama-70b
    REST:        POST /api/optimize/distributed {"model": "llama-70b"}

Usage:
    # Python
    from tools.unified_api import UnifiedAPI
    api = UnifiedAPI()
    result = api.suggest_optimizations(model_size=70)
    
    # CLI
    python -m tools.unified_api suggest --model-size 70
    
    # REST (via dashboard)
    curl -X POST http://localhost:8765/api/suggest \\
         -d '{"model_size": 70}'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
from functools import wraps


# =============================================================================
# API REGISTRY - Track all exposed endpoints
# =============================================================================

_REGISTRY: Dict[str, Dict[str, Any]] = {}


def api_endpoint(
    name: str,
    category: str,
    description: str,
    cli_command: Optional[str] = None,
    rest_path: Optional[str] = None,
):
    """
    Decorator to register an API endpoint.
    Ensures the function is exposed through all interfaces.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Register endpoint
        _REGISTRY[name] = {
            "name": name,
            "category": category,
            "description": description,
            "cli_command": cli_command or name.replace("_", "-"),
            "rest_path": rest_path or f"/api/{name.replace('_', '/')}",
            "function": func,
            "parameters": _extract_parameters(func),
        }
        
        wrapper._api_name = name
        return wrapper
    return decorator


def _extract_parameters(func: Callable) -> List[Dict[str, Any]]:
    """Extract parameter info from function signature."""
    import inspect
    sig = inspect.signature(func)
    params = []
    
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        
        param_info = {
            "name": name,
            "required": param.default is inspect.Parameter.empty,
            "default": None if param.default is inspect.Parameter.empty else param.default,
        }
        
        # Try to get type hint
        if param.annotation is not inspect.Parameter.empty:
            param_info["type"] = str(param.annotation).replace("typing.", "")
        
        params.append(param_info)
    
    return params


# =============================================================================
# UNIFIED API CLASS
# =============================================================================

class UnifiedAPI:
    """
    Unified API providing access to all optimization capabilities.
    
    This class is the single source of truth for all optimization operations.
    All methods decorated with @api_endpoint are automatically exposed through
    CLI, REST, and Python interfaces.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._init_components()
    
    def _init_components(self):
        """Initialize component modules lazily."""
        self._mcts_optimizer = None
        self._llm_oracle = None
        self._parallelism_planner = None
        self._profiler = None
        self._distributed_optimizer = None
        self._vllm_optimizer = None
        self._llm_engine = None
    
    # =========================================================================
    # SYSTEM DIAGNOSTICS
    # =========================================================================
    
    @api_endpoint(
        name="system_info",
        category="diagnostics",
        description="Get comprehensive system information",
        cli_command="system-info",
        rest_path="/api/system/info",
    )
    def system_info(self, quick: bool = False) -> Dict[str, Any]:
        """Get detailed system information including GPU, CPU, memory."""
        from tools.optimization_search.llm_oracle import ContextCollector
        
        collector = ContextCollector()
        hardware = collector.get_hardware_context()
        software = collector.get_software_context()
        
        result = {
            "hardware": hardware,
            "software": software,
            "timestamp": time.time(),
        }
        
        if not quick:
            # Add more detailed info
            result["capabilities"] = self._get_capabilities(hardware)
        
        return result
    
    def _get_capabilities(self, hardware: Dict) -> Dict[str, bool]:
        """Determine available capabilities based on hardware."""
        arch = hardware.get("gpu_arch", "").lower()
        return {
            "fp8": arch in ["hopper", "blackwell"],
            "bf16": arch not in ["volta", "turing", "pascal"],
            "flash_attention_3": arch in ["hopper", "blackwell"],
            "flash_attention_2": arch in ["ampere", "hopper", "blackwell"],
            "tensor_cores": arch not in ["volta", "pascal"],
            "nvlink": hardware.get("has_nvlink", False),
            "tma": arch in ["hopper", "blackwell"],
            "warp_specialization": arch in ["hopper", "blackwell"],
        }
    
    @api_endpoint(
        name="gpu_status",
        category="diagnostics",
        description="Get current GPU status and utilization",
        cli_command="gpu-status",
        rest_path="/api/gpu/status",
    )
    def gpu_status(self) -> Dict[str, Any]:
        """Get current GPU status."""
        import subprocess
        
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            
            gpus = []
            for i, line in enumerate(result.stdout.strip().split('\n')):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 6:
                    gpus.append({
                        "id": i,
                        "name": parts[0],
                        "memory_used_mb": int(parts[1]),
                        "memory_total_mb": int(parts[2]),
                        "utilization_pct": int(parts[3]),
                        "temperature_c": int(parts[4]),
                        "power_w": float(parts[5]),
                    })
            
            return {"gpus": gpus, "count": len(gpus)}
        except Exception as e:
            return {"error": str(e)}
    
    # =========================================================================
    # OPTIMIZATION SUGGESTIONS
    # =========================================================================
    
    @api_endpoint(
        name="suggest_optimizations",
        category="optimization",
        description="Get AI-powered optimization suggestions",
        cli_command="suggest",
        rest_path="/api/optimize/suggest",
    )
    def suggest_optimizations(
        self,
        model_size: Optional[float] = None,
        model_name: Optional[str] = None,
        profile_path: Optional[str] = None,
        constraints: Optional[Dict] = None,
        num_suggestions: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get optimization suggestions using LLM oracle.
        
        Args:
            model_size: Model size in billions of parameters
            model_name: Model name (e.g., "llama-70b")
            profile_path: Path to profile JSON
            constraints: Optimization constraints
            num_suggestions: Number of suggestions to return
        
        Returns:
            List of optimization suggestions
        """
        from tools.optimization_search import LLMOracle
        
        model_config = {}
        if model_size:
            model_config["parameters_billions"] = model_size
        if model_name:
            model_config["name"] = model_name
        
        profile_data = {}
        if profile_path and Path(profile_path).exists():
            with open(profile_path) as f:
                profile_data = json.load(f)
        
        oracle = LLMOracle()
        suggestions = oracle.suggest_optimizations(
            model_config=model_config,
            profile_data=profile_data,
            constraints=constraints,
            num_suggestions=num_suggestions,
        )
        
        return [s.to_dict() for s in suggestions]
    
    @api_endpoint(
        name="search_optimal_config",
        category="optimization",
        description="Use MCTS to search for optimal configuration",
        cli_command="search",
        rest_path="/api/optimize/search",
    )
    def search_optimal_config(
        self,
        model_size: float = 70,
        num_gpus: int = 8,
        gpu_memory: float = 80,
        gpu_arch: str = "hopper",
        optimization_goal: str = "throughput",
        budget: int = 100,
    ) -> Dict[str, Any]:
        """
        Use MCTS to search for optimal compound optimization configuration.
        
        Args:
            model_size: Model size in billions
            num_gpus: Number of GPUs
            gpu_memory: GPU memory in GB
            gpu_arch: GPU architecture
            optimization_goal: "throughput", "memory", or "balanced"
            budget: Search budget (rollouts)
        
        Returns:
            Optimal configuration with statistics
        """
        from tools.optimization_search import MCTSOptimizer
        
        model_config = {
            "parameters_billions": model_size,
            "num_layers": int(model_size * 1.2),
            "hidden_size": int((model_size * 1e9 / 100) ** 0.5 * 128),
        }
        
        hardware_config = {
            "num_gpus": num_gpus,
            "gpu_memory_gb": gpu_memory,
            "gpu_arch": gpu_arch,
            "has_nvlink": True,
        }
        
        optimizer = MCTSOptimizer(hardware_config, model_config)
        return optimizer.search(
            budget=budget,
            optimization_goal=optimization_goal,
            verbose=self.verbose,
        )
    
    @api_endpoint(
        name="ask",
        category="optimization",
        description="Ask any performance-related question",
        cli_command="ask",
        rest_path="/api/ask",
    )
    def ask(self, question: str, context: Optional[Dict] = None) -> str:
        """
        Ask the LLM oracle any performance-related question.
        
        Args:
            question: Your question
            context: Additional context
        
        Returns:
            Answer from the oracle
        """
        from tools.optimization_search import LLMOracle
        oracle = LLMOracle()
        return oracle.ask(question, context)
    
    # =========================================================================
    # DISTRIBUTED TRAINING
    # =========================================================================
    
    @api_endpoint(
        name="optimize_distributed",
        category="distributed",
        description="Optimize distributed training configuration",
        cli_command="distributed",
        rest_path="/api/optimize/distributed",
    )
    def optimize_distributed(
        self,
        model_size: float = 70,
        num_nodes: int = 1,
        gpus_per_node: int = 8,
        seq_length: int = 4096,
        batch_size: int = 1024,
        optimization_goal: str = "throughput",
    ) -> Dict[str, Any]:
        """
        Optimize distributed training configuration.
        
        Args:
            model_size: Model parameters in billions
            num_nodes: Number of nodes
            gpus_per_node: GPUs per node
            seq_length: Sequence length
            batch_size: Target batch size
            optimization_goal: "throughput", "memory", or "latency"
        
        Returns:
            Optimal parallelism configuration
        """
        try:
            from tools.parallelism_planner import auto_tune_config
            
            model_config = {
                "parameters_billions": model_size,
                "max_sequence_length": seq_length,
                "hidden_size": int((model_size * 1e9 / 100) ** 0.5 * 128),
                "num_layers": int(model_size * 1.2),
            }
            
            hardware_config = {
                "gpu_memory_gb": 80,
                "num_gpus": num_nodes * gpus_per_node,
                "has_nvlink": True,
            }
            
            return auto_tune_config(
                model_config=model_config,
                hardware_config=hardware_config,
                target_batch_size=batch_size,
                optimization_goal=optimization_goal,
            )
        except ImportError:
            return {"error": "Parallelism planner not available"}
    
    @api_endpoint(
        name="optimize_rlhf",
        category="distributed",
        description="Optimize RLHF training configuration",
        cli_command="rlhf",
        rest_path="/api/optimize/rlhf",
    )
    def optimize_rlhf(
        self,
        model_size: float = 70,
        algorithm: str = "ppo",
        num_gpus: int = 8,
    ) -> Dict[str, Any]:
        """
        Optimize RLHF training configuration.
        
        Args:
            model_size: Model parameters in billions
            algorithm: RLHF algorithm (ppo, dpo, grpo, etc.)
            num_gpus: Number of GPUs
        
        Returns:
            RLHF optimization configuration
        """
        try:
            from tools.parallelism_planner.rl_optimization import RLHFOptimizer, RLAlgorithm
            
            algo_map = {
                "ppo": RLAlgorithm.PPO,
                "dpo": RLAlgorithm.DPO,
                "grpo": RLAlgorithm.GRPO,
                "rloo": RLAlgorithm.RLOO,
                "kto": RLAlgorithm.KTO,
            }
            
            optimizer = RLHFOptimizer()
            result = optimizer.optimize(
                model_params_b=model_size,
                algorithm=algo_map.get(algorithm, RLAlgorithm.PPO),
                num_gpus=num_gpus,
            )
            
            return result.to_dict() if hasattr(result, 'to_dict') else result
        except ImportError:
            return {"error": "RLHF optimizer not available"}
    
    # =========================================================================
    # INFERENCE OPTIMIZATION
    # =========================================================================
    
    @api_endpoint(
        name="optimize_vllm",
        category="inference",
        description="Optimize vLLM serving configuration",
        cli_command="vllm",
        rest_path="/api/optimize/vllm",
    )
    def optimize_vllm(
        self,
        model_size: float = 70,
        num_gpus: int = 8,
        optimization_goal: str = "throughput",
        max_seq_len: int = 8192,
    ) -> Dict[str, Any]:
        """
        Optimize vLLM serving configuration.
        
        Args:
            model_size: Model parameters in billions
            num_gpus: Number of GPUs
            optimization_goal: "throughput", "latency", or "sla"
            max_seq_len: Maximum sequence length
        
        Returns:
            vLLM configuration with launch command
        """
        try:
            from tools.parallelism_planner.vllm_optimization import get_vllm_optimization
            
            model_config = {
                "name": f"model-{model_size}b",
                "parameters_billions": model_size,
                "max_sequence_length": max_seq_len,
            }
            
            hardware_config = {
                "gpu_memory_gb": 80,
                "num_gpus": num_gpus,
                "gpu_arch": "hopper",
            }
            
            return get_vllm_optimization(
                model_config=model_config,
                hardware_config=hardware_config,
                optimization_goal=optimization_goal,
            )
        except ImportError:
            return {"error": "vLLM optimizer not available"}
    
    @api_endpoint(
        name="optimize_inference",
        category="inference",
        description="General inference optimization recommendations",
        cli_command="inference",
        rest_path="/api/optimize/inference",
    )
    def optimize_inference(
        self,
        model_size: float = 70,
        framework: str = "auto",
        target_latency_ms: Optional[float] = None,
        target_throughput_tps: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Get inference optimization recommendations.
        
        Args:
            model_size: Model parameters in billions
            framework: Target framework (vllm, tensorrt, auto)
            target_latency_ms: Target latency in ms
            target_throughput_tps: Target throughput in tokens/sec
        
        Returns:
            Inference optimization configuration
        """
        from tools.optimization_search import LLMOracle
        
        oracle = LLMOracle()
        context = {
            "model_size": model_size,
            "framework": framework,
            "targets": {
                "latency_ms": target_latency_ms,
                "throughput_tps": target_throughput_tps,
            }
        }
        
        suggestions = oracle.suggest_optimizations(
            model_config={"parameters_billions": model_size},
            constraints=context["targets"],
            num_suggestions=5,
        )
        
        return {
            "model_size": model_size,
            "framework": framework,
            "suggestions": [s.to_dict() for s in suggestions],
            "recommended_framework": self._recommend_framework(model_size, framework),
        }
    
    def _recommend_framework(self, model_size: float, preferred: str) -> str:
        """Recommend inference framework."""
        if preferred != "auto":
            return preferred
        
        if model_size > 100:
            return "tensorrt-llm"  # Best for very large models
        elif model_size > 30:
            return "vllm"  # Great balance
        else:
            return "vllm"  # Good for smaller models too
    
    # =========================================================================
    # COMPOUND OPTIMIZATIONS
    # =========================================================================
    
    @api_endpoint(
        name="compound_optimizations",
        category="optimization",
        description="Generate compound optimization strategies",
        cli_command="compound",
        rest_path="/api/optimize/compound",
    )
    def compound_optimizations(
        self,
        model_size: float = 70,
        num_layers: int = 80,
        hidden_size: int = 8192,
        seq_length: int = 4096,
        batch_size: int = 1,
        num_gpus: int = 8,
        gpu_arch: str = "hopper",
        optimization_goal: str = "throughput",
    ) -> Dict[str, Any]:
        """
        Generate compound optimization strategies combining multiple techniques.
        
        Args:
            model_size: Model parameters in billions
            num_layers: Number of layers
            hidden_size: Hidden dimension
            seq_length: Sequence length
            batch_size: Batch size
            num_gpus: Number of GPUs
            gpu_arch: GPU architecture
            optimization_goal: "throughput", "memory", or "balanced"
        
        Returns:
            Compound optimization with all techniques
        """
        try:
            from tools.parallelism_planner.advanced_optimizations import (
                CompoundOptimizationGenerator,
                get_advanced_optimization_report,
            )
            
            model_config = {
                "parameters_billions": model_size,
                "num_layers": num_layers,
                "hidden_size": hidden_size,
                "max_sequence_length": seq_length,
                "batch_size": batch_size,
            }
            
            hardware_config = {
                "gpu_arch": gpu_arch,
                "gpu_memory_gb": 80,
                "num_gpus": num_gpus,
                "has_nvlink": True,
            }
            
            return get_advanced_optimization_report(
                model_config=model_config,
                hardware_config=hardware_config,
                optimization_goal=optimization_goal,
            )
        except ImportError:
            return {"error": "Advanced optimizations not available"}
    
    # =========================================================================
    # PROFILING
    # =========================================================================
    
    @api_endpoint(
        name="analyze_profile",
        category="profiling",
        description="Analyze a profile and get optimization recommendations",
        cli_command="analyze",
        rest_path="/api/profile/analyze",
    )
    def analyze_profile(
        self,
        profile_path: str,
        code_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a profile file and get recommendations.
        
        Args:
            profile_path: Path to profile JSON
            code_path: Optional path to source code
        
        Returns:
            Analysis with recommendations
        """
        from tools.optimization_search import LLMOracle, ContextCollector
        
        profile_data = {}
        if Path(profile_path).exists():
            with open(profile_path) as f:
                profile_data = json.load(f)
        else:
            return {"error": f"Profile not found: {profile_path}"}
        
        # Analyze profile
        analysis = ContextCollector.analyze_profile(profile_data)
        
        # Get suggestions
        oracle = LLMOracle()
        suggestions = oracle.suggest_optimizations(
            profile_data=profile_data,
            num_suggestions=5,
        )
        
        return {
            "profile_path": profile_path,
            "analysis": analysis,
            "suggestions": [s.to_dict() for s in suggestions],
        }
    
    # =========================================================================
    # UTILITIES
    # =========================================================================
    
    @api_endpoint(
        name="validate_config",
        category="utilities",
        description="Validate an optimization configuration",
        cli_command="validate",
        rest_path="/api/validate",
    )
    def validate_config(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict] = None,
        model_size: float = 70,
    ) -> Dict[str, Any]:
        """
        Validate an optimization configuration.
        
        Args:
            config_path: Path to config JSON
            config: Config dict (alternative to path)
            model_size: Model size in billions
        
        Returns:
            Validation result
        """
        from tools.optimization_search import LLMOracle
        
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                config = json.load(f)
        
        if not config:
            return {"error": "No config provided"}
        
        oracle = LLMOracle()
        return oracle.validate_config(
            config=config,
            model_config={"parameters_billions": model_size},
        )
    
    @api_endpoint(
        name="list_endpoints",
        category="utilities",
        description="List all available API endpoints",
        cli_command="endpoints",
        rest_path="/api/endpoints",
    )
    def list_endpoints(self) -> List[Dict[str, Any]]:
        """List all available API endpoints."""
        return [
            {
                "name": info["name"],
                "category": info["category"],
                "description": info["description"],
                "cli": info["cli_command"],
                "rest": info["rest_path"],
                "parameters": info["parameters"],
            }
            for info in _REGISTRY.values()
        ]
    
    @api_endpoint(
        name="health",
        category="utilities",
        description="Check API health and dependencies",
        cli_command="health",
        rest_path="/api/health",
    )
    def health(self) -> Dict[str, Any]:
        """Check API health."""
        status = {
            "status": "healthy",
            "timestamp": time.time(),
            "components": {},
        }
        
        # Check components
        components = [
            ("mcts_optimizer", "tools.optimization_search.mcts_optimizer"),
            ("llm_oracle", "tools.optimization_search.llm_oracle"),
            ("parallelism_planner", "tools.parallelism_planner"),
            ("llm_engine", "tools.llm_engine"),
        ]
        
        for name, module in components:
            try:
                __import__(module)
                status["components"][name] = "available"
            except ImportError:
                status["components"][name] = "unavailable"
        
        return status


# =============================================================================
# CLI INTERFACE
# =============================================================================

def build_cli_parser() -> argparse.ArgumentParser:
    """Build CLI parser from registered endpoints."""
    parser = argparse.ArgumentParser(
        description="🌐 Unified Performance Optimization API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m tools.unified_api suggest --model-size 70
    python -m tools.unified_api search --model-size 70 --goal throughput
    python -m tools.unified_api ask "Why is my attention slow?"
    python -m tools.unified_api vllm --model-size 70 --num-gpus 8
    python -m tools.unified_api endpoints
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Group endpoints by category
    categories = {}
    for info in _REGISTRY.values():
        cat = info["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(info)
    
    # Create subparsers for each endpoint
    for info in _REGISTRY.values():
        cmd = info["cli_command"]
        sp = subparsers.add_parser(cmd, help=info["description"])
        
        for param in info["parameters"]:
            name = param["name"]
            cli_name = f"--{name.replace('_', '-')}"
            
            kwargs = {"help": f"{name} parameter"}
            
            if param.get("type"):
                type_str = param["type"]
                if "float" in type_str:
                    kwargs["type"] = float
                elif "int" in type_str:
                    kwargs["type"] = int
                elif "bool" in type_str:
                    kwargs["action"] = "store_true"
            
            if param.get("default") is not None:
                kwargs["default"] = param["default"]
            
            if param.get("required"):
                kwargs["required"] = True
            
            # Special case for 'question' parameter
            if name == "question":
                sp.add_argument("question_text", nargs="+", help="Your question")
            else:
                sp.add_argument(cli_name, **kwargs)
    
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    return parser


def run_cli():
    """Run the CLI interface."""
    # Initialize API to register endpoints
    api = UnifiedAPI()
    
    parser = build_cli_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Find matching endpoint
    endpoint_info = None
    for info in _REGISTRY.values():
        if info["cli_command"] == args.command:
            endpoint_info = info
            break
    
    if not endpoint_info:
        print(f"Unknown command: {args.command}")
        return
    
    # Build kwargs from args
    kwargs = {}
    for param in endpoint_info["parameters"]:
        name = param["name"]
        arg_name = name.replace("-", "_")
        
        if name == "question" and hasattr(args, "question_text"):
            kwargs["question"] = " ".join(args.question_text)
        elif hasattr(args, arg_name):
            value = getattr(args, arg_name)
            if value is not None:
                kwargs[name] = value
    
    # Set verbose if requested
    api.verbose = getattr(args, "verbose", False)
    
    # Call the function
    func = endpoint_info["function"]
    try:
        result = func(api, **kwargs)
        
        if getattr(args, "json", False):
            print(json.dumps(result, indent=2))
        else:
            _pretty_print(result, args.command)
    
    except Exception as e:
        if getattr(args, "verbose", False):
            import traceback
            traceback.print_exc()
        else:
            print(f"Error: {e}")


def _pretty_print(result: Any, command: str):
    """Pretty print result for CLI."""
    if isinstance(result, str):
        print(result)
    elif isinstance(result, list):
        if command == "endpoints":
            print("\n🌐 Available API Endpoints")
            print("=" * 60)
            
            categories = {}
            for item in result:
                cat = item.get("category", "other")
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(item)
            
            for cat, items in categories.items():
                print(f"\n📁 {cat.upper()}")
                for item in items:
                    print(f"   • {item['cli']}: {item['description']}")
                    print(f"     REST: {item['rest']}")
        else:
            for i, item in enumerate(result, 1):
                if isinstance(item, dict):
                    print(f"\n{i}. {item.get('title', item.get('name', 'Item'))}")
                    for k, v in item.items():
                        if k not in ("title", "name"):
                            print(f"   {k}: {v}")
                else:
                    print(f"{i}. {item}")
    elif isinstance(result, dict):
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        elif "best_config" in result:
            # MCTS result
            print("\n🎯 Optimal Configuration Found")
            print("=" * 60)
            print(f"Score: {result.get('best_score', 0):.4f}")
            print(f"Speedup: {result.get('estimated_speedup', 1):.2f}x")
            print(f"\nActions: {', '.join(result.get('best_actions', []))}")
            print(f"\nConfig: {json.dumps(result.get('best_config', {}), indent=2)}")
        else:
            print(json.dumps(result, indent=2))
    else:
        print(result)


# =============================================================================
# REST API HANDLER (for integration with dashboard)
# =============================================================================

def handle_rest_request(path: str, method: str, data: Dict) -> Dict[str, Any]:
    """
    Handle a REST API request.
    Called by the dashboard server.
    
    Args:
        path: Request path (e.g., "/api/optimize/suggest")
        method: HTTP method
        data: Request data (JSON body)
    
    Returns:
        Response data
    """
    api = UnifiedAPI()
    
    # Find matching endpoint
    for info in _REGISTRY.values():
        if info["rest_path"] == path:
            func = info["function"]
            try:
                return {"success": True, "data": func(api, **data)}
            except Exception as e:
                return {"success": False, "error": str(e)}
    
    return {"success": False, "error": f"Unknown endpoint: {path}"}


def get_rest_endpoints() -> List[Dict[str, Any]]:
    """Get all REST endpoints for OpenAPI spec generation."""
    # Initialize to populate registry
    UnifiedAPI()
    
    return [
        {
            "path": info["rest_path"],
            "method": "POST",
            "description": info["description"],
            "parameters": info["parameters"],
        }
        for info in _REGISTRY.values()
    ]


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    run_cli()


