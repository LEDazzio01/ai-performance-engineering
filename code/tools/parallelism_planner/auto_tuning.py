"""
Auto-Tuning Module for Parallelism Planner

Automatic configuration optimization:
- Batch size finder (find max batch that fits)
- Auto-tune parallelism configuration
- Gradient accumulation optimizer
- Memory headroom analysis
- Configuration search and optimization
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import math


@dataclass
class BatchSizeResult:
    """Result of batch size search."""
    max_batch_size: int
    memory_at_max: float
    memory_headroom_gb: float
    throughput_estimate: float
    
    # Recommended settings
    recommended_batch_size: int  # With headroom for safety
    recommended_grad_accum: int
    effective_batch_size: int
    
    # Search details
    search_iterations: int
    memory_curve: List[Dict[str, Any]]  # [(batch_size, memory_gb)]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_batch_size": self.max_batch_size,
            "memory_at_max_gb": self.memory_at_max,
            "memory_headroom_gb": self.memory_headroom_gb,
            "throughput_estimate_tps": self.throughput_estimate,
            "recommended": {
                "batch_size": self.recommended_batch_size,
                "gradient_accumulation": self.recommended_grad_accum,
                "effective_batch_size": self.effective_batch_size,
            },
            "search_iterations": self.search_iterations,
            "memory_curve": self.memory_curve,
        }


@dataclass
class GradAccumResult:
    """Result of gradient accumulation optimization."""
    optimal_grad_accum: int
    optimal_micro_batch: int
    effective_batch_size: int
    
    # Trade-offs
    memory_per_gpu_gb: float
    estimated_throughput_tps: float
    pipeline_efficiency: float  # For PP
    
    # Alternatives
    alternatives: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "optimal": {
                "gradient_accumulation": self.optimal_grad_accum,
                "micro_batch_size": self.optimal_micro_batch,
                "effective_batch_size": self.effective_batch_size,
            },
            "memory_per_gpu_gb": self.memory_per_gpu_gb,
            "estimated_throughput_tps": self.estimated_throughput_tps,
            "pipeline_efficiency": self.pipeline_efficiency,
            "alternatives": self.alternatives,
        }


@dataclass
class AutoTuneResult:
    """Result of auto-tuning search."""
    best_config: Dict[str, Any]
    best_score: float
    
    # All evaluated configurations
    evaluated_configs: List[Dict[str, Any]]
    
    # Search statistics
    total_iterations: int
    search_time_seconds: float
    
    # Recommendations
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_config": self.best_config,
            "best_score": self.best_score,
            "evaluated_configs": self.evaluated_configs,
            "search_statistics": {
                "total_iterations": self.total_iterations,
                "search_time_seconds": self.search_time_seconds,
            },
            "recommendations": self.recommendations,
        }


class BatchSizeFinder:
    """Finds the maximum batch size that fits in GPU memory."""
    
    def __init__(
        self,
        gpu_memory_gb: float = 80,
        headroom_pct: float = 0.1,
    ):
        self.gpu_memory_gb = gpu_memory_gb
        self.headroom_pct = headroom_pct
        self.available_memory = gpu_memory_gb * (1 - headroom_pct)
    
    def find_max_batch_size(
        self,
        model_params_b: float,
        seq_length: int,
        hidden_size: int,
        num_layers: int,
        tp: int = 1,
        pp: int = 1,
        dp: int = 1,
        precision_bytes: int = 2,  # BF16
        include_optimizer: bool = True,
        target_effective_batch: int = 1024,
    ) -> BatchSizeResult:
        """
        Find the maximum micro-batch size that fits in memory.
        
        Uses binary search to find the maximum batch size.
        """
        
        # Calculate fixed memory (model + optimizer)
        param_memory = (model_params_b * 1e9 * precision_bytes) / (tp * pp) / 1e9
        
        if include_optimizer:
            # AdamW: 8 bytes per param (2 moments in FP32)
            optimizer_memory = (model_params_b * 1e9 * 8) / (tp * pp * dp) / 1e9
        else:
            optimizer_memory = 0
        
        fixed_memory = param_memory + optimizer_memory
        available_for_activations = self.available_memory - fixed_memory
        
        if available_for_activations <= 0:
            return BatchSizeResult(
                max_batch_size=0,
                memory_at_max=fixed_memory,
                memory_headroom_gb=0,
                throughput_estimate=0,
                recommended_batch_size=0,
                recommended_grad_accum=0,
                effective_batch_size=0,
                search_iterations=0,
                memory_curve=[],
            )
        
        # Binary search for max batch size
        memory_curve = []
        min_batch = 1
        max_batch = 128  # Start with reasonable max
        
        # Activation memory per sample
        # Rough estimate: seq_length * hidden_size * num_layers * 2 (for gradients)
        activation_per_sample = (seq_length * hidden_size * num_layers * 2 * precision_bytes) / (tp * pp) / 1e9
        
        # Adjust for gradient checkpointing if memory is tight
        if activation_per_sample > available_for_activations:
            # Assume checkpointing reduces activation memory by ~10x
            activation_per_sample /= 10
        
        iterations = 0
        while min_batch < max_batch:
            mid_batch = (min_batch + max_batch + 1) // 2
            activation_memory = activation_per_sample * mid_batch
            total_memory = fixed_memory + activation_memory
            
            memory_curve.append({
                "batch_size": mid_batch,
                "memory_gb": total_memory,
                "fits": total_memory <= self.available_memory,
            })
            
            if total_memory <= self.available_memory:
                min_batch = mid_batch
            else:
                max_batch = mid_batch - 1
            
            iterations += 1
            if iterations > 20:
                break
        
        max_batch_size = min_batch
        memory_at_max = fixed_memory + activation_per_sample * max_batch_size
        memory_headroom = self.gpu_memory_gb - memory_at_max
        
        # Recommended batch size (with safety margin)
        recommended_batch = max(1, int(max_batch_size * 0.9))
        
        # Calculate gradient accumulation for target effective batch
        if recommended_batch > 0:
            recommended_grad_accum = max(1, target_effective_batch // (recommended_batch * dp))
        else:
            recommended_grad_accum = 1
        
        effective_batch = recommended_batch * dp * recommended_grad_accum
        
        # Estimate throughput
        throughput_estimate = self._estimate_throughput(
            model_params_b, recommended_batch, seq_length, tp, pp, dp
        )
        
        return BatchSizeResult(
            max_batch_size=max_batch_size,
            memory_at_max=memory_at_max,
            memory_headroom_gb=memory_headroom,
            throughput_estimate=throughput_estimate,
            recommended_batch_size=recommended_batch,
            recommended_grad_accum=recommended_grad_accum,
            effective_batch_size=effective_batch,
            search_iterations=iterations,
            memory_curve=memory_curve,
        )
    
    def _estimate_throughput(
        self,
        model_params_b: float,
        batch_size: int,
        seq_length: int,
        tp: int,
        pp: int,
        dp: int,
    ) -> float:
        """Rough throughput estimate in tokens/second."""
        num_gpus = tp * pp * dp
        
        # Assume H100-level performance
        base_tflops = 1979
        peak_throughput = (base_tflops * 1e12 * num_gpus) / (6 * model_params_b * 1e9)
        
        # Apply efficiency factors
        efficiency = 0.45
        efficiency *= 0.95 ** (tp - 1)
        efficiency *= (1 - (pp - 1) / (pp * 4)) if pp > 1 else 1
        
        return peak_throughput * efficiency


class GradientAccumulationOptimizer:
    """Optimizes gradient accumulation settings."""
    
    def __init__(self, gpu_memory_gb: float = 80):
        self.gpu_memory_gb = gpu_memory_gb
        self.batch_finder = BatchSizeFinder(gpu_memory_gb)
    
    def optimize(
        self,
        model_params_b: float,
        seq_length: int,
        hidden_size: int,
        num_layers: int,
        target_effective_batch: int,
        tp: int = 1,
        pp: int = 1,
        dp: int = 1,
    ) -> GradAccumResult:
        """
        Find optimal gradient accumulation and micro-batch size.
        
        Balances memory usage, throughput, and pipeline efficiency.
        """
        
        # Find max batch size
        batch_result = self.batch_finder.find_max_batch_size(
            model_params_b=model_params_b,
            seq_length=seq_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            tp=tp,
            pp=pp,
            dp=dp,
            target_effective_batch=target_effective_batch,
        )
        
        max_micro_batch = batch_result.max_batch_size
        
        # Find all valid (micro_batch, grad_accum) pairs
        alternatives = []
        
        for micro_batch in range(1, max_micro_batch + 1):
            # Calculate grad accum needed
            grad_accum = target_effective_batch // (micro_batch * dp)
            if grad_accum < 1:
                continue
            
            actual_effective = micro_batch * dp * grad_accum
            
            # Memory estimate
            memory = self._estimate_memory(
                model_params_b, micro_batch, seq_length, hidden_size, num_layers, tp, pp, dp
            )
            
            # Pipeline efficiency (more micro-batches = better)
            num_microbatches = grad_accum * dp
            if pp > 1:
                pipeline_eff = 1 - ((pp - 1) / num_microbatches)
            else:
                pipeline_eff = 1.0
            
            # Throughput estimate
            throughput = self._estimate_throughput(
                model_params_b, micro_batch, seq_length, tp, pp, dp
            ) * pipeline_eff
            
            # Score: balance throughput and pipeline efficiency
            score = throughput * pipeline_eff
            
            alternatives.append({
                "micro_batch_size": micro_batch,
                "gradient_accumulation": grad_accum,
                "effective_batch_size": actual_effective,
                "memory_gb": memory,
                "pipeline_efficiency": pipeline_eff,
                "throughput_estimate": throughput,
                "score": score,
            })
        
        if not alternatives:
            return GradAccumResult(
                optimal_grad_accum=1,
                optimal_micro_batch=1,
                effective_batch_size=dp,
                memory_per_gpu_gb=self.gpu_memory_gb,
                estimated_throughput_tps=0,
                pipeline_efficiency=1.0,
                alternatives=[],
            )
        
        # Find optimal (highest score)
        best = max(alternatives, key=lambda x: x["score"])
        
        # Sort alternatives by score
        alternatives.sort(key=lambda x: x["score"], reverse=True)
        
        return GradAccumResult(
            optimal_grad_accum=best["gradient_accumulation"],
            optimal_micro_batch=best["micro_batch_size"],
            effective_batch_size=best["effective_batch_size"],
            memory_per_gpu_gb=best["memory_gb"],
            estimated_throughput_tps=best["throughput_estimate"],
            pipeline_efficiency=best["pipeline_efficiency"],
            alternatives=alternatives[:5],  # Top 5 alternatives
        )
    
    def _estimate_memory(
        self,
        model_params_b: float,
        batch_size: int,
        seq_length: int,
        hidden_size: int,
        num_layers: int,
        tp: int,
        pp: int,
        dp: int,
    ) -> float:
        """Estimate memory in GB."""
        param_mem = (model_params_b * 1e9 * 2) / (tp * pp) / 1e9
        optimizer_mem = (model_params_b * 1e9 * 8) / (tp * pp * dp) / 1e9
        activation_mem = (batch_size * seq_length * hidden_size * num_layers * 2 * 2) / (tp * pp) / 1e9
        return param_mem + optimizer_mem + activation_mem
    
    def _estimate_throughput(
        self,
        model_params_b: float,
        batch_size: int,
        seq_length: int,
        tp: int,
        pp: int,
        dp: int,
    ) -> float:
        """Estimate throughput in tokens/second."""
        num_gpus = tp * pp * dp
        base_tflops = 1979
        peak = (base_tflops * 1e12 * num_gpus) / (6 * model_params_b * 1e9)
        efficiency = 0.45 * (0.95 ** (tp - 1))
        return peak * efficiency


class AutoTuner:
    """
    Automatic configuration optimization using search.
    
    Searches for optimal parallelism configuration based on:
    - Memory constraints
    - Throughput optimization
    - Communication efficiency
    """
    
    def __init__(
        self,
        gpu_memory_gb: float = 80,
        num_gpus: int = 8,
        has_nvlink: bool = True,
    ):
        self.gpu_memory_gb = gpu_memory_gb
        self.num_gpus = num_gpus
        self.has_nvlink = has_nvlink
    
    def auto_tune(
        self,
        model_params_b: float,
        seq_length: int,
        hidden_size: int,
        num_layers: int,
        target_batch_size: int = 1024,
        optimization_goal: str = "throughput",  # "throughput", "memory", "latency"
        search_budget: int = 50,
    ) -> AutoTuneResult:
        """
        Automatically find optimal configuration.
        
        Args:
            model_params_b: Model size in billions
            seq_length: Sequence length
            hidden_size: Model hidden size
            num_layers: Number of layers
            target_batch_size: Target effective batch size
            optimization_goal: What to optimize for
            search_budget: Maximum configurations to evaluate
        """
        
        import time
        start_time = time.time()
        
        evaluated = []
        
        # Generate candidate configurations
        candidates = self._generate_candidates()
        
        for i, config in enumerate(candidates[:search_budget]):
            tp, pp, dp = config["tp"], config["pp"], config["dp"]
            
            # Check if config is valid
            if tp * pp * dp != self.num_gpus:
                continue
            
            # Estimate memory
            memory = self._estimate_memory(
                model_params_b, seq_length, hidden_size, num_layers, tp, pp, dp
            )
            
            # Skip if doesn't fit
            if memory > self.gpu_memory_gb:
                continue
            
            # Estimate throughput
            throughput = self._estimate_throughput(
                model_params_b, seq_length, tp, pp, dp
            )
            
            # Calculate score based on goal
            if optimization_goal == "throughput":
                score = throughput
            elif optimization_goal == "memory":
                score = (self.gpu_memory_gb - memory) / self.gpu_memory_gb * 100
            elif optimization_goal == "latency":
                score = -pp  # Lower PP = lower latency
            else:
                score = throughput * (1 - memory / self.gpu_memory_gb)
            
            evaluated.append({
                "config": config,
                "memory_gb": memory,
                "throughput_tps": throughput,
                "score": score,
            })
        
        # Find best
        if not evaluated:
            return AutoTuneResult(
                best_config={"tp": 1, "pp": 1, "dp": self.num_gpus},
                best_score=0,
                evaluated_configs=[],
                total_iterations=len(candidates[:search_budget]),
                search_time_seconds=time.time() - start_time,
                recommendations=["No valid configuration found - model may be too large"],
            )
        
        evaluated.sort(key=lambda x: x["score"], reverse=True)
        best = evaluated[0]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(best, evaluated)
        
        return AutoTuneResult(
            best_config=best["config"],
            best_score=best["score"],
            evaluated_configs=evaluated[:10],  # Top 10
            total_iterations=len(evaluated),
            search_time_seconds=time.time() - start_time,
            recommendations=recommendations,
        )
    
    def _generate_candidates(self) -> List[Dict[str, Any]]:
        """Generate candidate configurations to evaluate."""
        candidates = []
        
        # All valid TP/PP/DP combinations
        for tp in [1, 2, 4, 8]:
            for pp in [1, 2, 4, 8]:
                for dp in [1, 2, 4, 8, 16, 32, 64]:
                    if tp * pp * dp == self.num_gpus:
                        candidates.append({
                            "tp": tp,
                            "pp": pp,
                            "dp": dp,
                        })
        
        return candidates
    
    def _estimate_memory(
        self,
        model_params_b: float,
        seq_length: int,
        hidden_size: int,
        num_layers: int,
        tp: int,
        pp: int,
        dp: int,
    ) -> float:
        """Estimate memory per GPU."""
        param_mem = (model_params_b * 1e9 * 2) / (tp * pp) / 1e9
        optimizer_mem = (model_params_b * 1e9 * 8) / (tp * pp * dp) / 1e9
        activation_mem = (seq_length * hidden_size * num_layers * 2 * 2) / (tp * pp) / 1e9
        return param_mem + optimizer_mem + activation_mem
    
    def _estimate_throughput(
        self,
        model_params_b: float,
        seq_length: int,
        tp: int,
        pp: int,
        dp: int,
    ) -> float:
        """Estimate throughput."""
        num_gpus = tp * pp * dp
        base_tflops = 1979
        peak = (base_tflops * 1e12 * num_gpus) / (6 * model_params_b * 1e9)
        
        # Efficiency factors
        efficiency = 0.45
        efficiency *= 0.95 ** (tp - 1)  # TP overhead
        efficiency *= (1 - (pp - 1) / (pp * 4)) if pp > 1 else 1  # PP bubble
        efficiency *= 0.98 if self.has_nvlink else 0.85  # Interconnect
        
        return peak * efficiency
    
    def _generate_recommendations(
        self,
        best: Dict[str, Any],
        all_configs: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate recommendations based on search results."""
        recommendations = []
        
        cfg = best["config"]
        
        if cfg["tp"] > 1:
            recommendations.append(f"TP={cfg['tp']} for model sharding across NVLink domain")
        
        if cfg["pp"] > 1:
            recommendations.append(f"PP={cfg['pp']} to fit model in memory (pipeline bubble ~{(cfg['pp']-1)/(cfg['pp']*4)*100:.0f}%)")
        
        if cfg["dp"] > 1:
            recommendations.append(f"DP={cfg['dp']} for throughput scaling")
        
        # Check for close alternatives
        if len(all_configs) > 1:
            second = all_configs[1]
            if second["score"] > best["score"] * 0.95:
                recommendations.append(
                    f"Alternative: TP{second['config']['tp']}_PP{second['config']['pp']}_DP{second['config']['dp']} "
                    f"has similar performance"
                )
        
        return recommendations


def find_max_batch_size(
    model_config: Dict[str, Any],
    hardware_config: Dict[str, Any],
    parallelism_config: Dict[str, Any],
    target_effective_batch: int = 1024,
) -> Dict[str, Any]:
    """
    Find maximum batch size that fits in GPU memory.
    """
    
    finder = BatchSizeFinder(
        gpu_memory_gb=hardware_config.get("gpu_memory_gb", 80),
        headroom_pct=0.1,
    )
    
    result = finder.find_max_batch_size(
        model_params_b=model_config.get("parameters_billions", 70),
        seq_length=model_config.get("max_sequence_length", 4096),
        hidden_size=model_config.get("hidden_size", 8192),
        num_layers=model_config.get("num_layers", 80),
        tp=parallelism_config.get("tensor_parallel", 1),
        pp=parallelism_config.get("pipeline_parallel", 1),
        dp=parallelism_config.get("data_parallel", 8),
        target_effective_batch=target_effective_batch,
    )
    
    return result.to_dict()


def optimize_gradient_accumulation(
    model_config: Dict[str, Any],
    hardware_config: Dict[str, Any],
    parallelism_config: Dict[str, Any],
    target_effective_batch: int = 1024,
) -> Dict[str, Any]:
    """
    Optimize gradient accumulation settings.
    """
    
    optimizer = GradientAccumulationOptimizer(
        gpu_memory_gb=hardware_config.get("gpu_memory_gb", 80),
    )
    
    result = optimizer.optimize(
        model_params_b=model_config.get("parameters_billions", 70),
        seq_length=model_config.get("max_sequence_length", 4096),
        hidden_size=model_config.get("hidden_size", 8192),
        num_layers=model_config.get("num_layers", 80),
        target_effective_batch=target_effective_batch,
        tp=parallelism_config.get("tensor_parallel", 1),
        pp=parallelism_config.get("pipeline_parallel", 1),
        dp=parallelism_config.get("data_parallel", 8),
    )
    
    return result.to_dict()


def auto_tune_config(
    model_config: Dict[str, Any],
    hardware_config: Dict[str, Any],
    target_batch_size: int = 1024,
    optimization_goal: str = "throughput",
) -> Dict[str, Any]:
    """
    Automatically tune parallelism configuration.
    """
    
    tuner = AutoTuner(
        gpu_memory_gb=hardware_config.get("gpu_memory_gb", 80),
        num_gpus=hardware_config.get("num_gpus", 8),
        has_nvlink=hardware_config.get("has_nvlink", True),
    )
    
    result = tuner.auto_tune(
        model_params_b=model_config.get("parameters_billions", 70),
        seq_length=model_config.get("max_sequence_length", 4096),
        hidden_size=model_config.get("hidden_size", 8192),
        num_layers=model_config.get("num_layers", 80),
        target_batch_size=target_batch_size,
        optimization_goal=optimization_goal,
    )
    
    return result.to_dict()



