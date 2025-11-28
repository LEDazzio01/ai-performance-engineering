#!/usr/bin/env python3
"""
Sharding Strategy Recommendations

Provides recommendations for ZeRO, FSDP, and Hybrid Sharding (HSDP)
based on model size, memory constraints, and cluster topology.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .model_analyzer import ModelArchitecture


class ShardingStrategy(Enum):
    """Available sharding strategies."""
    NONE = "none"                    # No sharding (replicated DP)
    ZERO_1 = "zero_1"               # Optimizer state sharding
    ZERO_2 = "zero_2"               # + Gradient sharding
    ZERO_3 = "zero_3"               # + Parameter sharding
    FSDP_FULL = "fsdp_full"         # PyTorch FSDP (like ZeRO-3)
    FSDP_GRAD_OP = "fsdp_grad_op"   # FSDP with gradient + optimizer sharding
    FSDP_NO_SHARD = "fsdp_no_shard" # FSDP wrapper without sharding
    HSDP = "hsdp"                    # Hybrid Sharding (FSDP within node, DDP across)


class CPUOffloadStrategy(Enum):
    """CPU offload strategies."""
    NONE = "none"
    OPTIMIZER = "optimizer"         # Offload optimizer states
    PARAMS = "params"               # Offload parameters (ZeRO-Infinity)
    FULL = "full"                   # Offload everything possible


@dataclass
class ShardingConfig:
    """Configuration for a sharding strategy."""
    
    strategy: ShardingStrategy
    
    # FSDP-specific options
    sharding_group_size: int = 0       # 0 = shard across all ranks
    use_orig_params: bool = True       # PyTorch 2.0+ original params
    limit_all_gathers: bool = True     # Limit concurrent all-gathers
    forward_prefetch: bool = True      # Prefetch next layer during forward
    backward_prefetch: str = "pre"     # "pre" or "post" backward prefetch
    
    # CPU offload
    cpu_offload: CPUOffloadStrategy = CPUOffloadStrategy.NONE
    
    # Mixed precision
    param_dtype: str = "bf16"
    reduce_dtype: str = "fp32"
    buffer_dtype: str = "bf16"
    
    # Checkpointing
    activation_checkpointing: bool = False
    checkpoint_every_n_layers: int = 1
    
    # Advanced options
    sync_module_states: bool = True
    use_flash_attention: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "sharding_group_size": self.sharding_group_size,
            "use_orig_params": self.use_orig_params,
            "limit_all_gathers": self.limit_all_gathers,
            "forward_prefetch": self.forward_prefetch,
            "backward_prefetch": self.backward_prefetch,
            "cpu_offload": self.cpu_offload.value,
            "param_dtype": self.param_dtype,
            "reduce_dtype": self.reduce_dtype,
            "buffer_dtype": self.buffer_dtype,
            "activation_checkpointing": self.activation_checkpointing,
            "checkpoint_every_n_layers": self.checkpoint_every_n_layers,
        }


@dataclass
class ShardingAnalysis:
    """Analysis of memory usage with sharding."""
    
    strategy: ShardingStrategy
    
    # Memory breakdown per GPU (in GB)
    model_params_gb: float
    optimizer_states_gb: float
    gradients_gb: float
    activations_gb: float
    buffers_gb: float
    total_memory_gb: float
    
    # Memory savings
    memory_reduction_factor: float  # vs no sharding
    
    # Communication overhead
    all_gather_volume_gb: float     # Volume per step
    reduce_scatter_volume_gb: float
    total_comm_volume_gb: float
    estimated_comm_time_ms: float
    
    # Efficiency metrics
    memory_efficiency: float        # 0-1
    communication_overhead: float   # 0-1 (fraction of step time)
    
    fits_in_memory: bool
    memory_headroom_gb: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "model_params_gb": self.model_params_gb,
            "optimizer_states_gb": self.optimizer_states_gb,
            "gradients_gb": self.gradients_gb,
            "activations_gb": self.activations_gb,
            "buffers_gb": self.buffers_gb,
            "total_memory_gb": self.total_memory_gb,
            "memory_reduction_factor": self.memory_reduction_factor,
            "all_gather_volume_gb": self.all_gather_volume_gb,
            "reduce_scatter_volume_gb": self.reduce_scatter_volume_gb,
            "total_comm_volume_gb": self.total_comm_volume_gb,
            "estimated_comm_time_ms": self.estimated_comm_time_ms,
            "memory_efficiency": self.memory_efficiency,
            "communication_overhead": self.communication_overhead,
            "fits_in_memory": self.fits_in_memory,
            "memory_headroom_gb": self.memory_headroom_gb,
        }


@dataclass
class ShardingRecommendation:
    """A sharding strategy recommendation."""
    
    config: ShardingConfig
    analysis: ShardingAnalysis
    score: float
    rationale: List[str]
    warnings: List[str]
    framework_config: Dict[str, Any]  # Ready-to-use config for PyTorch/DeepSpeed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "analysis": self.analysis.to_dict(),
            "score": self.score,
            "rationale": self.rationale,
            "warnings": self.warnings,
            "framework_config": self.framework_config,
        }


class ShardingOptimizer:
    """Optimizes sharding strategy based on model and cluster."""
    
    # Memory multipliers for different sharding strategies
    SHARDING_MEMORY_FACTORS = {
        ShardingStrategy.NONE: {
            "params": 1.0,      # Full model on each GPU
            "optimizer": 1.0,   # Full optimizer states
            "gradients": 1.0,   # Full gradients
        },
        ShardingStrategy.ZERO_1: {
            "params": 1.0,
            "optimizer": 0.0,   # Sharded across DP ranks (1/N)
            "gradients": 1.0,
        },
        ShardingStrategy.ZERO_2: {
            "params": 1.0,
            "optimizer": 0.0,
            "gradients": 0.0,   # Sharded
        },
        ShardingStrategy.ZERO_3: {
            "params": 0.0,      # Sharded
            "optimizer": 0.0,
            "gradients": 0.0,
        },
        ShardingStrategy.FSDP_FULL: {
            "params": 0.0,
            "optimizer": 0.0,
            "gradients": 0.0,
        },
        ShardingStrategy.FSDP_GRAD_OP: {
            "params": 1.0,
            "optimizer": 0.0,
            "gradients": 0.0,
        },
        ShardingStrategy.FSDP_NO_SHARD: {
            "params": 1.0,
            "optimizer": 1.0,
            "gradients": 1.0,
        },
    }
    
    def __init__(self):
        pass
    
    def analyze_sharding(
        self,
        model: ModelArchitecture,
        strategy: ShardingStrategy,
        world_size: int,
        gpu_memory_gb: float,
        batch_size: int,
        seq_length: int,
        bandwidth_gbps: float = 400,
        activation_checkpointing: bool = False,
        sharding_group_size: int = 0,
    ) -> ShardingAnalysis:
        """Analyze memory usage for a sharding strategy."""
        
        # Handle HSDP (uses sharding_group_size)
        effective_shard_size = world_size if sharding_group_size == 0 else sharding_group_size
        
        # Base memory requirements (FP32 training with BF16 params)
        param_bytes = model.total_params_billion * 1e9 * 2  # BF16 params
        optimizer_bytes = model.total_params_billion * 1e9 * 8  # FP32 momentum + variance
        gradient_bytes = model.total_params_billion * 1e9 * 4  # FP32 gradients
        
        # Get sharding factors
        if strategy == ShardingStrategy.HSDP:
            factors = self.SHARDING_MEMORY_FACTORS[ShardingStrategy.FSDP_FULL]
        else:
            factors = self.SHARDING_MEMORY_FACTORS.get(
                strategy, self.SHARDING_MEMORY_FACTORS[ShardingStrategy.NONE]
            )
        
        # Apply sharding (divide by effective shard size where applicable)
        params_per_gpu = param_bytes * (factors["params"] + (1 - factors["params"]) / effective_shard_size)
        optimizer_per_gpu = optimizer_bytes * (factors["optimizer"] + (1 - factors["optimizer"]) / effective_shard_size)
        gradients_per_gpu = gradient_bytes * (factors["gradients"] + (1 - factors["gradients"]) / effective_shard_size)
        
        # Activation memory (rough estimate)
        activation_bytes = batch_size * seq_length * model.hidden_size * model.num_layers * 4
        if activation_checkpointing:
            activation_bytes *= 0.3  # ~70% reduction with checkpointing
        
        # Buffer memory (communication buffers, etc.)
        buffer_bytes = min(param_bytes * 0.1, 2 * 1024**3)  # ~10% or 2GB max
        
        # Convert to GB
        params_gb = params_per_gpu / (1024**3)
        optimizer_gb = optimizer_per_gpu / (1024**3)
        gradients_gb = gradients_per_gpu / (1024**3)
        activations_gb = activation_bytes / (1024**3)
        buffers_gb = buffer_bytes / (1024**3)
        
        total_gb = params_gb + optimizer_gb + gradients_gb + activations_gb + buffers_gb
        
        # Memory reduction factor
        no_shard_total = (param_bytes + optimizer_bytes + gradient_bytes + activation_bytes) / (1024**3)
        reduction_factor = no_shard_total / total_gb if total_gb > 0 else 1.0
        
        # Communication analysis for ZeRO-3/FSDP
        all_gather_volume = 0.0
        reduce_scatter_volume = 0.0
        
        if strategy in (ShardingStrategy.ZERO_3, ShardingStrategy.FSDP_FULL, ShardingStrategy.HSDP):
            # All-gather params before each layer (forward + backward = 2x)
            all_gather_volume = param_bytes * 2 / (1024**3)
            # Reduce-scatter gradients after backward
            reduce_scatter_volume = gradient_bytes / (1024**3)
        elif strategy == ShardingStrategy.ZERO_2:
            # Only reduce-scatter gradients
            reduce_scatter_volume = gradient_bytes / (1024**3)
        elif strategy == ShardingStrategy.ZERO_1:
            # All-reduce optimizer step
            reduce_scatter_volume = optimizer_bytes / effective_shard_size / (1024**3)
        
        total_comm = all_gather_volume + reduce_scatter_volume
        
        # Estimate communication time
        bandwidth_GBps = bandwidth_gbps / 8
        comm_time_ms = total_comm / bandwidth_GBps * 1000 if bandwidth_GBps > 0 else 0
        
        # Compute approximate step time for efficiency calculation
        # Rough: 2 PFLOPS for B200, ~1 PFLOPS for H100
        gpu_tflops = 2000  # Assume B200-class
        flops_per_step = model.total_params_billion * 1e9 * 6 * batch_size * seq_length  # 6 for fwd+bwd
        compute_time_ms = flops_per_step / (gpu_tflops * 1e12) * 1000
        
        total_time_ms = compute_time_ms + comm_time_ms
        comm_overhead = comm_time_ms / total_time_ms if total_time_ms > 0 else 0
        
        headroom = gpu_memory_gb - total_gb
        fits = headroom > 2.0
        
        return ShardingAnalysis(
            strategy=strategy,
            model_params_gb=params_gb,
            optimizer_states_gb=optimizer_gb,
            gradients_gb=gradients_gb,
            activations_gb=activations_gb,
            buffers_gb=buffers_gb,
            total_memory_gb=total_gb,
            memory_reduction_factor=reduction_factor,
            all_gather_volume_gb=all_gather_volume,
            reduce_scatter_volume_gb=reduce_scatter_volume,
            total_comm_volume_gb=total_comm,
            estimated_comm_time_ms=comm_time_ms,
            memory_efficiency=min(1.0, (total_gb / gpu_memory_gb) if gpu_memory_gb > 0 else 0),
            communication_overhead=comm_overhead,
            fits_in_memory=fits,
            memory_headroom_gb=headroom,
        )
    
    def recommend(
        self,
        model: ModelArchitecture,
        world_size: int,
        gpu_memory_gb: float,
        batch_size: int = 1,
        seq_length: int = 2048,
        bandwidth_gbps: float = 400,
        prefer_simplicity: bool = False,
        gpus_per_node: int = 8,
    ) -> List[ShardingRecommendation]:
        """Generate sharding recommendations."""
        
        recommendations = []
        
        strategies_to_try = [
            (ShardingStrategy.NONE, 0, False),
            (ShardingStrategy.ZERO_1, 0, False),
            (ShardingStrategy.ZERO_2, 0, False),
            (ShardingStrategy.ZERO_3, 0, False),
            (ShardingStrategy.FSDP_FULL, 0, False),
            (ShardingStrategy.FSDP_GRAD_OP, 0, False),
            (ShardingStrategy.FSDP_FULL, 0, True),  # With activation checkpointing
        ]
        
        # Add HSDP if multi-node
        if world_size > gpus_per_node:
            strategies_to_try.extend([
                (ShardingStrategy.HSDP, gpus_per_node, False),  # Shard within node
                (ShardingStrategy.HSDP, gpus_per_node, True),   # With checkpointing
            ])
        
        for strategy, shard_group, use_checkpoint in strategies_to_try:
            analysis = self.analyze_sharding(
                model=model,
                strategy=strategy,
                world_size=world_size,
                gpu_memory_gb=gpu_memory_gb,
                batch_size=batch_size,
                seq_length=seq_length,
                bandwidth_gbps=bandwidth_gbps,
                activation_checkpointing=use_checkpoint,
                sharding_group_size=shard_group,
            )
            
            score, rationale, warnings = self._score_analysis(
                analysis, strategy, model, prefer_simplicity, shard_group
            )
            
            config = ShardingConfig(
                strategy=strategy,
                sharding_group_size=shard_group,
                activation_checkpointing=use_checkpoint,
            )
            
            framework_config = self._generate_framework_config(
                strategy, config, model, world_size
            )
            
            recommendations.append(ShardingRecommendation(
                config=config,
                analysis=analysis,
                score=score,
                rationale=rationale,
                warnings=warnings,
                framework_config=framework_config,
            ))
        
        # Sort by score, then filter to those that fit
        recommendations.sort(key=lambda r: r.score, reverse=True)
        
        # Keep top recommendations that fit
        fitting = [r for r in recommendations if r.analysis.fits_in_memory]
        if not fitting:
            # Return best non-fitting with warning
            return recommendations[:3]
        
        return fitting[:5]
    
    def _score_analysis(
        self,
        analysis: ShardingAnalysis,
        strategy: ShardingStrategy,
        model: ModelArchitecture,
        prefer_simplicity: bool,
        shard_group: int,
    ) -> Tuple[float, List[str], List[str]]:
        """Score a sharding analysis."""
        
        score = 50.0
        rationale = []
        warnings = []
        
        # Memory fit (critical)
        if analysis.fits_in_memory:
            score += 25
            if analysis.memory_headroom_gb > 20:
                score += 10
                rationale.append(f"Excellent memory headroom ({analysis.memory_headroom_gb:.1f} GB)")
            elif analysis.memory_headroom_gb > 10:
                score += 5
                rationale.append(f"Good memory headroom ({analysis.memory_headroom_gb:.1f} GB)")
            else:
                warnings.append(f"Tight memory ({analysis.memory_headroom_gb:.1f} GB headroom)")
        else:
            score -= 30
            warnings.append(f"Does not fit in memory (needs {-analysis.memory_headroom_gb:.1f} GB more)")
        
        # Communication overhead
        if analysis.communication_overhead < 0.1:
            score += 10
            rationale.append("Low communication overhead")
        elif analysis.communication_overhead < 0.2:
            score += 5
        elif analysis.communication_overhead > 0.3:
            score -= 5
            warnings.append(f"High communication overhead ({analysis.communication_overhead:.1%})")
        
        # Simplicity bonus
        if prefer_simplicity:
            if strategy == ShardingStrategy.NONE:
                score += 15
                rationale.append("Simplest configuration (no sharding)")
            elif strategy in (ShardingStrategy.ZERO_1, ShardingStrategy.FSDP_NO_SHARD):
                score += 10
                rationale.append("Simple sharding (optimizer only)")
            elif strategy == ShardingStrategy.HSDP:
                score += 5
                rationale.append("Hybrid sharding balances communication and memory")
        
        # HSDP bonus for multi-node
        if strategy == ShardingStrategy.HSDP and shard_group > 0:
            score += 5
            rationale.append(f"HSDP: Full sharding within {shard_group} GPUs, replicated across nodes")
        
        # ZeRO-3/FSDP for large models
        if model.total_params_billion > 30:
            if strategy in (ShardingStrategy.ZERO_3, ShardingStrategy.FSDP_FULL):
                score += 5
                rationale.append("Full sharding recommended for large model")
        
        return score, rationale, warnings
    
    def _generate_framework_config(
        self,
        strategy: ShardingStrategy,
        config: ShardingConfig,
        model: ModelArchitecture,
        world_size: int,
    ) -> Dict[str, Any]:
        """Generate ready-to-use framework configuration."""
        
        if strategy in (ShardingStrategy.ZERO_1, ShardingStrategy.ZERO_2, ShardingStrategy.ZERO_3):
            # DeepSpeed configuration
            return self._generate_deepspeed_config(strategy, config, model, world_size)
        else:
            # PyTorch FSDP configuration
            return self._generate_fsdp_config(strategy, config, model, world_size)
    
    def _generate_deepspeed_config(
        self,
        strategy: ShardingStrategy,
        config: ShardingConfig,
        model: ModelArchitecture,
        world_size: int,
    ) -> Dict[str, Any]:
        """Generate DeepSpeed ZeRO configuration."""
        
        zero_stage = {
            ShardingStrategy.ZERO_1: 1,
            ShardingStrategy.ZERO_2: 2,
            ShardingStrategy.ZERO_3: 3,
        }.get(strategy, 0)
        
        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
            },
            "bf16": {
                "enabled": config.param_dtype == "bf16",
            },
            "zero_optimization": {
                "stage": zero_stage,
                "offload_optimizer": {
                    "device": "cpu" if config.cpu_offload in (CPUOffloadStrategy.OPTIMIZER, CPUOffloadStrategy.FULL) else "none",
                },
                "offload_param": {
                    "device": "cpu" if config.cpu_offload in (CPUOffloadStrategy.PARAMS, CPUOffloadStrategy.FULL) else "none",
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 5e8,
                "stage3_prefetch_bucket_size": 5e8,
                "stage3_param_persistence_threshold": 1e6,
            },
            "activation_checkpointing": {
                "partition_activations": config.activation_checkpointing,
                "cpu_checkpointing": False,
                "contiguous_memory_optimization": True,
                "number_checkpoints": model.num_layers // config.checkpoint_every_n_layers if config.activation_checkpointing else 0,
            },
            "wall_clock_breakdown": False,
        }
        
        return {"deepspeed": ds_config}
    
    def _generate_fsdp_config(
        self,
        strategy: ShardingStrategy,
        config: ShardingConfig,
        model: ModelArchitecture,
        world_size: int,
    ) -> Dict[str, Any]:
        """Generate PyTorch FSDP configuration."""
        
        from torch.distributed.fsdp import ShardingStrategy as FSDPStrategy
        
        sharding_strategy_map = {
            ShardingStrategy.FSDP_FULL: "FULL_SHARD",
            ShardingStrategy.FSDP_GRAD_OP: "SHARD_GRAD_OP",
            ShardingStrategy.FSDP_NO_SHARD: "NO_SHARD",
            ShardingStrategy.HSDP: "HYBRID_SHARD",
        }
        
        fsdp_config = {
            "sharding_strategy": sharding_strategy_map.get(strategy, "FULL_SHARD"),
            "cpu_offload": config.cpu_offload != CPUOffloadStrategy.NONE,
            "mixed_precision": {
                "param_dtype": config.param_dtype,
                "reduce_dtype": config.reduce_dtype,
                "buffer_dtype": config.buffer_dtype,
            },
            "use_orig_params": config.use_orig_params,
            "limit_all_gathers": config.limit_all_gathers,
            "forward_prefetch": config.forward_prefetch,
            "backward_prefetch": config.backward_prefetch.upper(),
            "sync_module_states": config.sync_module_states,
        }
        
        if strategy == ShardingStrategy.HSDP and config.sharding_group_size > 0:
            fsdp_config["sharding_group_size"] = config.sharding_group_size
        
        # Activation checkpointing config
        checkpoint_config = None
        if config.activation_checkpointing:
            checkpoint_config = {
                "checkpoint_impl": "reentrant",
                "checkpoint_every_n_layers": config.checkpoint_every_n_layers,
            }
        
        return {
            "fsdp": fsdp_config,
            "activation_checkpointing": checkpoint_config,
        }
    
    def format_recommendations(self, recommendations: List[ShardingRecommendation]) -> str:
        """Format recommendations as a human-readable report."""
        
        lines = [
            "=" * 80,
            "SHARDING STRATEGY RECOMMENDATIONS",
            "=" * 80,
        ]
        
        for i, rec in enumerate(recommendations, 1):
            a = rec.analysis
            c = rec.config
            
            strategy_name = c.strategy.value.upper()
            if c.activation_checkpointing:
                strategy_name += " + Checkpointing"
            if c.sharding_group_size > 0:
                strategy_name += f" (group={c.sharding_group_size})"
            
            lines.extend([
                "",
                f"{'─' * 80}",
                f"OPTION {i}: {strategy_name}  [Score: {rec.score:.0f}/100]",
                f"{'─' * 80}",
                "",
                f"  Memory per GPU:",
                f"    Parameters:      {a.model_params_gb:6.1f} GB",
                f"    Optimizer:       {a.optimizer_states_gb:6.1f} GB",
                f"    Gradients:       {a.gradients_gb:6.1f} GB",
                f"    Activations:     {a.activations_gb:6.1f} GB",
                f"    Total:           {a.total_memory_gb:6.1f} GB",
                f"    Headroom:        {a.memory_headroom_gb:6.1f} GB {'✓' if a.fits_in_memory else '✗'}",
                "",
                f"  Communication:",
                f"    All-gather:      {a.all_gather_volume_gb:6.2f} GB/step",
                f"    Reduce-scatter:  {a.reduce_scatter_volume_gb:6.2f} GB/step",
                f"    Overhead:        {a.communication_overhead:.1%}",
                "",
                f"  Memory reduction: {a.memory_reduction_factor:.1f}x vs no sharding",
            ])
            
            if rec.rationale:
                lines.append("")
                lines.append("  ✓ Rationale:")
                for r in rec.rationale:
                    lines.append(f"    • {r}")
            
            if rec.warnings:
                lines.append("")
                lines.append("  ⚠ Warnings:")
                for w in rec.warnings:
                    lines.append(f"    • {w}")
        
        lines.extend(["", "=" * 80])
        return "\n".join(lines)


if __name__ == "__main__":
    from .model_analyzer import ModelAnalyzer
    
    analyzer = ModelAnalyzer()
    model = analyzer.analyze("llama-3.1-70b")
    
    optimizer = ShardingOptimizer()
    recommendations = optimizer.recommend(
        model=model,
        world_size=8,
        gpu_memory_gb=80,
        batch_size=4,
        seq_length=4096,
        bandwidth_gbps=900,
        gpus_per_node=8,
    )
    
    print(optimizer.format_recommendations(recommendations))

