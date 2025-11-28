#!/usr/bin/env python3
"""
Sharding Strategy Recommendations

Supports ZeRO (1/2/3), FSDP, and HSDP (Hybrid Sharded Data Parallel)
with memory and communication cost modeling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .model_analyzer import ModelArchitecture


class ShardingStrategy(Enum):
    """Data parallel sharding strategies."""
    
    # No sharding - full replication
    NO_SHARD = "no_shard"
    
    # DeepSpeed ZeRO stages
    ZERO_1 = "zero_1"  # Optimizer state sharding
    ZERO_2 = "zero_2"  # + Gradient sharding
    ZERO_3 = "zero_3"  # + Parameter sharding
    
    # PyTorch FSDP strategies
    FSDP_FULL = "fsdp_full"  # Full sharding (like ZeRO-3)
    FSDP_SHARD_GRAD_OP = "fsdp_shard_grad_op"  # Shard grad + optimizer (like ZeRO-2)
    FSDP_NO_SHARD = "fsdp_no_shard"  # No sharding (DDP)
    
    # Hybrid strategies
    HSDP = "hsdp"  # Hybrid: FSDP within node, replicated across nodes
    ZERO_PP = "zero_pp"  # ZeRO with Pipeline Parallelism


@dataclass
class ShardingConfig:
    """Configuration for a sharding strategy."""
    
    strategy: ShardingStrategy
    
    # Sharding dimensions
    shard_size: int  # Number of ranks to shard across
    replicate_size: int  # Number of replicas
    
    # FSDP specific
    sharding_factor: float = 1.0  # 1.0 = full shard, 0.5 = half
    backward_prefetch: bool = True
    forward_prefetch: bool = False
    limit_all_gathers: bool = False
    use_orig_params: bool = True
    
    # ZeRO specific
    zero_stage: int = 0
    offload_optimizer: bool = False
    offload_params: bool = False
    contiguous_gradients: bool = True
    overlap_comm: bool = True
    
    # CPU offload settings
    cpu_offload: bool = False
    nvme_offload: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "shard_size": self.shard_size,
            "replicate_size": self.replicate_size,
            "sharding_factor": self.sharding_factor,
            "zero_stage": self.zero_stage,
            "cpu_offload": self.cpu_offload,
            "nvme_offload": self.nvme_offload,
        }


@dataclass
class ShardingAnalysis:
    """Analysis of memory and communication for a sharding strategy."""
    
    # Memory breakdown per GPU (GB)
    params_memory_gb: float
    gradients_memory_gb: float
    optimizer_memory_gb: float
    activations_memory_gb: float
    total_memory_gb: float
    
    # Memory savings compared to no sharding
    memory_savings_factor: float
    
    # Communication costs
    all_gather_volume_gb: float  # For ZeRO-3/FSDP forward
    reduce_scatter_volume_gb: float  # For ZeRO-3/FSDP backward
    all_reduce_volume_gb: float  # For gradients (ZeRO-1/2)
    
    # Estimated overhead
    communication_overhead_pct: float
    
    # Tradeoffs
    pros: List[str]
    cons: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory": {
                "params_gb": self.params_memory_gb,
                "gradients_gb": self.gradients_memory_gb,
                "optimizer_gb": self.optimizer_memory_gb,
                "activations_gb": self.activations_memory_gb,
                "total_gb": self.total_memory_gb,
            },
            "memory_savings_factor": self.memory_savings_factor,
            "communication": {
                "all_gather_gb": self.all_gather_volume_gb,
                "reduce_scatter_gb": self.reduce_scatter_volume_gb,
                "all_reduce_gb": self.all_reduce_volume_gb,
                "overhead_pct": self.communication_overhead_pct,
            },
            "pros": self.pros,
            "cons": self.cons,
        }


@dataclass
class ShardingRecommendation:
    """Recommended sharding configuration."""
    
    config: ShardingConfig
    analysis: ShardingAnalysis
    score: float
    
    # Framework-specific configs
    deepspeed_config: Dict[str, Any] = field(default_factory=dict)
    fsdp_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "analysis": self.analysis.to_dict(),
            "score": self.score,
            "deepspeed_config": self.deepspeed_config,
            "fsdp_config": self.fsdp_config,
        }


class ShardingOptimizer:
    """Optimizes sharding strategy selection."""
    
    # Memory multipliers for optimizer states
    # AdamW: momentum (fp32) + variance (fp32) = 8 bytes per param
    # + master weights (fp32) = 4 bytes per param
    ADAMW_MEMORY_MULTIPLIER = 12  # bytes per parameter
    
    # BF16 model: 2 bytes per param
    BF16_PARAM_BYTES = 2
    FP32_PARAM_BYTES = 4
    
    def __init__(self):
        pass
    
    def analyze_strategy(
        self,
        model: ModelArchitecture,
        strategy: ShardingStrategy,
        dp_size: int,
        batch_size: int = 1,
        seq_length: int = 2048,
        mixed_precision: bool = True,
    ) -> ShardingAnalysis:
        """Analyze memory and communication for a sharding strategy."""
        
        total_params = model.total_params_billion * 1e9
        param_bytes = self.BF16_PARAM_BYTES if mixed_precision else self.FP32_PARAM_BYTES
        
        # Base memory calculations (no sharding)
        base_params_gb = total_params * param_bytes / 1e9
        base_grads_gb = total_params * param_bytes / 1e9
        base_opt_gb = total_params * self.ADAMW_MEMORY_MULTIPLIER / 1e9
        
        # Activation memory (rough estimate)
        act_per_layer = batch_size * seq_length * model.hidden_size * param_bytes
        base_act_gb = act_per_layer * model.num_layers / 1e9
        
        # Apply sharding based on strategy
        if strategy == ShardingStrategy.NO_SHARD:
            params_gb = base_params_gb
            grads_gb = base_grads_gb
            opt_gb = base_opt_gb
            all_reduce_gb = base_grads_gb  # Full gradient all-reduce
            all_gather_gb = 0
            reduce_scatter_gb = 0
            
        elif strategy in (ShardingStrategy.ZERO_1, ShardingStrategy.FSDP_NO_SHARD):
            # Only optimizer states sharded
            params_gb = base_params_gb
            grads_gb = base_grads_gb
            opt_gb = base_opt_gb / dp_size
            all_reduce_gb = base_grads_gb
            all_gather_gb = 0
            reduce_scatter_gb = 0
            
        elif strategy in (ShardingStrategy.ZERO_2, ShardingStrategy.FSDP_SHARD_GRAD_OP):
            # Optimizer + gradients sharded
            params_gb = base_params_gb
            grads_gb = base_grads_gb / dp_size
            opt_gb = base_opt_gb / dp_size
            all_reduce_gb = 0
            all_gather_gb = 0
            reduce_scatter_gb = base_grads_gb  # Reduce-scatter gradients
            
        elif strategy in (ShardingStrategy.ZERO_3, ShardingStrategy.FSDP_FULL):
            # Everything sharded
            params_gb = base_params_gb / dp_size
            grads_gb = base_grads_gb / dp_size
            opt_gb = base_opt_gb / dp_size
            all_reduce_gb = 0
            # All-gather params in forward, reduce-scatter grads in backward
            all_gather_gb = base_params_gb * 2  # Forward + backward
            reduce_scatter_gb = base_grads_gb
            
        elif strategy == ShardingStrategy.HSDP:
            # Hybrid: shard within node (8 GPUs), replicate across nodes
            intra_node_size = min(8, dp_size)
            inter_node_size = dp_size // intra_node_size
            
            # Params/grads sharded within node
            params_gb = base_params_gb / intra_node_size
            grads_gb = base_grads_gb / intra_node_size
            opt_gb = base_opt_gb / intra_node_size
            
            # All-gather within node (fast NVLink)
            all_gather_gb = base_params_gb * 2 / inter_node_size
            # Reduce-scatter within node
            reduce_scatter_gb = base_grads_gb / inter_node_size
            # All-reduce across nodes for replicated shards
            all_reduce_gb = base_grads_gb / intra_node_size
            
        else:
            # Default to no sharding
            params_gb = base_params_gb
            grads_gb = base_grads_gb
            opt_gb = base_opt_gb
            all_reduce_gb = base_grads_gb
            all_gather_gb = 0
            reduce_scatter_gb = 0
        
        total_gb = params_gb + grads_gb + opt_gb + base_act_gb
        base_total = base_params_gb + base_grads_gb + base_opt_gb + base_act_gb
        savings_factor = base_total / total_gb if total_gb > 0 else 1.0
        
        # Estimate communication overhead
        total_comm_gb = all_reduce_gb + all_gather_gb + reduce_scatter_gb
        # Rough estimate: 1GB takes ~10ms on 100 GB/s link
        comm_time_estimate = total_comm_gb * 10  # ms
        # Assume compute time scales with model size
        compute_time_estimate = model.total_params_billion * 2  # ms rough
        comm_overhead = comm_time_estimate / (comm_time_estimate + compute_time_estimate) * 100
        
        # Build pros/cons
        pros, cons = self._get_strategy_tradeoffs(strategy, dp_size)
        
        return ShardingAnalysis(
            params_memory_gb=params_gb,
            gradients_memory_gb=grads_gb,
            optimizer_memory_gb=opt_gb,
            activations_memory_gb=base_act_gb,
            total_memory_gb=total_gb,
            memory_savings_factor=savings_factor,
            all_gather_volume_gb=all_gather_gb,
            reduce_scatter_volume_gb=reduce_scatter_gb,
            all_reduce_volume_gb=all_reduce_gb,
            communication_overhead_pct=comm_overhead,
            pros=pros,
            cons=cons,
        )
    
    def _get_strategy_tradeoffs(
        self, strategy: ShardingStrategy, dp_size: int
    ) -> Tuple[List[str], List[str]]:
        """Get pros and cons for a strategy."""
        
        if strategy == ShardingStrategy.NO_SHARD:
            return (
                ["Simplest implementation", "No communication overhead"],
                ["Highest memory usage", "Limited by single GPU memory"],
            )
        
        elif strategy == ShardingStrategy.ZERO_1:
            return (
                ["Low overhead", "~33% memory savings", "Simple gradient sync"],
                ["Still need full model copy", "Limited memory savings"],
            )
        
        elif strategy == ShardingStrategy.ZERO_2:
            return (
                ["Good memory savings (~50%)", "Moderate overhead"],
                ["Still need full params", "Reduce-scatter adds latency"],
            )
        
        elif strategy == ShardingStrategy.ZERO_3:
            return (
                [f"Max memory savings ({dp_size}x)", "Can train huge models"],
                ["High communication volume", "All-gather in forward pass"],
            )
        
        elif strategy == ShardingStrategy.FSDP_FULL:
            return (
                ["Native PyTorch support", f"~{dp_size}x memory reduction", "Good overlap"],
                ["Requires careful module wrapping", "Communication overhead"],
            )
        
        elif strategy == ShardingStrategy.HSDP:
            return (
                ["Best of both worlds", "Intra-node NVLink, inter-node IB"],
                ["More complex setup", "Requires multi-node"],
            )
        
        return ([], [])
    
    def recommend(
        self,
        model: ModelArchitecture,
        dp_size: int,
        gpu_memory_gb: float,
        batch_size: int = 1,
        seq_length: int = 2048,
        num_nodes: int = 1,
        gpus_per_node: int = 8,
        prefer_throughput: bool = True,
    ) -> List[ShardingRecommendation]:
        """Recommend sharding strategies for a model.
        
        Args:
            model: Model architecture
            dp_size: Data parallel size
            gpu_memory_gb: Available GPU memory
            batch_size: Micro-batch size
            seq_length: Sequence length
            num_nodes: Number of nodes
            gpus_per_node: GPUs per node
            prefer_throughput: Prefer throughput over memory efficiency
            
        Returns:
            Sorted list of recommendations
        """
        recommendations = []
        
        strategies_to_try = [
            ShardingStrategy.NO_SHARD,
            ShardingStrategy.ZERO_1,
            ShardingStrategy.ZERO_2,
            ShardingStrategy.ZERO_3,
            ShardingStrategy.FSDP_FULL,
        ]
        
        # Add HSDP if multi-node
        if num_nodes > 1:
            strategies_to_try.append(ShardingStrategy.HSDP)
        
        for strategy in strategies_to_try:
            analysis = self.analyze_strategy(
                model, strategy, dp_size, batch_size, seq_length
            )
            
            # Skip if doesn't fit
            if analysis.total_memory_gb > gpu_memory_gb * 0.9:  # 90% headroom
                continue
            
            # Calculate score
            score = self._score_strategy(
                analysis, gpu_memory_gb, prefer_throughput, num_nodes
            )
            
            # Generate framework configs
            ds_config = self._generate_deepspeed_config(strategy, model, dp_size)
            fsdp_config = self._generate_fsdp_config(strategy, dp_size, gpus_per_node)
            
            config = ShardingConfig(
                strategy=strategy,
                shard_size=dp_size if strategy in (
                    ShardingStrategy.ZERO_3, ShardingStrategy.FSDP_FULL
                ) else 1,
                replicate_size=1 if strategy in (
                    ShardingStrategy.ZERO_3, ShardingStrategy.FSDP_FULL
                ) else dp_size,
                zero_stage={
                    ShardingStrategy.ZERO_1: 1,
                    ShardingStrategy.ZERO_2: 2,
                    ShardingStrategy.ZERO_3: 3,
                }.get(strategy, 0),
            )
            
            recommendations.append(ShardingRecommendation(
                config=config,
                analysis=analysis,
                score=score,
                deepspeed_config=ds_config,
                fsdp_config=fsdp_config,
            ))
        
        # Sort by score
        recommendations.sort(key=lambda r: r.score, reverse=True)
        
        return recommendations
    
    def _score_strategy(
        self,
        analysis: ShardingAnalysis,
        gpu_memory_gb: float,
        prefer_throughput: bool,
        num_nodes: int,
    ) -> float:
        """Score a strategy (0-100)."""
        score = 50.0
        
        # Memory efficiency (0-30 points)
        memory_usage_pct = analysis.total_memory_gb / gpu_memory_gb
        if memory_usage_pct < 0.5:
            score += 30
        elif memory_usage_pct < 0.7:
            score += 20
        elif memory_usage_pct < 0.85:
            score += 10
        
        # Communication efficiency (0-30 points)
        if analysis.communication_overhead_pct < 5:
            score += 30
        elif analysis.communication_overhead_pct < 10:
            score += 20
        elif analysis.communication_overhead_pct < 20:
            score += 10
        
        # Throughput preference (0-20 points)
        if prefer_throughput:
            # Lower savings = less communication = higher throughput
            if analysis.memory_savings_factor < 2:
                score += 20
            elif analysis.memory_savings_factor < 4:
                score += 10
        else:
            # Memory preference
            if analysis.memory_savings_factor > 4:
                score += 20
            elif analysis.memory_savings_factor > 2:
                score += 10
        
        # Multi-node bonus for HSDP
        if num_nodes > 1 and analysis.all_reduce_volume_gb < analysis.all_gather_volume_gb:
            score += 10
        
        return min(100, score)
    
    def _generate_deepspeed_config(
        self,
        strategy: ShardingStrategy,
        model: ModelArchitecture,
        dp_size: int,
    ) -> Dict[str, Any]:
        """Generate DeepSpeed config for strategy."""
        
        if strategy not in (
            ShardingStrategy.ZERO_1, ShardingStrategy.ZERO_2, ShardingStrategy.ZERO_3
        ):
            return {}
        
        zero_stage = {
            ShardingStrategy.ZERO_1: 1,
            ShardingStrategy.ZERO_2: 2,
            ShardingStrategy.ZERO_3: 3,
        }[strategy]
        
        config = {
            "zero_optimization": {
                "stage": zero_stage,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 5e8,
                "allgather_bucket_size": 5e8,
            },
            "bf16": {
                "enabled": True,
            },
            "gradient_accumulation_steps": 1,
            "train_micro_batch_size_per_gpu": 1,
        }
        
        if zero_stage == 3:
            config["zero_optimization"].update({
                "stage3_prefetch_bucket_size": 5e8,
                "stage3_param_persistence_threshold": 1e6,
                "stage3_gather_16bit_weights_on_model_save": True,
            })
        
        return config
    
    def _generate_fsdp_config(
        self,
        strategy: ShardingStrategy,
        dp_size: int,
        gpus_per_node: int,
    ) -> Dict[str, Any]:
        """Generate PyTorch FSDP config for strategy."""
        
        if strategy not in (
            ShardingStrategy.FSDP_FULL, ShardingStrategy.FSDP_SHARD_GRAD_OP,
            ShardingStrategy.FSDP_NO_SHARD, ShardingStrategy.HSDP
        ):
            return {}
        
        from_str = {
            ShardingStrategy.FSDP_FULL: "FULL_SHARD",
            ShardingStrategy.FSDP_SHARD_GRAD_OP: "SHARD_GRAD_OP",
            ShardingStrategy.FSDP_NO_SHARD: "NO_SHARD",
            ShardingStrategy.HSDP: "HYBRID_SHARD",
        }
        
        config = {
            "sharding_strategy": from_str.get(strategy, "FULL_SHARD"),
            "mixed_precision": {
                "param_dtype": "torch.bfloat16",
                "reduce_dtype": "torch.float32",
                "buffer_dtype": "torch.bfloat16",
            },
            "backward_prefetch": "BACKWARD_PRE",
            "forward_prefetch": False,
            "use_orig_params": True,
            "limit_all_gathers": True,
        }
        
        if strategy == ShardingStrategy.HSDP:
            # Configure mesh for HSDP
            num_nodes = dp_size // gpus_per_node
            config["device_mesh"] = {
                "shape": [num_nodes, gpus_per_node],
                "names": ["replicate", "shard"],
            }
        
        return config
    
    def format_recommendations(
        self, recommendations: List[ShardingRecommendation]
    ) -> str:
        """Format recommendations as human-readable text."""
        if not recommendations:
            return "No valid sharding strategies found."
        
        lines = [
            "=" * 80,
            "SHARDING STRATEGY RECOMMENDATIONS",
            "=" * 80,
        ]
        
        for i, rec in enumerate(recommendations, 1):
            a = rec.analysis
            c = rec.config
            
            lines.extend([
                "",
                f"{'─' * 80}",
                f"OPTION {i}: {c.strategy.value.upper()}  [Score: {rec.score:.0f}/100]",
                f"{'─' * 80}",
                "",
                f"  Memory per GPU: {a.total_memory_gb:.1f} GB",
                f"    - Parameters: {a.params_memory_gb:.1f} GB",
                f"    - Gradients: {a.gradients_memory_gb:.1f} GB",
                f"    - Optimizer: {a.optimizer_memory_gb:.1f} GB",
                f"    - Activations: {a.activations_memory_gb:.1f} GB",
                "",
                f"  Memory savings: {a.memory_savings_factor:.1f}x",
                f"  Communication overhead: {a.communication_overhead_pct:.1f}%",
            ])
            
            if a.pros:
                lines.append("")
                lines.append("  ✓ Pros:")
                for pro in a.pros:
                    lines.append(f"    • {pro}")
            
            if a.cons:
                lines.append("")
                lines.append("  ⚠ Cons:")
                for con in a.cons:
                    lines.append(f"    • {con}")
        
        lines.extend(["", "=" * 80])
        
        return "\n".join(lines)



