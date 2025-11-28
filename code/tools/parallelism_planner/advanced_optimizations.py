"""
Advanced Optimizations Module for Parallelism Planner

Provides compound techniques and advanced optimization recommendations:
- Mixed precision strategies (FP8, BF16, FP16, dynamic scaling)
- Gradient checkpointing (selective, block-wise, memory-adaptive)
- Communication-computation overlap
- Memory-efficient optimizers (8-bit Adam, Adafactor, CAME)
- Pipeline scheduling (1F1B, Interleaved, Zero-bubble)
- Kernel fusion recommendations
- Flash Attention integration
- Async tensor parallelism
- Speculative decoding optimization
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import math


class PrecisionMode(Enum):
    """Precision modes for training/inference."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"  # For forward
    FP8_E5M2 = "fp8_e5m2"  # For backward
    TF32 = "tf32"
    MIXED_FP16 = "mixed_fp16"
    MIXED_BF16 = "mixed_bf16"
    MIXED_FP8 = "mixed_fp8"


class CheckpointingStrategy(Enum):
    """Activation checkpointing strategies."""
    NONE = "none"
    FULL = "full"                    # Checkpoint every layer
    SELECTIVE = "selective"          # Checkpoint expensive ops only
    BLOCK_WISE = "block_wise"        # Checkpoint every N layers
    MEMORY_ADAPTIVE = "memory_adaptive"  # Adapt based on available memory
    OFFLOAD = "offload"              # Offload to CPU


class PipelineSchedule(Enum):
    """Pipeline parallelism scheduling strategies."""
    GPIPE = "gpipe"                  # All forward then all backward
    ONE_F_ONE_B = "1f1b"             # Interleaved micro-batches
    INTERLEAVED = "interleaved"      # Virtual pipeline stages
    ZERO_BUBBLE = "zero_bubble"      # Minimize pipeline bubbles
    BREADTH_FIRST = "breadth_first"  # Megatron-style


class OverlapStrategy(Enum):
    """Communication-computation overlap strategies."""
    NONE = "none"
    ASYNC_ALLREDUCE = "async_allreduce"     # Overlap gradient allreduce with backward
    BUCKETED = "bucketed"                    # Bucket gradients for efficiency
    PIPELINING = "pipelining"               # Pipeline communication
    DOUBLE_BUFFER = "double_buffer"         # Double buffering for overlap


@dataclass
class PrecisionRecommendation:
    """Recommendation for precision settings."""
    mode: PrecisionMode
    compute_dtype: str
    param_dtype: str
    grad_dtype: str
    memory_savings_pct: float
    throughput_boost_pct: float
    accuracy_impact: str  # "none", "minimal", "moderate", "significant"
    hardware_requirements: List[str]
    config: Dict[str, Any]
    rationale: str


@dataclass
class CheckpointingRecommendation:
    """Recommendation for activation checkpointing."""
    strategy: CheckpointingStrategy
    checkpoint_layers: List[int]
    checkpoint_ratio: float
    memory_savings_gb: float
    recompute_overhead_pct: float
    config: Dict[str, Any]
    rationale: str


@dataclass
class PipelineRecommendation:
    """Recommendation for pipeline scheduling."""
    schedule: PipelineSchedule
    num_microbatches: int
    virtual_stages: int
    bubble_overhead_pct: float
    memory_per_stage_gb: float
    throughput_efficiency: float
    config: Dict[str, Any]
    rationale: str


@dataclass
class OptimizerRecommendation:
    """Recommendation for memory-efficient optimizer."""
    name: str
    memory_savings_gb: float
    state_memory_per_param: float  # bytes per parameter
    convergence_notes: str
    config: Dict[str, Any]
    rationale: str


@dataclass  
class CompoundOptimization:
    """A compound optimization combining multiple techniques."""
    name: str
    techniques: List[str]
    total_memory_savings_gb: float
    total_throughput_boost_pct: float
    complexity: str  # "low", "medium", "high"
    compatibility_notes: List[str]
    config: Dict[str, Any]
    implementation_steps: List[str]


class PrecisionOptimizer:
    """Recommends optimal precision settings based on hardware and workload."""
    
    def __init__(self, gpu_arch: str, gpu_memory_gb: float):
        self.gpu_arch = gpu_arch
        self.gpu_memory_gb = gpu_memory_gb
        
        # Hardware capabilities
        self.supports_fp8 = gpu_arch.lower() in ['hopper', 'blackwell', 'h100', 'h200', 'b100', 'b200', 'gb200']
        self.supports_bf16 = gpu_arch.lower() not in ['volta', 'turing', 'pascal']
        self.supports_tf32 = gpu_arch.lower() in ['ampere', 'hopper', 'blackwell', 'a100', 'h100', 'h200', 'b100', 'b200', 'gb200']
    
    def recommend(
        self,
        model_params_b: float,
        is_training: bool = True,
        accuracy_priority: str = "balanced"  # "high", "balanced", "speed"
    ) -> PrecisionRecommendation:
        """
        Recommend optimal precision settings.
        
        Args:
            model_params_b: Model parameters in billions
            is_training: Whether this is for training or inference
            accuracy_priority: Trade-off preference
        """
        
        # FP8 for Hopper/Blackwell when speed is priority
        if self.supports_fp8 and accuracy_priority == "speed":
            return PrecisionRecommendation(
                mode=PrecisionMode.MIXED_FP8,
                compute_dtype="fp8_e4m3",
                param_dtype="bf16",
                grad_dtype="fp8_e5m2",
                memory_savings_pct=50,
                throughput_boost_pct=80,
                accuracy_impact="minimal",
                hardware_requirements=["Hopper/Blackwell GPU", "CUDA 12.0+", "PyTorch 2.1+"],
                config={
                    "transformer_engine": True,
                    "fp8_format": "HYBRID",
                    "fp8_margin": 0,
                    "fp8_interval": 1,
                    "fp8_amax_history_len": 1024,
                    "fp8_amax_compute_algo": "max"
                },
                rationale="FP8 on Hopper/Blackwell provides 2x compute throughput with minimal accuracy loss using Transformer Engine"
            )
        
        # BF16 for most modern training
        if self.supports_bf16 and is_training:
            return PrecisionRecommendation(
                mode=PrecisionMode.MIXED_BF16,
                compute_dtype="bf16",
                param_dtype="bf16",
                grad_dtype="fp32",
                memory_savings_pct=50,
                throughput_boost_pct=40,
                accuracy_impact="none",
                hardware_requirements=["Ampere+ GPU", "CUDA 11.0+"],
                config={
                    "bf16": {"enabled": True},
                    "fp16": {"enabled": False},
                    "torch_dtype": "bfloat16",
                    "gradient_checkpointing": True
                },
                rationale="BF16 provides memory savings with no accuracy loss due to same exponent range as FP32"
            )
        
        # FP16 with loss scaling for older hardware
        if not self.supports_bf16 and is_training:
            return PrecisionRecommendation(
                mode=PrecisionMode.MIXED_FP16,
                compute_dtype="fp16",
                param_dtype="fp16",
                grad_dtype="fp32",
                memory_savings_pct=50,
                throughput_boost_pct=35,
                accuracy_impact="minimal",
                hardware_requirements=["Volta+ GPU"],
                config={
                    "fp16": {
                        "enabled": True,
                        "loss_scale": 0,
                        "initial_scale_power": 16,
                        "loss_scale_window": 1000,
                        "hysteresis": 2,
                        "min_loss_scale": 1
                    }
                },
                rationale="FP16 with dynamic loss scaling for hardware without BF16 support"
            )
        
        # Inference optimizations
        if not is_training:
            if self.supports_fp8:
                return PrecisionRecommendation(
                    mode=PrecisionMode.FP8_E4M3,
                    compute_dtype="fp8_e4m3",
                    param_dtype="fp8_e4m3",
                    grad_dtype="none",
                    memory_savings_pct=75,
                    throughput_boost_pct=100,
                    accuracy_impact="minimal",
                    hardware_requirements=["Hopper/Blackwell GPU"],
                    config={
                        "quantization": "fp8",
                        "kv_cache_dtype": "fp8"
                    },
                    rationale="FP8 inference with FP8 KV cache for maximum throughput"
                )
            else:
                return PrecisionRecommendation(
                    mode=PrecisionMode.BF16 if self.supports_bf16 else PrecisionMode.FP16,
                    compute_dtype="bf16" if self.supports_bf16 else "fp16",
                    param_dtype="bf16" if self.supports_bf16 else "fp16",
                    grad_dtype="none",
                    memory_savings_pct=50,
                    throughput_boost_pct=40,
                    accuracy_impact="none",
                    hardware_requirements=["Ampere+ GPU"] if self.supports_bf16 else ["Volta+ GPU"],
                    config={"torch_dtype": "bfloat16" if self.supports_bf16 else "float16"},
                    rationale="Half-precision inference for balanced speed and accuracy"
                )
        
        # Default: BF16 if supported, else FP16
        return PrecisionRecommendation(
            mode=PrecisionMode.MIXED_BF16 if self.supports_bf16 else PrecisionMode.MIXED_FP16,
            compute_dtype="bf16" if self.supports_bf16 else "fp16",
            param_dtype="bf16" if self.supports_bf16 else "fp16",
            grad_dtype="fp32",
            memory_savings_pct=50,
            throughput_boost_pct=40,
            accuracy_impact="none" if self.supports_bf16 else "minimal",
            hardware_requirements=["Ampere+ GPU"] if self.supports_bf16 else ["Volta+ GPU"],
            config={"bf16": {"enabled": True}} if self.supports_bf16 else {"fp16": {"enabled": True}},
            rationale="Standard mixed precision training"
        )


class CheckpointingOptimizer:
    """Recommends activation checkpointing strategy based on memory constraints."""
    
    def __init__(self, gpu_memory_gb: float, model_memory_gb: float):
        self.gpu_memory_gb = gpu_memory_gb
        self.model_memory_gb = model_memory_gb
        self.available_for_activations = gpu_memory_gb - model_memory_gb
    
    def recommend(
        self,
        num_layers: int,
        hidden_size: int,
        seq_length: int,
        batch_size: int,
        activation_memory_gb: float
    ) -> CheckpointingRecommendation:
        """
        Recommend checkpointing strategy based on memory constraints.
        """
        
        # Calculate per-layer activation memory
        per_layer_activation_gb = activation_memory_gb / num_layers if num_layers > 0 else 0
        
        # If activations fit comfortably, no checkpointing needed
        if activation_memory_gb < self.available_for_activations * 0.8:
            return CheckpointingRecommendation(
                strategy=CheckpointingStrategy.NONE,
                checkpoint_layers=[],
                checkpoint_ratio=0,
                memory_savings_gb=0,
                recompute_overhead_pct=0,
                config={
                    "gradient_checkpointing": False
                },
                rationale=f"Activations ({activation_memory_gb:.1f} GB) fit in available memory ({self.available_for_activations:.1f} GB)"
            )
        
        # Calculate how much we need to save
        memory_to_save = activation_memory_gb - (self.available_for_activations * 0.7)
        
        # Full checkpointing if we need to save a lot
        if memory_to_save > activation_memory_gb * 0.5:
            return CheckpointingRecommendation(
                strategy=CheckpointingStrategy.FULL,
                checkpoint_layers=list(range(num_layers)),
                checkpoint_ratio=1.0,
                memory_savings_gb=activation_memory_gb * 0.95,  # ~95% savings
                recompute_overhead_pct=33,  # ~33% recompute overhead
                config={
                    "gradient_checkpointing": True,
                    "gradient_checkpointing_policy": "full"
                },
                rationale=f"Need to save {memory_to_save:.1f} GB - using full checkpointing"
            )
        
        # Block-wise checkpointing for moderate savings
        checkpoint_every = max(2, int(num_layers / (memory_to_save / per_layer_activation_gb)))
        checkpoint_layers = list(range(0, num_layers, checkpoint_every))
        checkpoint_ratio = len(checkpoint_layers) / num_layers
        
        return CheckpointingRecommendation(
            strategy=CheckpointingStrategy.BLOCK_WISE,
            checkpoint_layers=checkpoint_layers,
            checkpoint_ratio=checkpoint_ratio,
            memory_savings_gb=memory_to_save * 0.9,
            recompute_overhead_pct=int(checkpoint_ratio * 33),
            config={
                "gradient_checkpointing": True,
                "gradient_checkpointing_policy": "block",
                "checkpoint_every_n_layers": checkpoint_every
            },
            rationale=f"Block-wise checkpointing every {checkpoint_every} layers for balanced memory/speed"
        )


class PipelineScheduleOptimizer:
    """Recommends pipeline scheduling strategies."""
    
    def recommend(
        self,
        pp_degree: int,
        num_layers: int,
        microbatch_size: int,
        global_batch_size: int,
        memory_per_layer_gb: float,
        is_training: bool = True
    ) -> PipelineRecommendation:
        """
        Recommend optimal pipeline scheduling.
        """
        
        if pp_degree <= 1:
            return PipelineRecommendation(
                schedule=PipelineSchedule.GPIPE,
                num_microbatches=1,
                virtual_stages=1,
                bubble_overhead_pct=0,
                memory_per_stage_gb=memory_per_layer_gb * num_layers,
                throughput_efficiency=1.0,
                config={},
                rationale="No pipeline parallelism - single stage"
            )
        
        layers_per_stage = num_layers // pp_degree
        
        # Calculate optimal number of microbatches
        # Rule of thumb: microbatches >= 4 * pp_degree for low bubble
        min_microbatches = pp_degree * 4
        num_microbatches = max(min_microbatches, global_batch_size // microbatch_size)
        
        # Ensure divisibility
        if num_microbatches < pp_degree:
            num_microbatches = pp_degree
        
        # Calculate bubble overhead
        # Bubble = (pp - 1) / num_microbatches for 1F1B
        bubble_ratio = (pp_degree - 1) / num_microbatches
        bubble_overhead_pct = bubble_ratio * 100
        
        # Memory per stage
        memory_per_stage = memory_per_layer_gb * layers_per_stage
        
        # Zero-bubble for large PP degrees with enough microbatches
        if pp_degree >= 4 and num_microbatches >= pp_degree * 8:
            return PipelineRecommendation(
                schedule=PipelineSchedule.ZERO_BUBBLE,
                num_microbatches=num_microbatches,
                virtual_stages=2,  # 2 virtual stages per physical stage
                bubble_overhead_pct=bubble_overhead_pct * 0.1,  # ~90% reduction
                memory_per_stage_gb=memory_per_stage * 1.5,  # Higher memory for zero-bubble
                throughput_efficiency=0.95,
                config={
                    "pipeline_schedule": "zero_bubble",
                    "virtual_pipeline_model_parallel_size": 2,
                    "num_microbatches": num_microbatches
                },
                rationale=f"Zero-bubble scheduling minimizes {bubble_overhead_pct:.1f}% bubble to ~{bubble_overhead_pct*0.1:.1f}%"
            )
        
        # Interleaved for moderate PP
        if pp_degree >= 2 and num_microbatches >= pp_degree * 4:
            virtual_stages = min(4, num_layers // (pp_degree * 2))
            return PipelineRecommendation(
                schedule=PipelineSchedule.INTERLEAVED,
                num_microbatches=num_microbatches,
                virtual_stages=virtual_stages,
                bubble_overhead_pct=bubble_overhead_pct / virtual_stages,
                memory_per_stage_gb=memory_per_stage * 1.2,
                throughput_efficiency=1 - (bubble_overhead_pct / virtual_stages / 100),
                config={
                    "pipeline_schedule": "interleaved",
                    "virtual_pipeline_model_parallel_size": virtual_stages,
                    "num_microbatches": num_microbatches
                },
                rationale=f"Interleaved scheduling with {virtual_stages} virtual stages reduces bubble"
            )
        
        # Default: 1F1B
        return PipelineRecommendation(
            schedule=PipelineSchedule.ONE_F_ONE_B,
            num_microbatches=num_microbatches,
            virtual_stages=1,
            bubble_overhead_pct=bubble_overhead_pct,
            memory_per_stage_gb=memory_per_stage,
            throughput_efficiency=1 - bubble_ratio,
            config={
                "pipeline_schedule": "1f1b",
                "num_microbatches": num_microbatches
            },
            rationale=f"1F1B scheduling with {bubble_overhead_pct:.1f}% bubble overhead"
        )


class MemoryEfficientOptimizerRecommender:
    """Recommends memory-efficient optimizers."""
    
    OPTIMIZER_SPECS = {
        "adamw": {
            "state_bytes_per_param": 8,  # 2 FP32 moments
            "convergence": "excellent",
            "description": "Standard AdamW - best convergence"
        },
        "adamw_8bit": {
            "state_bytes_per_param": 2,  # Quantized moments
            "convergence": "very_good",
            "description": "8-bit AdamW via bitsandbytes - 75% memory reduction"
        },
        "adafactor": {
            "state_bytes_per_param": 0.5,  # Factored moments
            "convergence": "good",
            "description": "Adafactor - ~94% memory reduction, good for LLMs"
        },
        "came": {
            "state_bytes_per_param": 0.5,
            "convergence": "good", 
            "description": "CAME optimizer - confidence-guided adaptive memory efficient"
        },
        "lion": {
            "state_bytes_per_param": 4,  # Single FP32 momentum
            "convergence": "good",
            "description": "Lion - 50% memory reduction, may need LR tuning"
        },
        "sophia": {
            "state_bytes_per_param": 4,
            "convergence": "excellent",
            "description": "Sophia - second-order info with first-order cost"
        },
        "sgd_momentum": {
            "state_bytes_per_param": 4,  # Single momentum
            "convergence": "moderate",
            "description": "SGD with momentum - simple, 50% less memory"
        }
    }
    
    def recommend(
        self,
        model_params_b: float,
        available_memory_gb: float,
        convergence_priority: str = "balanced"  # "high", "balanced", "memory"
    ) -> OptimizerRecommendation:
        """
        Recommend memory-efficient optimizer based on constraints.
        """
        
        # Calculate memory requirements for each optimizer
        params_count = model_params_b * 1e9
        
        options = []
        for name, spec in self.OPTIMIZER_SPECS.items():
            state_memory_gb = (params_count * spec['state_bytes_per_param']) / 1e9
            
            # Check if it fits
            if state_memory_gb < available_memory_gb:
                options.append({
                    "name": name,
                    "state_memory_gb": state_memory_gb,
                    "convergence": spec['convergence'],
                    "description": spec['description'],
                    "bytes_per_param": spec['state_bytes_per_param']
                })
        
        if not options:
            # Nothing fits - recommend Adafactor as last resort
            return OptimizerRecommendation(
                name="adafactor",
                memory_savings_gb=model_params_b * 7.5,  # vs AdamW
                state_memory_per_param=0.5,
                convergence_notes="May need learning rate warmup and careful tuning",
                config={
                    "optimizer": "adafactor",
                    "scale_parameter": True,
                    "relative_step": True,
                    "warmup_init": True
                },
                rationale="Only option that fits - Adafactor with factored second moments"
            )
        
        # Sort by preference based on priority
        if convergence_priority == "high":
            # Prefer convergence
            order = {"excellent": 0, "very_good": 1, "good": 2, "moderate": 3}
            options.sort(key=lambda x: order.get(x['convergence'], 4))
        elif convergence_priority == "memory":
            # Prefer memory efficiency
            options.sort(key=lambda x: x['state_memory_gb'])
        else:
            # Balanced - prefer very_good convergence with good memory
            options.sort(key=lambda x: (
                0 if x['convergence'] in ['excellent', 'very_good'] else 1,
                x['state_memory_gb']
            ))
        
        best = options[0]
        adamw_memory = (params_count * 8) / 1e9
        
        return OptimizerRecommendation(
            name=best['name'],
            memory_savings_gb=adamw_memory - best['state_memory_gb'],
            state_memory_per_param=best['bytes_per_param'],
            convergence_notes=best['description'],
            config=self._get_optimizer_config(best['name']),
            rationale=f"{best['name']} saves {adamw_memory - best['state_memory_gb']:.1f} GB optimizer states"
        )
    
    def _get_optimizer_config(self, name: str) -> Dict[str, Any]:
        """Get configuration for optimizer."""
        configs = {
            "adamw": {
                "optimizer": "adamw",
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "weight_decay": 0.01
            },
            "adamw_8bit": {
                "optimizer": "adamw_bnb_8bit",
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "weight_decay": 0.01,
                "optim_bits": 8
            },
            "adafactor": {
                "optimizer": "adafactor",
                "lr": None,  # Relative step
                "scale_parameter": True,
                "relative_step": True,
                "warmup_init": True
            },
            "lion": {
                "optimizer": "lion",
                "lr": 3e-5,  # 3-10x lower than Adam
                "betas": [0.9, 0.99],
                "weight_decay": 0.1
            },
            "came": {
                "optimizer": "came",
                "lr": 1e-4,
                "betas": [0.9, 0.999, 0.9999],
                "weight_decay": 0.01
            },
            "sophia": {
                "optimizer": "sophia",
                "lr": 2e-4,
                "betas": [0.96, 0.99],
                "weight_decay": 0.1,
                "rho": 0.04
            },
            "sgd_momentum": {
                "optimizer": "sgd",
                "lr": 1e-2,
                "momentum": 0.9,
                "weight_decay": 0.01
            }
        }
        return configs.get(name, configs["adamw"])


class KernelFusionRecommender:
    """Recommends kernel fusion and Flash Attention settings."""
    
    def __init__(self, gpu_arch: str, seq_length: int):
        self.gpu_arch = gpu_arch
        self.seq_length = seq_length
        
        # Flash Attention support
        self.supports_flash_attn_2 = gpu_arch.lower() in [
            'ampere', 'hopper', 'blackwell', 'a100', 'h100', 'h200', 'b100', 'b200', 'gb200'
        ]
        self.supports_flash_attn_3 = gpu_arch.lower() in ['hopper', 'blackwell', 'h100', 'h200', 'b100', 'b200', 'gb200']
    
    def recommend(self) -> Dict[str, Any]:
        """Get kernel fusion recommendations."""
        
        recommendations = {
            "flash_attention": {},
            "fused_kernels": [],
            "torch_compile": {},
            "custom_kernels": []
        }
        
        # Flash Attention
        if self.supports_flash_attn_3:
            recommendations["flash_attention"] = {
                "version": "flash_attn_3",
                "enabled": True,
                "causal": True,
                "rationale": "Flash Attention 3 on Hopper/Blackwell - optimal for long sequences"
            }
        elif self.supports_flash_attn_2:
            recommendations["flash_attention"] = {
                "version": "flash_attn_2",
                "enabled": True,
                "causal": True,
                "rationale": "Flash Attention 2 - memory efficient attention"
            }
        else:
            recommendations["flash_attention"] = {
                "version": "sdpa",
                "enabled": True,
                "rationale": "PyTorch SDPA - fallback efficient attention"
            }
        
        # Fused kernels
        recommendations["fused_kernels"] = [
            {
                "name": "fused_layer_norm",
                "benefit": "Reduced memory traffic",
                "config": {"fused_layer_norm": True}
            },
            {
                "name": "fused_bias_gelu",
                "benefit": "Single kernel for bias + activation",
                "config": {"bias_gelu_fusion": True}
            },
            {
                "name": "fused_softmax",
                "benefit": "Numerically stable fused softmax",
                "config": {"fused_softmax": True}
            },
            {
                "name": "fused_rope",
                "benefit": "Rotary position embedding fusion",
                "config": {"rope_fusion": True}
            }
        ]
        
        # Torch compile
        recommendations["torch_compile"] = {
            "enabled": True,
            "mode": "reduce-overhead" if self.seq_length < 4096 else "max-autotune",
            "backend": "inductor",
            "dynamic": self.seq_length > 2048,
            "rationale": "torch.compile with Inductor backend for automatic fusion"
        }
        
        # Custom kernels for specific operations
        if self.seq_length >= 8192:
            recommendations["custom_kernels"].append({
                "name": "ring_attention",
                "use_case": "Long sequence training",
                "config": {"ring_attention": True, "ring_size": 8}
            })
        
        return recommendations


class CommunicationOptimizer:
    """Optimizes communication patterns for distributed training."""
    
    def recommend(
        self,
        dp_degree: int,
        tp_degree: int,
        pp_degree: int,
        has_nvlink: bool,
        has_infiniband: bool,
        model_params_b: float
    ) -> Dict[str, Any]:
        """
        Recommend communication optimization strategies.
        """
        
        recommendations = {
            "overlap_strategy": None,
            "bucketing": {},
            "compression": {},
            "topology_aware": {}
        }
        
        # Overlap strategies
        if dp_degree > 1:
            recommendations["overlap_strategy"] = {
                "strategy": OverlapStrategy.ASYNC_ALLREDUCE.value,
                "enabled": True,
                "config": {
                    "overlap_comm": True,
                    "contiguous_gradients": True
                },
                "rationale": "Overlap gradient allreduce with backward pass"
            }
        
        # Gradient bucketing
        if dp_degree > 1:
            bucket_size_mb = 25 if has_nvlink else 50
            recommendations["bucketing"] = {
                "enabled": True,
                "bucket_size_mb": bucket_size_mb,
                "config": {
                    "bucket_cap_mb": bucket_size_mb,
                    "find_unused_parameters": False
                },
                "rationale": f"Bucket size {bucket_size_mb}MB for efficient allreduce"
            }
        
        # Gradient compression for large models over slow networks
        if dp_degree > 1 and model_params_b > 10 and not has_infiniband:
            recommendations["compression"] = {
                "enabled": True,
                "method": "powersgd",
                "rank": 2,
                "config": {
                    "gradient_compression": True,
                    "compression_method": "powersgd",
                    "powersgd_rank": 2
                },
                "rationale": "PowerSGD compression for bandwidth-limited training"
            }
        
        # Topology-aware communication
        if has_nvlink and tp_degree > 1:
            recommendations["topology_aware"] = {
                "tp_communication": "nvlink",
                "dp_communication": "infiniband" if has_infiniband else "pcie",
                "pp_communication": "infiniband" if has_infiniband else "pcie",
                "nccl_config": {
                    "NCCL_NVLS_ENABLE": "1" if has_nvlink else "0",
                    "NCCL_IB_DISABLE": "0" if has_infiniband else "1",
                    "NCCL_SOCKET_NTHREADS": "4",
                    "NCCL_NSOCKS_PERTHREAD": "4"
                }
            }
        
        return recommendations


class CompoundOptimizationGenerator:
    """
    Generates compound optimizations by combining multiple techniques.
    
    This is where we create the "advanced" configurations that combine:
    - Precision + Checkpointing + Pipeline + Optimizer optimizations
    """
    
    def __init__(
        self,
        gpu_arch: str,
        gpu_memory_gb: float,
        num_gpus: int,
        has_nvlink: bool
    ):
        self.gpu_arch = gpu_arch
        self.gpu_memory_gb = gpu_memory_gb
        self.num_gpus = num_gpus
        self.has_nvlink = has_nvlink
    
    def generate_optimal_compound(
        self,
        model_params_b: float,
        num_layers: int,
        hidden_size: int,
        seq_length: int,
        batch_size: int,
        is_training: bool = True,
        optimization_goal: str = "throughput"  # "throughput", "memory", "balanced"
    ) -> CompoundOptimization:
        """
        Generate an optimal compound optimization combining multiple techniques.
        """
        
        techniques = []
        total_memory_savings = 0.0
        total_throughput_boost = 0.0
        compatibility_notes = []
        config = {}
        steps = []
        
        # 1. Precision optimization
        precision_opt = PrecisionOptimizer(self.gpu_arch, self.gpu_memory_gb)
        precision_rec = precision_opt.recommend(
            model_params_b,
            is_training,
            "speed" if optimization_goal == "throughput" else "balanced"
        )
        
        techniques.append(f"Mixed Precision ({precision_rec.mode.value})")
        total_memory_savings += model_params_b * 2 * (precision_rec.memory_savings_pct / 100)
        total_throughput_boost += precision_rec.throughput_boost_pct
        config.update(precision_rec.config)
        steps.append(f"Enable {precision_rec.mode.value} precision: {precision_rec.rationale}")
        
        # 2. Activation checkpointing
        model_memory_gb = model_params_b * 2  # BF16
        activation_memory_gb = (batch_size * seq_length * hidden_size * num_layers * 2) / 1e9
        
        checkpoint_opt = CheckpointingOptimizer(self.gpu_memory_gb, model_memory_gb)
        checkpoint_rec = checkpoint_opt.recommend(
            num_layers, hidden_size, seq_length, batch_size, activation_memory_gb
        )
        
        if checkpoint_rec.strategy != CheckpointingStrategy.NONE:
            techniques.append(f"Activation Checkpointing ({checkpoint_rec.strategy.value})")
            total_memory_savings += checkpoint_rec.memory_savings_gb
            config.update(checkpoint_rec.config)
            steps.append(f"Enable {checkpoint_rec.strategy.value} checkpointing: saves {checkpoint_rec.memory_savings_gb:.1f}GB")
            if checkpoint_rec.recompute_overhead_pct > 20:
                compatibility_notes.append(f"Recompute overhead: ~{checkpoint_rec.recompute_overhead_pct}%")
        
        # 3. Memory-efficient optimizer
        available_for_optimizer = self.gpu_memory_gb - model_memory_gb - 10  # 10GB headroom
        optimizer_rec = MemoryEfficientOptimizerRecommender().recommend(
            model_params_b,
            available_for_optimizer,
            "memory" if optimization_goal == "memory" else "balanced"
        )
        
        if optimizer_rec.name != "adamw":
            techniques.append(f"Memory-Efficient Optimizer ({optimizer_rec.name})")
            total_memory_savings += optimizer_rec.memory_savings_gb
            config["optimizer"] = optimizer_rec.config
            steps.append(f"Use {optimizer_rec.name}: saves {optimizer_rec.memory_savings_gb:.1f}GB optimizer states")
            compatibility_notes.append(optimizer_rec.convergence_notes)
        
        # 4. Kernel fusion
        kernel_rec = KernelFusionRecommender(self.gpu_arch, seq_length).recommend()
        techniques.append("Fused Kernels + Flash Attention")
        total_throughput_boost += 15  # Typical improvement
        config["kernels"] = kernel_rec
        steps.append(f"Enable {kernel_rec['flash_attention']['version']}: memory-efficient attention")
        
        # 5. Communication optimization (if distributed)
        if self.num_gpus > 1:
            comm_opt = CommunicationOptimizer().recommend(
                dp_degree=self.num_gpus // 8 if self.num_gpus >= 8 else 1,
                tp_degree=min(8, self.num_gpus) if self.has_nvlink else 1,
                pp_degree=1,
                has_nvlink=self.has_nvlink,
                has_infiniband=True,  # Assume IB for multi-node
                model_params_b=model_params_b
            )
            
            if comm_opt.get("overlap_strategy"):
                techniques.append("Async Communication Overlap")
                total_throughput_boost += 10
                config["communication"] = comm_opt
                steps.append("Enable communication-computation overlap")
        
        # Determine complexity
        if len(techniques) <= 2:
            complexity = "low"
        elif len(techniques) <= 4:
            complexity = "medium"
        else:
            complexity = "high"
        
        return CompoundOptimization(
            name=f"Optimal {optimization_goal.title()} Configuration",
            techniques=techniques,
            total_memory_savings_gb=total_memory_savings,
            total_throughput_boost_pct=total_throughput_boost,
            complexity=complexity,
            compatibility_notes=compatibility_notes,
            config=config,
            implementation_steps=steps
        )
    
    def generate_all_profiles(
        self,
        model_params_b: float,
        num_layers: int,
        hidden_size: int,
        seq_length: int,
        batch_size: int
    ) -> Dict[str, CompoundOptimization]:
        """
        Generate compound optimizations for all optimization goals.
        """
        return {
            "throughput": self.generate_optimal_compound(
                model_params_b, num_layers, hidden_size, seq_length, batch_size,
                optimization_goal="throughput"
            ),
            "memory": self.generate_optimal_compound(
                model_params_b, num_layers, hidden_size, seq_length, batch_size,
                optimization_goal="memory"
            ),
            "balanced": self.generate_optimal_compound(
                model_params_b, num_layers, hidden_size, seq_length, batch_size,
                optimization_goal="balanced"
            )
        }


def get_advanced_optimization_report(
    model_config: Dict[str, Any],
    hardware_config: Dict[str, Any],
    optimization_goal: str = "balanced"
) -> Dict[str, Any]:
    """
    Generate a comprehensive advanced optimization report.
    
    Args:
        model_config: Model configuration dict
        hardware_config: Hardware configuration dict
        optimization_goal: "throughput", "memory", or "balanced"
    
    Returns:
        Complete optimization report with all recommendations
    """
    
    # Extract configs
    model_params_b = model_config.get('parameters_billions', 7)
    num_layers = model_config.get('num_layers', 32)
    hidden_size = model_config.get('hidden_size', 4096)
    seq_length = model_config.get('max_sequence_length', 4096)
    batch_size = model_config.get('batch_size', 1)
    
    gpu_arch = hardware_config.get('gpu_arch', 'ampere')
    gpu_memory_gb = hardware_config.get('gpu_memory_gb', 80)
    num_gpus = hardware_config.get('num_gpus', 1)
    has_nvlink = hardware_config.get('has_nvlink', False)
    
    # Generate compound optimization
    generator = CompoundOptimizationGenerator(gpu_arch, gpu_memory_gb, num_gpus, has_nvlink)
    compound = generator.generate_optimal_compound(
        model_params_b, num_layers, hidden_size, seq_length, batch_size,
        optimization_goal=optimization_goal
    )
    
    # Generate individual recommendations
    precision_rec = PrecisionOptimizer(gpu_arch, gpu_memory_gb).recommend(
        model_params_b, is_training=True
    )
    
    kernel_rec = KernelFusionRecommender(gpu_arch, seq_length).recommend()
    
    return {
        "optimization_goal": optimization_goal,
        "compound_optimization": {
            "name": compound.name,
            "techniques": compound.techniques,
            "total_memory_savings_gb": compound.total_memory_savings_gb,
            "total_throughput_boost_pct": compound.total_throughput_boost_pct,
            "complexity": compound.complexity,
            "compatibility_notes": compound.compatibility_notes,
            "implementation_steps": compound.implementation_steps,
            "config": compound.config
        },
        "precision": {
            "mode": precision_rec.mode.value,
            "compute_dtype": precision_rec.compute_dtype,
            "param_dtype": precision_rec.param_dtype,
            "memory_savings_pct": precision_rec.memory_savings_pct,
            "throughput_boost_pct": precision_rec.throughput_boost_pct,
            "accuracy_impact": precision_rec.accuracy_impact,
            "rationale": precision_rec.rationale
        },
        "kernels": kernel_rec,
        "hardware_features_used": [
            f for f in [
                "FP8" if 'hopper' in gpu_arch.lower() or 'blackwell' in gpu_arch.lower() else None,
                "NVLink" if has_nvlink else None,
                "Flash Attention 3" if kernel_rec['flash_attention'].get('version') == 'flash_attn_3' else "Flash Attention 2"
            ] if f
        ]
    }



