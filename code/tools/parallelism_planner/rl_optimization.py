"""
RL/RLHF Optimization Module for Parallelism Planner

Specialized optimizations for Reinforcement Learning from Human Feedback:
- PPO/DPO/GRPO/RLOO specific configurations
- Actor-Critic memory optimization
- Reference model strategies (CPU offload, quantization, vLLM serving)
- Reward model optimization
- Multi-model memory planning (4 models: actor, critic, ref, reward)
- On-policy vs off-policy batch sizing
- Sample efficiency optimization
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


class RLAlgorithm(Enum):
    """RL algorithms for LLM training."""
    PPO = "ppo"           # Proximal Policy Optimization
    DPO = "dpo"           # Direct Preference Optimization  
    GRPO = "grpo"         # Group Relative Policy Optimization
    RLOO = "rloo"         # REINFORCE Leave-One-Out
    KTO = "kto"           # Kahneman-Tversky Optimization
    ORPO = "orpo"         # Odds Ratio Preference Optimization
    IPO = "ipo"           # Identity Preference Optimization


class ReferenceModelStrategy(Enum):
    """Strategies for handling reference model in RLHF."""
    FULL_GPU = "full_gpu"           # Keep on GPU (most memory)
    CPU_OFFLOAD = "cpu_offload"     # Offload to CPU
    QUANTIZED = "quantized"         # Quantize to INT8/INT4
    VLLM_SERVER = "vllm_server"     # Serve via vLLM
    SHARED_WEIGHTS = "shared_weights"  # Share with actor (LoRA)
    NONE = "none"                   # DPO doesn't need ref during forward


@dataclass
class RLModelMemoryPlan:
    """Memory plan for a single model in RL pipeline."""
    name: str
    params_billion: float
    memory_gb: float
    strategy: str
    device: str  # "gpu", "cpu", "vllm"
    quantization: Optional[str] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class RLMemoryPlan:
    """Complete memory plan for RL training."""
    algorithm: RLAlgorithm
    total_gpu_memory_needed: float
    total_cpu_memory_needed: float
    
    # Individual model plans
    actor: RLModelMemoryPlan
    critic: Optional[RLModelMemoryPlan]
    reference: Optional[RLModelMemoryPlan]
    reward: Optional[RLModelMemoryPlan]
    
    # Batch settings
    ppo_batch_size: int
    ppo_mini_batch_size: int
    generation_batch_size: int
    
    # Memory breakdown
    memory_breakdown: Dict[str, float]
    
    # Recommendations
    recommendations: List[str]
    warnings: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm.value,
            "total_gpu_memory_needed_gb": self.total_gpu_memory_needed,
            "total_cpu_memory_needed_gb": self.total_cpu_memory_needed,
            "models": {
                "actor": self._model_to_dict(self.actor),
                "critic": self._model_to_dict(self.critic) if self.critic else None,
                "reference": self._model_to_dict(self.reference) if self.reference else None,
                "reward": self._model_to_dict(self.reward) if self.reward else None,
            },
            "batch_settings": {
                "ppo_batch_size": self.ppo_batch_size,
                "ppo_mini_batch_size": self.ppo_mini_batch_size,
                "generation_batch_size": self.generation_batch_size,
            },
            "memory_breakdown": self.memory_breakdown,
            "recommendations": self.recommendations,
            "warnings": self.warnings,
        }
    
    def _model_to_dict(self, model: RLModelMemoryPlan) -> Dict[str, Any]:
        return {
            "name": model.name,
            "params_billion": model.params_billion,
            "memory_gb": model.memory_gb,
            "strategy": model.strategy,
            "device": model.device,
            "quantization": model.quantization,
            "notes": model.notes,
        }


@dataclass
class RLTrainingConfig:
    """Complete RL training configuration."""
    algorithm: RLAlgorithm
    
    # Parallelism
    actor_tp: int
    actor_pp: int
    critic_tp: int
    critic_pp: int
    dp: int
    
    # Memory optimization
    reference_strategy: ReferenceModelStrategy
    use_peft: bool
    peft_config: Optional[Dict[str, Any]]
    
    # Training settings
    ppo_epochs: int
    generation_config: Dict[str, Any]
    
    # DeepSpeed config
    deepspeed_config: Dict[str, Any]
    
    # Launch commands
    trl_command: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm.value,
            "parallelism": {
                "actor_tp": self.actor_tp,
                "actor_pp": self.actor_pp,
                "critic_tp": self.critic_tp,
                "critic_pp": self.critic_pp,
                "dp": self.dp,
            },
            "reference_strategy": self.reference_strategy.value,
            "use_peft": self.use_peft,
            "peft_config": self.peft_config,
            "ppo_epochs": self.ppo_epochs,
            "generation_config": self.generation_config,
            "deepspeed_config": self.deepspeed_config,
            "trl_command": self.trl_command,
        }


class RLOptimizer:
    """Optimizes RL/RLHF training configurations."""
    
    # Memory multipliers for different RL algorithms
    ALGORITHM_MEMORY_MULTIPLIERS = {
        RLAlgorithm.PPO: 4.0,    # Actor + Critic + Ref + Reward
        RLAlgorithm.DPO: 2.0,    # Policy + Reference (but ref only for loss)
        RLAlgorithm.GRPO: 2.5,   # Policy + Reference + Reward
        RLAlgorithm.RLOO: 2.0,   # Policy + Reference
        RLAlgorithm.KTO: 2.0,    # Policy + Reference
        RLAlgorithm.ORPO: 1.5,   # Policy + smaller overhead
        RLAlgorithm.IPO: 2.0,    # Policy + Reference
    }
    
    def __init__(self, gpu_memory_gb: float = 80, num_gpus: int = 8):
        self.gpu_memory_gb = gpu_memory_gb
        self.num_gpus = num_gpus
        self.total_gpu_memory = gpu_memory_gb * num_gpus
    
    def plan_memory(
        self,
        model_params_b: float,
        algorithm: RLAlgorithm = RLAlgorithm.PPO,
        critic_params_b: Optional[float] = None,
        reward_params_b: Optional[float] = None,
        use_peft: bool = False,
        peft_rank: int = 16,
    ) -> RLMemoryPlan:
        """
        Plan memory allocation for RL training.
        
        Args:
            model_params_b: Actor/policy model parameters in billions
            algorithm: RL algorithm
            critic_params_b: Critic model size (defaults to same as actor)
            reward_params_b: Reward model size
            use_peft: Whether to use LoRA/PEFT
            peft_rank: LoRA rank if using PEFT
        """
        
        critic_params = critic_params_b or model_params_b
        reward_params = reward_params_b or model_params_b
        
        # Base memory per model (BF16)
        actor_base_mem = model_params_b * 2  # Weights only
        critic_base_mem = critic_params * 2
        ref_base_mem = model_params_b * 2
        reward_base_mem = reward_params * 2
        
        # Determine strategies based on available memory
        recommendations = []
        warnings = []
        
        # Check if everything fits on GPU
        total_model_memory = actor_base_mem
        needs_cpu_offload = False
        needs_quantization = False
        
        if algorithm == RLAlgorithm.PPO:
            # PPO needs all 4 models
            total_model_memory = actor_base_mem + critic_base_mem + ref_base_mem + reward_base_mem
            
            # Add optimizer states (actor + critic only, 8 bytes per param for AdamW)
            optimizer_memory = (model_params_b + critic_params) * 8
            total_model_memory += optimizer_memory
            
            if total_model_memory > self.total_gpu_memory * 0.8:
                needs_cpu_offload = True
                if total_model_memory > self.total_gpu_memory * 1.5:
                    needs_quantization = True
        
        elif algorithm == RLAlgorithm.DPO:
            # DPO: policy trains, reference for KL (can be computed offline)
            total_model_memory = actor_base_mem * 2 + model_params_b * 8  # policy + ref + optimizer
            
        else:  # GRPO, RLOO, etc.
            total_model_memory = actor_base_mem * 2.5 + model_params_b * 8
        
        # Determine reference model strategy
        if use_peft:
            ref_strategy = ReferenceModelStrategy.SHARED_WEIGHTS
            ref_device = "gpu"
            ref_mem = 0  # Shared with actor
            recommendations.append("Using shared weights for reference (LoRA)")
        elif needs_quantization:
            ref_strategy = ReferenceModelStrategy.QUANTIZED
            ref_device = "gpu"
            ref_mem = ref_base_mem * 0.25  # INT4
            recommendations.append("Reference model quantized to INT4 to fit in memory")
        elif needs_cpu_offload:
            ref_strategy = ReferenceModelStrategy.CPU_OFFLOAD
            ref_device = "cpu"
            ref_mem = 0  # On CPU
            recommendations.append("Reference model offloaded to CPU")
        else:
            ref_strategy = ReferenceModelStrategy.FULL_GPU
            ref_device = "gpu"
            ref_mem = ref_base_mem
        
        # Build model plans
        actor = RLModelMemoryPlan(
            name="actor",
            params_billion=model_params_b,
            memory_gb=actor_base_mem + (model_params_b * 8 if not use_peft else peft_rank * 0.001),
            strategy="full" if not use_peft else "lora",
            device="gpu",
            notes=["Includes optimizer states"] if not use_peft else ["LoRA adapters only"]
        )
        
        critic = None
        if algorithm == RLAlgorithm.PPO:
            critic = RLModelMemoryPlan(
                name="critic",
                params_billion=critic_params,
                memory_gb=critic_base_mem + critic_params * 8,
                strategy="full",
                device="gpu",
                notes=["Value head for PPO"]
            )
        
        reference = RLModelMemoryPlan(
            name="reference",
            params_billion=model_params_b,
            memory_gb=ref_mem,
            strategy=ref_strategy.value,
            device=ref_device,
            quantization="int4" if needs_quantization else None,
            notes=["For KL divergence computation"]
        )
        
        reward = None
        if algorithm in [RLAlgorithm.PPO, RLAlgorithm.GRPO]:
            reward = RLModelMemoryPlan(
                name="reward",
                params_billion=reward_params,
                memory_gb=reward_base_mem,
                strategy="frozen",
                device="gpu",
                notes=["Frozen reward model"]
            )
        
        # Calculate batch settings
        available_for_batch = self.total_gpu_memory - total_model_memory * 0.9
        generation_batch = max(1, int(available_for_batch / (model_params_b * 0.1)))
        ppo_batch = generation_batch * 4
        ppo_mini_batch = max(1, generation_batch // 4)
        
        # Memory breakdown
        memory_breakdown = {
            "actor_weights": actor_base_mem,
            "actor_optimizer": model_params_b * 8 if not use_peft else peft_rank * 0.001,
            "critic_weights": critic_base_mem if critic else 0,
            "critic_optimizer": critic_params * 8 if critic else 0,
            "reference": ref_mem,
            "reward": reward_base_mem if reward else 0,
            "activations": model_params_b * 0.5,  # Estimate
            "kv_cache": model_params_b * 0.2,  # For generation
        }
        
        # Warnings
        if total_model_memory > self.total_gpu_memory * 0.95:
            warnings.append(f"Memory very tight: {total_model_memory:.1f}GB needed, {self.total_gpu_memory:.1f}GB available")
        
        if algorithm == RLAlgorithm.PPO and not use_peft and model_params_b > 13:
            warnings.append("Consider using LoRA for large models to reduce memory")
        
        # Final recommendations
        if algorithm == RLAlgorithm.PPO:
            recommendations.append("Use PPO with GAE for stable training")
            recommendations.append(f"Recommended PPO epochs: 4")
        elif algorithm == RLAlgorithm.DPO:
            recommendations.append("DPO is simpler than PPO - no reward model or critic needed")
            recommendations.append("Use beta=0.1 as starting point")
        elif algorithm == RLAlgorithm.GRPO:
            recommendations.append("GRPO uses group-relative rewards - good for math/coding")
        
        return RLMemoryPlan(
            algorithm=algorithm,
            total_gpu_memory_needed=total_model_memory,
            total_cpu_memory_needed=ref_base_mem if ref_device == "cpu" else 0,
            actor=actor,
            critic=critic,
            reference=reference,
            reward=reward,
            ppo_batch_size=ppo_batch,
            ppo_mini_batch_size=ppo_mini_batch,
            generation_batch_size=generation_batch,
            memory_breakdown=memory_breakdown,
            recommendations=recommendations,
            warnings=warnings,
        )
    
    def generate_config(
        self,
        model_name: str,
        model_params_b: float,
        algorithm: RLAlgorithm = RLAlgorithm.PPO,
        num_gpus: int = 8,
        use_peft: bool = True,
        peft_rank: int = 16,
    ) -> RLTrainingConfig:
        """
        Generate complete RL training configuration.
        """
        
        memory_plan = self.plan_memory(
            model_params_b, algorithm, use_peft=use_peft, peft_rank=peft_rank
        )
        
        # Determine parallelism
        if model_params_b > 70:
            actor_tp = min(8, num_gpus)
            actor_pp = max(1, num_gpus // actor_tp // 2)
        elif model_params_b > 30:
            actor_tp = min(4, num_gpus)
            actor_pp = 1
        else:
            actor_tp = 1
            actor_pp = 1
        
        critic_tp = actor_tp
        critic_pp = actor_pp
        dp = num_gpus // (actor_tp * actor_pp)
        
        # Reference strategy
        if use_peft:
            ref_strategy = ReferenceModelStrategy.SHARED_WEIGHTS
        elif memory_plan.total_gpu_memory_needed > self.total_gpu_memory * 0.8:
            ref_strategy = ReferenceModelStrategy.CPU_OFFLOAD
        else:
            ref_strategy = ReferenceModelStrategy.FULL_GPU
        
        # PEFT config
        peft_config = None
        if use_peft:
            peft_config = {
                "r": peft_rank,
                "lora_alpha": peft_rank * 2,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "task_type": "CAUSAL_LM",
            }
        
        # Generation config
        generation_config = {
            "max_new_tokens": 512,
            "do_sample": True,
            "top_p": 0.9,
            "temperature": 0.7,
            "num_return_sequences": 1,
        }
        
        # DeepSpeed config
        deepspeed_config = {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": memory_plan.ppo_mini_batch_size,
            "gradient_accumulation_steps": max(1, memory_plan.ppo_batch_size // (memory_plan.ppo_mini_batch_size * dp)),
            "zero_optimization": {
                "stage": 3 if model_params_b > 30 else 2,
                "offload_optimizer": {
                    "device": "cpu" if ref_strategy == ReferenceModelStrategy.CPU_OFFLOAD else "none",
                },
                "offload_param": {
                    "device": "cpu" if ref_strategy == ReferenceModelStrategy.CPU_OFFLOAD else "none",
                },
                "overlap_comm": True,
            },
            "bf16": {"enabled": True},
            "gradient_clipping": 1.0,
        }
        
        # TRL launch command
        if algorithm == RLAlgorithm.PPO:
            trl_command = f"""accelerate launch --config_file accelerate_config.yaml \\
    --num_processes {num_gpus} \\
    ppo_trainer.py \\
    --model_name {model_name} \\
    --reward_model_name {model_name}-rm \\
    --ppo_epochs 4 \\
    --mini_batch_size {memory_plan.ppo_mini_batch_size} \\
    --batch_size {memory_plan.ppo_batch_size} \\
    {"--use_peft" if use_peft else ""} \\
    --bf16"""
        elif algorithm == RLAlgorithm.DPO:
            trl_command = f"""accelerate launch --config_file accelerate_config.yaml \\
    --num_processes {num_gpus} \\
    dpo_trainer.py \\
    --model_name {model_name} \\
    --beta 0.1 \\
    --batch_size {memory_plan.ppo_batch_size} \\
    {"--use_peft" if use_peft else ""} \\
    --bf16"""
        else:
            trl_command = f"# Configure {algorithm.value} training with trl library"
        
        return RLTrainingConfig(
            algorithm=algorithm,
            actor_tp=actor_tp,
            actor_pp=actor_pp,
            critic_tp=critic_tp,
            critic_pp=critic_pp,
            dp=dp,
            reference_strategy=ref_strategy,
            use_peft=use_peft,
            peft_config=peft_config,
            ppo_epochs=4 if algorithm == RLAlgorithm.PPO else 1,
            generation_config=generation_config,
            deepspeed_config=deepspeed_config,
            trl_command=trl_command,
        )


def get_rl_optimization(
    model_config: Dict[str, Any],
    hardware_config: Dict[str, Any],
    rl_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Get RL/RLHF optimization recommendations.
    """
    
    rl = rl_config or {}
    algorithm_str = rl.get("algorithm", "ppo").upper()
    algorithm = RLAlgorithm[algorithm_str] if algorithm_str in RLAlgorithm.__members__ else RLAlgorithm.PPO
    
    optimizer = RLOptimizer(
        gpu_memory_gb=hardware_config.get("gpu_memory_gb", 80),
        num_gpus=hardware_config.get("num_gpus", 8),
    )
    
    memory_plan = optimizer.plan_memory(
        model_params_b=model_config.get("parameters_billions", 70),
        algorithm=algorithm,
        use_peft=rl.get("use_peft", True),
        peft_rank=rl.get("peft_rank", 16),
    )
    
    training_config = optimizer.generate_config(
        model_name=model_config.get("name", "model"),
        model_params_b=model_config.get("parameters_billions", 70),
        algorithm=algorithm,
        num_gpus=hardware_config.get("num_gpus", 8),
        use_peft=rl.get("use_peft", True),
    )
    
    return {
        "algorithm": algorithm.value,
        "memory_plan": memory_plan.to_dict(),
        "training_config": training_config.to_dict(),
        "comparison": _get_algorithm_comparison(),
    }


def _get_algorithm_comparison() -> List[Dict[str, Any]]:
    """Get comparison of RL algorithms."""
    return [
        {
            "algorithm": "PPO",
            "memory_multiplier": "4x",
            "complexity": "high",
            "stability": "high",
            "sample_efficiency": "low",
            "best_for": "General RLHF, complex reward signals",
        },
        {
            "algorithm": "DPO",
            "memory_multiplier": "2x",
            "complexity": "low",
            "stability": "high",
            "sample_efficiency": "high",
            "best_for": "Preference learning, limited compute",
        },
        {
            "algorithm": "GRPO",
            "memory_multiplier": "2.5x",
            "complexity": "medium",
            "stability": "medium",
            "sample_efficiency": "medium",
            "best_for": "Math/coding tasks, group comparisons",
        },
        {
            "algorithm": "KTO",
            "memory_multiplier": "2x",
            "complexity": "low",
            "stability": "high",
            "sample_efficiency": "high",
            "best_for": "Binary feedback, simple preferences",
        },
    ]



