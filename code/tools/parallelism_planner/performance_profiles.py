"""
Performance Profiles Module for Parallelism Planner

Provides workload-specific optimization profiles:
- Training (pre-training, fine-tuning, RLHF)
- Inference (batch, streaming, real-time)
- Long-context scenarios
- MoE-specific optimizations
- Cost-optimized profiles
- Research/experimentation profiles

Each profile combines parallelism, precision, and optimization settings
tuned for specific workload characteristics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class WorkloadType(Enum):
    """Types of workloads with different optimization needs."""
    PRETRAINING = "pretraining"
    FINETUNING = "finetuning"
    RLHF = "rlhf"
    INSTRUCTION_TUNING = "instruction_tuning"
    INFERENCE_BATCH = "inference_batch"
    INFERENCE_STREAMING = "inference_streaming"
    INFERENCE_REALTIME = "inference_realtime"
    LONG_CONTEXT = "long_context"
    MOE_TRAINING = "moe_training"
    DISTILLATION = "distillation"
    EVALUATION = "evaluation"


class OptimizationPriority(Enum):
    """Optimization priority for profile selection."""
    THROUGHPUT = "throughput"          # Maximize tokens/second
    LATENCY = "latency"                # Minimize response time
    MEMORY = "memory"                  # Minimize memory usage
    COST = "cost"                      # Minimize $/token
    QUALITY = "quality"                # Maximize training stability
    BALANCED = "balanced"              # Balance all factors


@dataclass
class PerformanceProfile:
    """A complete performance profile for a specific workload."""
    name: str
    workload_type: WorkloadType
    priority: OptimizationPriority
    description: str
    
    # Parallelism settings
    parallelism: Dict[str, Any]
    
    # Precision settings
    precision: Dict[str, Any]
    
    # Memory optimizations
    memory: Dict[str, Any]
    
    # Batch/sequence settings
    batch_settings: Dict[str, Any]
    
    # Communication settings
    communication: Dict[str, Any]
    
    # Framework-specific configs
    deepspeed_config: Optional[Dict[str, Any]] = None
    megatron_args: Optional[List[str]] = None
    accelerate_config: Optional[Dict[str, Any]] = None
    
    # Performance expectations
    expected_throughput_tps: Optional[float] = None
    expected_memory_gb: Optional[float] = None
    expected_mfu: Optional[float] = None  # Model FLOPS Utilization
    
    # Warnings and notes
    warnings: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)


class ProfileGenerator:
    """Generates performance profiles based on workload and hardware."""
    
    def __init__(
        self,
        gpu_arch: str,
        gpu_memory_gb: float,
        num_gpus: int,
        has_nvlink: bool = False,
        has_infiniband: bool = False,
        num_nodes: int = 1
    ):
        self.gpu_arch = gpu_arch.lower()
        self.gpu_memory_gb = gpu_memory_gb
        self.num_gpus = num_gpus
        self.has_nvlink = has_nvlink
        self.has_infiniband = has_infiniband
        self.num_nodes = num_nodes
        self.total_gpus = num_gpus * num_nodes
        
        # Determine capabilities
        self.supports_fp8 = self.gpu_arch in ['hopper', 'blackwell', 'h100', 'h200', 'b100', 'b200', 'gb200']
        self.supports_bf16 = self.gpu_arch not in ['volta', 'pascal']
    
    def generate_pretraining_profile(
        self,
        model_params_b: float,
        seq_length: int = 4096,
        global_batch_size: int = 1024
    ) -> PerformanceProfile:
        """Generate profile optimized for large-scale pre-training."""
        
        # Calculate optimal parallelism
        # For pretraining: maximize throughput, use large batches
        
        # TP within NVLink domain (typically 8 GPUs)
        tp = min(8, self.num_gpus) if self.has_nvlink else 1
        
        # PP for very large models
        if model_params_b > 70:
            pp = min(8, self.total_gpus // tp)
        elif model_params_b > 30:
            pp = min(4, self.total_gpus // tp)
        else:
            pp = 1
        
        # DP to fill remaining GPUs
        dp = self.total_gpus // (tp * pp)
        
        # Context parallel for long sequences
        cp = 2 if seq_length >= 32768 and self.has_nvlink else 1
        
        # Gradient accumulation
        microbatch_size = 1 if model_params_b > 30 else 2
        grad_accum = global_batch_size // (dp * microbatch_size)
        
        return PerformanceProfile(
            name="Large-Scale Pretraining",
            workload_type=WorkloadType.PRETRAINING,
            priority=OptimizationPriority.THROUGHPUT,
            description=f"Optimized for {model_params_b}B parameter pretraining on {self.total_gpus} GPUs",
            parallelism={
                "data_parallel": dp,
                "tensor_parallel": tp,
                "pipeline_parallel": pp,
                "context_parallel": cp,
                "sequence_parallel": tp > 1
            },
            precision={
                "compute_dtype": "fp8" if self.supports_fp8 else "bf16",
                "param_dtype": "bf16",
                "grad_dtype": "fp32",
                "loss_scaling": "dynamic"
            },
            memory={
                "gradient_checkpointing": True,
                "checkpointing_strategy": "selective",
                "optimizer": "adamw_8bit" if model_params_b > 30 else "adamw",
                "cpu_offload": False,
                "nvme_offload": False
            },
            batch_settings={
                "global_batch_size": global_batch_size,
                "micro_batch_size": microbatch_size,
                "gradient_accumulation_steps": grad_accum,
                "sequence_length": seq_length
            },
            communication={
                "overlap_comm_compute": True,
                "bucket_size_mb": 25,
                "async_allreduce": True,
                "gradient_compression": False
            },
            deepspeed_config=self._get_deepspeed_config(
                zero_stage=1 if pp > 1 else 2,
                grad_accum=grad_accum,
                microbatch=microbatch_size
            ),
            megatron_args=self._get_megatron_args(tp, pp, dp, seq_length),
            expected_mfu=0.45 if self.supports_fp8 else 0.38,
            warnings=[
                "Use interleaved PP scheduling for lower bubble overhead" if pp > 2 else None,
                "Consider gradient compression for multi-node training" if self.num_nodes > 2 else None
            ],
            prerequisites=[
                "PyTorch 2.0+",
                "Flash Attention 2+",
                "Transformer Engine" if self.supports_fp8 else None,
                "NCCL 2.18+" if self.has_nvlink else None
            ]
        )
    
    def generate_finetuning_profile(
        self,
        model_params_b: float,
        seq_length: int = 2048,
        batch_size: int = 32,
        use_lora: bool = False
    ) -> PerformanceProfile:
        """Generate profile optimized for fine-tuning."""
        
        if use_lora:
            # LoRA: minimal parallelism needed
            dp = self.total_gpus
            tp = 1
            pp = 1
            optimizer = "adamw_8bit"
            checkpointing = model_params_b > 13
        else:
            # Full fine-tuning
            tp = min(8, self.num_gpus) if self.has_nvlink and model_params_b > 30 else 1
            pp = 1  # PP adds complexity for fine-tuning
            dp = self.total_gpus // tp
            optimizer = "adamw_8bit" if model_params_b > 30 else "adamw"
            checkpointing = model_params_b > 7
        
        return PerformanceProfile(
            name=f"Fine-tuning {'(LoRA)' if use_lora else '(Full)'}",
            workload_type=WorkloadType.FINETUNING,
            priority=OptimizationPriority.QUALITY,
            description=f"Fine-tuning {model_params_b}B model with {'LoRA' if use_lora else 'full weights'}",
            parallelism={
                "data_parallel": dp,
                "tensor_parallel": tp,
                "pipeline_parallel": pp,
                "context_parallel": 1,
                "sequence_parallel": tp > 1
            },
            precision={
                "compute_dtype": "bf16",
                "param_dtype": "bf16" if not use_lora else "fp32",
                "grad_dtype": "fp32",
                "loss_scaling": "dynamic"
            },
            memory={
                "gradient_checkpointing": checkpointing,
                "checkpointing_strategy": "full" if checkpointing else "none",
                "optimizer": optimizer,
                "lora_rank": 16 if use_lora else None,
                "lora_alpha": 32 if use_lora else None
            },
            batch_settings={
                "global_batch_size": batch_size,
                "micro_batch_size": max(1, batch_size // (dp * 4)),
                "gradient_accumulation_steps": 4,
                "sequence_length": seq_length
            },
            communication={
                "overlap_comm_compute": True,
                "bucket_size_mb": 25,
                "async_allreduce": True,
                "gradient_compression": False
            },
            deepspeed_config=self._get_deepspeed_config(
                zero_stage=2 if not use_lora else 0,
                grad_accum=4,
                microbatch=max(1, batch_size // (dp * 4))
            ),
            expected_mfu=0.35,
            warnings=[
                "Consider LoRA for faster iteration" if not use_lora and model_params_b > 13 else None,
                "Use higher rank (32-64) for complex adaptations" if use_lora else None
            ],
            prerequisites=[
                "peft library" if use_lora else None,
                "PyTorch 2.0+"
            ]
        )
    
    def generate_rlhf_profile(
        self,
        model_params_b: float,
        seq_length: int = 2048
    ) -> PerformanceProfile:
        """Generate profile optimized for RLHF training."""
        
        # RLHF has special memory requirements (actor, critic, ref, reward models)
        # Typically need 4x memory of single model
        
        # Conservative parallelism
        tp = min(8, self.num_gpus) if self.has_nvlink else min(4, self.num_gpus)
        pp = 1  # PP complicates RLHF
        dp = self.total_gpus // tp
        
        return PerformanceProfile(
            name="RLHF Training",
            workload_type=WorkloadType.RLHF,
            priority=OptimizationPriority.MEMORY,
            description=f"RLHF training for {model_params_b}B model (needs 4x memory for actor/critic/ref/reward)",
            parallelism={
                "data_parallel": dp,
                "tensor_parallel": tp,
                "pipeline_parallel": pp,
                "context_parallel": 1,
                "sequence_parallel": tp > 1
            },
            precision={
                "compute_dtype": "bf16",
                "param_dtype": "bf16",
                "grad_dtype": "fp32",
                "reference_model_dtype": "bf16"  # Can use lower precision
            },
            memory={
                "gradient_checkpointing": True,
                "checkpointing_strategy": "full",
                "optimizer": "adamw_8bit",
                "cpu_offload": model_params_b > 30,
                "offload_optimizer_states": True,
                "offload_gradients": model_params_b > 30
            },
            batch_settings={
                "global_batch_size": 64,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 64 // dp,
                "sequence_length": seq_length,
                "ppo_epochs": 4,
                "ppo_batch_size": 16
            },
            communication={
                "overlap_comm_compute": True,
                "bucket_size_mb": 50,
                "async_allreduce": True,
                "gradient_compression": False
            },
            deepspeed_config=self._get_deepspeed_rlhf_config(model_params_b),
            expected_mfu=0.25,  # RLHF is less efficient due to sampling
            warnings=[
                "RLHF requires ~4x memory of standard training",
                "Consider freezing reference model on CPU",
                "Use ZeRO-3 with CPU offload for large models"
            ],
            prerequisites=[
                "trl library",
                "DeepSpeed RLHF support",
                "Large CPU memory for offload"
            ]
        )
    
    def generate_inference_profile(
        self,
        model_params_b: float,
        seq_length: int = 4096,
        mode: str = "batch"  # "batch", "streaming", "realtime"
    ) -> PerformanceProfile:
        """Generate profile optimized for inference."""
        
        # Inference: no optimizer states, no gradients
        # Focus on minimizing latency or maximizing throughput
        
        if mode == "realtime":
            # Single GPU, minimal latency
            tp = 1
            pp = 1
            dp = 1
            priority = OptimizationPriority.LATENCY
            batch_size = 1
        elif mode == "streaming":
            # Streaming: moderate parallelism
            tp = min(4, self.num_gpus) if self.has_nvlink else 1
            pp = 1
            dp = self.num_gpus // tp
            priority = OptimizationPriority.LATENCY
            batch_size = 8
        else:
            # Batch: maximize throughput
            tp = min(8, self.num_gpus) if self.has_nvlink else 1
            pp = 1
            dp = self.num_gpus // tp
            priority = OptimizationPriority.THROUGHPUT
            batch_size = 32
        
        # KV cache optimization
        kv_cache_dtype = "fp8" if self.supports_fp8 else "bf16"
        
        return PerformanceProfile(
            name=f"Inference ({mode.title()})",
            workload_type=WorkloadType.INFERENCE_BATCH if mode == "batch" else WorkloadType.INFERENCE_STREAMING,
            priority=priority,
            description=f"{mode.title()} inference for {model_params_b}B model",
            parallelism={
                "data_parallel": dp,
                "tensor_parallel": tp,
                "pipeline_parallel": pp,
                "context_parallel": 1,
                "sequence_parallel": False  # Not needed for inference
            },
            precision={
                "compute_dtype": "fp8" if self.supports_fp8 else "bf16",
                "param_dtype": "bf16",
                "kv_cache_dtype": kv_cache_dtype,
                "attention_dtype": "fp8" if self.supports_fp8 else "bf16"
            },
            memory={
                "gradient_checkpointing": False,  # No gradients in inference
                "kv_cache_optimization": True,
                "paged_attention": True,
                "continuous_batching": mode != "realtime",
                "speculative_decoding": mode == "realtime" and model_params_b < 30
            },
            batch_settings={
                "max_batch_size": batch_size,
                "max_sequence_length": seq_length,
                "max_new_tokens": 512,
                "prefill_chunk_size": 2048
            },
            communication={
                "overlap_comm_compute": tp > 1,
                "async_allreduce": False,
                "gradient_compression": False
            },
            expected_throughput_tps=self._estimate_inference_throughput(model_params_b, tp, batch_size),
            warnings=[
                "Use vLLM or TensorRT-LLM for production inference",
                "Consider quantization (AWQ/GPTQ) for memory efficiency"
            ],
            prerequisites=[
                "vLLM or TensorRT-LLM",
                "Flash Attention 2+",
                "PagedAttention support"
            ]
        )
    
    def generate_long_context_profile(
        self,
        model_params_b: float,
        seq_length: int = 131072  # 128K
    ) -> PerformanceProfile:
        """Generate profile optimized for long-context training/inference."""
        
        # Long context requires context parallelism and ring attention
        
        # Context parallel for long sequences
        if seq_length >= 65536:
            cp = min(8, self.num_gpus) if self.has_nvlink else 4
        elif seq_length >= 32768:
            cp = min(4, self.num_gpus) if self.has_nvlink else 2
        else:
            cp = 2 if self.has_nvlink else 1
        
        tp = min(8 // cp, self.num_gpus // cp) if self.has_nvlink else 1
        pp = 1
        dp = self.total_gpus // (tp * cp * pp)
        
        return PerformanceProfile(
            name="Long Context",
            workload_type=WorkloadType.LONG_CONTEXT,
            priority=OptimizationPriority.MEMORY,
            description=f"Long context ({seq_length//1024}K) processing for {model_params_b}B model",
            parallelism={
                "data_parallel": dp,
                "tensor_parallel": tp,
                "pipeline_parallel": pp,
                "context_parallel": cp,
                "sequence_parallel": True,
                "ring_attention": True
            },
            precision={
                "compute_dtype": "bf16",
                "param_dtype": "bf16",
                "kv_cache_dtype": "fp8" if self.supports_fp8 else "bf16",
                "attention_dtype": "bf16"
            },
            memory={
                "gradient_checkpointing": True,
                "checkpointing_strategy": "full",
                "kv_cache_optimization": True,
                "sliding_window_attention": seq_length > 65536,
                "flash_attention": True
            },
            batch_settings={
                "global_batch_size": 4,  # Small batches for long context
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "sequence_length": seq_length,
                "context_window": seq_length
            },
            communication={
                "overlap_comm_compute": True,
                "ring_allreduce": True,
                "async_p2p": True,
                "bucket_size_mb": 100  # Larger buckets for ring attention
            },
            warnings=[
                "Long context requires O(n²) memory for attention without optimization",
                "Use Ring Attention or Striped Attention for linear memory",
                f"KV cache alone will use ~{self._estimate_kv_cache(model_params_b, seq_length):.1f}GB per GPU"
            ],
            prerequisites=[
                "Flash Attention 2+",
                "Ring Attention support",
                "Context Parallelism support (Megatron-LM)"
            ]
        )
    
    def generate_moe_profile(
        self,
        model_params_b: float,
        num_experts: int = 8,
        top_k: int = 2,
        seq_length: int = 4096
    ) -> PerformanceProfile:
        """Generate profile optimized for MoE model training."""
        
        # MoE-specific: Expert Parallelism
        
        # EP should divide number of experts evenly
        ep_candidates = [e for e in [1, 2, 4, 8, num_experts] if num_experts % e == 0]
        ep = max([e for e in ep_candidates if e <= self.num_gpus])
        
        tp = min(8 // ep, self.num_gpus // ep) if self.has_nvlink else 1
        pp = 1
        dp = self.total_gpus // (tp * ep * pp)
        
        return PerformanceProfile(
            name="MoE Training",
            workload_type=WorkloadType.MOE_TRAINING,
            priority=OptimizationPriority.THROUGHPUT,
            description=f"MoE training for {model_params_b}B model with {num_experts} experts (top-{top_k})",
            parallelism={
                "data_parallel": dp,
                "tensor_parallel": tp,
                "pipeline_parallel": pp,
                "context_parallel": 1,
                "expert_parallel": ep,
                "sequence_parallel": tp > 1
            },
            precision={
                "compute_dtype": "bf16",
                "param_dtype": "bf16",
                "grad_dtype": "fp32",
                "router_dtype": "fp32"  # Router benefits from FP32
            },
            memory={
                "gradient_checkpointing": True,
                "checkpointing_strategy": "selective",
                "optimizer": "adamw_8bit",
                "expert_capacity_factor": 1.25
            },
            batch_settings={
                "global_batch_size": 512,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 512 // (dp * 2),
                "sequence_length": seq_length,
                "top_k_experts": top_k
            },
            communication={
                "overlap_comm_compute": True,
                "bucket_size_mb": 25,
                "async_allreduce": True,
                "all_to_all_for_experts": True
            },
            warnings=[
                "MoE requires balanced expert routing for efficiency",
                "Consider auxiliary load balancing loss",
                "Expert communication can bottleneck without high-bandwidth interconnect"
            ],
            prerequisites=[
                "Megablocks or Tutel for efficient MoE",
                "High-bandwidth interconnect for all-to-all"
            ]
        )
    
    def generate_cost_optimized_profile(
        self,
        model_params_b: float,
        target_cost_per_token: float = 0.0001,
        seq_length: int = 4096
    ) -> PerformanceProfile:
        """Generate profile optimized for cost efficiency."""
        
        # Cost optimization: balance throughput vs GPU-hours
        
        # Use fewer GPUs with higher utilization
        min_gpus_for_memory = max(1, int(model_params_b * 2.5 / self.gpu_memory_gb) + 1)
        optimal_gpus = max(min_gpus_for_memory, 8)  # At least 8 for efficiency
        
        tp = min(optimal_gpus, 8) if self.has_nvlink else 1
        pp = 1
        dp = optimal_gpus // tp
        
        return PerformanceProfile(
            name="Cost Optimized",
            workload_type=WorkloadType.PRETRAINING,
            priority=OptimizationPriority.COST,
            description=f"Cost-optimized training for {model_params_b}B model targeting ${target_cost_per_token:.5f}/token",
            parallelism={
                "data_parallel": dp,
                "tensor_parallel": tp,
                "pipeline_parallel": pp,
                "context_parallel": 1,
                "sequence_parallel": tp > 1
            },
            precision={
                "compute_dtype": "bf16",
                "param_dtype": "bf16",
                "grad_dtype": "fp32",
                "loss_scaling": "dynamic"
            },
            memory={
                "gradient_checkpointing": True,
                "checkpointing_strategy": "block",
                "optimizer": "adafactor",  # Memory efficient
                "cpu_offload": model_params_b > 30
            },
            batch_settings={
                "global_batch_size": 256,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 256 // dp,
                "sequence_length": seq_length
            },
            communication={
                "overlap_comm_compute": True,
                "gradient_compression": self.num_nodes > 1,
                "async_allreduce": True
            },
            warnings=[
                "Adafactor may need learning rate tuning",
                "CPU offload adds latency but saves GPU memory/cost",
                "Consider spot instances for further cost reduction"
            ],
            prerequisites=[
                "Spot/preemptible instance support",
                "Checkpoint saving for preemption handling"
            ]
        )
    
    def _get_deepspeed_config(
        self,
        zero_stage: int,
        grad_accum: int,
        microbatch: int
    ) -> Dict[str, Any]:
        """Generate DeepSpeed configuration."""
        return {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": microbatch,
            "gradient_accumulation_steps": grad_accum,
            "zero_optimization": {
                "stage": zero_stage,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 5e7,
                "stage3_prefetch_bucket_size": 5e7 if zero_stage == 3 else None,
                "stage3_param_persistence_threshold": 1e5 if zero_stage == 3 else None
            },
            "bf16": {"enabled": True},
            "gradient_clipping": 1.0,
            "steps_per_print": 100
        }
    
    def _get_deepspeed_rlhf_config(self, model_params_b: float) -> Dict[str, Any]:
        """Generate DeepSpeed config for RLHF."""
        return {
            "train_batch_size": "auto",
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                } if model_params_b > 30 else None,
                "overlap_comm": True,
                "contiguous_gradients": True
            },
            "bf16": {"enabled": True},
            "gradient_clipping": 1.0
        }
    
    def _get_megatron_args(
        self,
        tp: int,
        pp: int,
        dp: int,
        seq_length: int
    ) -> List[str]:
        """Generate Megatron-LM arguments."""
        return [
            f"--tensor-model-parallel-size {tp}",
            f"--pipeline-model-parallel-size {pp}",
            f"--data-parallel-size {dp}",
            f"--seq-length {seq_length}",
            "--use-flash-attn",
            "--bf16",
            "--overlap-grad-reduce",
            "--overlap-param-gather"
        ]
    
    def _estimate_inference_throughput(
        self,
        model_params_b: float,
        tp: int,
        batch_size: int
    ) -> float:
        """Estimate inference throughput in tokens/second."""
        # Rough estimation based on model size and parallelism
        base_throughput = 1000 / model_params_b  # Tokens/second for single GPU
        tp_efficiency = 0.9 ** (tp - 1)  # Diminishing returns
        return base_throughput * tp * batch_size * tp_efficiency
    
    def _estimate_kv_cache(
        self,
        model_params_b: float,
        seq_length: int
    ) -> float:
        """Estimate KV cache size in GB."""
        # KV cache = 2 * num_layers * seq_length * hidden_size * 2 (K and V) * bytes
        # Rough estimate based on model size
        num_layers = int(model_params_b * 2.5)  # Approximate
        hidden_size = int((model_params_b * 1e9 / num_layers / 12) ** 0.5)  # Approximate
        bytes_per_element = 2  # BF16
        
        kv_cache_bytes = 2 * num_layers * seq_length * hidden_size * 2 * bytes_per_element
        return kv_cache_bytes / 1e9


class ProfileSelector:
    """Selects the best profile based on user requirements."""
    
    def __init__(self, generator: ProfileGenerator):
        self.generator = generator
    
    def select_profile(
        self,
        model_params_b: float,
        workload: str,
        seq_length: int = 4096,
        priority: str = "balanced",
        **kwargs
    ) -> PerformanceProfile:
        """
        Select the best profile for the given requirements.
        
        Args:
            model_params_b: Model size in billions
            workload: Workload type string
            seq_length: Sequence length
            priority: Optimization priority
            **kwargs: Additional workload-specific parameters
        """
        
        workload = workload.lower()
        
        if workload in ['pretrain', 'pretraining']:
            return self.generator.generate_pretraining_profile(
                model_params_b,
                seq_length,
                kwargs.get('batch_size', 1024)
            )
        elif workload in ['finetune', 'finetuning', 'sft']:
            return self.generator.generate_finetuning_profile(
                model_params_b,
                seq_length,
                kwargs.get('batch_size', 32),
                kwargs.get('use_lora', False)
            )
        elif workload in ['rlhf', 'ppo', 'dpo']:
            return self.generator.generate_rlhf_profile(
                model_params_b,
                seq_length
            )
        elif workload in ['inference', 'infer']:
            return self.generator.generate_inference_profile(
                model_params_b,
                seq_length,
                kwargs.get('mode', 'batch')
            )
        elif workload in ['long_context', 'long-context', 'longcontext']:
            return self.generator.generate_long_context_profile(
                model_params_b,
                seq_length
            )
        elif workload in ['moe', 'mixture_of_experts']:
            return self.generator.generate_moe_profile(
                model_params_b,
                kwargs.get('num_experts', 8),
                kwargs.get('top_k', 2),
                seq_length
            )
        elif workload in ['cost', 'cost_optimized', 'budget']:
            return self.generator.generate_cost_optimized_profile(
                model_params_b,
                kwargs.get('target_cost_per_token', 0.0001),
                seq_length
            )
        else:
            # Default to balanced pretraining
            return self.generator.generate_pretraining_profile(
                model_params_b,
                seq_length
            )


def get_performance_profile(
    model_params_b: float,
    workload: str,
    hardware_config: Dict[str, Any],
    seq_length: int = 4096,
    **kwargs
) -> Dict[str, Any]:
    """
    Get a performance profile for the given model and workload.
    
    Args:
        model_params_b: Model size in billions
        workload: Workload type (pretraining, finetuning, inference, etc.)
        hardware_config: Hardware configuration dict
        seq_length: Sequence length
        **kwargs: Additional parameters
    
    Returns:
        Performance profile as a dictionary
    """
    
    generator = ProfileGenerator(
        gpu_arch=hardware_config.get('gpu_arch', 'ampere'),
        gpu_memory_gb=hardware_config.get('gpu_memory_gb', 80),
        num_gpus=hardware_config.get('num_gpus', 8),
        has_nvlink=hardware_config.get('has_nvlink', True),
        has_infiniband=hardware_config.get('has_infiniband', False),
        num_nodes=hardware_config.get('num_nodes', 1)
    )
    
    selector = ProfileSelector(generator)
    profile = selector.select_profile(
        model_params_b,
        workload,
        seq_length,
        **kwargs
    )
    
    return {
        "name": profile.name,
        "workload_type": profile.workload_type.value,
        "priority": profile.priority.value,
        "description": profile.description,
        "parallelism": profile.parallelism,
        "precision": profile.precision,
        "memory": profile.memory,
        "batch_settings": profile.batch_settings,
        "communication": profile.communication,
        "deepspeed_config": profile.deepspeed_config,
        "megatron_args": profile.megatron_args,
        "expected_throughput_tps": profile.expected_throughput_tps,
        "expected_memory_gb": profile.expected_memory_gb,
        "expected_mfu": profile.expected_mfu,
        "warnings": [w for w in profile.warnings if w],
        "prerequisites": [p for p in profile.prerequisites if p]
    }


def list_available_profiles() -> List[Dict[str, str]]:
    """List all available performance profiles."""
    return [
        {"name": "pretraining", "description": "Large-scale pre-training with maximum throughput"},
        {"name": "finetuning", "description": "Fine-tuning with quality-focused settings"},
        {"name": "rlhf", "description": "RLHF/PPO training with memory optimization"},
        {"name": "inference", "description": "Inference with latency/throughput optimization"},
        {"name": "long_context", "description": "Long-context (32K-128K+) processing"},
        {"name": "moe", "description": "Mixture-of-Experts training"},
        {"name": "cost_optimized", "description": "Cost-efficient training configuration"}
    ]



