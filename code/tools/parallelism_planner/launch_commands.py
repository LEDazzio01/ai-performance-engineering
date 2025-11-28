#!/usr/bin/env python3
"""
Framework Launch Command Generator

Generates ready-to-use launch commands and configuration files for:
- PyTorch torchrun
- DeepSpeed
- Megatron-LM
- Accelerate
- FSDP
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path

from .strategy_optimizer import ParallelismStrategy
from .sharding_strategies import ShardingStrategy, ShardingConfig


@dataclass
class LaunchConfig:
    """Configuration for distributed launch."""
    
    # Distributed configuration
    num_nodes: int = 1
    gpus_per_node: int = 8
    master_addr: str = "localhost"
    master_port: int = 29500
    
    # Parallelism
    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    cp_size: int = 1
    ep_size: int = 1
    
    # Sharding
    sharding: ShardingStrategy = ShardingStrategy.NO_SHARD
    
    # Training config
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    
    # Model
    model_name: str = ""
    
    # Environment
    nccl_debug: bool = False
    cuda_visible_devices: Optional[str] = None
    
    @property
    def world_size(self) -> int:
        return self.num_nodes * self.gpus_per_node
    
    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps * self.dp_size


class TorchrunGenerator:
    """Generate torchrun launch commands."""
    
    def generate(
        self,
        config: LaunchConfig,
        script: str = "train.py",
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate torchrun command.
        
        Args:
            config: Launch configuration
            script: Training script to run
            extra_args: Additional script arguments
            
        Returns:
            Complete torchrun command
        """
        parts = ["torchrun"]
        
        # Distributed config
        if config.num_nodes > 1:
            parts.extend([
                f"--nnodes={config.num_nodes}",
                f"--nproc_per_node={config.gpus_per_node}",
                f"--rdzv_backend=c10d",
                f"--rdzv_endpoint={config.master_addr}:{config.master_port}",
            ])
        else:
            parts.append(f"--nproc_per_node={config.gpus_per_node}")
        
        # Add script
        parts.append(script)
        
        # Script arguments
        if config.tp_size > 1:
            parts.append(f"--tensor-parallel-size {config.tp_size}")
        if config.pp_size > 1:
            parts.append(f"--pipeline-parallel-size {config.pp_size}")
        if config.micro_batch_size > 1:
            parts.append(f"--micro-batch-size {config.micro_batch_size}")
        if config.gradient_accumulation_steps > 1:
            parts.append(f"--gradient-accumulation-steps {config.gradient_accumulation_steps}")
        
        # Extra arguments
        if extra_args:
            for key, value in extra_args.items():
                if isinstance(value, bool):
                    if value:
                        parts.append(f"--{key}")
                else:
                    parts.append(f"--{key} {value}")
        
        return " \\\n    ".join(parts)
    
    def generate_multi_node(
        self,
        config: LaunchConfig,
        script: str = "train.py",
        node_rank: int = 0,
    ) -> str:
        """Generate torchrun command for specific node in multi-node setup."""
        parts = [
            "torchrun",
            f"--nnodes={config.num_nodes}",
            f"--nproc_per_node={config.gpus_per_node}",
            f"--node_rank={node_rank}",
            f"--master_addr={config.master_addr}",
            f"--master_port={config.master_port}",
            script,
        ]
        
        return " \\\n    ".join(parts)


class DeepSpeedGenerator:
    """Generate DeepSpeed launch commands and config files."""
    
    def generate_command(
        self,
        config: LaunchConfig,
        script: str = "train.py",
        config_file: str = "ds_config.json",
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate deepspeed launch command."""
        parts = ["deepspeed"]
        
        # Multi-node
        if config.num_nodes > 1:
            parts.extend([
                f"--num_nodes={config.num_nodes}",
                f"--num_gpus={config.gpus_per_node}",
                f"--master_addr={config.master_addr}",
                f"--master_port={config.master_port}",
            ])
        else:
            parts.append(f"--num_gpus={config.gpus_per_node}")
        
        # Add script and config
        parts.extend([
            script,
            f"--deepspeed_config {config_file}",
        ])
        
        # Extra arguments
        if extra_args:
            for key, value in extra_args.items():
                if isinstance(value, bool):
                    if value:
                        parts.append(f"--{key}")
                else:
                    parts.append(f"--{key} {value}")
        
        return " \\\n    ".join(parts)
    
    def generate_config(
        self,
        config: LaunchConfig,
        optimizer: str = "AdamW",
        learning_rate: float = 1e-4,
        warmup_steps: int = 100,
        fp16: bool = False,
        bf16: bool = True,
    ) -> Dict[str, Any]:
        """Generate DeepSpeed configuration file content."""
        
        # Determine ZeRO stage from sharding strategy
        zero_stage = 0
        if config.sharding == ShardingStrategy.ZERO_1:
            zero_stage = 1
        elif config.sharding in (ShardingStrategy.ZERO_2, ShardingStrategy.FSDP_SHARD_GRAD_OP):
            zero_stage = 2
        elif config.sharding in (ShardingStrategy.ZERO_3, ShardingStrategy.FSDP_FULL):
            zero_stage = 3
        
        ds_config = {
            "train_micro_batch_size_per_gpu": config.micro_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "steps_per_print": 100,
            
            "optimizer": {
                "type": optimizer,
                "params": {
                    "lr": learning_rate,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01,
                },
            },
            
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": learning_rate,
                    "warmup_num_steps": warmup_steps,
                    "total_num_steps": 10000,
                },
            },
            
            "gradient_clipping": 1.0,
        }
        
        # Precision
        if bf16:
            ds_config["bf16"] = {"enabled": True}
        elif fp16:
            ds_config["fp16"] = {
                "enabled": True,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1,
            }
        
        # ZeRO optimization
        if zero_stage > 0:
            zero_config = {
                "stage": zero_stage,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 5e8,
                "allgather_bucket_size": 5e8,
            }
            
            if zero_stage == 3:
                zero_config.update({
                    "stage3_prefetch_bucket_size": 5e8,
                    "stage3_param_persistence_threshold": 1e6,
                    "stage3_max_live_parameters": 1e9,
                    "stage3_max_reuse_distance": 1e9,
                    "stage3_gather_16bit_weights_on_model_save": True,
                })
            
            ds_config["zero_optimization"] = zero_config
        
        # Tensor parallelism (if using Megatron-DeepSpeed)
        if config.tp_size > 1 or config.pp_size > 1:
            ds_config["tensor_parallel"] = {
                "tp_size": config.tp_size,
            }
            ds_config["pipeline"] = {
                "pipe_parallel_size": config.pp_size,
                "num_micro_batches": config.gradient_accumulation_steps,
            }
        
        # Activation checkpointing
        ds_config["activation_checkpointing"] = {
            "partition_activations": True,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": True,
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
            "profile": False,
        }
        
        return ds_config
    
    def generate_hostfile(
        self,
        hostnames: List[str],
        slots_per_host: int = 8,
    ) -> str:
        """Generate DeepSpeed hostfile content."""
        lines = []
        for hostname in hostnames:
            lines.append(f"{hostname} slots={slots_per_host}")
        return "\n".join(lines)


class AccelerateGenerator:
    """Generate HuggingFace Accelerate config and commands."""
    
    def generate_config(
        self,
        config: LaunchConfig,
    ) -> Dict[str, Any]:
        """Generate accelerate config.yaml content."""
        
        # Determine distributed type
        distributed_type = "MULTI_GPU"
        if config.num_nodes > 1:
            distributed_type = "MULTI_NODE"
        if config.sharding in (ShardingStrategy.FSDP_FULL, ShardingStrategy.FSDP_SHARD_GRAD_OP):
            distributed_type = "FSDP"
        if config.sharding in (ShardingStrategy.ZERO_1, ShardingStrategy.ZERO_2, ShardingStrategy.ZERO_3):
            distributed_type = "DEEPSPEED"
        
        acc_config = {
            "compute_environment": "LOCAL_MACHINE",
            "distributed_type": distributed_type,
            "mixed_precision": "bf16",
            "num_machines": config.num_nodes,
            "num_processes": config.world_size,
            "machine_rank": 0,
            "main_process_ip": config.master_addr,
            "main_process_port": config.master_port,
            "use_cpu": False,
        }
        
        # FSDP config
        if "FSDP" in distributed_type:
            acc_config["fsdp_config"] = {
                "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                "fsdp_backward_prefetch_policy": "BACKWARD_PRE",
                "fsdp_forward_prefetch": False,
                "fsdp_offload_params": False,
                "fsdp_sharding_strategy": "FULL_SHARD" if config.sharding == ShardingStrategy.FSDP_FULL else "SHARD_GRAD_OP",
                "fsdp_state_dict_type": "FULL_STATE_DICT",
                "fsdp_use_orig_params": True,
            }
        
        return acc_config
    
    def generate_command(
        self,
        config: LaunchConfig,
        script: str = "train.py",
        config_file: str = "accelerate_config.yaml",
    ) -> str:
        """Generate accelerate launch command."""
        parts = [
            "accelerate launch",
            f"--config_file {config_file}",
            script,
        ]
        
        return " \\\n    ".join(parts)


class MegatronGenerator:
    """Generate Megatron-LM launch arguments."""
    
    def generate_args(
        self,
        config: LaunchConfig,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate Megatron-LM arguments."""
        
        args = {
            # Distributed
            "tensor-model-parallel-size": config.tp_size,
            "pipeline-model-parallel-size": config.pp_size,
            "distributed-backend": "nccl",
            
            # Training
            "micro-batch-size": config.micro_batch_size,
            "global-batch-size": config.effective_batch_size,
            
            # Precision
            "bf16": True,
            
            # Optimization
            "use-flash-attn-v2": True,
            "use-distributed-optimizer": config.sharding != ShardingStrategy.NO_SHARD,
            "overlap-grad-reduce": True,
            "overlap-param-gather": True,
        }
        
        # Context parallel
        if config.cp_size > 1:
            args["context-parallel-size"] = config.cp_size
        
        # Expert parallel
        if config.ep_size > 1:
            args["expert-model-parallel-size"] = config.ep_size
        
        # Model config
        if model_config:
            args.update(model_config)
        
        return args
    
    def generate_command(
        self,
        config: LaunchConfig,
        script: str = "pretrain_gpt.py",
        megatron_args: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate Megatron-LM launch command."""
        args = megatron_args or self.generate_args(config)
        
        parts = [
            "python -m torch.distributed.run",
            f"--nnodes={config.num_nodes}",
            f"--nproc_per_node={config.gpus_per_node}",
            f"--master_addr={config.master_addr}",
            f"--master_port={config.master_port}",
            script,
        ]
        
        # Add Megatron args
        for key, value in args.items():
            if isinstance(value, bool):
                if value:
                    parts.append(f"--{key}")
            else:
                parts.append(f"--{key} {value}")
        
        return " \\\n    ".join(parts)


class LaunchCommandGenerator:
    """Unified interface for generating launch commands."""
    
    def __init__(self):
        self.torchrun = TorchrunGenerator()
        self.deepspeed = DeepSpeedGenerator()
        self.accelerate = AccelerateGenerator()
        self.megatron = MegatronGenerator()
    
    def from_strategy(
        self,
        strategy: ParallelismStrategy,
        num_nodes: int = 1,
        gpus_per_node: int = 8,
        sharding: ShardingStrategy = ShardingStrategy.NO_SHARD,
    ) -> LaunchConfig:
        """Create LaunchConfig from ParallelismStrategy."""
        return LaunchConfig(
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            tp_size=strategy.tp,
            pp_size=strategy.pp,
            dp_size=strategy.dp,
            cp_size=strategy.cp,
            ep_size=strategy.ep,
            sharding=sharding,
            micro_batch_size=strategy.micro_batch_size,
            gradient_accumulation_steps=strategy.gradient_accumulation,
        )
    
    def generate_all(
        self,
        config: LaunchConfig,
        script: str = "train.py",
    ) -> Dict[str, Any]:
        """Generate all launch commands and configs.
        
        Returns:
            Dictionary with all framework commands and configs
        """
        return {
            "torchrun": {
                "command": self.torchrun.generate(config, script),
                "multi_node_commands": [
                    self.torchrun.generate_multi_node(config, script, i)
                    for i in range(config.num_nodes)
                ] if config.num_nodes > 1 else None,
            },
            "deepspeed": {
                "command": self.deepspeed.generate_command(config, script),
                "config": self.deepspeed.generate_config(config),
            },
            "accelerate": {
                "command": self.accelerate.generate_command(config, script),
                "config": self.accelerate.generate_config(config),
            },
            "megatron": {
                "command": self.megatron.generate_command(config, script),
                "args": self.megatron.generate_args(config),
            },
            "environment": self._generate_env_vars(config),
        }
    
    def _generate_env_vars(self, config: LaunchConfig) -> Dict[str, str]:
        """Generate recommended environment variables."""
        env = {
            "NCCL_DEBUG": "WARN" if config.nccl_debug else "ERROR",
            "NCCL_IB_DISABLE": "0",  # Enable InfiniBand
            "NCCL_NET_GDR_LEVEL": "2",  # GPUDirect RDMA
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "OMP_NUM_THREADS": "1",
        }
        
        if config.num_nodes > 1:
            env.update({
                "MASTER_ADDR": config.master_addr,
                "MASTER_PORT": str(config.master_port),
            })
        
        if config.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
        
        return env
    
    def format_launch_guide(
        self,
        config: LaunchConfig,
        script: str = "train.py",
    ) -> str:
        """Generate human-readable launch guide."""
        all_cmds = self.generate_all(config, script)
        
        lines = [
            "=" * 80,
            "DISTRIBUTED TRAINING LAUNCH GUIDE",
            "=" * 80,
            "",
            f"Configuration:",
            f"  Nodes: {config.num_nodes}",
            f"  GPUs per node: {config.gpus_per_node}",
            f"  Total GPUs: {config.world_size}",
            f"  TP={config.tp_size} × PP={config.pp_size} × DP={config.dp_size}",
            f"  Sharding: {config.sharding.value}",
            "",
            "=" * 80,
            "OPTION 1: PyTorch torchrun",
            "=" * 80,
            "",
            all_cmds["torchrun"]["command"],
            "",
        ]
        
        if config.num_nodes > 1:
            lines.extend([
                "For multi-node, run on each node:",
                "",
            ])
            for i, cmd in enumerate(all_cmds["torchrun"]["multi_node_commands"][:2]):
                lines.extend([f"# Node {i}:", cmd, ""])
        
        lines.extend([
            "=" * 80,
            "OPTION 2: DeepSpeed",
            "=" * 80,
            "",
            all_cmds["deepspeed"]["command"],
            "",
            "# Save this as ds_config.json:",
            json.dumps(all_cmds["deepspeed"]["config"], indent=2),
            "",
            "=" * 80,
            "OPTION 3: HuggingFace Accelerate",
            "=" * 80,
            "",
            all_cmds["accelerate"]["command"],
            "",
            "# Save accelerate_config.yaml:",
            "# " + str(all_cmds["accelerate"]["config"]),
            "",
            "=" * 80,
            "ENVIRONMENT VARIABLES",
            "=" * 80,
            "",
        ])
        
        for key, value in all_cmds["environment"].items():
            lines.append(f"export {key}={value}")
        
        lines.extend(["", "=" * 80])
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Demo
    config = LaunchConfig(
        num_nodes=2,
        gpus_per_node=8,
        tp_size=4,
        pp_size=2,
        dp_size=2,
        sharding=ShardingStrategy.ZERO_3,
        micro_batch_size=1,
        gradient_accumulation_steps=8,
        master_addr="node0",
    )
    
    generator = LaunchCommandGenerator()
    print(generator.format_launch_guide(config, "train.py"))
