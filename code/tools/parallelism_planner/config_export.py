"""
Configuration Export Module for Parallelism Planner

Export complete training configurations in various formats:
- DeepSpeed JSON config
- Accelerate YAML config
- Megatron-LM arguments
- FSDP config
- Environment variables
- Complete training script
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ExportedConfig:
    """A complete exported configuration."""
    name: str
    description: str
    
    # Framework configs
    deepspeed_config: Optional[Dict[str, Any]] = None
    accelerate_config: Optional[Dict[str, Any]] = None
    fsdp_config: Optional[Dict[str, Any]] = None
    megatron_args: Optional[List[str]] = None
    
    # Environment
    environment_variables: Dict[str, str] = field(default_factory=dict)
    
    # Launch commands
    torchrun_command: Optional[str] = None
    deepspeed_command: Optional[str] = None
    
    # Training script template
    training_script: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "deepspeed_config": self.deepspeed_config,
            "accelerate_config": self.accelerate_config,
            "fsdp_config": self.fsdp_config,
            "megatron_args": self.megatron_args,
            "environment_variables": self.environment_variables,
            "torchrun_command": self.torchrun_command,
            "deepspeed_command": self.deepspeed_command,
            "training_script": self.training_script,
        }


class ConfigExporter:
    """Exports complete training configurations."""
    
    def export_full_config(
        self,
        model_name: str,
        model_params_b: float,
        num_nodes: int,
        gpus_per_node: int,
        tp: int,
        pp: int,
        dp: int,
        batch_size: int,
        seq_length: int,
        precision: str = "bf16",
        optimizer: str = "adamw",
        learning_rate: float = 1e-4,
        gradient_checkpointing: bool = True,
        zero_stage: int = 2,
        master_addr: str = "localhost",
        master_port: int = 29500,
    ) -> ExportedConfig:
        """
        Export a complete training configuration.
        """
        
        total_gpus = num_nodes * gpus_per_node
        
        # DeepSpeed config
        deepspeed_config = self._generate_deepspeed_config(
            batch_size, dp, zero_stage, precision, optimizer, learning_rate, gradient_checkpointing
        )
        
        # Accelerate config
        accelerate_config = self._generate_accelerate_config(
            num_nodes, gpus_per_node, precision, zero_stage
        )
        
        # FSDP config
        fsdp_config = self._generate_fsdp_config(precision)
        
        # Megatron args
        megatron_args = self._generate_megatron_args(
            tp, pp, dp, seq_length, batch_size, precision
        )
        
        # Environment variables
        env_vars = self._generate_environment_variables(
            num_nodes, gpus_per_node, master_addr, master_port, tp
        )
        
        # Launch commands
        torchrun_cmd = self._generate_torchrun_command(
            num_nodes, gpus_per_node, master_addr, master_port, tp, pp
        )
        
        deepspeed_cmd = self._generate_deepspeed_command(
            num_nodes, gpus_per_node, master_addr
        )
        
        # Training script template
        training_script = self._generate_training_script(
            model_name, precision, gradient_checkpointing, learning_rate, tp, pp
        )
        
        return ExportedConfig(
            name=f"{model_name}_training_config",
            description=f"Training config for {model_name} ({model_params_b:.1f}B) on {total_gpus} GPUs",
            deepspeed_config=deepspeed_config,
            accelerate_config=accelerate_config,
            fsdp_config=fsdp_config,
            megatron_args=megatron_args,
            environment_variables=env_vars,
            torchrun_command=torchrun_cmd,
            deepspeed_command=deepspeed_cmd,
            training_script=training_script,
        )
    
    def _generate_deepspeed_config(
        self,
        batch_size: int,
        dp: int,
        zero_stage: int,
        precision: str,
        optimizer: str,
        learning_rate: float,
        gradient_checkpointing: bool,
    ) -> Dict[str, Any]:
        """Generate DeepSpeed config."""
        
        micro_batch = max(1, batch_size // dp // 8)
        grad_accum = max(1, batch_size // (micro_batch * dp))
        
        config = {
            "train_batch_size": batch_size,
            "train_micro_batch_size_per_gpu": micro_batch,
            "gradient_accumulation_steps": grad_accum,
            "gradient_clipping": 1.0,
            "steps_per_print": 100,
            "zero_optimization": {
                "stage": zero_stage,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 5e7,
            },
        }
        
        # Precision
        if precision == "bf16":
            config["bf16"] = {"enabled": True}
        elif precision == "fp16":
            config["fp16"] = {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 16,
            }
        
        # ZeRO-3 specific
        if zero_stage == 3:
            config["zero_optimization"].update({
                "stage3_prefetch_bucket_size": 5e7,
                "stage3_param_persistence_threshold": 1e5,
                "stage3_gather_16bit_weights_on_model_save": True,
            })
        
        # Optimizer
        if optimizer == "adamw":
            config["optimizer"] = {
                "type": "AdamW",
                "params": {
                    "lr": learning_rate,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01,
                }
            }
        
        # Gradient checkpointing
        if gradient_checkpointing:
            config["activation_checkpointing"] = {
                "partition_activations": True,
                "contiguous_memory_optimization": True,
            }
        
        return config
    
    def _generate_accelerate_config(
        self,
        num_nodes: int,
        gpus_per_node: int,
        precision: str,
        zero_stage: int,
    ) -> Dict[str, Any]:
        """Generate Accelerate config."""
        
        return {
            "compute_environment": "LOCAL_MACHINE",
            "distributed_type": "DEEPSPEED" if zero_stage > 0 else "MULTI_GPU",
            "downcast_bf16": "no",
            "machine_rank": 0,
            "main_training_function": "main",
            "mixed_precision": precision,
            "num_machines": num_nodes,
            "num_processes": num_nodes * gpus_per_node,
            "rdzv_backend": "static",
            "same_network": True,
            "tpu_env": [],
            "tpu_use_cluster": False,
            "tpu_use_sudo": False,
            "use_cpu": False,
            "deepspeed_config": {
                "zero_stage": zero_stage,
                "offload_optimizer_device": "none",
                "offload_param_device": "none",
            } if zero_stage > 0 else None,
        }
    
    def _generate_fsdp_config(self, precision: str) -> Dict[str, Any]:
        """Generate FSDP config."""
        
        return {
            "sharding_strategy": "FULL_SHARD",
            "cpu_offload": False,
            "mixed_precision": {
                "param_dtype": "torch.bfloat16" if precision == "bf16" else "torch.float16",
                "reduce_dtype": "torch.bfloat16" if precision == "bf16" else "torch.float16",
                "buffer_dtype": "torch.bfloat16" if precision == "bf16" else "torch.float16",
            },
            "backward_prefetch": "BACKWARD_PRE",
            "forward_prefetch": True,
            "use_orig_params": True,
            "limit_all_gathers": True,
            "activation_checkpointing": True,
        }
    
    def _generate_megatron_args(
        self,
        tp: int,
        pp: int,
        dp: int,
        seq_length: int,
        batch_size: int,
        precision: str,
    ) -> List[str]:
        """Generate Megatron-LM arguments."""
        
        micro_batch = max(1, batch_size // dp // 8)
        
        args = [
            f"--tensor-model-parallel-size {tp}",
            f"--pipeline-model-parallel-size {pp}",
            f"--micro-batch-size {micro_batch}",
            f"--global-batch-size {batch_size}",
            f"--seq-length {seq_length}",
            "--use-flash-attn",
            "--overlap-grad-reduce",
            "--overlap-param-gather",
        ]
        
        if precision == "bf16":
            args.append("--bf16")
        elif precision == "fp16":
            args.append("--fp16")
        
        return args
    
    def _generate_environment_variables(
        self,
        num_nodes: int,
        gpus_per_node: int,
        master_addr: str,
        master_port: int,
        tp: int,
    ) -> Dict[str, str]:
        """Generate recommended environment variables."""
        
        env = {
            # Distributed
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": str(master_port),
            "WORLD_SIZE": str(num_nodes * gpus_per_node),
            
            # NCCL
            "NCCL_DEBUG": "WARN",
            "NCCL_TIMEOUT": "1800",
            "NCCL_IB_TIMEOUT": "23",
            
            # CUDA
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            
            # Performance
            "OMP_NUM_THREADS": "8",
            "TOKENIZERS_PARALLELISM": "false",
        }
        
        # NVLink optimizations for TP
        if tp > 1:
            env["NCCL_NVLS_ENABLE"] = "1"
            env["NCCL_P2P_LEVEL"] = "NVL"
        
        return env
    
    def _generate_torchrun_command(
        self,
        num_nodes: int,
        gpus_per_node: int,
        master_addr: str,
        master_port: int,
        tp: int,
        pp: int,
    ) -> str:
        """Generate torchrun launch command."""
        
        cmd = f"""torchrun \\
    --nnodes={num_nodes} \\
    --nproc_per_node={gpus_per_node} \\
    --rdzv_backend=c10d \\
    --rdzv_endpoint={master_addr}:{master_port} \\
    train.py \\
    --tensor-parallel-size {tp} \\
    --pipeline-parallel-size {pp}"""
        
        return cmd
    
    def _generate_deepspeed_command(
        self,
        num_nodes: int,
        gpus_per_node: int,
        master_addr: str,
    ) -> str:
        """Generate DeepSpeed launch command."""
        
        cmd = f"""deepspeed \\
    --num_nodes={num_nodes} \\
    --num_gpus={gpus_per_node} \\
    --master_addr={master_addr} \\
    train.py \\
    --deepspeed_config ds_config.json"""
        
        return cmd
    
    def _generate_training_script(
        self,
        model_name: str,
        precision: str,
        gradient_checkpointing: bool,
        learning_rate: float,
        tp: int,
        pp: int,
    ) -> str:
        """Generate training script template."""
        
        script = f'''#!/usr/bin/env python3
"""
Auto-generated training script for {model_name}

Generated by Parallelism Strategy Advisor
"""

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

def main():
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        "{model_name}",
        torch_dtype=torch.{"bfloat16" if precision == "bf16" else "float16"},
        {"use_flash_attention_2=True," if precision in ["bf16", "fp16"] else ""}
    )
    
    {"# Enable gradient checkpointing" if gradient_checkpointing else ""}
    {"model.gradient_checkpointing_enable()" if gradient_checkpointing else ""}
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate={learning_rate},
        num_train_epochs=1,
        {"bf16=True," if precision == "bf16" else "fp16=True,"}
        logging_steps=10,
        save_strategy="steps",
        save_steps=1000,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  # Define your dataset
    )
    
    # Train
    trainer.train()

if __name__ == "__main__":
    import os
    main()
'''
        return script
    
    def export_to_files(
        self,
        config: ExportedConfig,
        output_dir: str,
    ) -> Dict[str, str]:
        """Export configuration to files."""
        
        os.makedirs(output_dir, exist_ok=True)
        files_created = {}
        
        # DeepSpeed config
        if config.deepspeed_config:
            ds_path = os.path.join(output_dir, "ds_config.json")
            with open(ds_path, "w") as f:
                json.dump(config.deepspeed_config, f, indent=2)
            files_created["deepspeed_config"] = ds_path
        
        # Environment script
        if config.environment_variables:
            env_path = os.path.join(output_dir, "env.sh")
            with open(env_path, "w") as f:
                f.write("#!/bin/bash\n# Environment variables for distributed training\n\n")
                for key, value in config.environment_variables.items():
                    f.write(f'export {key}="{value}"\n')
            files_created["environment_script"] = env_path
        
        # Training script
        if config.training_script:
            script_path = os.path.join(output_dir, "train.py")
            with open(script_path, "w") as f:
                f.write(config.training_script)
            files_created["training_script"] = script_path
        
        # Launch script
        launch_path = os.path.join(output_dir, "launch.sh")
        with open(launch_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("# Launch script for distributed training\n\n")
            f.write("source env.sh\n\n")
            if config.torchrun_command:
                f.write("# Option 1: torchrun\n")
                f.write(config.torchrun_command + "\n\n")
            if config.deepspeed_command:
                f.write("# Option 2: DeepSpeed\n")
                f.write(f"# {config.deepspeed_command}\n")
        files_created["launch_script"] = launch_path
        
        return files_created


def export_training_config(
    model_config: Dict[str, Any],
    hardware_config: Dict[str, Any],
    parallelism_config: Dict[str, Any],
    training_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Export a complete training configuration.
    """
    
    training = training_config or {}
    
    exporter = ConfigExporter()
    config = exporter.export_full_config(
        model_name=model_config.get("name", "model"),
        model_params_b=model_config.get("parameters_billions", 70),
        num_nodes=hardware_config.get("num_nodes", 1),
        gpus_per_node=hardware_config.get("gpus_per_node", 8),
        tp=parallelism_config.get("tensor_parallel", 1),
        pp=parallelism_config.get("pipeline_parallel", 1),
        dp=parallelism_config.get("data_parallel", 8),
        batch_size=training.get("batch_size", 256),
        seq_length=model_config.get("max_sequence_length", 4096),
        precision=training.get("precision", "bf16"),
        optimizer=training.get("optimizer", "adamw"),
        learning_rate=training.get("learning_rate", 1e-4),
        gradient_checkpointing=training.get("gradient_checkpointing", True),
        zero_stage=training.get("zero_stage", 2),
    )
    
    return config.to_dict()



